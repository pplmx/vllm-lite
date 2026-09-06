//! Engine lifecycle methods: health checks, error state, cancel handling, and `add_request`.
//!
//! The main run loop is `engine/run.rs`; this file is the supporting
//! surface that callers and tests poke at directly.

// Sub-module for health, error, cancel, and add_request methods on Engine.
// See mod.rs for the Engine struct definition.

use crate::engine::Engine;
use crate::types::Request;
use tokio::sync::mpsc;
use vllm_traits::{Batch, FinishReason, SampledToken, SeqId};

impl Engine {
    /// Returns `true` if the engine is considered healthy and ready to process
    /// requests. The current heuristic treats an engine as unhealthy once it has
    /// accumulated 10 or more errors — callers should stop issuing new requests
    /// and may want to inspect [`Engine::get_last_error`] for diagnostics.
    pub const fn is_healthy(&self) -> bool {
        self.error_count < 10
    }

    /// Returns the most recent error message recorded by the engine, or `None`
    /// if no error has occurred since the engine was constructed.
    ///
    /// This is a lightweight accessor intended for health probes and CLI status
    /// reporting. It does not clear the error; subsequent errors overwrite the
    /// stored value.
    pub fn get_last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Check every sequence in `batch.seq_ids` for a matched
    /// `stop_token_sequences` suffix and finalize matches with
    /// [`FinishReason::Stop`].
    ///
    /// **Must be called after `scheduler.update()`** so `seq.tokens`
    /// includes the freshly generated token(s), and **after the token-send
    /// loop** so the matched token reaches the client before the response
    /// channel is dropped.
    ///
    /// The check uses the full `seq.tokens[prompt_len..]` slice (not just
    /// the current step's token) so stop sequences that span step
    /// boundaries are caught. For each match, `finish_sequence` marks the
    /// sequence finished in the scheduler (releasing KV blocks) and
    /// [`Self::finalize_finished`] delivers `FinishReason::Stop` to the
    /// HTTP handler.
    ///
    /// Extracted from `step_regular` so all three step paths — regular
    /// (`batch.rs`), speculative (`spec_dispatch/dispatch.rs`), and
    /// CUDA-graph (`graph_step.rs`) — share identical stop-detection
    /// logic. Pre-fix, only `step_regular` performed this check; the other
    /// two silently ignored `stop_token_sequences`, generating tokens past
    /// the stop point until `max_tokens` was hit.
    pub(crate) fn finalize_stop_sequences(&mut self, batch: &Batch) -> Vec<SeqId> {
        let mut newly_stopped: Vec<SeqId> = Vec::new();
        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            let Some(seq) = self.scheduler.get_sequence(*seq_id) else {
                continue;
            };
            // Suffix-match only — `matches_stop_sequences` compares the
            // TAIL of the slice, so borrow the generated region directly
            // instead of `to_vec()`-copying it. This runs on every step
            // of all three step paths for every sequence carrying stop
            // sequences; a full copy would be O(generated) per step
            // (quadratic over a request's lifetime) purely to feed a
            // suffix check. Slicing from `prompt_len` keeps the
            // prompt-exclusion semantic (a stop suffix in the prompt
            // must not match) — see the pinned regression test.
            let generated = &seq.tokens[seq.prompt_len..];
            // RIL ISS-075: the model's end-of-sentence token is a MODEL-level
            // stop signal that applies to every sequence regardless of its
            // per-request `stop_token_sequences`. Without it, a 'short'
            // answer burns the full `max_tokens` budget and reports
            // `FinishReason::Length` instead of `Stop`.
            let matches_eos = self
                .eos_token_id
                .is_some_and(|eos| generated.last() == Some(&eos));
            // Defensive `.get`: synthetic batches may carry an empty
            // `sampling_params` (the Batch docs call that "equivalent
            // to greedy decoding") — mirror the verifier path so a
            // seq/params length mismatch degrades instead of panicking.
            // Empty stops (fallthrough arm) make `matches_stop_sequences`
            // a no-op, so the EOS check above is unaffected.
            let stops: &[Vec<u32>] = match batch
                .sampling_params
                .get(i)
                .and_then(|p| p.stop_token_sequences.as_ref())
            {
                Some(s) if !s.is_empty() => s,
                _ => &[],
            };
            if matches_eos || crate::sampling::matches_stop_sequences(generated, stops) {
                newly_stopped.push(*seq_id);
            }
        }
        for seq_id in &newly_stopped {
            self.scheduler.finish_sequence(*seq_id);
            self.finalize_finished(*seq_id, FinishReason::Stop);
        }
        newly_stopped
    }

    /// Configure the model's end-of-sentence token id (RIL ISS-075).
    ///
    /// Called by the server after engine construction (same hook pattern as
    /// `configure_speculative`); the core default is `None` so engines
    /// constructed without a checkpoint (tests, mocks) keep the legacy
    /// run-to-`max_tokens` behavior.
    pub const fn set_eos_token_id(&mut self, eos_token_id: Option<u32>) {
        self.eos_token_id = eos_token_id;
    }

    /// Notify any registered handler of the [`FinishReason`] for `seq_id`,
    /// then drop both the finish-reason sender and the matching token
    /// response channel.
    ///
    /// Used by the regular and speculative step paths (`scheduler/batch.rs`,
    /// `engine/spec_dispatch/dispatch.rs`, `engine/graph_step.rs`) just
    /// before they drop `response_txs`. Centralising the helper keeps
    /// the three sites consistent — pre-fix each one dropped the
    /// channel without telling the handler *why*, and the HTTP layer
    /// hardcoded `finish_reason = "stop"` for every response.
    ///
    /// The `send` is best-effort: if the handler already dropped the
    /// oneshot (e.g. it gave up waiting after a client disconnect),
    /// the `Result` is ignored.
    pub(crate) fn finalize_finished(&mut self, seq_id: SeqId, reason: FinishReason) {
        // RIL ISS-082: balance the `record_request_start` from
        // `add_request` so the `requests_in_flight` gauge is live.
        // `finalize_finished` is the single finalization point for every
        // Stop / Length / Cancelled completion, so one decrement per
        // admitted request. The counter saturates at 0, so an unbalanced
        // call cannot wrap to u64::MAX.
        self.scheduler.metrics.record_request_end();
        if let Some(tx) = self.finish_reason_txs.remove(&seq_id) {
            let _ = tx.send(reason);
        }
        self.response_txs.remove(&seq_id);
        // RIL ISS-034: tell the target model the sequence is done so it can
        // drop per-sequence state (e.g. the Qwen3.5 hybrid GDN recurrent
        // map). Best-effort: a poisoned lock just skips the cleanup.
        if let Ok(mut model) = crate::sync::lock_mutex(&self.target_model) {
            model.on_sequence_finished(seq_id);
        }
        // Draft models (legacy single draft + registry-loaded external/self-
        // spec drafts) hold per-sequence state too (a Qwen3.5 hybrid used as
        // a draft leaks its GDN map the same way); notify them as well.
        if let Some(draft) = &self.draft_model
            && let Ok(mut backend) = draft.lock()
        {
            backend.on_sequence_finished(seq_id);
        }
        self.draft_registry.notify_sequence_finished(seq_id);
    }

    /// Cancel an in-flight or queued request identified by `seq_id`.
    ///
    /// Returns `true` if a request with that id was found and removed from the
    /// scheduler (and its response channel was dropped), `false` otherwise. The
    /// caller is responsible for sending any partial-result notifications to
    /// the client before invoking this method; once cancelled, no further
    /// tokens will be produced for the sequence.
    ///
    /// If the cancelled sequence had a `finish_reason_tx` registered
    /// (i.e. the HTTP layer asked for one), this method sends
    /// [`FinishReason::Cancelled`] through it before removing the
    /// channel — so the handler can distinguish "client cancelled"
    /// from a channel close that happened for some other reason.
    pub fn cancel_request(&mut self, seq_id: SeqId) -> bool {
        let canceled = self.scheduler.cancel_request(seq_id);
        if canceled {
            self.finalize_finished(seq_id, FinishReason::Cancelled);
            self.scheduler.metrics.remove_per_request(seq_id);
        }
        canceled
    }

    /// RIL ISS-110: cancel every sequence whose response channel returned
    /// `TrySendError::Closed` during the token-emission loop (the client
    /// dropped the receiver — a disconnect). `Closed` is the engine's ONLY
    /// disconnect signal on non-streaming paths (the HTTP layer builds no
    /// `CancelOnDrop` guard there), so pre-fix an aborted request silently
    /// generated into a closed channel for its entire `max_tokens` budget —
    /// burning CPU/tokens/KV per abandonment. Called by every step path
    /// (regular, speculative, CUDA-graph) right after their send loop.
    pub(crate) fn cancel_on_closed_channels(&mut self, disconnected: &[SeqId]) {
        for &seq_id in disconnected {
            let _ = self.cancel_request(seq_id);
        }
    }

    /// Submit a new generation request to the engine.
    ///
    /// The returned `SeqId` can be used with [`Engine::cancel_request`] or for
    /// correlation with tokens emitted on `response_tx`. Tokens generated by
    /// the model for this request will be sent on `response_tx` as they are
    /// produced; the channel must remain open for the lifetime of the request.
    ///
    /// If `req.prompt` is empty the request is rejected: `last_error` is set
    /// and `0` is returned. (Sequence id 0 is reserved as a sentinel for the
    /// "no request allocated" case.)
    ///
    /// **No `FinishReason` notification**: this method does not accept a
    /// finish-reason oneshot. Callers that need the OpenAI-correct
    /// `finish_reason` (HTTP streaming handlers) should send an
    /// `EngineMessage::AddRequest` over the engine mailbox instead — the
    /// message variant carries the optional `finish_reason_tx` field. Tests
    /// and other in-process callers that talk to the engine directly don't
    /// need it: when the response channel closes without a reason, the
    /// caller falls back to `"stop"`.
    ///
    /// **P36 v0.3 wire-type follow-up engine wire-through:**
    /// `response_tx` carries [`SampledToken`] (token + logprob +
    /// `top_logprobs`) instead of a bare `TokenId` so the HTTP layer
    /// can render `OpenAI`'s `choices[].logprobs` shape.
    pub fn add_request(&mut self, req: Request, response_tx: mpsc::Sender<SampledToken>) -> SeqId {
        // Validate prompt is not empty
        if req.prompt.is_empty() {
            self.last_error = Some("prompt cannot be empty".to_string());
            return 0;
        }

        // RIL ISS-038: stateful backends (Qwen3.5 hybrid GDN) cannot resume
        // from a prefix-cache hit — their recurrent state is not in the KV
        // cache — so route them through the no-prefix path (full prefill).
        let supports_prefix_caching = crate::sync::lock_mutex(&self.target_model)
            .map_or(true, |model| model.supports_prefix_caching());
        let seq_id = if supports_prefix_caching {
            self.scheduler.add_request(req)
        } else {
            self.scheduler.add_request_without_prefix_cache(req)
        };
        self.response_txs.insert(seq_id, response_tx);
        // RIL ISS-082: count ADMITTED requests only — a rejected admission
        // (`seq_id == 0`, e.g. an empty prompt that slipped through) never
        // reaches `finalize_finished`, so it must not get a start either.
        // `finalize_finished` balances this via `record_request_end`.
        if seq_id != 0 {
            self.scheduler.metrics.record_request_start();
        }
        seq_id
    }
}
