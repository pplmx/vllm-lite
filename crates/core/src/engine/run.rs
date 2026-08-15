//! Engine main run loop: tick → schedule → forward → update, plus the `has_pending` accessor.
//!
//! The loop runs on a dedicated OS thread; cancellation is
//! cooperatively checked at the top of each tick.

// Sub-module for the main run loop and has_pending accessor on Engine.
// See mod.rs for the Engine struct definition.

use crate::engine::Engine;
use crate::error::EngineError;
use crate::sync::lock_mutex;
use crate::types::EngineMessage;
use std::panic::{AssertUnwindSafe, catch_unwind};
use tokio::sync::mpsc;
use tracing::error;

impl Engine {
    /// Run the engine's main loop, draining `msg_rx` and stepping the
    /// scheduler until a `Shutdown` message is received.
    ///
    /// The loop is single-threaded and non-async: incoming messages are
    /// drained with `try_recv`, then one model step is executed if any
    /// sequence is pending, then the thread sleeps for the duration
    /// produced by the current [`crate::engine::SleepPolicy`]. This pattern gives
    /// back-pressure-friendly batching without an async runtime.
    ///
    /// This call blocks the current thread and never returns except on
    /// `EngineMessage::Shutdown`. Spawn it on a dedicated worker thread
    /// (the `vllm-server` crate does this for you).
    pub fn run(&mut self, mut msg_rx: mpsc::Receiver<EngineMessage>) {
        let mut step_count = 0u64;
        loop {
            while let Ok(msg) = msg_rx.try_recv() {
                match msg {
                    EngineMessage::AddRequest {
                        request,
                        response_tx,
                        seq_id_tx,
                        finish_reason_tx,
                        request_id,
                    } => self.handle_add_request(
                        *request,
                        response_tx,
                        seq_id_tx,
                        finish_reason_tx,
                        request_id.as_ref(),
                    ),
                    EngineMessage::CancelRequest { seq_id } => {
                        // Production-readiness recommendation: when an
                        // HTTP client disconnects mid-stream, the
                        // handler sends CancelRequest to release the
                        // sequence's KV blocks and response channel.
                        // Unknown seq_id is a no-op (race with natural
                        // completion) — `cancel_request` returns
                        // false and we drop the message silently.
                        let _ = self.cancel_request(seq_id);
                    }
                    EngineMessage::GetMetrics { response_tx } => {
                        let (used, total) = self.scheduler.get_kv_cache_usage();
                        self.scheduler.metrics.record_kv_cache_usage(used, total);
                        self.scheduler
                            .metrics
                            .record_prefix_cache_nodes(self.scheduler.prefix_cache().len());
                        let _ = response_tx.send(self.scheduler.metrics.snapshot());
                    }
                    EngineMessage::GetEmbeddings {
                        input_tokens,
                        response_tx,
                    } => self.handle_get_embeddings(&input_tokens, &response_tx),
                    EngineMessage::Shutdown => return,
                }
            }

            if self.scheduler.has_pending() {
                step_count += 1;
                // RIL ISS-046: the engine thread is single-threaded; a panic
                // inside the target model's forward (a GPU kernel fault, an
                // assert in an architecture impl, an aborted candle op) would
                // otherwise unwind `run()` and kill the whole server. The
                // speculative draft path already guards its forwards with
                // catch_unwind; guard the canonical step the same way.
                // `AssertUnwindSafe` is required because the closure captures
                // `&mut self` (the standard `UnwindSafe` bound refuses it) —
                // the deliberate contract is "panics in foreign backend code
                // become step errors, never process death". catch_unwind
                // cannot catch aborts/double-panics: a truly crashing backend
                // still terminates the process.
                let result = catch_unwind(AssertUnwindSafe(|| {
                    if self.cuda_graph_enabled() && !self.speculative_mode {
                        self.step_with_graph()
                    } else {
                        self.step()
                    }
                }))
                .unwrap_or_else(|_| {
                    Err(EngineError::ModelError(
                        "engine step panicked inside the model forward; caught and recovered"
                            .to_string(),
                    ))
                });
                if let Err(e) = result {
                    self.error_count += 1;
                    self.last_error = Some(e.to_string());
                    error!(step = step_count, error = %e, "Engine step error");
                    // ISS-045: a panic bypasses `step()`'s own recovery, so
                    // the step may have stranded Prefilling/Waiting sequences
                    // in `running` — release and re-queue them (idempotent —
                    // a no-op when `step` already cleaned up).
                    self.scheduler.requeue_stuck_prefills();
                } else {
                    // A successful step clears the error backoff marker so a
                    // transient failure doesn't leave the loop sleeping at
                    // the max interval forever (RIL ISS-046).
                    self.last_error = None;
                }
            }

            // RIL ISS-046: a step that errored (caught panic, or a model
            // forward that now returns LockPoisoned after its mutex was
            // poisoned by the panic) must not be re-attempted at the 1 ms
            // busy interval forever — that would hot-loop a dead model.
            // Sleep the max interval so the loop stays responsive to new
            // work / shutdown but stops burning CPU on a broken forward.
            // `SleepPolicy::next_interval` with `has_work = true` returns
            // the 1 ms base; the error path overrides it.
            let interval = if self.last_error.is_some() {
                self.sleep_policy.max_interval
            } else {
                self.sleep_policy
                    .next_interval(self.scheduler.has_pending())
            };
            std::thread::sleep(std::time::Duration::from_millis(interval));
        }
    }

    /// Handle an `AddRequest` message: enter the request-id tracing span,
    /// admit the request, and reply on the `seq_id` / `finish_reason` channels.
    fn handle_add_request(
        &mut self,
        request: crate::types::Request,
        response_tx: tokio::sync::mpsc::Sender<vllm_traits::SampledToken>,
        seq_id_tx: Option<tokio::sync::oneshot::Sender<vllm_traits::SeqId>>,
        finish_reason_tx: Option<tokio::sync::oneshot::Sender<vllm_traits::FinishReason>>,
        request_id: Option<&String>,
    ) {
        // Production-readiness §6 (日志与追踪): when an HTTP handler
        // forwards a `request_id`, enter a tracing::info_span! so every
        // engine-side log line for this HTTP request carries the same
        // correlation id. When `request_id` is `None` (test fixtures,
        // non-HTTP callers) the span still enters, rendering as `null`.
        let _request_id_span = tracing::info_span!("engine.add_request", request_id = request_id);

        // P38: deref Box<Request> → Request (add_request public API
        // still takes Request by value). The caller treats seq_id 0 as
        // "do not bother cancelling" (rejection e.g. empty prompt).
        let seq_id = self.add_request(request, response_tx);
        if seq_id != 0
            && let Some(tx) = finish_reason_tx
        {
            // RIL ISS-076 / TASK-090: only track the finish-reason sender for
            // ADMITTED sequences. `add_request` rejects empty prompts by
            // returning 0; no sequence is ever id 0, so an insert under key 0
            // would never be removed by `finalize_finished` — a permanent
            // oneshot leak (and a hang for a direct engine caller awaiting
            // `finish_reason_rx`). The server layer rejects empty prompts
            // first; this guard keeps the map clean for direct-engine callers
            // too.
            self.finish_reason_txs.insert(seq_id, tx);
        }
        if let Some(tx) = seq_id_tx {
            let _ = tx.send(seq_id);
        }
    }

    /// Handle a `GetEmbeddings` message: call `model.embed` and send the
    /// result (or log the error).
    fn handle_get_embeddings(
        &self,
        input_tokens: &[Vec<vllm_traits::TokenId>],
        response_tx: &tokio::sync::mpsc::UnboundedSender<Vec<Vec<f32>>>,
    ) {
        let positions: Vec<Vec<usize>> = input_tokens
            .iter()
            .map(|tokens| (0..tokens.len()).collect())
            .collect();
        match lock_mutex(&self.target_model)
            .and_then(|mut model| model.embed(input_tokens, &positions).map_err(Into::into))
        {
            Ok(embeddings) => {
                let _ = response_tx.send(embeddings);
            }
            Err(e) => {
                error!(error = %e, "Embeddings error");
            }
        }
    }

    /// Returns `true` if the scheduler currently has at least one waiting or
    /// running sequence. Useful for tests and external monitors that want to
    /// know whether calling [`Engine::step`] would do meaningful work.
    pub fn has_pending(&self) -> bool {
        self.scheduler.has_pending()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;
    use tokio::sync::mpsc;
    use vllm_testing::StubModel;

    /// RIL ISS-076 / TASK-090: an admission-rejected request (empty prompt
    /// -> `add_request` returns `seq_id` 0) must NOT leave an orphaned
    /// `finish_reason_tx` in the engine's map. Pre-fix `handle_add_request`
    /// inserted it at key 0 unconditionally, and no sequence is ever id 0,
    /// so `finalize_finished` never removes it — the oneshot sender leaked
    /// forever and a direct engine caller awaiting `finish_reason_rx` would
    /// hang. Server layers reject empty prompts first, so this is a
    /// direct-engine-only leak; keep it from poisoning the map regardless.
    #[test]
    fn test_handle_add_request_rejected_admission_leaks_no_finish_reason() {
        let mut engine = Engine::new(StubModel::returning(42), None);
        let (tx, _rx) = mpsc::channel(64);
        let (reason_tx, _reason_rx) = tokio::sync::oneshot::channel();
        let (seq_id_tx, _seq_id_rx) = tokio::sync::oneshot::channel();

        // Empty prompt: `add_request` rejects it and returns 0.
        engine.handle_add_request(
            Request::new(7, vec![], 5),
            tx,
            Some(seq_id_tx),
            Some(reason_tx),
            None,
        );

        assert!(
            engine.finish_reason_txs.is_empty(),
            "a rejected admission must not leave an orphaned finish_reason_tx \
             under key 0 (RIL ISS-076)"
        );
    }
}
