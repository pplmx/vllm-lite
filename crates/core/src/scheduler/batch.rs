//! `Batch` value type: the per-step plan handed to the model layer, with token ids, block ids, and per-sequence metadata.
//!
//! Constructed by `BatchComposer` (`scheduler/batch_composer/`); the
//! model layer's `forward` consumes it as-is.
use crate::error::Result;
use crate::sampling::sample_batch_with_params;
use crate::sync::lock_mutex;
use vllm_traits::{BatchOutput, FinishReason, SampledToken, SeqId, TokenId};

/// Extract the last `vocab_size` logits per sequence (the "next" token position).
///
/// `forward_logits` returns one `Vec<f32>` per sequence; for decode each is
/// `1 * vocab`, for prefill each is `num_prompt_tokens * vocab`. The next
/// token always comes from the last position's logits.
fn extract_per_seq_logits(logits_list: &[Vec<f32>], vocab_size: usize) -> Vec<Vec<f32>> {
    logits_list
        .iter()
        .map(|seq_logits| {
            let start = seq_logits.len().saturating_sub(vocab_size);
            seq_logits[start..].to_vec()
        })
        .collect()
}

impl crate::engine::Engine {
    /// Regular (non-speculative) decode step.
    ///
    /// ARCH-02 (technical due diligence): previously this function
    /// called `model.forward`, which chose the next token greedily
    /// inside the model layer regardless of any per-request sampling
    /// parameters the HTTP layer had accepted. The seam is now:
    ///
    /// ```text
    /// forward_logits → take last-position logits per seq →
    /// sample_batch_with_params(batch.sampling_params)
    /// ```
    ///
    /// `forward` is still available for callers (and tests) that want
    /// the legacy greedy path, but the engine's hot path no longer uses
    /// it.
    ///
    /// **P36 v0.3 wire-type follow-up engine wire-through:** returns
    /// `Vec<(SeqId, SampledToken)>` instead of `Vec<(SeqId, TokenId)>`
    /// so the per-sequence response channel can carry the sampled
    /// token's `logprob` + `top_logprobs` alongside the token itself.
    pub(crate) fn step_regular(&mut self) -> Result<Vec<(SeqId, SampledToken)>> {
        let start = std::time::Instant::now();
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = batch.seq_ids.len();
        let total_tokens: usize = batch.input_tokens.iter().map(std::vec::Vec::len).sum();

        tracing::debug!(
            batch_size = batch_size,
            total_tokens = total_tokens,
            is_prefill = ?batch.is_prefill,
            phase = ?batch.phase,
            "Engine step: processing batch"
        );

        tracing::debug!(
            seq_count = batch.seq_ids.len(),
            total_input_tokens = batch.total_tokens,
            "Processing batch"
        );

        // ARCH-02: switch from `model.forward` (greedy internally) to
        // `model.forward_logits` + engine-side sampling. The per-seq
        // params now ride along on `Batch`.
        // Acquire the model lock only for the forward pass; release it
        // before sampling so other workers can access the model.
        let (logits_list, vocab_size) = {
            let mut model = lock_mutex(&self.target_model)?;
            let logits_list = model.forward_logits(
                &batch.seq_ids,
                &batch.input_tokens,
                &batch.positions,
                &batch.kv_block_ids,
                &batch.num_computed_tokens,
                &batch.is_prefill,
            )?;
            let vocab_size = model.vocab_size();
            // Release the model lock immediately after the last use so
            // sampling and output construction don't hold it needlessly.
            drop(model);
            (logits_list, vocab_size)
        };
        let per_seq: Vec<Vec<f32>> = extract_per_seq_logits(&logits_list, vocab_size);

        // Gather seen tokens (already-generated portion of each sequence)
        // so `repeat_penalty` can penalise them. Prefill yields an empty
        // seen-set, which makes repeat-penalty a no-op as expected.
        let seen_tokens: Vec<Vec<TokenId>> = batch
            .seq_ids
            .iter()
            .map(|sid| {
                self.scheduler
                    .get_sequence(*sid)
                    .map(|s| s.tokens[s.prompt_len..].to_vec())
                    .unwrap_or_default()
            })
            .collect();
        let next_tokens = sample_batch_with_params(&per_seq, &batch.sampling_params, &seen_tokens);
        let logits_per_seq = per_seq;

        let output = BatchOutput {
            seq_ids: batch.seq_ids.clone(),
            next_tokens,
        };

        tracing::debug!(
            output_tokens = output.next_tokens.len(),
            first_output = output.next_tokens.first().map(|s| s.token),
            "Engine step: output tokens"
        );

        let input_counts: Vec<usize> = batch.input_tokens.iter().map(std::vec::Vec::len).collect();

        // RIL ISS-053: a mid-chunk prefill's predicted token is stale (the
        // real next prompt token is re-fed on the next chunk) — it must
        // reach neither the client response channel nor the step result.
        // Staleness is derived from the batch slice against the sequence's
        // full prompt, read while the sequence is still in `running`
        // (`update` requeues partial prefills before the send below).
        let stale_mask = batch
            .seq_ids
            .iter()
            .enumerate()
            .map(|(i, sid)| {
                batch.is_prefill[i]
                    && self.scheduler.get_sequence(*sid).is_some_and(|s| {
                        batch.num_computed_tokens[i] + batch.input_tokens[i].len() < s.prompt_len
                    })
            })
            .collect::<Vec<_>>();

        self.scheduler
            .update(&batch.seq_ids, &output.next_tokens, &input_counts);

        let mut disconnected = Vec::new();
        let results = self.send_and_collect_results(
            &batch.seq_ids,
            &output.next_tokens,
            &stale_mask,
            &mut disconnected,
        );

        // RIL ISS-110: cancel sequences whose response channel closed
        // mid-step (client disconnect) so they don't burn the rest of their
        // `max_tokens` budget into a dead channel.
        self.cancel_on_closed_channels(&disconnected);

        // Keep `logits_per_seq` alive through this point for structural
        // symmetry with the CUDA-Graph path (P36); it is not consumed.
        let _ = logits_per_seq;
        self.finalize_and_record(&batch, &results, start);

        Ok(results)
    }

    /// Send one sampled token to its sequence's response channel, watching
    /// for a full mailbox.
    ///
    /// RIL ISS-074 / TASK-089: a `Full` channel means the consumer drains
    /// slower than generation — the token is lost to the client stream.
    /// Pre-fix the `let _ = try_send(...)` ignored the error, so the gap was
    /// a permanent silent hole. Now it is logged AND counted so operators can
    /// detect the loss on `/metrics`.
    ///
    /// RIL ISS-110: a `Closed` channel (receiver dropped — client gone) is
    /// the engine's ONLY disconnect signal on the non-streaming paths (they
    /// build no `CancelOnDrop` guard), so the sequence is recorded in
    /// `disconnected` and the caller cancels it — pre-fix it was a silent
    /// no-op and an aborted request generated into a closed channel for its
    /// whole `max_tokens` budget. `Ok` leaves the token in the stream.
    pub(crate) fn try_send_token(
        &self,
        seq_id: SeqId,
        sampled: &SampledToken,
        tx: &tokio::sync::mpsc::Sender<SampledToken>,
        disconnected: &mut Vec<SeqId>,
    ) {
        match tx.try_send(sampled.clone()) {
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!(
                    seq_id = %seq_id,
                    token = %sampled.token,
                    "response channel full; dropped token from client stream (consumer too slow)"
                );
                self.scheduler.metrics.record_dropped_token();
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!(
                    seq_id = %seq_id,
                    token = %sampled.token,
                    "response channel closed; cancelling sequence (client disconnected)"
                );
                disconnected.push(seq_id);
            }
            Ok(()) => {}
        }
    }

    /// Send sampled tokens to each sequence's response channel and collect
    /// them into the return vec. Idempotent: if a channel is missing (e.g.
    /// the sequence was already finalized) the token is still returned.
    ///
    /// `stale[i] == true` marks a mid-chunk prefill whose predicted token is
    /// NOT generated output (RIL ISS-053): it is neither sent to the channel
    /// nor included in the returned results, so the client stream, step
    /// results, and output-token metrics all see only real generated tokens.
    /// Send sampled tokens to each sequence's response channel and collect
    /// them into the return vec. Idempotent: if a channel is missing (e.g.
    /// the sequence was already finalized) the token is still returned.
    ///
    /// `stale[i] == true` marks a mid-chunk prefill whose predicted token is
    /// NOT generated output (RIL ISS-053): it is neither sent to the channel
    /// nor included in the returned results, so the client stream, step
    /// results, and output-token metrics all see only real generated tokens.
    ///
    /// RIL ISS-110: sequences whose channel returned `Closed` are appended
    /// to `disconnected` so the caller can cancel them (client disconnect).
    pub(crate) fn send_and_collect_results(
        &self,
        seq_ids: &[SeqId],
        next_tokens: &[SampledToken],
        stale: &[bool],
        disconnected: &mut Vec<SeqId>,
    ) -> Vec<(SeqId, SampledToken)> {
        let mut results = Vec::with_capacity(seq_ids.len());
        for ((seq_id, sampled), &is_stale) in seq_ids.iter().zip(next_tokens.iter()).zip(stale) {
            if is_stale {
                tracing::debug!(
                    seq_id = %seq_id,
                    token = %sampled.token,
                    "Drawing stale mid-chunk prediction (not real output); not emitting"
                );
                continue;
            }
            tracing::debug!(seq_id = %seq_id, token = %sampled.token, "Sending token to channel");
            if let Some(tx) = self.response_txs.get(seq_id) {
                self.try_send_token(*seq_id, sampled, tx, disconnected);
            }
            results.push((*seq_id, sampled.clone()));
        }
        results
    }

    /// Finalize stop-sequence and length-completed sequences, clear finished
    /// entries, and record batch metrics + latency.
    ///
    /// `results` is the emitted per-sequence token list (stale mid-chunk
    /// prefill predictions already masked out) — its length is the number of
    /// tokens actually generated this step, which is what `tokens_total`
    /// ("Total tokens generated") counts (RIL ISS-083). Pre-fix the
    /// regular and speculative paths recorded the sum of *input* token
    /// lengths instead, so a prefill over-counted the metric by the full
    /// prompt length and the CUDA-graph path (which correctly counted
    /// emitted results) disagreed with them for the same workload.
    pub(crate) fn finalize_and_record(
        &mut self,
        batch: &vllm_traits::Batch,
        results: &[(vllm_traits::SeqId, vllm_traits::SampledToken)],
        start: std::time::Instant,
    ) {
        // P38 v0.3 wire-type engine wire-through: stop-sequence
        // finalization. Runs after `scheduler.update` (so `seq.tokens`
        // includes the new token) and after the token-send loop above
        // (so the matched token reaches the client before the channel
        // is dropped). Matched sequences get `FinishReason::Stop`.
        self.finalize_stop_sequences(batch);

        let finished = self.scheduler.finished_sequences();
        for seq in &finished {
            tracing::debug!(seq_id = seq.id, "Sequence finished");
            // Tell the handler *why* the channel is closing before
            // dropping it. Sequences finalized above via
            // `finalize_stop_sequences` already had their txs removed
            // (idempotent `remove` → no-op), so this second pass only
            // affects max_tokens completions.
            self.finalize_finished(seq.id, FinishReason::Length);
        }
        self.scheduler.clear_finished();

        if !batch.seq_ids.is_empty() {
            // RIL ISS-083: count generated (output) tokens, not input tokens.
            self.scheduler
                .metrics
                .record_tokens(u64::try_from(results.len()).unwrap_or(0));
            // RIL ISS-095: split the step into prefill (input positions
            // processed) vs decode (1 token per decode seq) so the
            // prefill/decode throughput gauges are live, not pinned at 0.
            self.scheduler.metrics.record_batch_phase_tokens(batch);
            self.scheduler
                .metrics
                .record_batch_size(batch.seq_ids.len());
        }

        // invariant: elapsed millis fits in f64 mantissa (< 2^52 ms ≈ 142 years).
        #[allow(clippy::cast_precision_loss)]
        let elapsed = start.elapsed().as_millis() as f64;
        if elapsed > 0.0 {
            self.scheduler.metrics.record_latency(elapsed);
        }

        // suppress unused-variable lint when sampling diagnostics are
        // stripped in release builds; keep the structure symmetric with
        // the CUDA-Graph path for future logging.
    }

    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// Run one scheduling step (regular or speculative depending on engine configuration).
    ///
    /// **P36 v0.3 wire-type follow-up engine wire-through:** returns
    /// `Vec<(SeqId, SampledToken)>` so callers can surface
    /// per-token logprobs without re-running the softmax.
    pub fn step(&mut self) -> Result<Vec<(SeqId, SampledToken)>> {
        let result = if self.speculative_mode
            && (self.draft_model.is_some() || self.draft_resolver.is_some())
        {
            let max_draft = self.adaptive_decoder.as_ref().map_or(
                self.max_draft_tokens,
                super::super::speculative::adaptive::AdaptiveSpeculativeDecoder::current_max_draft_tokens,
            );
            self.step_speculative_inner(max_draft)
        } else {
            self.step_regular()
        };
        // RIL ISS-045: a forward error propagates with `?` before
        // `update()` runs, leaving freshly-admitted Prefilling sequences
        // stranded in `running` (never re-scheduled, KV pinned, scheduler
        // spinning, client hang). Roll them back before surfacing the
        // error so a transient failure cannot permanently wedge the
        // engine; the request retries on a later round.
        if result.is_err() {
            self.scheduler.requeue_stuck_prefills();
        }
        self.record_step_metrics();
        result
    }

    /// Record post-step observability so the `/metrics` snapshot reflects
    /// current allocator state without a `GetMetrics` round-trip (RIL
    /// ISS-100). `kv_cache_usage_percent` was pinned at 0.000 under a
    /// /metrics-only scrape because the sole production writer
    /// (`record_kv_cache_usage`) lived inside the `EngineMessage::GetMetrics`
    /// arm of `run()` — the /metrics handler renders the snapshot directly.
    /// `get_kv_cache_usage` is O(1) (an allocator counter), so calling it
    /// every step is free. Called from every step path (regular,
    /// speculative, and the CUDA-graph `step_with_graph`), on success and
    /// error alike — the allocator state is whatever the step left behind.
    ///
    /// RIL ISS-120: `prefix_cache_nodes` had the SAME GetMetrics-only
    /// writer (run.rs) and was omitted from the ISS-100 fix, so it pinned
    /// at 0 under a /metrics-/debug-metrics-only scrape. `RadixTree::len`
    /// is O(1) (a stored `entry_count` counter), so refreshing it every
    /// step is free too.
    pub(crate) fn record_step_metrics(&self) {
        let (used, total) = self.scheduler.get_kv_cache_usage();
        self.scheduler.metrics.record_kv_cache_usage(used, total);
        self.scheduler
            .metrics
            .record_prefix_cache_nodes(self.scheduler.prefix_cache().len());
    }
}
