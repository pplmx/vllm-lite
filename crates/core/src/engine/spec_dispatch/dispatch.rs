//! Top-level speculative dispatch.
//!
//! [`Engine::step_speculative_inner`](super::super::Engine::step_speculative_inner)
//! is the speculative counterpart to `Engine::step`. It:
//!
//! 1. Builds the batch from the scheduler.
//! 2. Warms up draft KV caches after prefill.
//! 3. Generates draft tokens (per-seq via resolver, or batched legacy).
//! 4. Verifies drafts against target logits, accepting/rejecting.
//! 5. Rolls back KV cache for rejected drafts.
//! 6. Updates the scheduler with the produced tokens.
//! 7. Records speculative metrics (efficiency, accuracy, per-request rates).

use crate::error::Result;
use vllm_traits::{BatchPhase, FinishReason, SampledToken, SeqId};

impl crate::engine::Engine {
    /// Speculative decode step (called from `Engine::step` when speculative mode is on).
    ///
    /// **P36 v0.3 wire-type follow-up engine wire-through:** returns
    /// `Vec<(SeqId, SampledToken)>`. Speculative-accepted draft
    /// tokens carry a placeholder `SampledToken` with
    /// `logprob = 0.0` + `top_logprobs = vec![]` (the logprob of an
    /// accepted draft would require re-running the target forward
    /// pass at each accepted position, which is non-trivial;
    /// computed-logprob for speculative accepted tokens is deferred
    /// to a future iteration). The bonus token (sampled by the
    /// verifier) carries the full `SampledToken` from
    /// `sample_one_with_params`. The HTTP handler detects the
    /// placeholder via `logprob == 0.0 && top_logprobs.is_empty()`
    /// and suppresses `ChatChoice::logprobs` output for any
    /// sequence containing one.
    pub(crate) fn step_speculative_inner(
        &mut self,
        max_draft: usize,
    ) -> Result<Vec<(SeqId, SampledToken)>> {
        let start = std::time::Instant::now();
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        // Warmup draft KV cache after prefill (Plans 17.4-A, 17.4-E)
        if batch.phase == BatchPhase::Prefill
            && self.speculative_mode
            && let Err(e) = self.warmup_draft_kv(&batch)
        {
            tracing::warn!(error = %e, "Draft warmup failed, continuing without warmup");
        }

        let draft_outputs = if self.draft_resolver.is_some() {
            // v18.0 per-request dispatch: resolve each seq's draft via the
            // resolver, then run draft generation per-seq. Mixed-routing
            // (RTE-03) and FALL-02 (runtime errors) live here.
            // `generate_per_seq_drafts` always returns Ok — per-seq errors
            // are caught internally and degrade the affected sequence. If
            // a future batch-wide failure mode is added, restore the Err
            // arm here to fall back to non-speculative decode.
            self.generate_per_seq_drafts(&batch, max_draft)
        } else {
            self.generate_batched_drafts(&batch, max_draft)?
        };

        let (verified, accepted_counts) =
            self.verify_draft_tokens_logits(&batch, &draft_outputs)?;

        // Roll back KV cache for rejected drafts
        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            let drafts = &draft_outputs[i];
            let accepted = accepted_counts[i];
            let rejected = drafts.len().saturating_sub(accepted);
            if rejected > 0 {
                self.scheduler.memory_rollback(*seq_id, rejected);
            }
        }

        // H-16 (PERF-05): pre-size `results` to the verified-sequence
        // count so the per-iteration push below does not reallocate.
        let mut results = Vec::with_capacity(verified.len());
        for (seq_id, sampled) in &verified {
            if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.try_send(sampled.clone());
            }
            results.push((*seq_id, sampled.clone()));
        }

        // Multi-token scheduler input tracking (Plan 17.1-E)
        let seq_ids: Vec<SeqId> = results.iter().map(|(id, _)| *id).collect();
        let sampled: Vec<SampledToken> = results.iter().map(|(_, s)| s.clone()).collect();
        // Build input_counts from accepted counts + 1 bonus token
        let input_counts: Vec<usize> = accepted_counts
            .iter()
            .map(|&accepted| accepted + 1) // accepted drafts + the target token
            .collect();
        self.scheduler.update(&seq_ids, &sampled, &input_counts);

        // P38 v0.3 wire-type engine wire-through: stop-sequence
        // finalization. Must run after `scheduler.update` (so
        // `seq.tokens` includes the new tokens) and after the token-send
        // loop above (so matched tokens reach the client).
        self.finalize_stop_sequences(&batch);

        // Track accuracy in adaptive decoder and record adjustment events
        if let Some(ref mut decoder) = self.adaptive_decoder {
            let total_draft: usize = draft_outputs.iter().map(std::vec::Vec::len).sum();
            let total_accepted: usize = accepted_counts.iter().sum();
            if decoder.record_verification(total_draft, total_accepted) {
                self.scheduler.metrics.record_speculative_adjustment();
            }
        }

        // Record speculative efficiency metric (Plan 17.4-F / MTRC-02)
        let total_draft: usize = draft_outputs.iter().map(std::vec::Vec::len).sum();
        let total_accepted: usize = accepted_counts.iter().sum();
        let total_tokens_step = total_draft + total_accepted;
        if total_tokens_step > 0 {
            // invariant: draft/accepted counts are bounded per-step; precision loss
            // is acceptable for the efficiency ratio metric.
            #[allow(clippy::cast_precision_loss)]
            let efficiency = total_draft as f64 / total_tokens_step as f64;
            self.scheduler
                .metrics
                .record_speculative_efficiency(efficiency);
        }

        // Record per-request acceptance rate (Plan 17.4-F / MTRC-01)
        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            let seq_drafts = draft_outputs[i].len();
            let seq_accepted = accepted_counts[i];
            self.scheduler
                .metrics
                .record_per_request_acceptance(*seq_id, seq_accepted, seq_drafts);
        }

        let finished = self.scheduler.finished_sequences();
        for seq in &finished {
            // Tell the handler the sequence stopped, then drop the
            // matching token channel. Sequences finalized above via
            // `finalize_stop_sequences` already had their txs removed
            // (idempotent `remove` → no-op), so this second pass only
            // affects max_tokens completions.
            self.finalize_finished(seq.id, FinishReason::Length);
            self.scheduler.metrics.remove_per_request(seq.id);
        }
        self.scheduler.clear_finished();

        if !batch.seq_ids.is_empty() {
            let total_tokens: usize = batch.input_tokens.iter().map(std::vec::Vec::len).sum();
            self.scheduler
                .metrics
                .record_tokens(u64::try_from(total_tokens).unwrap_or(0));
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

        Ok(results)
    }
}
