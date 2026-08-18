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
use vllm_traits::{Batch, BatchPhase, FinishReason, SampledToken, SeqId};

impl crate::engine::Engine {
    /// Speculative decode step (called from `Engine::step` when speculative mode is on).
    ///
    /// **P36 v0.3 wire-type follow-up engine wire-through:** returns
    /// `Vec<(SeqId, SampledToken)>`. Speculative-accepted draft
    /// tokens carry a placeholder `SampledToken` with
    /// `logprob = NEG_INFINITY` + `top_logprobs = vec![]` (the
    /// logprob of an accepted draft would require re-running the
    /// target forward pass at each accepted position, which is
    /// non-trivial; computed-logprob for speculative accepted tokens
    /// is deferred to a future iteration). The bonus token (sampled
    /// by the verifier) carries the full `SampledToken` from
    /// `sample_one_with_params`. The HTTP handler detects the
    /// placeholder via a non-finite logprob
    /// (`!logprob.is_finite()`) and suppresses
    /// `ChatChoice::logprobs` output for any sequence containing one.
    pub(crate) fn step_speculative_inner(
        &mut self,
        max_draft: usize,
    ) -> Result<Vec<(SeqId, SampledToken)>> {
        let start = std::time::Instant::now();
        let mut batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        self.ensure_verification_blocks(&mut batch, max_draft);

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

        // RIL ISS-027: rejected-draft KV blocks are NOT rolled back here.
        // The verification span's blocks are pre-allocated
        // (`ensure_verification_blocks`, RIL ISS-026) and the rejected
        // drafts' KV simply sits beyond `num_computed_tokens` — it is never
        // read and gets overwritten as the sequence grows. The old
        // `memory_rollback(rejected)` ran its block math against the
        // pre-step `num_computed_tokens`: for a resumed prefill (partial
        // prefix-cache hit) it rewound the count and released the matched
        // prefix block (the freed region is never recomputed → garbage KV),
        // and for decode it could free a block holding real prefix KV when
        // the rewind crossed a block boundary.

        // H-16 (PERF-05): pre-size `results` to the verified-sequence
        // count so the per-iteration push below does not reallocate.
        // RIL ISS-057: a mid-chunk prefill's predicted token(s) are stale —
        // the real next prompt token is re-fed on the next chunk, so nothing
        // produced while the sequence is still mid-prompt may reach the
        // client stream. `verified` is a flat token list (a sequence can
        // contribute several entries: accepted drafts + bonus/rejection), so
        // staleness is keyed by seq_id via its batch entry (start + chunk <
        // prompt_len), read while the sequence is still in `running`
        // (update_speculative requeues partial prefills afterwards). The
        // final prefill chunk (transitions to Decoding) and decode sequences
        // are NOT masked.
        let stale_by_seq: std::collections::HashMap<SeqId, bool> = batch
            .seq_ids
            .iter()
            .enumerate()
            .map(|(i, sid)| {
                let stale = batch.is_prefill[i]
                    && self.scheduler.get_sequence(*sid).is_some_and(|s| {
                        batch.num_computed_tokens[i] + batch.input_tokens[i].len() < s.prompt_len
                    });
                (*sid, stale)
            })
            .collect();

        // Emit verified tokens to their client channels (gating stale
        // mid-chunk prefill predictions, RIL ISS-057) — `results` still
        // carries every entry so the fold below can advance frontiers.
        let results = self.emit_verified_tokens(&verified, &stale_by_seq);
        // Multi-token scheduler input tracking (Plan 17.1-E): fold the
        // emitted tokens into the scheduler per sequence, advancing
        // `num_computed_tokens` by the tokens whose KV was computed
        // (RIL ISS-025 / ISS-059).
        self.fold_speculative_update(&batch, &results, &accepted_counts, &stale_by_seq);

        // P38 v0.3 wire-type engine wire-through: stop-sequence
        // finalization. Must run after `scheduler.update` (so
        // `seq.tokens` includes the new tokens) and after the token-send
        // loop above (so matched tokens reach the client).
        self.finalize_stop_sequences(&batch);

        // Track accuracy in adaptive decoder and record adjustment events
        let total_draft: usize = draft_outputs.iter().map(std::vec::Vec::len).sum();
        let total_accepted: usize = accepted_counts.iter().sum();
        if let Some(ref mut decoder) = self.adaptive_decoder {
            if decoder.record_verification(total_draft, total_accepted) {
                self.scheduler.metrics.record_speculative_adjustment();
            }
        }

        // Record speculative efficiency metric (Plan 17.4-F / MTRC-02)
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
            // RIL ISS-083: count generated (output) tokens — the emitted
            // verified results — not the input token sum. `tokens_total`
            // ("Total tokens generated") must agree with the regular and
            // CUDA-graph step paths, which both count emitted results.
            self.scheduler
                .metrics
                .record_tokens(u64::try_from(results.len()).unwrap_or(0));
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

    /// Send each verified token to its sequence's client channel, gating out
    /// stale mid-chunk prefill predictions (RIL ISS-057). `verified` is a
    /// flat token list — a sequence can contribute several entries (accepted
    /// drafts + bonus/rejection) — and staleness is keyed by `seq_id` via its
    /// batch entry (start + chunk < `prompt_len`).
    ///
    /// `results` still carries every verified entry so the caller's
    /// per-sequence scheduler fold can advance the mid-chunk prefill's
    /// frontier — only client emission is gated here.
    fn emit_verified_tokens(
        &self,
        verified: &[(SeqId, SampledToken)],
        stale_by_seq: &std::collections::HashMap<SeqId, bool>,
    ) -> Vec<(SeqId, SampledToken)> {
        let mut results = Vec::with_capacity(verified.len());
        for (seq_id, sampled) in verified {
            let is_stale = stale_by_seq.get(seq_id).copied().unwrap_or(false);
            if is_stale {
                tracing::debug!(
                    seq_id = %seq_id,
                    token = %sampled.token,
                    "Drawing stale mid-chunk prediction (not real output); not emitting"
                );
            } else if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.try_send(sampled.clone());
            }
            results.push((*seq_id, sampled.clone()));
        }
        results
    }

    /// Fold the step's emitted tokens into the scheduler **per sequence**
    /// (Plan 17.1-E). `results` carries one entry per emitted token
    /// (accepted drafts + bonus/rejection token), and the scheduler must
    /// record every one of them; `num_computed_tokens` must advance by the
    /// number of tokens whose KV the target model actually computed during
    /// verification (`input_len + accepted`).
    ///
    /// RIL ISS-025: the old code flattened `results` into per-token
    /// `seq_ids`/`sampled` vectors but passed a per-sequence
    /// `input_counts` vector. `scheduler.update` zips the two together, so
    /// the fold truncated to one token per sequence — tokens after the first
    /// were streamed to the client but never recorded in `seq.tokens` — and
    /// `num_computed_tokens` advanced by `accepted+1` (only correct for
    /// decode batches with `input_len` == 1). Long prompts in speculative mode
    /// never completed prefill in one step and re-fed already-generated
    /// draft tokens back into subsequent prefill batches.
    fn fold_speculative_update(
        &mut self,
        batch: &Batch,
        results: &[(SeqId, SampledToken)],
        accepted_counts: &[usize],
        stale_by_seq: &std::collections::HashMap<SeqId, bool>,
    ) {
        let mut per_seq: Vec<(SeqId, Vec<SampledToken>)> = Vec::new();
        let mut seq_index: std::collections::HashMap<SeqId, usize> =
            std::collections::HashMap::new();
        for (seq_id, sampled) in results {
            if let Some(&i) = seq_index.get(seq_id) {
                per_seq[i].1.push(sampled.clone());
            } else {
                seq_index.insert(*seq_id, per_seq.len());
                per_seq.push((*seq_id, vec![sampled.clone()]));
            }
        }

        for (seq_id, tokens) in &per_seq {
            let Some(i) = batch.seq_ids.iter().position(|sid| sid == seq_id) else {
                continue;
            };
            let chunk_len = batch.input_tokens.get(i).map_or(1, std::vec::Vec::len);
            // RIL ISS-059: a MID-chunk prefill's accepted drafts occupy real
            // upcoming prompt positions — the verifier only checked them
            // against the target model's continuation guess, never against
            // the prompt itself, so their KV is guessed content and the real
            // prompt tokens would be SKIPPED if the frontier advanced past
            // them (`requeue_partial_prefills`/ISS-054 preserves the inflated
            // frontier, so the next chunk composes from it). Advance the
            // frontier by the REAL chunk only; the final chunk (completes the
            // prompt) and decode sequences keep the full
            // `chunk_len + accepted` — there the drafts are genuine generated
            // continuation starting at `prompt_len`. `stale_by_seq` encodes
            // exactly this mid-chunk predicate (start + chunk < prompt_len).
            let mid_chunk = stale_by_seq.get(seq_id).copied().unwrap_or(false);
            let input_count = if mid_chunk {
                chunk_len
            } else {
                chunk_len + accepted_counts[i]
            };
            self.scheduler.update_speculative(
                std::slice::from_ref(seq_id),
                std::slice::from_ref(tokens),
                std::slice::from_ref(&input_count),
            );
        }
    }

    /// Grow each batch sequence's block table to cover the full speculative
    /// verification span (`num_computed + input_len + max_draft` tokens)
    /// BEFORE the draft/target forward writes KV (RIL ISS-026).
    ///
    /// The batch composer pre-allocates blocks for `input_len` only; the
    /// verifier then processes `input_len + drafts` tokens, and
    /// `write_prefill_kv`'s missing-block fallback
    /// (`block_ids.get(block_idx).unwrap_or(0)`) silently wrote the overflow
    /// draft KV into block 0 — corrupting the prompt's first-block KV
    /// whenever the draft span crossed a block boundary (verified by a
    /// real-model regression test). The extra blocks stay owned by the
    /// sequence (released on finish); rejected-draft blocks are simply
    /// unused capacity beyond `num_computed_tokens` (RIL ISS-027 — the old
    /// post-verification `memory_rollback` is no longer needed and was
    /// actively harmful for resumed prefill / boundary-crossing decode).
    fn ensure_verification_blocks(&mut self, batch: &mut vllm_traits::Batch, max_draft: usize) {
        for (i, seq_id) in batch.seq_ids.iter().copied().enumerate() {
            let base = batch.num_computed_tokens.get(i).copied().unwrap_or(0);
            let span = batch.input_tokens.get(i).map_or(0, std::vec::Vec::len) + max_draft;
            let blocks_needed = (base + span).div_ceil(vllm_traits::BLOCK_SIZE);
            if batch.kv_block_ids[i].len() < blocks_needed {
                self.scheduler
                    .ensure_blocks_for_tokens(seq_id, blocks_needed);
                if let Some(seq) = self.scheduler.get_sequence(seq_id) {
                    batch.kv_block_ids[i].clone_from(seq.kv_blocks.as_ref());
                }
            }
        }
    }
}
