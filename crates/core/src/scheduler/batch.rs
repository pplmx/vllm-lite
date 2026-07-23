//! `Batch` value type: the per-step plan handed to the model layer, with token ids, block ids, and per-sequence metadata.
//!
//! Constructed by `BatchComposer` (`scheduler/batch_composer/`); the
//! model layer's `forward` consumes it as-is.
use crate::error::Result;
use crate::sampling::sample_batch_with_params;
use crate::sync::lock_mutex;
use vllm_traits::{BatchOutput, FinishReason, SampledToken, SeqId, TokenId};

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
            (logits_list, vocab_size)
        };
        // forward_logits returns one Vec<f32> per sequence. For
        // decode each is 1 * vocab; for prefill each is
        // num_prompt_tokens * vocab. The "next" token always comes
        // from the last position's logits.
        let per_seq: Vec<Vec<f32>> = logits_list
            .iter()
            .map(|seq_logits| {
                let start = seq_logits.len().saturating_sub(vocab_size);
                seq_logits[start..].to_vec()
            })
            .collect();
        // Gather seen tokens (already-generated portion of each
        // sequence) so `repeat_penalty` can penalise them. Prefill
        // yields an empty seen-set, which makes repeat-penalty a
        // no-op as expected.
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
            next_tokens: next_tokens.clone(),
        };

        tracing::debug!(
            output_tokens = output.next_tokens.len(),
            first_output = output.next_tokens.first().map(|s| s.token),
            "Engine step: output tokens"
        );

        let input_counts: Vec<usize> = batch.input_tokens.iter().map(std::vec::Vec::len).collect();

        self.scheduler
            .update(&batch.seq_ids, &output.next_tokens, &input_counts);

        let mut results = Vec::new();
        for (seq_id, sampled) in batch.seq_ids.iter().zip(output.next_tokens.iter()) {
            tracing::debug!(seq_id = %seq_id, token = %sampled.token, "Sending token to channel");
            if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.try_send(sampled.clone());
            }
            results.push((*seq_id, sampled.clone()));
        }

        // P38 v0.3 wire-type engine wire-through: stop-sequence
        // finalization. Runs after `scheduler.update` (so `seq.tokens`
        // includes the new token) and after the token-send loop above
        // (so the matched token reaches the client before the channel
        // is dropped). Matched sequences get `FinishReason::Stop`.
        self.finalize_stop_sequences(&batch);

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

        // suppress unused-variable lint when sampling diagnostics are
        // stripped in release builds; keep the structure symmetric with
        // the CUDA-Graph path for future logging.
        let _ = logits_per_seq;

        Ok(results)
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
        if self.speculative_mode && self.draft_model.is_some() {
            let max_draft = self
                .adaptive_decoder
                .as_ref()
                .map_or(self.max_draft_tokens, super::super::speculative::adaptive::AdaptiveSpeculativeDecoder::current_max_draft_tokens);
            self.step_speculative_inner(max_draft)
        } else {
            self.step_regular()
        }
    }
}
