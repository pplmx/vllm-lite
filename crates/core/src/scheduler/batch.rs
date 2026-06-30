use crate::error::Result;
use crate::sync::lock_mutex;
use vllm_traits::{SeqId, TokenId};

impl crate::engine::Engine {
    /// Regular (non-speculative) decode step.
    pub(crate) fn step_regular(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
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

        let output = lock_mutex(&self.target_model)?.forward(
            &batch.seq_ids,
            &batch.input_tokens,
            &batch.positions,
            &batch.kv_block_ids,
            &batch.num_computed_tokens,
            &batch.is_prefill,
        )?;

        tracing::debug!(
            output_tokens = output.next_tokens.len(),
            first_output = output.next_tokens.first(),
            "Engine step: output tokens"
        );

        let input_counts: Vec<usize> = batch.input_tokens.iter().map(std::vec::Vec::len).collect();

        self.scheduler
            .update(&batch.seq_ids, &output.next_tokens, &input_counts);

        let mut results = Vec::new();
        for (seq_id, token) in batch.seq_ids.iter().zip(output.next_tokens.iter()) {
            tracing::debug!(seq_id = %seq_id, token = %token, "Sending token to channel");
            if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.try_send(*token);
            }
            results.push((*seq_id, *token));
        }

        let finished = self.scheduler.finished_sequences();
        for seq in &finished {
            tracing::debug!(seq_id = seq.id, "Sequence finished");
            if let Some(tx) = self.response_txs.remove(&seq.id) {
                drop(tx);
            }
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

        Ok(results)
    }

    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// Run one scheduling step (regular or speculative depending on engine configuration).
    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
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
