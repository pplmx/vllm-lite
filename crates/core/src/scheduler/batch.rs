use crate::error::Result;
use vllm_traits::{ModelBackend, SeqId, TokenId};

impl<M: ModelBackend + 'static> crate::engine::Engine<M> {
    /// Backward-compatible non-speculative step.
    /// Delegates to the unified `step(Some(0))` path.
    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        self.step_internal()
    }

    /// Internal non-speculative step implementation.
    pub(crate) fn step_internal(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let start = std::time::Instant::now();
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = batch.seq_ids.len();
        let total_tokens: usize = batch.input_tokens.iter().map(|t| t.len()).sum();

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

        let output = self.target_model.lock().unwrap().forward(
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

        let input_counts: Vec<usize> = batch.input_tokens.iter().map(|v| v.len()).collect();

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
            self.metrics.record_tokens(total_tokens as u64);
            self.metrics.record_batch_size(batch.seq_ids.len());
        }

        let elapsed = start.elapsed().as_millis() as f64;
        if elapsed > 0.0 {
            self.metrics.record_latency(elapsed);
        }

        Ok(results)
    }
}
