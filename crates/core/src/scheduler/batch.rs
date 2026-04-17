use crate::error::Result;
use vllm_traits::{ModelBackend, SeqId, TokenId};

impl<M: ModelBackend + 'static> crate::engine::Engine<M> {
    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let start = std::time::Instant::now();
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = batch.seq_ids.len();
        let total_tokens: usize = batch.input_tokens.iter().map(|t| t.len()).sum();
        eprintln!(
            "ENGINE step: batch_size={}, total_tokens={}, is_prefill={:?}, phase={:?}",
            batch_size, total_tokens, batch.is_prefill, batch.phase
        );

        // Debug first sequence's tokens and positions
        if let Some(first_idx) = batch.seq_ids.first().copied() {
            if let Some(idx) = batch.seq_ids.iter().position(|&id| id == first_idx) {
                if idx < batch.input_tokens.len() {
                    eprintln!(
                        "ENGINE step: first seq_id={}, input_tokens={:?}, positions={:?}, num_computed={}",
                        first_idx,
                        &batch.input_tokens[idx][..batch.input_tokens[idx].len().min(15)],
                        batch.positions.get(idx).map(|p| &p[..p.len().min(15)]),
                        batch.num_computed_tokens.get(idx).copied().unwrap_or(0)
                    );
                }
            }
        }

        let output = self.target_model.lock().unwrap().forward(
            &batch.seq_ids,
            &batch.input_tokens,
            &batch.positions,
            &batch.kv_block_ids,
            &batch.num_computed_tokens,
            &batch.is_prefill,
        )?;

        eprintln!("ENGINE step: output tokens = {:?}", output.next_tokens);

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
            eprintln!("ENGINE step: sending token {} to seq_id {}", token, seq_id);
            if let Some(tx) = self.response_txs.get(seq_id) {
                tracing::debug!(seq_id = %seq_id, token = %token, "Sending token to channel");
                let _ = tx.try_send(*token); // Use try_send to avoid blocking
            }
            results.push((*seq_id, *token));
        }

        let finished = self.scheduler.finished_sequences();
        for seq in &finished {
            eprintln!("ENGINE step: sequence {} finished", seq.id);
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
