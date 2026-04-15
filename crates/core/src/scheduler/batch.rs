use crate::error::Result;
use vllm_traits::{ModelBackend, SeqId, TokenId};

impl<M: ModelBackend + 'static> crate::engine::Engine<M> {
    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let start = std::time::Instant::now();
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            eprintln!("DEBUG step: batch is empty, skipping");
            return Ok(vec![]);
        }

        eprintln!("DEBUG step: batch seq_ids={:?}, input_tokens_count={}, positions_len={}, num_computed={:?}",
            batch.seq_ids,
            batch.input_tokens.len(),
            batch.positions.len(),
            batch.num_computed_tokens
        );

        let output = self.target_model.lock().unwrap().forward(
            &batch.seq_ids,
            &batch.input_tokens,
            &batch.positions,
            &batch.kv_block_ids,
            &batch.num_computed_tokens,
            &batch.is_prefill,
        )?;

        let input_counts: Vec<usize> = batch.input_tokens.iter().map(|v| v.len()).collect();

        self.scheduler
            .update(&batch.seq_ids, &output.next_tokens, &input_counts);

        let mut results = Vec::new();
        for (seq_id, token) in batch.seq_ids.iter().zip(output.next_tokens.iter()) {
            if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.try_send(*token); // Use try_send to avoid blocking
            }
            results.push((*seq_id, *token));
        }

        let finished = self.scheduler.finished_sequences();
        for seq in &finished {
            if let Some(tx) = self.response_txs.remove(&seq.id) {
                drop(tx);
            }
        }
        self.scheduler.clear_finished();

        if !batch.seq_ids.is_empty() {
            let total_tokens: usize = batch.input_tokens.iter().map(|t| t.len()).sum();
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
