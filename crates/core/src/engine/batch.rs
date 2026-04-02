use crate::error::Result;
use vllm_traits::{ModelBackend, SeqId, TokenId};

impl<M: ModelBackend> super::Engine<M> {
    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let start = std::time::Instant::now();
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let output =
            self.target_model
                .forward(&batch.seq_ids, &batch.input_tokens, &batch.positions)?;

        let input_counts: Vec<usize> = batch
            .input_tokens
            .iter()
            .map(|v| v.len())
            .collect::<Vec<_>>();

        self.scheduler
            .update(&batch.seq_ids, &output.next_tokens, &input_counts);

        let mut results = Vec::new();
        for (seq_id, token) in batch.seq_ids.iter().zip(output.next_tokens.iter()) {
            if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.send(*token);
            }
            results.push((*seq_id, *token));
        }

        for seq in self.scheduler.finished_sequences() {
            self.response_txs.remove(&seq.id);
        }

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

    #[allow(dead_code)]
    pub fn greedy_sample(logits: &[f32]) -> TokenId {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as TokenId)
            .unwrap_or(0)
    }
}
