use crate::error::Result;
use vllm_traits::{Batch, ModelBackend, SeqId, TokenId};

impl<M: ModelBackend> super::Engine<M> {
    pub fn step_speculative(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let start = std::time::Instant::now();
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let draft_outputs = self.generate_draft_tokens(&batch)?;

        let verified_outputs = self.verify_draft_tokens(&batch, &draft_outputs)?;

        let mut results = Vec::new();
        for (seq_id, token) in verified_outputs {
            if let Some(tx) = self.response_txs.get(&seq_id) {
                let _ = tx.try_send(token); // Use try_send to avoid blocking
            }
            results.push((seq_id, token));
        }

        let seq_ids: Vec<SeqId> = results.iter().map(|(id, _)| *id).collect();
        let tokens: Vec<TokenId> = results.iter().map(|(_, t)| *t).collect();
        let input_counts: Vec<usize> = vec![1; tokens.len()];
        self.scheduler.update(&seq_ids, &tokens, &input_counts);

        let finished = self.scheduler.finished_sequences();
        for seq in &finished {
            self.response_txs.remove(&seq.id);
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

    fn generate_draft_tokens(&mut self, batch: &Batch) -> Result<Vec<Vec<TokenId>>> {
        let mut draft_outputs = Vec::new();

        for (i, ((seq_id, tokens), positions)) in batch
            .seq_ids
            .iter()
            .zip(batch.input_tokens.iter())
            .zip(batch.positions.iter())
            .enumerate()
        {
            let mut draft = Vec::new();
            let mut current_tokens = tokens.clone();
            let mut current_positions = positions.clone();

            for _ in 0..self.max_draft_tokens {
                let output = self.draft_model.lock().unwrap().forward(
                    &[*seq_id],
                    std::slice::from_ref(&current_tokens),
                    std::slice::from_ref(&current_positions),
                    std::slice::from_ref(&batch.kv_block_ids[i]),
                    std::slice::from_ref(&batch.num_computed_tokens[i]),
                    std::slice::from_ref(&batch.is_prefill[i]),
                )?;
                let token = *output.next_tokens.first().unwrap_or(&0);
                draft.push(token);
                current_tokens.push(token);
                current_positions.push(current_positions.len());
            }
            draft_outputs.push(draft);
        }

        Ok(draft_outputs)
    }

    fn verify_draft_tokens(
        &mut self,
        batch: &Batch,
        draft_outputs: &[Vec<TokenId>],
    ) -> Result<Vec<(SeqId, TokenId)>> {
        let mut results = Vec::new();

        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            let drafts = &draft_outputs[i];

            if drafts.is_empty() {
                let target_output = self.target_model.lock().unwrap().forward(
                    &[*seq_id],
                    std::slice::from_ref(&batch.input_tokens[i]),
                    std::slice::from_ref(&batch.positions[i]),
                    std::slice::from_ref(&batch.kv_block_ids[i]),
                    std::slice::from_ref(&batch.num_computed_tokens[i]),
                    std::slice::from_ref(&batch.is_prefill[i]),
                )?;
                if let Some(&token) = target_output.next_tokens.first() {
                    results.push((*seq_id, token));
                }
                continue;
            }

            let mut verify_tokens = batch.input_tokens[i].clone();
            verify_tokens.extend(drafts.iter().cloned());

            let verify_positions: Vec<usize> = (0..verify_tokens.len()).collect();
            let verify_kv_block_ids: Vec<Vec<usize>> =
                vec![batch.kv_block_ids[i].clone(); verify_tokens.len()];
            let verify_num_computed: Vec<usize> =
                vec![batch.num_computed_tokens[i] + drafts.len(); verify_tokens.len()];
            let verify_is_prefill: Vec<bool> = vec![false; verify_tokens.len()];

            let target_output = self.target_model.lock().unwrap().forward(
                &[*seq_id],
                std::slice::from_ref(&verify_tokens),
                std::slice::from_ref(&verify_positions),
                &verify_kv_block_ids,
                &verify_num_computed,
                &verify_is_prefill,
            )?;

            let target_tokens = &target_output.next_tokens;

            for (j, &draft_token) in drafts.iter().enumerate() {
                if j < target_tokens.len() && target_tokens[j] == draft_token {
                    results.push((*seq_id, draft_token));
                } else {
                    break;
                }
            }

            let target_idx = drafts.len();
            if target_idx < target_tokens.len() {
                results.push((*seq_id, target_tokens[target_idx]));
            } else if let Some(&first) = target_tokens.first() {
                results.push((*seq_id, first));
            }
        }

        Ok(results)
    }
}
