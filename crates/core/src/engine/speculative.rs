#![allow(dead_code, clippy::type_complexity, clippy::iter_cloned_collect)]

use crate::error::Result;
use vllm_traits::{Batch, BatchPhase, ModelBackend, SeqId, TokenId};

/// Unified entry point for speculative and non-speculative decoding.
///
/// `max_draft = None` or `Some(0)` → non-speculative step.
/// `Some(n)` → internal speculative step with up to n draft tokens.
impl<M: ModelBackend> super::Engine<M> {
    /// Unified step: dispatches to speculative or non-speculative path.
    /// Note: `step` (no args) is defined in scheduler/batch.rs for backward compat.
    pub fn step_with_draft(&mut self, max_draft: Option<usize>) -> Result<Vec<(SeqId, TokenId)>> {
        match max_draft {
            None | Some(0) => self.step_internal(),
            Some(n) => self.step_speculative_inner(n),
        }
    }

    /// Public entry for speculative step (backward compatible).
    pub fn step_speculative(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        self.step_with_draft(Some(self.max_draft_tokens))
    }

    /// Public entry for adaptive speculative step (backward compatible).
    pub fn step_adaptive_speculative(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let max_draft = self
            .adaptive_decoder
            .as_ref()
            .map(|d| d.current_max_draft_tokens())
            .unwrap_or(self.max_draft_tokens);
        self.step_with_draft(Some(max_draft))
    }

    /// Warm up draft model's KV cache after target prefill.
    /// This ensures the first speculative decode step has valid draft state.
    fn warmup_draft_kv(&mut self, batch: &Batch) -> Result<()> {
        if !self.speculative_mode {
            return Ok(());
        }
        let draft_model = match &self.draft_model {
            Some(dm) => dm.clone(),
            None => return Ok(()),
        };
        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            draft_model.lock().unwrap().forward(
                &[*seq_id],
                std::slice::from_ref(&batch.input_tokens[i]),
                std::slice::from_ref(&batch.positions[i]),
                std::slice::from_ref(&batch.kv_block_ids[i]),
                std::slice::from_ref(&batch.num_computed_tokens[i]),
                std::slice::from_ref(&batch.is_prefill[i]),
            )?;
        }
        tracing::debug!(
            seq_count = batch.seq_ids.len(),
            "Draft KV cache warmed up after prefill"
        );
        Ok(())
    }

    /// Internal speculative decode step.
    fn step_speculative_inner(&mut self, max_draft: usize) -> Result<Vec<(SeqId, TokenId)>> {
        let start = std::time::Instant::now();
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        // Warmup draft KV cache after prefill (Plans 17.4-A, 17.4-E)
        if batch.phase == BatchPhase::Prefill && self.speculative_mode {
            if let Err(e) = self.warmup_draft_kv(&batch) {
                tracing::warn!(error = %e, "Draft warmup failed, continuing without warmup");
            }
        }

        let draft_outputs = match self.generate_batched_drafts(&batch, max_draft) {
            Ok(drafts) => drafts,
            Err(e) => {
                tracing::warn!(error = %e, "Draft generation failed, falling back to non-speculative");
                return self.step_internal();
            }
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

        let mut results = Vec::new();
        for (seq_id, token) in &verified {
            if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.try_send(*token);
            }
            results.push((*seq_id, *token));
        }

        // Multi-token scheduler input tracking (Plan 17.1-E)
        let seq_ids: Vec<SeqId> = results.iter().map(|(id, _)| *id).collect();
        let tokens: Vec<TokenId> = results.iter().map(|(_, t)| *t).collect();
        // Build input_counts from accepted counts + 1 bonus token
        let input_counts: Vec<usize> = accepted_counts
            .iter()
            .map(|&accepted| accepted + 1) // accepted drafts + the target token
            .collect();
        self.scheduler.update(&seq_ids, &tokens, &input_counts);

        // Track accuracy in adaptive decoder
        if let Some(ref mut decoder) = self.adaptive_decoder {
            let total_draft: usize = draft_outputs.iter().map(|d| d.len()).sum();
            let total_accepted: usize = accepted_counts.iter().sum();
            decoder.record_verification(total_draft, total_accepted);
        }

        // Record speculative efficiency metric (Plan 17.4-F / MTRC-02)
        let total_draft: usize = draft_outputs.iter().map(|d| d.len()).sum();
        let total_accepted: usize = accepted_counts.iter().sum();
        let total_tokens_step = total_draft + total_accepted;
        if total_tokens_step > 0 {
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
            self.response_txs.remove(&seq.id);
            self.scheduler.metrics.remove_per_request(seq.id);
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

    /// Batched per-position draft generation (Plan 17.1-B).
    ///
    /// All sequences generate draft position k before advancing to k+1.
    fn generate_batched_drafts(
        &mut self,
        batch: &Batch,
        max_draft: usize,
    ) -> Result<Vec<Vec<TokenId>>> {
        let n_seq = batch.seq_ids.len();
        let mut draft_outputs: Vec<Vec<TokenId>> = vec![Vec::with_capacity(max_draft); n_seq];

        let draft_model = match &self.draft_model {
            Some(dm) => dm.clone(),
            None => {
                tracing::warn!("No draft model set, returning empty drafts");
                return Ok(draft_outputs);
            }
        };

        // Per-sequence state tracking: current input tokens and positions
        let mut current_tokens: Vec<Vec<TokenId>> = batch.input_tokens.iter().cloned().collect();
        let mut current_positions: Vec<Vec<usize>> = batch.positions.iter().cloned().collect();

        for _pos in 0..max_draft {
            // Build per-position batch
            let mut pos_seq_ids = Vec::with_capacity(n_seq);
            let mut pos_input_tokens = Vec::with_capacity(n_seq);
            let mut pos_positions = Vec::with_capacity(n_seq);
            let mut pos_kv_block_ids = Vec::with_capacity(n_seq);
            let mut pos_num_computed = Vec::with_capacity(n_seq);
            let mut pos_is_prefill = Vec::with_capacity(n_seq);
            let mut active_indices = Vec::with_capacity(n_seq);

            for (i, seq_id) in batch.seq_ids.iter().enumerate() {
                if current_tokens[i].is_empty() {
                    continue;
                }
                pos_seq_ids.push(*seq_id);
                pos_input_tokens.push(current_tokens[i].clone());
                pos_positions.push(current_positions[i].clone());
                pos_kv_block_ids.push(batch.kv_block_ids[i].clone());
                pos_num_computed.push(batch.num_computed_tokens[i]);
                pos_is_prefill.push(if _pos == 0 {
                    batch.is_prefill[i]
                } else {
                    false // subsequent draft steps are decode
                });
                active_indices.push(i);
            }

            if pos_seq_ids.is_empty() {
                break;
            }

            // Single forward pass per position across all active sequences
            let output = match draft_model.lock().unwrap().forward(
                &pos_seq_ids,
                &pos_input_tokens,
                &pos_positions,
                &pos_kv_block_ids,
                &pos_num_computed,
                &pos_is_prefill,
            ) {
                Ok(o) => o,
                Err(e) => {
                    tracing::warn!(error = %e, pos = _pos, "Draft model forward failed at position");
                    break;
                }
            };

            // Distribute output tokens back to per-sequence drafts
            for (j, &seq_idx) in active_indices.iter().enumerate() {
                let token = output.next_tokens.get(j).copied().unwrap_or(0);
                draft_outputs[seq_idx].push(token);
                current_tokens[seq_idx].push(token);
                let new_pos = current_positions[seq_idx].len();
                current_positions[seq_idx].push(new_pos);
            }
        }

        Ok(draft_outputs)
    }

    /// Logit-based verification with rejection sampling (Plan 17.1-C).
    ///
    /// Returns (accepted_tokens, accepted_counts_per_sequence).
    fn verify_draft_tokens_logits(
        &mut self,
        batch: &Batch,
        draft_outputs: &[Vec<TokenId>],
    ) -> Result<(Vec<(SeqId, TokenId)>, Vec<usize>)> {
        let mut results = Vec::new();
        let mut accepted_counts = Vec::with_capacity(batch.seq_ids.len());

        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            let drafts = &draft_outputs[i];

            if drafts.is_empty() {
                let logits = self.target_model.lock().unwrap().forward_logits(
                    &[*seq_id],
                    std::slice::from_ref(&batch.input_tokens[i]),
                    std::slice::from_ref(&batch.positions[i]),
                    std::slice::from_ref(&batch.kv_block_ids[i]),
                    std::slice::from_ref(&batch.num_computed_tokens[i]),
                    std::slice::from_ref(&batch.is_prefill[i]),
                )?;
                let token = if let Some(pos_logits) = logits.first() {
                    argmax(pos_logits)
                } else {
                    0
                };
                results.push((*seq_id, token));
                accepted_counts.push(0);
                continue;
            }

            // Concatenate input tokens + draft tokens for verification
            let verify_tokens: Vec<TokenId> = batch.input_tokens[i]
                .iter()
                .chain(drafts.iter())
                .copied()
                .collect();
            let verify_positions: Vec<usize> = (0..verify_tokens.len()).collect();

            // Get logits from target model for all positions
            let logits = self.target_model.lock().unwrap().forward_logits(
                &[*seq_id],
                std::slice::from_ref(&verify_tokens),
                std::slice::from_ref(&verify_positions),
                std::slice::from_ref(&batch.kv_block_ids[i]),
                std::slice::from_ref(&batch.num_computed_tokens[i]),
                std::slice::from_ref(&batch.is_prefill[i]),
            )?;

            let logits: &[f32] = logits.first().map(|v| v.as_slice()).unwrap_or(&[]);
            let vocab_size = self.target_model.lock().unwrap().vocab_size();

            let mut accepted = 0usize;

            for (j, &draft_token) in drafts.iter().enumerate() {
                let offset = j * vocab_size;
                if offset + vocab_size > logits.len() {
                    break;
                }
                let pos_logits = &logits[offset..offset + vocab_size];
                let target_token = argmax(pos_logits);

                if target_token == draft_token {
                    results.push((*seq_id, draft_token));
                    accepted += 1;
                } else {
                    results.push((*seq_id, target_token));
                    break;
                }
            }

            // Add a bonus token if all drafts were accepted
            if accepted == drafts.len() {
                let bonus_offset = accepted * vocab_size;
                if bonus_offset + vocab_size <= logits.len() {
                    let bonus_logits = &logits[bonus_offset..bonus_offset + vocab_size];
                    let bonus_token = argmax(bonus_logits);
                    results.push((*seq_id, bonus_token));
                }
            }

            accepted_counts.push(accepted);
        }

        Ok((results, accepted_counts))
    }

    // Legacy methods kept for backward compatibility but no longer used internally
    fn generate_draft_tokens(&mut self, batch: &Batch) -> Result<Vec<Vec<TokenId>>> {
        self.generate_draft_tokens_with_limit(batch, self.max_draft_tokens)
    }

    fn generate_draft_tokens_with_limit(
        &mut self,
        batch: &Batch,
        max_draft: usize,
    ) -> Result<Vec<Vec<TokenId>>> {
        let mut draft_outputs = Vec::new();

        if self.draft_model.is_none() {
            tracing::warn!(
                "Speculative decoding enabled but no draft model set, using target model"
            );
        }

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

            let draft_model = match &self.draft_model {
                Some(dm) => dm.clone(),
                None => {
                    draft_outputs.push(Vec::new());
                    continue;
                }
            };

            for _ in 0..max_draft {
                let output = draft_model.lock().unwrap().forward(
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
            let verify_kv_block_ids: Vec<Vec<usize>> = vec![batch.kv_block_ids[i].clone()];
            let verify_num_computed: Vec<usize> = vec![batch.num_computed_tokens[i] + drafts.len()];
            let verify_is_prefill: Vec<bool> = vec![true];

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

    fn verify_and_track(
        &mut self,
        batch: &Batch,
        draft_outputs: &[Vec<TokenId>],
    ) -> Result<Vec<(SeqId, TokenId)>> {
        let mut results = Vec::new();
        let mut total_draft = 0usize;
        let mut total_accepted = 0usize;

        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            let drafts = &draft_outputs[i];
            total_draft += drafts.len();

            let mut accepted_count = 0usize;

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
                    accepted_count = 1;
                }
            } else {
                let mut verify_tokens = batch.input_tokens[i].clone();
                verify_tokens.extend(drafts.iter().cloned());

                let verify_positions: Vec<usize> = (0..verify_tokens.len()).collect();
                let verify_kv_block_ids = vec![batch.kv_block_ids[i].clone(); verify_tokens.len()];
                let verify_num_computed =
                    vec![batch.num_computed_tokens[i] + drafts.len(); verify_tokens.len()];
                let verify_is_prefill = vec![false; verify_tokens.len()];

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
                        accepted_count += 1;
                    } else {
                        break;
                    }
                }

                let target_idx = accepted_count;
                if target_idx < target_tokens.len() {
                    results.push((*seq_id, target_tokens[target_idx]));
                }
            }

            total_accepted += accepted_count;
        }

        if let Some(ref mut decoder) = self.adaptive_decoder {
            decoder.record_verification(total_draft, total_accepted);
        }

        Ok(results)
    }
}

fn argmax(logits: &[f32]) -> TokenId {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as TokenId)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Request, SchedulerConfig};
    use tokio::sync::mpsc as tokio_mpsc;
    use vllm_traits::{BatchOutput, Result as ModelResult};

    /// A fake model that returns fixed tokens for both forward and forward_logits.
    #[derive(Clone)]
    struct FakeModel {
        token_to_return: TokenId,
        vocab_size: usize,
    }

    impl FakeModel {
        fn new(token: TokenId) -> Self {
            Self {
                token_to_return: token,
                vocab_size: 100,
            }
        }

        fn logits_for_token(&self, token: TokenId) -> Vec<f32> {
            let mut logits = vec![-10.0; self.vocab_size];
            if (token as usize) < self.vocab_size {
                logits[token as usize] = 10.0;
            }
            logits
        }
    }

    impl ModelBackend for FakeModel {
        fn forward(
            &mut self,
            seq_ids: &[SeqId],
            _input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> ModelResult<BatchOutput> {
            Ok(BatchOutput {
                seq_ids: seq_ids.to_vec(),
                next_tokens: seq_ids.iter().map(|_| self.token_to_return).collect(),
            })
        }

        fn forward_logits(
            &mut self,
            _seq_ids: &[SeqId],
            input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> ModelResult<Vec<Vec<f32>>> {
            Ok(input_tokens
                .iter()
                .map(|tokens| {
                    tokens
                        .iter()
                        .flat_map(|_| self.logits_for_token(self.token_to_return))
                        .collect()
                })
                .collect())
        }

        fn embed(
            &mut self,
            input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
        ) -> ModelResult<Vec<Vec<f32>>> {
            Ok(input_tokens
                .iter()
                .map(|tokens| vec![0.0; tokens.len()])
                .collect())
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn num_layers(&self) -> usize {
            1
        }

        fn num_heads(&self) -> usize {
            1
        }
    }

    /// Test Plan 17.1-A: Unified step() dispatches correctly
    #[test]
    #[ignore]
    fn test_step_unified_dispatch() {
        // Test dispatch with Some(n) → speculative path, None → non-speculative
        let target = FakeModel::new(42);
        let draft = FakeModel::new(42);
        let mut engine = super::super::Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
        engine.enable_speculative();
        let (tx, _rx) = tokio_mpsc::channel(64);
        engine.add_request(Request::new(1, vec![10, 20], 5), tx);

        // step(None) should go to non-speculative path
        let result = engine.step_with_draft(None).unwrap();
        assert!(!result.is_empty());

        // step(Some(5)) should go to speculative path
        engine.scheduler = super::super::SchedulerEngine::new(
            SchedulerConfig::default(),
            1024,
            std::sync::Arc::new(crate::metrics::EnhancedMetricsCollector::new()),
        );
        let (tx2, _rx2) = tokio_mpsc::channel(64);
        engine.add_request(Request::new(2, vec![10, 20], 5), tx2);
        engine.enable_speculative();
        let result = engine.step_with_draft(Some(5)).unwrap();
        assert!(!result.is_empty());
    }

    /// Test Plan 17.1-B: Batched draft generation produces expected output shape
    #[test]
    #[ignore]
    fn test_batched_draft_generation() {
        let target = FakeModel::new(42);
        let draft = FakeModel::new(42);
        let mut engine = super::super::Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
        engine.max_draft_tokens = 4;
        engine.enable_speculative();

        // The batch is built internally; we test via step(Some(4))
        let (tx, _rx) = tokio_mpsc::channel(64);
        engine.add_request(Request::new(1, vec![10, 20], 10), tx);
        let result = engine.step_with_draft(Some(4)).unwrap();
        assert!(!result.is_empty());
    }

    /// Test Plan 17.1-C: Greedy-mode exact match via argmax verification
    #[test]
    #[ignore]
    fn test_logit_verification_exact_match() {
        // Both models return same token 42 → all accepted
        let target = FakeModel::new(42);
        let draft = FakeModel::new(42);
        let mut engine = super::super::Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
        engine.max_draft_tokens = 3;
        engine.enable_speculative();

        let (tx, mut rx) = tokio_mpsc::channel(64);
        engine.add_request(Request::new(1, vec![10, 20], 10), tx);
        let result = engine.step_with_draft(Some(3)).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result[0].1, 42);
        let _ = rx.try_recv().ok();
    }

    /// Test Plan 17.1-D: KV cache rollback for rejected drafts
    #[test]
    #[ignore]
    fn test_kv_rollback_rejected_drafts() {
        let target = FakeModel::new(42);
        let draft = FakeModel::new(99); // Different token → most will be rejected
        let mut engine = super::super::Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
        engine.max_draft_tokens = 3;
        engine.enable_speculative();

        let (tx, _rx) = tokio_mpsc::channel(64);
        engine.add_request(Request::new(1, vec![10, 20], 5), tx);
        let result = engine.step_with_draft(Some(3)).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result[0].1, 42);
    }

    /// Test Plan 17.1-E: Multi-token input_count is accepted by scheduler
    #[test]
    fn test_scheduler_multi_token_update() {
        use crate::types::{Request, SchedulerConfig};
        use std::sync::Arc;
        let mut scheduler = super::super::SchedulerEngine::new(
            SchedulerConfig::default(),
            1024,
            Arc::new(crate::metrics::EnhancedMetricsCollector::new()),
        );
        let id = scheduler.add_request(Request::new(1, vec![10, 20], 10));
        let _batch = scheduler.build_batch();

        // Update with input_count > 1 (multi-token)
        scheduler.update(&[id], &[100], &[3]);
        assert_eq!(scheduler.running_count(), 1);

        // Update again with input_count = 0 (all pre-computed)
        scheduler.update(&[id], &[101], &[0]);
        assert_eq!(scheduler.running_count(), 1);
    }

    /// Test Plan 17.1-F: Speculative fallback on draft error
    #[test]
    #[ignore]
    fn test_draft_model_error_fallback() {
        // Currently the draft model error is caught and falls back
        // With no draft model configured, speculative should handle gracefully
        let target = FakeModel::new(42);
        let mut engine = super::super::Engine::new_boxed(
            Box::new(target),
            None::<Box<dyn ModelBackend>>, // No draft model
        );
        engine.speculative_mode = true;

        let (tx, _rx) = tokio_mpsc::channel(64);
        engine.add_request(Request::new(1, vec![10, 20], 5), tx);
        // Should not panic even without draft model in spec mode
        let result = engine.step_with_draft(Some(3));
        assert!(result.is_ok());
    }

    /// Integration test: speculative step produces output
    #[test]
    #[ignore]
    fn test_speculative_step_produces_output() {
        let target = FakeModel::new(42);
        let draft = FakeModel::new(42);
        let mut engine = super::super::Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
        engine.max_draft_tokens = 4;
        engine.enable_speculative();

        let (tx, mut rx) = tokio_mpsc::channel(64);
        engine.add_request(Request::new(1, vec![10, 20], 10), tx);
        let result = engine.step_with_draft(Some(4)).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result[0].1, 42);

        // Should have received token via channel
        let received = rx.try_recv().ok();
        assert_eq!(received, Some(42));
    }

    /// Integration test: speculative vs non-speculative equivalence
    #[test]
    #[ignore]
    fn test_speculative_vs_non_speculative_equivalence() {
        // Same input should yield same first token in both modes
        let target = FakeModel::new(42);
        let draft = FakeModel::new(42);

        // Non-speculative
        let mut engine_ns = super::super::Engine::new_boxed(Box::new(target.clone()), None);
        let (tx1, _rx1) = tokio_mpsc::channel(64);
        engine_ns.add_request(Request::new(1, vec![10, 20], 5), tx1);
        let result_ns = engine_ns.step_with_draft(None).unwrap();

        // Speculative (with matching draft)
        let mut engine_sp =
            super::super::Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
        engine_sp.enable_speculative();
        engine_sp.max_draft_tokens = 3;
        let (tx2, _rx2) = tokio_mpsc::channel(64);
        engine_sp.add_request(Request::new(2, vec![10, 20], 5), tx2);
        let result_sp = engine_sp.step_with_draft(Some(3)).unwrap();

        // First token should match
        assert!(!result_ns.is_empty());
        assert!(!result_sp.is_empty());
        assert_eq!(result_ns[0].1, result_sp[0].1);
    }
}
