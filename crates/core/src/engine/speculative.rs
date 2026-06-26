#![allow(clippy::type_complexity, clippy::iter_cloned_collect)]

use crate::error::Result;
use crate::sync::lock_mutex;
use vllm_traits::{Batch, BatchPhase, SeqId, TokenId};

impl super::Engine {
    /// Warm up draft model's KV cache after target prefill.
    /// This ensures the first speculative decode step has valid draft state.
    pub(crate) fn warmup_draft_kv(&mut self, batch: &Batch) -> Result<()> {
        if !self.speculative_mode {
            return Ok(());
        }
        let draft_model = match &self.draft_model {
            Some(dm) => dm.clone(),
            None => return Ok(()),
        };
        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            lock_mutex(&draft_model)?.forward(
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

    /// Speculative decode step (called from `Engine::step` when speculative mode is on).
    pub(crate) fn step_speculative_inner(
        &mut self,
        max_draft: usize,
    ) -> Result<Vec<(SeqId, TokenId)>> {
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
                return self.step_regular();
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

        // Track accuracy in adaptive decoder and record adjustment events
        if let Some(ref mut decoder) = self.adaptive_decoder {
            let total_draft: usize = draft_outputs.iter().map(|d| d.len()).sum();
            let total_accepted: usize = accepted_counts.iter().sum();
            if decoder.record_verification(total_draft, total_accepted) {
                self.scheduler.metrics.record_speculative_adjustment();
            }
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
            self.scheduler.metrics.record_tokens(total_tokens as u64);
            self.scheduler
                .metrics
                .record_batch_size(batch.seq_ids.len());
        }

        let elapsed = start.elapsed().as_millis() as f64;
        if elapsed > 0.0 {
            self.scheduler.metrics.record_latency(elapsed);
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
            let output = match lock_mutex(&draft_model)?.forward(
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
                let logits = lock_mutex(&self.target_model)?.forward_logits(
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
            let logits = lock_mutex(&self.target_model)?.forward_logits(
                &[*seq_id],
                std::slice::from_ref(&verify_tokens),
                std::slice::from_ref(&verify_positions),
                std::slice::from_ref(&batch.kv_block_ids[i]),
                std::slice::from_ref(&batch.num_computed_tokens[i]),
                std::slice::from_ref(&batch.is_prefill[i]),
            )?;

            let logits: &[f32] = logits.first().map(|v| v.as_slice()).unwrap_or(&[]);
            let vocab_size = lock_mutex(&self.target_model)?.vocab_size();

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
    use vllm_traits::{BatchOutput, ModelBackend, Result as ModelResult};

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

    /// Wrapper around FakeModel that counts forward/forward_logits invocations.
    /// Used to verify warmup_draft_kv calls draft model per sequence.
    /// `Arc<AtomicUsize>` + Clone enable inspecting call count after the model
    /// has been moved into the engine (the engine clones the Arc internally).
    #[derive(Clone)]
    struct CounterModel {
        inner: FakeModel,
        forward_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl CounterModel {
        fn new(token: TokenId) -> Self {
            Self {
                inner: FakeModel::new(token),
                forward_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }
        fn forward_count(&self) -> usize {
            self.forward_count
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl ModelBackend for CounterModel {
        fn forward(
            &mut self,
            seq_ids: &[SeqId],
            input_tokens: &[Vec<TokenId>],
            positions: &[Vec<usize>],
            kv_block_ids: &[Vec<usize>],
            num_computed_tokens: &[usize],
            is_prefill: &[bool],
        ) -> ModelResult<BatchOutput> {
            self.forward_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.forward(
                seq_ids,
                input_tokens,
                positions,
                kv_block_ids,
                num_computed_tokens,
                is_prefill,
            )
        }

        fn forward_logits(
            &mut self,
            seq_ids: &[SeqId],
            input_tokens: &[Vec<TokenId>],
            positions: &[Vec<usize>],
            kv_block_ids: &[Vec<usize>],
            num_computed_tokens: &[usize],
            is_prefill: &[bool],
        ) -> ModelResult<Vec<Vec<f32>>> {
            self.forward_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.forward_logits(
                seq_ids,
                input_tokens,
                positions,
                kv_block_ids,
                num_computed_tokens,
                is_prefill,
            )
        }

        fn embed(
            &mut self,
            input_tokens: &[Vec<TokenId>],
            positions: &[Vec<usize>],
        ) -> ModelResult<Vec<Vec<f32>>> {
            self.inner.embed(input_tokens, positions)
        }

        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }

        fn num_layers(&self) -> usize {
            self.inner.num_layers()
        }

        fn num_heads(&self) -> usize {
            self.inner.num_heads()
        }
    }

    /// Test Plan 17.4-A: warmup_draft_kv invokes draft model once per sequence.
    /// Fast unit test (no #[ignore]): directly constructs a Prefill batch and
    /// calls warmup_draft_kv to verify the contract independently of step().
    #[test]
    fn test_warmup_draft_kv_invokes_draft_per_sequence() {
        let target = FakeModel::new(42);
        let draft = CounterModel::new(42);
        let draft_count_before = draft.forward_count();
        let mut engine =
            super::super::Engine::new_boxed(Box::new(target), Some(Box::new(draft.clone())));
        engine.enable_speculative();

        // Construct a Prefill batch with 3 sequences.
        let batch = vllm_traits::types::Batch {
            seq_ids: vec![1, 2, 3],
            input_tokens: vec![vec![10, 20], vec![30], vec![40, 50, 60]],
            positions: vec![vec![0, 1], vec![0], vec![0, 1, 2]],
            kv_block_ids: vec![vec![0], vec![0], vec![0]],
            num_computed_tokens: vec![0, 0, 0],
            is_prefill: vec![true, true, true],
            phase: vllm_traits::BatchPhase::Prefill,
            total_tokens: 6,
            max_seq_len: 3,
        };

        // Execute warmup directly.
        engine
            .warmup_draft_kv(&batch)
            .expect("warmup_draft_kv should succeed");

        // Verify: draft model forward() called once per seq_id.
        let calls = draft.forward_count() - draft_count_before;
        assert_eq!(
            calls, 3,
            "warmup_draft_kv should invoke draft.forward() exactly once per seq_id (got {})",
            calls
        );
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
        let result = engine.step().unwrap();
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
        let result = engine.step().unwrap();
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
        let result = engine.step().unwrap();
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
        let result = engine.step().unwrap();
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
        let result = engine.step().unwrap();
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
        let result = engine.step();
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
        let result = engine.step().unwrap();
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
        let result_ns = engine_ns.step().unwrap();

        // Speculative (with matching draft)
        let mut engine_sp =
            super::super::Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
        engine_sp.enable_speculative();
        engine_sp.max_draft_tokens = 3;
        let (tx2, _rx2) = tokio_mpsc::channel(64);
        engine_sp.add_request(Request::new(2, vec![10, 20], 5), tx2);
        let result_sp = engine_sp.step().unwrap();

        // First token should match
        assert!(!result_ns.is_empty());
        assert!(!result_sp.is_empty());
        assert_eq!(result_ns[0].1, result_sp[0].1);
    }
}
