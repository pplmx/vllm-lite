//! Self-speculation implementation using reduced layer count
//!
//! This module provides self-speculation where the draft model uses
//! a subset of layers from the target model, sharing weights (zero copy).

use crate::speculative::config::SpeculationConfig;
use crate::speculative::verifier::{
    DraftVerifier, Result as VerifierResult, VerificationResult, VerifierError,
};
use crate::types::{SeqId, TokenId};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

pub struct SelfSpeculativeModel<M: ModelBackend> {
    model: M,
    #[allow(dead_code)]
    config: SpeculationConfig,
    draft_layer_count: usize,
    draft_kv_block_ids: HashMap<SeqId, Vec<usize>>,
}

impl<M: ModelBackend> SelfSpeculativeModel<M> {
    pub fn new(model: M, config: SpeculationConfig) -> Self {
        let total_layers = model.num_layers();
        let draft_layer_count = config
            .draft_layers
            .unwrap_or_else(|| (total_layers as f32 * 0.125).max(1.0) as usize);
        Self {
            model,
            config,
            draft_layer_count,
            draft_kv_block_ids: HashMap::new(),
        }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn mut_model(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn draft_layer_count(&self) -> usize {
        self.draft_layer_count
    }

    pub fn set_draft_layer_count(&mut self, count: usize) {
        self.draft_layer_count = count;
    }

    pub fn draft_kv_block_ids(&self) -> &HashMap<SeqId, Vec<usize>> {
        &self.draft_kv_block_ids
    }

    pub fn clear_draft_kv(&mut self) {
        self.draft_kv_block_ids.clear();
    }

    pub fn remove_draft_seq(&mut self, seq_id: SeqId) {
        self.draft_kv_block_ids.remove(&seq_id);
    }

    #[allow(dead_code)]
    fn sample_token(&self, logits: &[f32]) -> TokenId {
        let temperature = self.config.temperature;
        if temperature == 0.0 {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _): (usize, &f32)| idx as TokenId)
                .unwrap_or(0)
        } else {
            use std::time::{SystemTime, UNIX_EPOCH};
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .subsec_nanos();
            let r: f32 = nanos as f32 / u32::MAX as f32;

            let probs: Vec<f32> = logits.iter().map(|&x| (x / temperature).exp()).collect();
            let sum: f32 = probs.iter().sum();
            let probs: Vec<f32> = probs.iter().map(|&x| x / sum).collect();
            let mut cumsum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    return i as TokenId;
                }
            }
            probs.len().saturating_sub(1) as TokenId
        }
    }

    #[allow(dead_code)]
    fn top_k_filter(&self, logits: &mut [f32], k: usize) {
        if k == 0 {
            return;
        }
        let threshold_idx = k.min(logits.len().saturating_sub(1));
        let threshold = logits
            .iter()
            .enumerate()
            .nth(threshold_idx)
            .map(|(_idx, _)| {
                let mut sorted = logits.to_vec();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                sorted[k.saturating_sub(1)]
            })
            .unwrap_or(f32::NEG_INFINITY);

        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    #[allow(dead_code)]
    fn top_p_filter(&self, logits: &mut [f32], p: f32) {
        if p >= 1.0 {
            return;
        }
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let mut cumsum = 0.0;
        for &idx in indices.iter() {
            let prob = (logits[idx] as f64).exp() as f32;
            cumsum += prob;
            if cumsum > p {
                for &j in indices.iter().skip_while(|&&i| i == idx) {
                    logits[j] = f32::NEG_INFINITY;
                }
                break;
            }
        }
    }
}

impl<M: ModelBackend> DraftVerifier for SelfSpeculativeModel<M> {
    fn generate_draft(
        &mut self,
        batch: &vllm_traits::types::Batch,
        num_tokens: usize,
    ) -> VerifierResult<Vec<(SeqId, Vec<TokenId>)>> {
        if num_tokens == 0 || batch.is_empty() {
            return Ok(vec![]);
        }

        let mut drafts: Vec<(SeqId, Vec<TokenId>)> = Vec::new();

        for batch_idx in 0..batch.seq_ids.len() {
            let seq_id = batch.seq_ids[batch_idx];
            let input_tokens = &batch.input_tokens[batch_idx];
            let num_computed = batch.num_computed_tokens[batch_idx];

            let draft_block_ids = self
                .draft_kv_block_ids
                .entry(seq_id)
                .or_insert_with(|| batch.kv_block_ids[batch_idx].clone());

            let mut current_tokens: Vec<TokenId> = input_tokens.to_vec();
            let mut draft_tokens: Vec<TokenId> = Vec::with_capacity(num_tokens);

            // Use position tracking to compute positions for each draft step
            let mut current_num_computed = num_computed;

            for _step in 0..num_tokens {
                let last_token = vec![*current_tokens.last().unwrap_or(&0)];
                let step_position = vec![current_num_computed];

                let output = self
                    .model
                    .forward_to_layer(
                        &[seq_id],
                        &[last_token],
                        &[step_position],
                        std::slice::from_ref(draft_block_ids),
                        &[current_num_computed],
                        &[false],
                        self.draft_layer_count,
                    )
                    .map_err(|e| VerifierError::DraftGeneration(e.to_string()))?;

                let next_token = output.next_tokens.first().copied().unwrap_or(0);
                draft_tokens.push(next_token);
                current_tokens.push(next_token);
                current_num_computed += 1;
            }

            drafts.push((seq_id, draft_tokens));
        }

        Ok(drafts)
    }

    /// Verification is performed by `Engine::verify_draft_tokens_logits()`
    /// using `forward_logits()` with argmax comparison. This trait method is
    /// a stub — the engine bypasses the DraftVerifier trait for verification
    /// and implements its own logit-based path.
    fn verify(
        &self,
        _seq_id: SeqId,
        _draft_tokens: &[TokenId],
        _target_logits: &[f32],
    ) -> VerifierResult<VerificationResult> {
        Ok(VerificationResult::new(_seq_id, _draft_tokens.to_vec()))
    }

    fn accept(&mut self, _seq_id: SeqId, _accepted_count: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::config::SpeculationConfig;
    use vllm_traits::BatchOutput;

    #[test]
    fn test_draft_layer_count_calculation() {
        let config = SpeculationConfig::builder().draft_layers(4).build();
        assert_eq!(config.draft_layers, Some(4));
    }

    #[test]
    fn test_draft_layer_count_default() {
        let config = SpeculationConfig::default();
        assert!(config.draft_layers.is_none());
        assert!(config.self_speculation);
    }

    #[test]
    fn test_generate_draft_empty_batch() {
        let config = SpeculationConfig::default();
        let model = StubSpecModel;
        let mut ssm = SelfSpeculativeModel::new(model, config);
        let batch = vllm_traits::types::Batch::empty();
        let result = ssm.generate_draft(&batch, 5).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_generate_draft_zero_tokens() {
        let config = SpeculationConfig::default();
        let model = StubSpecModel;
        let mut ssm = SelfSpeculativeModel::new(model, config);
        let batch = vllm_traits::types::Batch {
            seq_ids: vec![1],
            input_tokens: vec![vec![10]],
            positions: vec![vec![0]],
            kv_block_ids: vec![vec![0]],
            num_computed_tokens: vec![0],
            is_prefill: vec![true],
            phase: vllm_traits::BatchPhase::Prefill,
            total_tokens: 1,
            max_seq_len: 1,
        };
        let result = ssm.generate_draft(&batch, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_generate_draft_returns_tokens() {
        let config = SpeculationConfig::default();
        let model = StubSpecModel;
        let mut ssm = SelfSpeculativeModel::new(model, config);
        let batch = vllm_traits::types::Batch {
            seq_ids: vec![1],
            input_tokens: vec![vec![10]],
            positions: vec![vec![0]],
            kv_block_ids: vec![vec![0]],
            num_computed_tokens: vec![0],
            is_prefill: vec![true],
            phase: vllm_traits::BatchPhase::Prefill,
            total_tokens: 1,
            max_seq_len: 1,
        };
        let result = ssm.generate_draft(&batch, 3).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1);
        assert_eq!(result[0].1.len(), 3);
    }

    #[test]
    fn test_greedy_argmax_sampling() {
        let config = SpeculationConfig::builder().temperature(0.0).build();
        let model = StubSpecModel;
        let ssm = SelfSpeculativeModel::new(model, config);
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let token = ssm.sample_token(&logits);
        assert_eq!(token, 3); // index 3 has highest value 0.8
    }

    #[test]
    fn test_sample_token_argmax_tie() {
        let config = SpeculationConfig::default();
        let model = StubSpecModel;
        let ssm = SelfSpeculativeModel::new(model, config);
        let logits = vec![0.1, 0.5, 0.3, 0.5, 0.2];
        let token = ssm.sample_token(&logits);
        // With temperature=0 and partial_cmp, max_by returns the last max
        assert_eq!(token, 3); // index 3 is the last occurrence of max value 0.5
    }

    #[test]
    fn test_draft_layer_count_is_used() {
        let config = SpeculationConfig::builder().draft_layers(3).build();
        let model = StubSpecModel;
        let ssm = SelfSpeculativeModel::new(model, config);
        assert_eq!(ssm.draft_layer_count(), 3);
    }

    #[test]
    fn test_draft_kv_cache_tracking() {
        let config = SpeculationConfig::default();
        let model = StubSpecModel;
        let mut ssm = SelfSpeculativeModel::new(model, config);
        let blocks = ssm.draft_kv_block_ids();
        assert!(blocks.is_empty());
        ssm.remove_draft_seq(42);
        assert!(ssm.draft_kv_block_ids().is_empty());
    }

    #[test]
    fn test_forward_to_layer_default_fallback() {
        // Verify that forward_to_layer default impl delegates to forward()
        let mut model = TrackingModel {
            forward_called: false,
            layers_processed: 0,
        };
        let _batch = vllm_traits::types::Batch {
            seq_ids: vec![1],
            input_tokens: vec![vec![10]],
            positions: vec![vec![0]],
            kv_block_ids: vec![vec![0]],
            num_computed_tokens: vec![0],
            is_prefill: vec![true],
            phase: vllm_traits::BatchPhase::Prefill,
            total_tokens: 1,
            max_seq_len: 1,
        };
        let _ = model.forward_to_layer(&[1], &[vec![10]], &[vec![0]], &[vec![0]], &[0], &[true], 4);
        assert!(model.forward_called);
    }

    struct StubSpecModel;

    impl ModelBackend for StubSpecModel {
        fn forward(
            &mut self,
            seq_ids: &[SeqId],
            _input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> vllm_traits::Result<BatchOutput> {
            Ok(BatchOutput {
                seq_ids: seq_ids.to_vec(),
                next_tokens: seq_ids.iter().map(|&id| (id * 10 + 1) as TokenId).collect(),
            })
        }

        fn forward_logits(
            &mut self,
            seq_ids: &[SeqId],
            _input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> vllm_traits::Result<Vec<Vec<f32>>> {
            Ok(vec![vec![0.0; 100]; seq_ids.len()])
        }

        fn embed(
            &mut self,
            _input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
        ) -> vllm_traits::Result<Vec<Vec<f32>>> {
            Ok(vec![])
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn num_layers(&self) -> usize {
            32
        }

        fn num_heads(&self) -> usize {
            32
        }
    }

    struct TrackingModel {
        forward_called: bool,
        #[allow(dead_code)]
        layers_processed: usize,
    }

    impl ModelBackend for TrackingModel {
        fn forward(
            &mut self,
            seq_ids: &[SeqId],
            _input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> vllm_traits::Result<BatchOutput> {
            self.forward_called = true;
            Ok(BatchOutput {
                seq_ids: seq_ids.to_vec(),
                next_tokens: seq_ids.iter().map(|_| 42).collect(),
            })
        }

        fn forward_logits(
            &mut self,
            seq_ids: &[SeqId],
            _input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> vllm_traits::Result<Vec<Vec<f32>>> {
            Ok(vec![vec![0.0; 100]; seq_ids.len()])
        }

        fn embed(
            &mut self,
            _input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
        ) -> vllm_traits::Result<Vec<Vec<f32>>> {
            Ok(vec![])
        }

        fn vocab_size(&self) -> usize {
            100
        }

        fn num_layers(&self) -> usize {
            32
        }

        fn num_heads(&self) -> usize {
            32
        }
    }
}
