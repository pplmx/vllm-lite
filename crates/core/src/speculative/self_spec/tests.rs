//! Unit tests for `SelfSpeculativeModel`: draft-layer count calculation,
//! draft generation (empty/zero/multi-token), greedy argmax sampling,
//! KV-cache tracking, and `forward_to_layer` fallback behaviour.

use super::*;
use crate::speculative::config::SpeculationConfig;
use crate::speculative::verifier::DraftVerifier;
use crate::types::{SeqId, TokenId};
use vllm_traits::{BatchOutput, ModelBackend};

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
    let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
    let token = crate::sampling::greedy_sample(&logits);
    assert_eq!(token, 3); // index 3 has highest value 0.8
}

#[test]
fn test_greedy_sample_argmax_tie() {
    let logits = vec![0.1, 0.5, 0.3, 0.5, 0.2];
    let token = crate::sampling::greedy_sample(&logits);
    assert_eq!(token, 1); // first max at index 1
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
            next_tokens: seq_ids
                .iter()
                .map(|&id| TokenId::try_from(id * 10 + 1).unwrap_or(0))
                .collect(),
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
