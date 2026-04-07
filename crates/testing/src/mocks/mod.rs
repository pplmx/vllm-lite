//! Unified mock implementations for testing.
//!
//! This module consolidates all mock ModelBackend implementations
//! previously scattered across the codebase.

use vllm_traits::{BatchOutput, ModelBackend, Result, SeqId, TokenId};

/// Stub model that returns seq_id as next token.
/// Used in: prefix_cache tests.
pub struct StubModel;

impl ModelBackend for StubModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| 1 as TokenId).collect(),
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
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| tokens.iter().map(|_| 0.0).collect())
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| tokens.iter().map(|_| 0.0).collect())
            .collect())
    }
}

/// Model that returns seq_id + 1 as next token.
/// Used in: integration tests.
pub struct IncrementModel;

impl ModelBackend for IncrementModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|id| *id as TokenId).collect(),
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
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| tokens.iter().map(|_| 0.0).collect())
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| tokens.iter().map(|_| 0.0).collect())
            .collect())
    }
}

/// Model that always returns a constant token.
/// Used in: integration tests.
#[derive(Clone)]
pub struct ConstModel {
    pub return_token: TokenId,
}

impl ConstModel {
    pub fn new(return_token: TokenId) -> Self {
        Self { return_token }
    }
}

impl ModelBackend for ConstModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| self.return_token).collect(),
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
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|t| t.iter().map(|_| 0.0).collect())
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|t| t.iter().map(|_| 0.0).collect())
            .collect())
    }
}

/// Model that returns seq_id % vocab_size as next token.
/// Used in: model tests.
pub struct FakeModel {
    vocab_size: usize,
}

impl FakeModel {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
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
    ) -> Result<BatchOutput> {
        let next_tokens: Vec<TokenId> = seq_ids
            .iter()
            .map(|&id| ((id as usize) % self.vocab_size) as TokenId)
            .collect();

        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
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
    ) -> Result<Vec<Vec<f32>>> {
        let vocab_size = 32000;
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                tokens
                    .iter()
                    .map(|_| rand::random::<f32>() * vocab_size as f32)
                    .collect()
            })
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| tokens.iter().map(|_| 0.0).collect())
            .collect())
    }
}

/// Model that never progresses (always returns same token).
/// Useful for testing timeouts and preemption.
pub struct NeverProgressModel {
    token: TokenId,
}

impl NeverProgressModel {
    pub fn new(token: TokenId) -> Self {
        Self { token }
    }
}

impl ModelBackend for NeverProgressModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| self.token).collect(),
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
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|t| t.iter().map(|_| 0.0).collect())
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|t| t.iter().map(|_| 0.0).collect())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increment_model() {
        let mut model = IncrementModel;
        let output = model
            .forward(
                &[1, 2],
                &[vec![1], vec![2]],
                &[vec![0], vec![0]],
                &[vec![0], vec![0]],
                &[0, 0],
                &[true, true],
            )
            .unwrap();
        assert_eq!(output.next_tokens, vec![1, 2]);
    }

    #[test]
    fn test_const_model() {
        let mut model = ConstModel::new(42);
        let output = model
            .forward(&[1], &[vec![1]], &[vec![0]], &[vec![0]], &[0], &[true])
            .unwrap();
        assert_eq!(output.next_tokens, vec![42]);
    }

    #[test]
    fn test_fake_model() {
        let mut model = FakeModel::new(100);
        let output = model
            .forward(
                &[1, 50, 99],
                &[vec![1], vec![2], vec![3]],
                &[vec![0], vec![0], vec![0]],
                &[vec![0], vec![0], vec![0]],
                &[0, 0, 0],
                &[true, true, true],
            )
            .unwrap();
        assert_eq!(output.next_tokens, vec![1, 50, 99]);
    }

    #[test]
    fn test_never_progress_model() {
        let mut model = NeverProgressModel::new(777);
        let output = model
            .forward(
                &[1, 2, 3],
                &[vec![1], vec![2], vec![3]],
                &[vec![0], vec![0], vec![0]],
                &[vec![0], vec![0], vec![0]],
                &[0, 0, 0],
                &[true, true, true],
            )
            .unwrap();
        assert_eq!(output.next_tokens, vec![777, 777, 777]);
    }

    #[test]
    fn test_stub_model() {
        let mut model = StubModel;
        let output = model
            .forward(&[1], &[vec![1]], &[vec![0]], &[vec![0]], &[0], &[true])
            .unwrap();
        assert_eq!(output.next_tokens, vec![1]);
    }
}
