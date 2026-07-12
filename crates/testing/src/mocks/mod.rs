//! Unified mock implementations for testing.
//!
//! `NeverProgressModel` is the timeout/preemption test fixture; it
//! is constructed by integration tests via `NeverProgressModel::new`.
#![allow(dead_code)]
//!
//! This module consolidates all mock `ModelBackend` implementations
//! previously scattered across the codebase.

use vllm_traits::{BatchOutput, ModelBackend, Result, SeqId, TokenId};

/// Deterministic stub that returns a fixed token for every sequence.
#[derive(Debug, Clone, Copy)]
pub struct StubModel {
    token: TokenId,
}

impl Default for StubModel {
    fn default() -> Self {
        Self::returning(1)
    }
}

impl StubModel {
    #[must_use]
    pub const fn returning(token: TokenId) -> Self {
        Self { token }
    }
}

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
        // ARCH-02: make `forward_logits` agree with the configured
        // `token` so engine tests that depend on a deterministic
        // returned token still see it after the engine switched to
        // `forward_logits` + engine-side sampling.
        let vocab_size = self.vocab_size();
        let peak = f32::from(u16::try_from(self.token).unwrap_or(u16::MAX));
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                let mut logits = Vec::with_capacity(tokens.len() * vocab_size);
                for _ in tokens {
                    let mut pos_logits = vec![-10.0; vocab_size];
                    pos_logits[self.token as usize] = peak;
                    logits.extend(pos_logits);
                }
                logits
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

    fn vocab_size(&self) -> usize {
        151_936
    }

    fn num_layers(&self) -> usize {
        32
    }

    fn num_heads(&self) -> usize {
        32
    }
}

/// Model that returns `seq_id` + 1 as next token.
#[derive(Debug)]
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
        // invariant: stub test seq IDs are small (test fixture); truncation
        // is not reachable in practice.
        #[allow(clippy::cast_possible_truncation)]
        let tokens: Vec<TokenId> = seq_ids.iter().map(|id| *id as TokenId).collect();
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: tokens,
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
        let vocab_size = self.vocab_size();
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                let mut logits = Vec::with_capacity(tokens.len() * vocab_size);
                for &t in tokens {
                    let mut pos_logits = vec![-10.0; vocab_size];
                    pos_logits[t as usize] = 10.0;
                    logits.extend(pos_logits);
                }
                logits
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

    fn vocab_size(&self) -> usize {
        151_936
    }

    fn num_layers(&self) -> usize {
        32
    }

    fn num_heads(&self) -> usize {
        32
    }
}

/// Model that always returns a constant token.
#[derive(Debug)]
/// Used in: integration tests.
#[derive(Clone)]
pub struct ConstModel {
    pub return_token: TokenId,
}

impl ConstModel {
    #[must_use]
    pub const fn new(return_token: TokenId) -> Self {
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
        let vocab_size = self.vocab_size();
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                let mut logits = Vec::with_capacity(tokens.len() * vocab_size);
                for &t in tokens {
                    let mut pos_logits = vec![-10.0; vocab_size];
                    pos_logits[t as usize] = 10.0;
                    logits.extend(pos_logits);
                }
                logits
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
            .map(|t| t.iter().map(|_| 0.0).collect())
            .collect())
    }

    fn vocab_size(&self) -> usize {
        151_936
    }

    fn num_layers(&self) -> usize {
        32
    }

    fn num_heads(&self) -> usize {
        32
    }
}
#[derive(Debug)]

/// Model that returns `seq_id` % `vocab_size` as next token.
/// Used in: model tests.
pub struct FakeModel {
    vocab_size: usize,
}

impl FakeModel {
    #[must_use]
    pub const fn new(vocab_size: usize) -> Self {
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
        // invariant: stub test seq IDs are small; usize and u32 truncation are
        // bounded by vocab_size (a real tokenizer vocab size).
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
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
        let vocab_size = self.vocab_size();
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                let mut logits = Vec::with_capacity(tokens.len() * vocab_size);
                for &t in tokens {
                    let mut pos_logits = vec![-10.0; vocab_size];
                    pos_logits[t as usize] = 10.0;
                    logits.extend(pos_logits);
                }
                logits
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

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn num_layers(&self) -> usize {
        32
    }

    fn num_heads(&self) -> usize {
        32
    }
}

/// Model that never progresses (always returns same token).
/// Useful for testing timeouts and preemption.
#[allow(dead_code)]
pub(crate) struct NeverProgressModel {
    token: TokenId,
}

impl NeverProgressModel {
    pub const fn new(token: TokenId) -> Self {
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
        let vocab_size = self.vocab_size();
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                let mut logits = Vec::with_capacity(tokens.len() * vocab_size);
                for _ in tokens {
                    let mut pos_logits = vec![-10.0; vocab_size];
                    pos_logits[self.token as usize] = 10.0;
                    logits.extend(pos_logits);
                }
                logits
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
            .map(|t| t.iter().map(|_| 0.0).collect())
            .collect())
    }

    fn vocab_size(&self) -> usize {
        151_936
    }

    fn num_layers(&self) -> usize {
        32
    }

    fn num_heads(&self) -> usize {
        32
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
        let mut model = StubModel::default();
        let output = model
            .forward(&[1], &[vec![1]], &[vec![0]], &[vec![0]], &[0], &[true])
            .unwrap();
        assert_eq!(output.next_tokens, vec![1]);
    }

    #[test]
    fn test_stub_model_custom_token() {
        let mut model = StubModel::returning(42);
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
        assert_eq!(output.next_tokens, vec![42, 42]);
    }
}
