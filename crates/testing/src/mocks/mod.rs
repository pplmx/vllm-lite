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
    /// Create a stub that always returns `token`.
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
            // P36: placeholder SampledToken (no logprob info on the
            // legacy model-layer `forward` path; engine uses
            // `forward_logits` + `sample_batch_with_params`).
            next_tokens: seq_ids
                .iter()
                .map(|_| vllm_traits::SampledToken {
                    token: self.token,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                })
                .collect(),
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
        // P36: placeholder SampledToken (no logprob info on the
        // legacy model-layer `forward` path; engine uses
        // `forward_logits` + `sample_batch_with_params`).
        #[allow(clippy::cast_possible_truncation)]
        let next_tokens: Vec<vllm_traits::SampledToken> = seq_ids
            .iter()
            .map(|id| vllm_traits::SampledToken {
                token: *id as TokenId,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            })
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
    /// Create a const model that always returns `return_token`.
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
            // P36: placeholder SampledToken (no logprob info on the
            // legacy model-layer `forward` path; engine uses
            // `forward_logits` + `sample_batch_with_params`).
            next_tokens: seq_ids
                .iter()
                .map(|_| vllm_traits::SampledToken {
                    token: self.return_token,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                })
                .collect(),
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
    /// Create a fake model with the given vocabulary size.
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
        // P36: placeholder SampledToken (no logprob info on the
        // legacy model-layer `forward` path; engine uses
        // `forward_logits` + `sample_batch_with_params`).
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let next_tokens: Vec<vllm_traits::SampledToken> = seq_ids
            .iter()
            .map(|&id| vllm_traits::SampledToken {
                token: ((id as usize) % self.vocab_size) as TokenId,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            })
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
    /// Create a model that stalls on every forward call with the given return token.
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
            // P36: placeholder SampledToken (no logprob info on the
            // legacy model-layer `forward` path; engine uses
            // `forward_logits` + `sample_batch_with_params`).
            next_tokens: seq_ids
                .iter()
                .map(|_| vllm_traits::SampledToken {
                    token: self.token,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                })
                .collect(),
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

/// A model whose `forward` and `forward_logits` agree on the next token.
///
/// `FakeModel` returns `seq_id % vocab_size` from `forward` but peaks the
/// logits at the INPUT token, so a draft produced through `forward` never
/// matches what the target's `forward_logits` predicts — every draft is
/// rejected and the speculative path degenerates to one token per step. Any
/// test that needs real multi-token speculative bursts (stop handling across
/// a burst, acceptance accounting) needs a fixture where the two agree.
///
/// Both paths emit `last_input_token + 1` (mod `vocab_size`), so the draft's
/// prediction equals the target's prediction at the same position and a
/// speculative step accepts the whole draft budget.
#[derive(Debug, Clone, Copy)]
pub struct ConsistentModel {
    vocab_size: usize,
}

impl ConsistentModel {
    /// Create a consistent model with the given vocabulary size.
    #[must_use]
    pub const fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    /// The token both `forward` and `forward_logits` predict for `token`.
    const fn next_of(token: TokenId, vocab_size: usize) -> TokenId {
        if vocab_size == 0 {
            return 0;
        }
        ((token as usize + 1) % vocab_size) as TokenId
    }
}

impl ModelBackend for ConsistentModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: (0..seq_ids.len())
                .map(|i| {
                    let last = input_tokens
                        .get(i)
                        .and_then(|t| t.last())
                        .copied()
                        .unwrap_or(0);
                    vllm_traits::SampledToken {
                        token: Self::next_of(last, self.vocab_size),
                        logprob: 0.0,
                        top_logprobs: Vec::new(),
                    }
                })
                .collect(),
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
            .map(|tokens| {
                let mut logits = Vec::with_capacity(tokens.len() * self.vocab_size);
                for &t in tokens {
                    let mut pos_logits = vec![-10.0_f32; self.vocab_size];
                    let next = Self::next_of(t, self.vocab_size) as usize;
                    if next < pos_logits.len() {
                        pos_logits[next] = 10.0;
                    }
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
        Ok(input_tokens.iter().map(|_| vec![0.0_f32; 8]).collect())
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
        // P36: forward returns placeholder SampledToken (no logprob
        // info on the legacy model-layer path).
        assert_eq!(
            output.next_tokens,
            vec![
                vllm_traits::SampledToken {
                    token: 1,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                },
                vllm_traits::SampledToken {
                    token: 2,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                },
            ]
        );
    }

    #[test]
    fn test_const_model() {
        let mut model = ConstModel::new(42);
        let output = model
            .forward(&[1], &[vec![1]], &[vec![0]], &[vec![0]], &[0], &[true])
            .unwrap();
        assert_eq!(
            output.next_tokens,
            vec![vllm_traits::SampledToken {
                token: 42,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            }]
        );
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
        assert_eq!(
            output.next_tokens,
            vec![
                vllm_traits::SampledToken {
                    token: 1,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                },
                vllm_traits::SampledToken {
                    token: 50,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                },
                vllm_traits::SampledToken {
                    token: 99,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                },
            ]
        );
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
        assert_eq!(
            output.next_tokens,
            vec![
                vllm_traits::SampledToken {
                    token: 777,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                },
                vllm_traits::SampledToken {
                    token: 777,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                },
                vllm_traits::SampledToken {
                    token: 777,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                },
            ]
        );
    }

    #[test]
    fn test_stub_model() {
        let mut model = StubModel::default();
        let output = model
            .forward(&[1], &[vec![1]], &[vec![0]], &[vec![0]], &[0], &[true])
            .unwrap();
        assert_eq!(
            output.next_tokens,
            vec![vllm_traits::SampledToken {
                // `StubModel::default()` returns `returning(1)` per
                // the Default impl above; pin the matching token id
                // here.
                token: 1,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            }]
        );
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
        assert_eq!(
            output.next_tokens,
            vec![
                vllm_traits::SampledToken {
                    token: 42,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                },
                vllm_traits::SampledToken {
                    token: 42,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                },
            ]
        );
    }

    // ── forward_logits ────────────────────────────────────────────────

    #[test]
    fn test_stub_model_forward_logits() {
        let mut model = StubModel::default(); // token = 1
        let logits = model
            .forward_logits(&[1], &[vec![10]], &[vec![0]], &[vec![0]], &[0], &[true])
            .unwrap();
        assert_eq!(logits.len(), 1);
        let vs = model.vocab_size();
        assert_eq!(logits[0].len(), vs);
        // peak at self.token (1), value = 1.0f32
        assert!((logits[0][1] - 1.0).abs() < f32::EPSILON);
        // all other positions are -10.0
        assert!((logits[0][0] + 10.0).abs() < f32::EPSILON);
        assert!((logits[0][2] + 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_increment_model_forward_logits() {
        let mut model = IncrementModel;
        let logits = model
            .forward_logits(
                &[1, 2],
                &[vec![5], vec![8]],
                &[vec![0], vec![0]],
                &[vec![0], vec![0]],
                &[0, 0],
                &[true, true],
            )
            .unwrap();
        assert_eq!(logits.len(), 2);
        let vs = model.vocab_size();
        // first seq has 1 token → vs logits, peak at index 5
        assert_eq!(logits[0].len(), vs);
        assert!((logits[0][5] - 10.0).abs() < f32::EPSILON);
        // second seq has 1 token → vs logits, peak at index 8
        assert_eq!(logits[1].len(), vs);
        assert!((logits[1][8] - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_const_model_forward_logits() {
        let mut model = ConstModel::new(42);
        let logits = model
            .forward_logits(&[1], &[vec![7]], &[vec![0]], &[vec![0]], &[0], &[true])
            .unwrap();
        assert_eq!(logits.len(), 1);
        let vs = model.vocab_size();
        assert_eq!(logits[0].len(), vs);
        // forward_logits peaks at the *input* token (7), not return_token (42)
        assert!((logits[0][7] - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fake_model_forward_logits() {
        let mut model = FakeModel::new(100);
        let logits = model
            .forward_logits(&[1], &[vec![3]], &[vec![0]], &[vec![0]], &[0], &[true])
            .unwrap();
        assert_eq!(logits.len(), 1);
        // FakeModel vocab_size is configurable
        assert_eq!(logits[0].len(), 100);
        assert!((logits[0][3] - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_never_progress_model_forward_logits() {
        let mut model = NeverProgressModel::new(777);
        let logits = model
            .forward_logits(&[1], &[vec![1]], &[vec![0]], &[vec![0]], &[0], &[true])
            .unwrap();
        assert_eq!(logits.len(), 1);
        let vs = model.vocab_size();
        assert_eq!(logits[0].len(), vs);
        assert!((logits[0][777] - 10.0).abs() < f32::EPSILON);
    }

    // ── embed ─────────────────────────────────────────────────────────

    #[test]
    fn test_stub_model_embed() {
        let mut model = StubModel::default();
        let embeddings = model.embed(&[vec![1, 2, 3]], &[vec![0, 1, 2]]).unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 3);
        assert!(embeddings[0].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_increment_model_embed() {
        let mut model = IncrementModel;
        let embeddings = model
            .embed(&[vec![1], vec![2]], &[vec![0], vec![0]])
            .unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 1);
        assert_eq!(embeddings[1].len(), 1);
        assert!(embeddings[0].iter().all(|&v| v == 0.0));
        assert!(embeddings[1].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_const_model_embed() {
        let mut model = ConstModel::new(42);
        let embeddings = model.embed(&[vec![5]], &[vec![0]]).unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 1);
        assert!(embeddings[0].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_fake_model_embed() {
        let mut model = FakeModel::new(100);
        let embeddings = model
            .embed(&[vec![1, 2, 3, 4, 5]], &[vec![0, 1, 2, 3, 4]])
            .unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 5);
        assert!(embeddings[0].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_never_progress_model_embed() {
        let mut model = NeverProgressModel::new(777);
        let embeddings = model.embed(&[vec![1]], &[vec![0]]).unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 1);
        assert!(embeddings[0].iter().all(|&v| v == 0.0));
    }

    // ── trait metadata ────────────────────────────────────────────────

    #[test]
    fn test_stub_model_metadata() {
        let model = StubModel::default();
        assert_eq!(model.vocab_size(), 151_936);
        assert_eq!(model.num_layers(), 32);
        assert_eq!(model.num_heads(), 32);
    }

    #[test]
    fn test_increment_model_metadata() {
        let model = IncrementModel;
        assert_eq!(model.vocab_size(), 151_936);
        assert_eq!(model.num_layers(), 32);
        assert_eq!(model.num_heads(), 32);
    }

    #[test]
    fn test_const_model_metadata() {
        let model = ConstModel::new(42);
        assert_eq!(model.vocab_size(), 151_936);
        assert_eq!(model.num_layers(), 32);
        assert_eq!(model.num_heads(), 32);
    }

    /// The whole point of the fixture: `forward` and `forward_logits` must
    /// predict the SAME token, which is what lets a speculative draft be
    /// accepted. `FakeModel` fails that (forward returns `seq_id % vocab`,
    /// logits peak at the input token), so every draft is rejected there.
    #[test]
    fn test_consistent_model_forward_and_logits_agree() {
        let mut model = ConsistentModel::new(1000);
        let seq_ids = vec![1u64];
        let input = vec![vec![42u32]];
        let out = model
            .forward(&seq_ids, &input, &[vec![0]], &[vec![0]], &[0], &[false])
            .expect("forward");
        let logits = model
            .forward_logits(&seq_ids, &input, &[vec![0]], &[vec![0]], &[0], &[false])
            .expect("logits");

        let from_forward = out.next_tokens[0].token;
        let best = logits[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite logits"))
            .map(|(i, _)| i)
            .expect("non-empty logits");
        assert_eq!(
            from_forward as usize, best,
            "forward and forward_logits must predict the same token, or speculative drafts can never be accepted"
        );
        assert_eq!(from_forward, 43, "42 -> 43 under the +1 rule");
    }

    /// Wrapping at the vocabulary edge must stay in range.
    #[test]
    fn test_consistent_model_wraps_at_vocab_edge() {
        let mut model = ConsistentModel::new(8);
        let out = model
            .forward(
                &[1u64],
                &[vec![7u32]],
                &[vec![0]],
                &[vec![0]],
                &[0],
                &[false],
            )
            .expect("forward");
        assert_eq!(out.next_tokens[0].token, 0, "7 + 1 wraps to 0");
    }

    #[test]
    fn test_fake_model_metadata() {
        let model = FakeModel::new(100);
        assert_eq!(model.vocab_size(), 100);
        assert_eq!(model.num_layers(), 32);
        assert_eq!(model.num_heads(), 32);
    }

    #[test]
    fn test_never_progress_model_metadata() {
        let model = NeverProgressModel::new(777);
        assert_eq!(model.vocab_size(), 151_936);
        assert_eq!(model.num_layers(), 32);
        assert_eq!(model.num_heads(), 32);
    }
}
