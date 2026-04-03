//! Gemma4 Attention implementation.

use vllm_traits::{BatchOutput, ModelBackend, Result as EngineResult, SeqId, TokenId};

/// Gemma4 attention module with hybrid local/Global attention.
pub struct Gemma4Attention;

impl Gemma4Attention {
    /// Create a new Gemma4Attention.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Gemma4Attention {
    fn default() -> Self {
        Self
    }
}

impl ModelBackend for Gemma4Attention {
    fn forward(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> EngineResult<BatchOutput> {
        todo!()
    }

    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> EngineResult<Vec<Vec<f32>>> {
        todo!()
    }

    fn embed(
        &mut self,
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> EngineResult<Vec<Vec<f32>>> {
        todo!()
    }
}
