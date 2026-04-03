//! Gemma4 Model implementation.

use crate::gemma4::block::Gemma4Block;
use vllm_traits::{BatchOutput, ModelBackend, Result as EngineResult, SeqId, TokenId};

/// Gemma4 model with transformer blocks.
pub struct Gemma4Model {
    _num_layers: usize,
    _layers: Vec<Gemma4Block>,
}

impl Gemma4Model {
    /// Create a new Gemma4Model.
    pub fn new(num_layers: usize) -> Self {
        let _layers = (0..num_layers).map(|_| Gemma4Block::new()).collect();
        Self {
            _num_layers: num_layers,
            _layers,
        }
    }
}

impl ModelBackend for Gemma4Model {
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
