//! Mixtral model implementation.

use crate::config::ModelConfig;
use vllm_traits::ModelBackend;
use vllm_traits::types::{BatchOutput, SeqId, TokenId};

pub struct MixtralModel {
    // Placeholder for model implementation
}

impl MixtralModel {
    pub fn from_weights(
        _config: ModelConfig,
        _device: candle_core::Device,
        _weights: std::collections::HashMap<String, candle_core::Tensor>,
        _num_kv_blocks: usize,
    ) -> candle_core::Result<Self> {
        todo!("MixtralModel from_weights not implemented")
    }
}

impl ModelBackend for MixtralModel {
    fn forward(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput, vllm_traits::ModelError> {
        todo!("MixtralModel forward not implemented")
    }

    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>, vllm_traits::ModelError> {
        todo!("MixtralModel forward_logits not implemented")
    }

    fn embed(
        &mut self,
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>, vllm_traits::ModelError> {
        todo!("MixtralModel embed not implemented")
    }
}
