//! Shared pre-norm decoder block: RMSNorm → RoPE GQA → residual → RMSNorm → SwiGLU → residual.
//!
//! Used by Llama, Mistral, and Qwen3 causal-LM stacks.

use crate::components::LnLayerNorm;
use crate::components::SwiGLU;
use crate::components::attention::RopeGqaAttention;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Result, Tensor};

/// Standard decoder layer with RoPE group-query attention and SwiGLU FFN.
pub struct RopeGqaDecoderBlock {
    input_layernorm: LnLayerNorm,
    post_attention_layernorm: LnLayerNorm,
    attention: RopeGqaAttention,
    mlp: SwiGLU,
}

impl RopeGqaDecoderBlock {
    pub fn new(
        input_layernorm: LnLayerNorm,
        post_attention_layernorm: LnLayerNorm,
        attention: RopeGqaAttention,
        mlp: SwiGLU,
    ) -> Self {
        Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    pub fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self
            .attention
            .forward_prefill(&x, kv_cache, layer_idx, block_ids, positions)?;
        let x = (&x + &residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        let residual = x.clone();
        let mut x = self.input_layernorm.forward(x)?;
        x = self.attention.forward_decode(
            &x,
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            positions,
        )?;
        if x.dims().len() == 3 && x.dims()[1] == 1 && residual.dims().len() == 2 {
            let dims = x.dims();
            let batch_size = dims[0];
            let hidden_size: usize = dims[2];
            x = x.reshape((batch_size, hidden_size))?;
        }
        x = (&x + &residual)?;

        let residual = x.clone();
        x = self.post_attention_layernorm.forward(&x)?;
        x = self.mlp.forward(&x)?;
        x.add(&residual)
    }
}

/// View any decoder-layer wrapper as the shared block.
pub trait AsDecoderBlock {
    fn as_decoder_block(&self) -> &RopeGqaDecoderBlock;
}

impl AsDecoderBlock for RopeGqaDecoderBlock {
    fn as_decoder_block(&self) -> &RopeGqaDecoderBlock {
        self
    }
}
