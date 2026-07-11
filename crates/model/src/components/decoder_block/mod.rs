#![allow(clippy::module_name_repetitions)]
//! Shared pre-norm decoder block: `RMSNorm` → `RoPE` GQA → residual → `RMSNorm` → `SwiGLU` → residual.
//!
//! Used by Llama, Mistral, and Qwen3 causal-LM stacks.
//! Registry blocks implement this trait via [`TransformerBlock`].
//!
//! [`TransformerBlock`]: crate::components::block::TransformerBlock

pub mod factory;

use crate::components::LnLayerNorm;
use crate::components::SwiGLU;
use crate::components::attention::RopeGqaAttention;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Result, Tensor};

pub use factory::{block_from_weights, new_block};

#[derive(Debug)]
/// Standard decoder layer with `RoPE` group-query attention and `SwiGLU` FFN.
pub struct RopeGqaDecoderBlock {
    input_layernorm: LnLayerNorm,
    post_attention_layernorm: LnLayerNorm,
    attention: RopeGqaAttention,
    mlp: SwiGLU,
}

impl RopeGqaDecoderBlock {
    #[must_use]
    pub const fn new(
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

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
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

    /// Run the prefill path: process the full prompt and cache its KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
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

    /// Run a chunked-prefill continuation against an existing KV prefix.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_prefill_continue(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
        num_computed_tokens: usize,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward_prefill_continue(
            &x,
            kv_cache,
            layer_idx,
            block_ids,
            positions,
            num_computed_tokens,
        )?;
        let x = (&x + &residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    /// Run the decode path: process one new token against cached KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
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

/// Paged-KV decoder layer: prefill and single-token decode with block table.
pub trait PagedDecoderBlock {
    /// Run the prefill path: process the full prompt and cache its KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor>;

    /// Run the decode path: process one new token against cached KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor>;
}

impl PagedDecoderBlock for RopeGqaDecoderBlock {
    fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        Self::forward_prefill(self, x, kv_cache, layer_idx, block_ids, positions)
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        Self::forward_decode(
            self,
            x,
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            positions,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::AttentionConfig;
    use candle_core::{DType, Device, Tensor};

    fn tiny_block(device: &Device) -> RopeGqaDecoderBlock {
        let hidden = 64usize;
        let input_ln_weight = Tensor::ones(hidden, DType::F32, device).unwrap();
        let input_ln_bias = Tensor::zeros(hidden, DType::F32, device).unwrap();
        let input_layernorm = LnLayerNorm::new(input_ln_weight, input_ln_bias, 1e-5);

        let post_ln_weight = Tensor::ones(hidden, DType::F32, device).unwrap();
        let post_ln_bias = Tensor::zeros(hidden, DType::F32, device).unwrap();
        let post_attention_layernorm = LnLayerNorm::new(post_ln_weight, post_ln_bias, 1e-5);

        let attention = RopeGqaAttention::new(
            hidden,
            4,
            2,
            16,
            10_000.0,
            None,
            AttentionConfig::default(),
            false,
        )
        .unwrap();
        let mlp = SwiGLU::new(hidden, 128, None).unwrap();

        RopeGqaDecoderBlock::new(input_layernorm, post_attention_layernorm, attention, mlp)
    }

    #[test]
    fn test_decoder_prefill_continue_matches_full_prefill() {
        let device = Device::Cpu;
        let block = tiny_block(&device);

        let seq_len = 6usize;
        let x = Tensor::ones((1, seq_len, 64), DType::F32, &device).unwrap();
        let block_ids: Vec<usize> = (0..seq_len).map(|i| i / 16).collect();
        let positions: Vec<usize> = (0..seq_len).collect();

        let mut full_cache =
            PagedKvCache::new(1, 4, 16, 32, device.clone(), false).unwrap();
        let full_out = block
            .forward_prefill(&x, &mut full_cache, 0, &block_ids, &positions)
            .unwrap();

        let mut chunked_cache =
            PagedKvCache::new(1, 4, 16, 32, device.clone(), false).unwrap();
        let chunk1 = x.narrow(1, 0, 3).unwrap();
        let pos1: Vec<usize> = (0..3).collect();
        block
            .forward_prefill(&chunk1, &mut chunked_cache, 0, &block_ids, &pos1)
            .unwrap();

        let chunk2 = x.narrow(1, 3, 3).unwrap();
        let pos2: Vec<usize> = (3..6).collect();
        let cont_out = block
            .forward_prefill_continue(
                &chunk2,
                &mut chunked_cache,
                0,
                &block_ids,
                &pos2,
                3,
            )
            .unwrap();

        let full_last = full_out.narrow(1, seq_len - 1, 1).unwrap();
        let cont_last = cont_out.narrow(1, 2, 1).unwrap();
        let diff = (full_last - cont_last)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "chunked continuation diverged: diff={diff}");
    }

    #[test]
    fn test_decoder_prefill_then_decode_shape() {
        let device = Device::Cpu;
        let block = tiny_block(&device);
        let mut kv_cache = PagedKvCache::new(1, 4, 16, 32, device.clone(), false).unwrap();

        let seq_len = 4usize;
        let x = Tensor::ones((1, seq_len, 64), DType::F32, &device).unwrap();
        let block_ids: Vec<usize> = (0..seq_len).map(|i| i / 16).collect();
        let positions: Vec<usize> = (0..seq_len).collect();

        let prefill_out = block
            .forward_prefill(&x, &mut kv_cache, 0, &block_ids, &positions)
            .unwrap();
        assert_eq!(prefill_out.dims(), &[1, seq_len, 64]);

        let decode_x = Tensor::ones((1, 64), DType::F32, &device).unwrap();
        let decode_out = block
            .forward_decode(&decode_x, &mut kv_cache, 0, &[0], seq_len, &[seq_len])
            .unwrap();
        assert_eq!(decode_out.dims(), &[1, 64]);
    }
}
