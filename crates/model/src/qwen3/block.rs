#![allow(clippy::type_complexity)]

use super::{
    attention::{AttentionConfig, GqaAttention},
    mlp::SwiGLU,
};
use crate::kv_cache::PagedKvCache;
use candle_core::{Module, Result, Tensor};
use candle_nn::LayerNorm;

pub struct TransformerBlock {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    attention: GqaAttention,
    mlp: SwiGLU,
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
        vb: Option<candle_nn::VarBuilder>,
        has_qk_norm: bool,
    ) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });

        let input_layernorm =
            candle_nn::layer_norm(hidden_size, rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            candle_nn::layer_norm(hidden_size, rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        let vb_attn = vb.pp("attn");
        let attention = GqaAttention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            Some(vb_attn),
            AttentionConfig::default(),
            has_qk_norm,
        )?;

        let vb_mlp = vb.pp("mlp");
        let mlp = SwiGLU::new(hidden_size, intermediate_size, Some(vb_mlp))?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_weights(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
        has_qk_norm: bool,
        weights: Option<(
            Option<Tensor>, // q_proj
            Option<Tensor>, // k_proj
            Option<Tensor>, // v_proj
            Option<Tensor>, // o_proj
            Option<Tensor>, // gate_proj
            Option<Tensor>, // up_proj
            Option<Tensor>, // down_proj
            Option<Tensor>, // input_layernorm
            Option<Tensor>, // post_attention_layernorm
            Option<Tensor>, // q_norm
            Option<Tensor>, // k_norm
        )>,
    ) -> Result<Self> {
        let Some((
            Some(q_w),
            Some(k_w),
            Some(v_w),
            Some(o_w),
            Some(gate_w),
            Some(up_w),
            Some(down_w),
            Some(input_ln_w),
            Some(post_attn_ln_w),
            q_norm_w,
            k_norm_w,
        )) = weights
        else {
            return Err(candle_core::Error::msg("Missing layer weights"));
        };

        let input_ln_dim = input_ln_w.dim(0).unwrap_or(hidden_size);
        let input_ln_bias = Tensor::zeros(input_ln_dim, input_ln_w.dtype(), input_ln_w.device())?;
        let input_layernorm = LayerNorm::new(input_ln_w.clone(), input_ln_bias, rms_norm_eps);

        let post_attn_dim = post_attn_ln_w.dim(0).unwrap_or(hidden_size);
        let post_attn_bias = Tensor::zeros(
            post_attn_dim,
            post_attn_ln_w.dtype(),
            post_attn_ln_w.device(),
        )?;
        let post_attention_layernorm =
            LayerNorm::new(post_attn_ln_w.clone(), post_attn_bias, rms_norm_eps);

        let attention = GqaAttention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            q_w,
            k_w,
            v_w,
            o_w,
            AttentionConfig::default(),
            has_qk_norm,
            q_norm_w,
            k_norm_w,
        )?;

        let mlp = SwiGLU::new_with_weights(hidden_size, intermediate_size, gate_w, up_w, down_w)?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        })
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
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self
            .attention
            .forward_prefill(&x, kv_cache, layer_idx, block_ids)?;
        let x = (&x + &residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (&x + &residual)?;

        Ok(x)
    }

    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward_decode(
            &x,
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
        )?;
        let x = (&x + &residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (&x + &residual)?;

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_transformer_block_forward() -> Result<()> {
        let device = Device::Cpu;
        let block = TransformerBlock::new(256, 4, 2, 64, 512, 1e-6, None, false)?;

        let x = Tensor::ones((1, 2, 256), DType::F32, &device)?;
        let output = block.forward(&x)?;

        assert_eq!(output.dims(), &[1, 2, 256]);
        Ok(())
    }

    #[test]
    fn test_transformer_block_batch_forward() -> Result<()> {
        let device = Device::Cpu;
        let block = TransformerBlock::new(128, 4, 2, 32, 256, 1e-6, None, false)?;

        let x = Tensor::ones((4, 3, 128), DType::F32, &device)?;
        let output = block.forward(&x)?;

        assert_eq!(output.dims(), &[4, 3, 128]);
        Ok(())
    }

    #[test]
    fn test_transformer_block_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let block = TransformerBlock::new(128, 4, 2, 32, 256, 1e-6, None, false)?;

        let x = Tensor::zeros((2, 1, 128), DType::F32, &device)?;
        let output = block.forward(&x)?;

        assert_eq!(output.dims(), &[2, 1, 128]);
        Ok(())
    }

    #[test]
    fn test_transformer_block_with_qk_norm() -> Result<()> {
        // Qwen3-0.6B uses q_norm/k_norm
        let device = Device::Cpu;
        let block = TransformerBlock::new(128, 4, 2, 32, 256, 1e-6, None, true)?;

        let x = Tensor::ones((1, 2, 128), DType::F32, &device)?;
        let output = block.forward(&x)?;

        assert_eq!(output.dims(), &[1, 2, 128]);
        Ok(())
    }

    #[test]
    fn test_transformer_block_with_custom_head_dim() -> Result<()> {
        // Qwen3-0.6B has head_dim=128 but hidden/heads=1024/16=64
        // This test verifies custom head_dim works
        let device = Device::Cpu;
        // hidden=1024, heads=16, kv_heads=8, head_dim=128, intermediate=3072
        let block = TransformerBlock::new(1024, 16, 8, 128, 3072, 1e-6, None, true)?;

        let x = Tensor::ones((1, 4, 1024), DType::F32, &device)?;
        let output = block.forward(&x)?;

        assert_eq!(output.dims(), &[1, 4, 1024]);
        Ok(())
    }
}
