#![allow(clippy::type_complexity)]

use super::attention::Qwen3Attention;
use crate::components::AttentionConfig;
use crate::components::LnLayerNorm;
use crate::components::SwiGLU;
use crate::kv_cache::PagedKvCache;
use candle_core::{Result, Tensor};
use vllm_dist::TensorParallelConfig;

pub struct TransformerBlock {
    input_layernorm: LnLayerNorm,
    post_attention_layernorm: LnLayerNorm,
    attention: Qwen3Attention,
    mlp: SwiGLU,
    #[allow(dead_code)]
    tp_config: Option<TensorParallelConfig>,
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        theta: f32,
        rms_norm_eps: f64,
        vb: Option<candle_nn::VarBuilder>,
        has_qk_norm: bool,
    ) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });
        let device = candle_core::Device::Cpu;

        let input_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let input_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
        let input_layernorm = LnLayerNorm::new(input_ln_weight, input_ln_bias, rms_norm_eps);

        let post_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let post_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
        let post_attention_layernorm = LnLayerNorm::new(post_ln_weight, post_ln_bias, rms_norm_eps);

        let vb_attn = vb.pp("attn");
        let attention = Qwen3Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
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
            tp_config: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_tp(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        theta: f32,
        rms_norm_eps: f64,
        tp_config: Option<TensorParallelConfig>,
        has_qk_norm: bool,
    ) -> Result<Self> {
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu);
        let device = candle_core::Device::Cpu;

        let input_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let input_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
        let input_layernorm = LnLayerNorm::new(input_ln_weight, input_ln_bias, rms_norm_eps);

        let post_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let post_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
        let post_attention_layernorm = LnLayerNorm::new(post_ln_weight, post_ln_bias, rms_norm_eps);

        let vb_attn = vb.pp("attn");
        let attention = Qwen3Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
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
            tp_config,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_weights(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        theta: f32,
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

        let input_ln_bias = Tensor::zeros(
            input_ln_w.dim(0).unwrap_or(hidden_size),
            input_ln_w.dtype(),
            input_ln_w.device(),
        )?;
        let input_layernorm = LnLayerNorm::new(input_ln_w, input_ln_bias, rms_norm_eps);

        let post_attn_bias = Tensor::zeros(
            post_attn_ln_w.dim(0).unwrap_or(hidden_size),
            post_attn_ln_w.dtype(),
            post_attn_ln_w.device(),
        )?;
        let post_attention_layernorm =
            LnLayerNorm::new(post_attn_ln_w, post_attn_bias, rms_norm_eps);

        let attention = Qwen3Attention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
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
            tp_config: None,
        })
    }

    pub fn from_weights(
        config: &crate::config::ModelConfig,
        layer_idx: usize,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let theta = config.rope_theta;
        let rms_norm_eps = config.rms_norm_eps;
        let has_qk_norm = false;

        let get_weight = |keys: &[&str]| -> Option<&Tensor> {
            for key in keys {
                if let Some(w) = weights.get(*key) {
                    return Some(w);
                }
            }
            None
        };

        let q_key = get_weight(&[
            &format!("model.layers.{}.self_attn.q_proj.weight", layer_idx),
            &format!("model.layers.{}.attn.q_proj.weight", layer_idx),
        ]);
        let k_key = get_weight(&[
            &format!("model.layers.{}.self_attn.k_proj.weight", layer_idx),
            &format!("model.layers.{}.attn.k_proj.weight", layer_idx),
        ]);
        let v_key = get_weight(&[
            &format!("model.layers.{}.self_attn.v_proj.weight", layer_idx),
            &format!("model.layers.{}.attn.v_proj.weight", layer_idx),
        ]);
        let o_key = get_weight(&[
            &format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
            &format!("model.layers.{}.attn.o_proj.weight", layer_idx),
        ]);

        let q_norm_key = format!("model.layers.{}.self_attn.q_norm.weight", layer_idx);
        let k_norm_key = format!("model.layers.{}.self_attn.k_norm.weight", layer_idx);
        let q_norm_weight = weights.get(&q_norm_key).cloned();
        let k_norm_weight = weights.get(&k_norm_key).cloned();

        let layer_weights = Some((
            q_key.cloned(),
            k_key.cloned(),
            v_key.cloned(),
            o_key.cloned(),
            weights
                .get(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))
                .cloned(),
            weights
                .get(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx))
                .cloned(),
            weights
                .get(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx))
                .cloned(),
            weights
                .get(&format!(
                    "model.layers.{}.input_layernorm.weight",
                    layer_idx
                ))
                .cloned(),
            weights
                .get(&format!(
                    "model.layers.{}.post_attention_layernorm.weight",
                    layer_idx
                ))
                .cloned(),
            q_norm_weight,
            k_norm_weight,
        ));

        Self::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            theta,
            rms_norm_eps,
            has_qk_norm,
            layer_weights,
        )
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
        let x = (&x + &residual)?;

        Ok(x)
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
        x = (&x + &residual)?;

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
        let block = TransformerBlock::new(256, 4, 2, 64, 512, 10000.0, 1e-6, None, false)?;

        let x = Tensor::ones((1, 2, 256), DType::F32, &device)?;
        let output = block.forward(&x)?;

        assert_eq!(output.dims(), &[1, 2, 256]);
        Ok(())
    }

    #[test]
    fn test_transformer_block_batch_forward() -> Result<()> {
        let device = Device::Cpu;
        let block = TransformerBlock::new(128, 4, 2, 32, 256, 10000.0, 1e-6, None, false)?;

        let x = Tensor::ones((4, 3, 128), DType::F32, &device)?;
        let output = block.forward(&x)?;

        assert_eq!(output.dims(), &[4, 3, 128]);
        Ok(())
    }

    #[test]
    fn test_transformer_block_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let block = TransformerBlock::new(128, 4, 2, 32, 256, 10000.0, 1e-6, None, false)?;

        let x = Tensor::zeros((2, 1, 128), DType::F32, &device)?;
        let output = block.forward(&x)?;

        assert_eq!(output.dims(), &[2, 1, 128]);
        Ok(())
    }

    #[test]
    fn test_transformer_block_with_qk_norm() -> Result<()> {
        // Qwen3-0.6B uses q_norm/k_norm
        let device = Device::Cpu;
        let block = TransformerBlock::new(128, 4, 2, 32, 256, 10000.0, 1e-6, None, true)?;

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
        let block = TransformerBlock::new(1024, 16, 8, 128, 3072, 10000.0, 1e-6, None, true)?;

        let x = Tensor::ones((1, 4, 1024), DType::F32, &device)?;
        let output = block.forward(&x)?;

        assert_eq!(output.dims(), &[1, 4, 1024]);
        Ok(())
    }

    #[test]
    fn test_transformer_block_forward_decode_3d_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let hidden_size = 256;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;

        let block = TransformerBlock::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            512,
            10000.0,
            1e-6,
            None,
            false,
        )?;
        let mut kv_cache =
            crate::kv_cache::PagedKvCache::new(1, num_heads, head_dim, 8, device.clone(), false)?;

        let x = Tensor::ones((1, hidden_size), DType::F32, &device)?;

        let block_ids: Vec<usize> = vec![0];
        let num_computed = 0;
        let positions = vec![0];

        let output =
            block.forward_decode(&x, &mut kv_cache, 0, &block_ids, num_computed, &positions)?;

        assert_eq!(output.dims(), &[1, hidden_size]);
        Ok(())
    }

    #[test]
    fn test_transformer_block_forward_decode_with_qk_norm() -> Result<()> {
        let device = Device::Cpu;
        let hidden_size = 1024;
        let num_heads = 16;
        let num_kv_heads = 8;
        let head_dim = 128;

        let block = TransformerBlock::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            3072,
            10000.0,
            1e-6,
            None,
            true,
        )?;
        let mut kv_cache =
            crate::kv_cache::PagedKvCache::new(1, num_heads, head_dim, 8, device.clone(), false)?;

        let x = Tensor::ones((1, hidden_size), DType::F32, &device)?;
        let block_ids: Vec<usize> = vec![0];
        let positions = vec![0];

        let output = block.forward_decode(&x, &mut kv_cache, 0, &block_ids, 0, &positions)?;

        assert_eq!(output.dims(), &[1, hidden_size]);
        Ok(())
    }

    #[test]
    fn test_transformer_block_decode_sequential_tokens() -> Result<()> {
        let device = Device::Cpu;
        let hidden_size = 512;
        let num_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 64;

        let block = TransformerBlock::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            1024,
            10000.0,
            1e-6,
            None,
            false,
        )?;
        let mut kv_cache =
            crate::kv_cache::PagedKvCache::new(1, num_heads, head_dim, 16, device.clone(), false)?;

        for step in 0..5 {
            let x = Tensor::ones((1, hidden_size), DType::F32, &device)?;
            let block_ids: Vec<usize> = vec![step / 8];
            let positions = vec![step];

            let output =
                block.forward_decode(&x, &mut kv_cache, 0, &block_ids, step, &positions)?;

            assert_eq!(
                output.dims(),
                &[1, hidden_size],
                "Step {} output shape mismatch",
                step
            );
        }

        Ok(())
    }

    #[test]
    fn test_transformer_block_decode_with_multiple_kv_blocks() -> Result<()> {
        let device = Device::Cpu;
        let hidden_size = 256;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;
        let block_size = 8;

        let block = TransformerBlock::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            512,
            10000.0,
            1e-6,
            None,
            false,
        )?;
        let mut kv_cache =
            crate::kv_cache::PagedKvCache::new(1, num_heads, head_dim, 4, device.clone(), false)?;

        for step in 0..24 {
            let x = Tensor::ones((1, hidden_size), DType::F32, &device)?;
            let block_id = step / block_size;
            let block_ids: Vec<usize> = vec![block_id];
            let positions = vec![step];

            let output =
                block.forward_decode(&x, &mut kv_cache, 0, &block_ids, step, &positions)?;

            assert_eq!(
                output.dims(),
                &[1, hidden_size],
                "Step {} output shape mismatch",
                step
            );
        }

        Ok(())
    }
}
