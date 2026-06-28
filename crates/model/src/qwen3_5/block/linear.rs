#![allow(non_snake_case)]
//! GDN-based linear attention block for Qwen3.5 hybrid layers.

use std::collections::HashMap;

use crate::components::gated_delta::{GatedDeltaConfig, GatedDeltaNet, GatedDeltaState};
use crate::qwen3_5::config::GdnLinearConfig;
use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, LayerNorm, Linear, VarBuilder, conv1d};

#[derive(Debug)]
/// `LinearAttentionBlock`: linear attention block.
pub struct LinearAttentionBlock {
    pub(crate) gdn: GatedDeltaNet,
}

impl LinearAttentionBlock {
    pub(crate) fn new(d_model: usize, gdn: GdnLinearConfig, vb: VarBuilder) -> CandleResult<Self> {
        let GdnLinearConfig {
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
        } = gdn;

        let config = GatedDeltaConfig {
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
        };

        let conv_cfg = candle_nn::Conv1dConfig {
            padding: conv_kernel_size - 1,
            groups: config.qkv_proj_dim(),
            ..Default::default()
        };
        let gdn = GatedDeltaNet::from_components(
            config,
            candle_nn::linear(d_model, config.qkv_proj_dim(), vb.pp("in_proj_qkv"))?,
            candle_nn::linear(d_model, config.value_dim(), vb.pp("in_proj_z"))?,
            candle_nn::linear(d_model, num_v_heads, vb.pp("in_proj_a"))?,
            candle_nn::linear(d_model, num_v_heads, vb.pp("in_proj_b"))?,
            conv1d(
                config.qkv_proj_dim(),
                config.qkv_proj_dim(),
                conv_kernel_size,
                conv_cfg,
                vb.pp("conv"),
            )?,
            Tensor::zeros(num_v_heads, DType::F32, vb.device())?,
            Tensor::zeros(num_v_heads, DType::F32, vb.device())?,
            candle_nn::linear(config.value_dim(), d_model, vb.pp("out_proj"))?,
            candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm"))?,
        );

        Ok(Self { gdn })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.gdn.forward(x)
    }

    pub fn forward_prefill(&self, x: &Tensor) -> CandleResult<(Tensor, GatedDeltaState)> {
        self.gdn.forward_prefill(x)
    }

    pub fn forward_decode(&self, x: &Tensor, state: &mut GatedDeltaState) -> CandleResult<Tensor> {
        self.gdn.forward_decode(x, state)
    }
}

impl LinearAttentionBlock {
    pub fn from_weights(
        prefix: &str,
        weights: &HashMap<String, Tensor>,
        d_model: usize,
    ) -> CandleResult<Self> {
        let in_proj_qkv_key = format!("{prefix}.linear_attn.in_proj_qkv.weight");
        let in_proj_z_key = format!("{prefix}.linear_attn.in_proj_z.weight");
        let in_proj_a_key = format!("{prefix}.linear_attn.in_proj_a.weight");
        let in_proj_b_key = format!("{prefix}.linear_attn.in_proj_b.weight");
        let a_log_key = format!("{prefix}.linear_attn.A_log");
        let dt_bias_key = format!("{prefix}.linear_attn.dt_bias");
        let conv_key = format!("{prefix}.linear_attn.conv1d.weight");
        let out_proj_key = format!("{prefix}.linear_attn.out_proj.weight");
        let norm_key = format!("{prefix}.linear_attn.norm.weight");

        let in_proj_qkv_w = weights
            .get(&in_proj_qkv_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {in_proj_qkv_key}")))?;
        let in_proj_z_w = weights
            .get(&in_proj_z_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {in_proj_z_key}")))?;
        let in_proj_a_w = weights
            .get(&in_proj_a_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {in_proj_a_key}")))?;
        let in_proj_b_w = weights
            .get(&in_proj_b_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {in_proj_b_key}")))?;
        let a_log_w = weights
            .get(&a_log_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {a_log_key}")))?;
        let dt_bias_w = weights
            .get(&dt_bias_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {dt_bias_key}")))?;
        let conv_w = weights
            .get(&conv_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {conv_key}")))?;
        let out_proj_w = weights
            .get(&out_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {out_proj_key}")))?;
        let norm_w = weights
            .get(&norm_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {norm_key}")))?;

        let num_v_heads = a_log_w.dims()[0];
        let value_dim = in_proj_z_w.dim(0).unwrap_or(d_model);
        let value_head_dim = value_dim / num_v_heads;
        let qkv_dim = in_proj_qkv_w.dim(0).unwrap_or(value_dim);
        let key_dim = (qkv_dim - value_dim) / 2;
        let key_head_dim = value_head_dim;
        let num_k_heads = key_dim / key_head_dim;
        let conv_kernel_size = conv_w.dims().get(2).copied().unwrap_or(4);

        let config = GatedDeltaConfig {
            num_k_heads,
            num_v_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
        };

        let conv_in = conv_w.dim(1).unwrap_or(config.qkv_proj_dim());
        let conv_cfg = candle_nn::Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: conv_in,
            cudnn_fwd_algo: None,
        };
        let conv = Conv1d::new(conv_w, None, conv_cfg);

        let norm_b = weights
            .get(&format!("{prefix}.linear_attn.norm.bias"))
            .cloned()
            .unwrap_or_else(|| {
                // invariant: tensor shape is derived from norm_w dimensions; allocation
                // of a fixed-size zero buffer on the same device as norm_w cannot fail.
                Tensor::zeros(
                    norm_w.dim(0).unwrap_or(d_model),
                    DType::F32,
                    norm_w.device(),
                )
                .expect("Failed to create norm bias")
            });

        let gdn = GatedDeltaNet::from_components(
            config,
            Linear::new(in_proj_qkv_w, None),
            Linear::new(in_proj_z_w, None),
            Linear::new(in_proj_a_w, None),
            Linear::new(in_proj_b_w, None),
            conv,
            a_log_w,
            dt_bias_w,
            Linear::new(out_proj_w, None),
            LayerNorm::new(norm_w, norm_b, 1e-5),
        );

        Ok(Self { gdn })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    use crate::qwen3_5::config::GdnLinearConfig;

    #[test]
    fn test_linear_attention_block_creation() {
        let device = Device::Cpu;
        let gdn = GdnLinearConfig {
            num_k_heads: 8,
            num_v_heads: 16,
            key_head_dim: 16,
            value_head_dim: 16,
            conv_kernel_size: 4,
        };
        let block =
            LinearAttentionBlock::new(1024, gdn, VarBuilder::zeros(DType::F32, &device)).unwrap();

        assert_eq!(block.gdn.config.num_v_heads, 16);
        assert_eq!(block.gdn.config.num_k_heads, 8);
    }

    #[test]
    fn test_linear_attention_block_forward_shape() {
        let device = Device::Cpu;
        let gdn = GdnLinearConfig {
            num_k_heads: 4,
            num_v_heads: 8,
            key_head_dim: 16,
            value_head_dim: 16,
            conv_kernel_size: 4,
        };
        let block =
            LinearAttentionBlock::new(64, gdn, VarBuilder::zeros(DType::F32, &device)).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (1, 6, 64), &device).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), x.dims());
    }
}
