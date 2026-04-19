//! SSM (State Space Model) / Mamba layer implementation.
//!
//! This module provides the core SSM layers used by Mamba-style models:
//! - SSMLayer / MambaBlock: Standard Mamba for Qwen3.5 Mamba-only models
//! - SSMHarmonicSSMLayer: Hybrid variant for Qwen3.5 attention+SSM models

use std::collections::HashMap;

use candle_core::{DType, Module, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, LayerNorm, Linear, VarBuilder, conv1d};
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct SSMConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
}

impl SSMConfig {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            d_state: 16,
            d_conv: 4,
            expand: 2,
        }
    }

    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    pub fn d_state(&self) -> usize {
        self.d_state
    }

    pub fn d_conv(&self) -> usize {
        self.d_conv
    }

    pub fn with_d_state(mut self, d_state: usize) -> Self {
        self.d_state = d_state;
        self
    }

    pub fn with_d_conv(mut self, d_conv: usize) -> Self {
        self.d_conv = d_conv;
        self
    }

    pub fn with_expand(mut self, expand: usize) -> Self {
        self.expand = expand;
        self
    }
}

pub struct SSMLayer {
    x_proj: Linear,
    a_log: Linear,
    d: Linear,
    conv: Conv1d,
    d_inner: usize,
    d_state: usize,
}

impl SSMLayer {
    pub fn new(config: &SSMConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d_inner = config.d_inner();

        let x_proj = candle_nn::linear(d_inner, d_inner * 3, vb.pp("x_proj"))?;
        let a_log = candle_nn::linear(d_inner, config.d_state * d_inner, vb.pp("A_log"))?;
        let d = candle_nn::linear(d_inner, d_inner, vb.pp("D"))?;

        let conv_cfg = candle_nn::Conv1dConfig {
            padding: config.d_conv - 1,
            ..Default::default()
        };
        let conv = conv1d(d_inner, d_inner, config.d_conv, conv_cfg, vb.pp("conv"))?;

        Ok(Self {
            x_proj,
            a_log,
            d,
            conv,
            d_inner,
            d_state: config.d_state,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor, Tensor, Tensor)> {
        let x_conv = x.transpose(1, 2)?;
        let x_conv = self.conv.forward(&x_conv)?;
        let x_conv = x_conv.transpose(1, 2)?;
        let x_conv = candle_nn::ops::silu(&x_conv)?;

        let x_ssm = self.x_proj.forward(&x_conv)?;

        let parts = x_ssm.chunk(3, 2)?;
        let delta = &parts[0];
        let b = &parts[1];
        let c = &parts[2];

        let delta = candle_nn::ops::silu(delta)?;

        Ok((delta, b.clone(), c.clone(), x_conv))
    }

    pub fn d_inner(&self) -> usize {
        self.d_inner
    }

    pub fn d_state(&self) -> usize {
        self.d_state
    }

    pub fn a_log(&self) -> &Linear {
        &self.a_log
    }

    pub fn d_linear(&self) -> &Linear {
        &self.d
    }

    pub fn from_weights(
        d_inner: usize,
        d_state: usize,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self, SSMError> {
        let x_proj_w = weights
            .get(&format!("model.layers.{}.mamba.x_proj.weight", layer_idx))
            .cloned()
            .ok_or_else(|| {
                SSMError::Msg(format!("Missing x_proj weight for layer {}", layer_idx))
            })?;
        let x_proj = candle_nn::Linear::new(x_proj_w, None);

        let a_log_w = weights
            .get(&format!("model.layers.{}.mamba.A_log.weight", layer_idx))
            .cloned()
            .ok_or_else(|| {
                SSMError::Msg(format!("Missing A_log weight for layer {}", layer_idx))
            })?;
        let a_log = candle_nn::Linear::new(a_log_w, None);

        let d_w = weights
            .get(&format!("model.layers.{}.mamba.D.weight", layer_idx))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing D weight for layer {}", layer_idx)))?;
        let d = candle_nn::Linear::new(d_w, None);

        let conv_w = weights
            .get(&format!("model.layers.{}.mamba.conv1d.weight", layer_idx))
            .cloned()
            .ok_or_else(|| {
                SSMError::Msg(format!("Missing conv1d weight for layer {}", layer_idx))
            })?;
        let d_conv = 4;
        let conv_cfg = candle_nn::Conv1dConfig {
            padding: d_conv - 1,
            ..Default::default()
        };
        let conv = Conv1d::new(conv_w, None, conv_cfg);

        Ok(Self {
            x_proj,
            a_log,
            d,
            conv,
            d_inner,
            d_state,
        })
    }
}

pub struct MambaBlock {
    input_proj: Linear,
    ssm: SSMLayer,
    output_proj: Linear,
    norm: LayerNorm,
}

impl MambaBlock {
    pub fn new(d_model: usize, d_state: usize, vb: VarBuilder) -> CandleResult<Self> {
        let config = SSMConfig::new(d_model).with_d_state(d_state);

        let input_proj = candle_nn::linear(d_model, config.d_inner() * 2, vb.pp("in_proj"))?;
        let ssm = SSMLayer::new(&config, vb.clone())?;
        let output_proj = candle_nn::linear(config.d_inner(), d_model, vb.pp("out_proj"))?;
        let norm = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            input_proj,
            ssm,
            output_proj,
            norm,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();

        let x_proj = self.input_proj.forward(x)?;
        let parts = x_proj.chunk(2, 2)?;
        let z = &parts[0];
        let x_inner = &parts[1];

        let (delta, b, c, x_conv) = self.ssm.forward(x_inner)?;

        let batch = x.dims()[0];
        let seq_len = x_conv.dims()[1];
        let d_inner = self.ssm.d_inner();
        let d_state = self.ssm.d_state();

        let a_log = self
            .ssm
            .a_log()
            .forward(&x_conv)?
            .reshape((batch, seq_len, d_state, d_inner))?;

        let mut h = Tensor::zeros((batch, d_state, d_inner), DType::F32, x.device())?;
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let _delta_t = delta.narrow(1, t, 1)?.squeeze(1)?;
            let x_t = x_conv.narrow(1, t, 1)?.squeeze(1)?;
            let b_t = b.narrow(1, t, 1)?.squeeze(1)?;
            let c_t = c.narrow(1, t, 1)?.squeeze(1)?;
            let a_t = a_log.narrow(1, t, 1)?.squeeze(1)?.exp()?;

            let bx = b_t.unsqueeze(1)?.broadcast_mul(&x_t.unsqueeze(1)?)?;
            let h_new = a_t.broadcast_mul(&h)?.broadcast_add(&bx)?;
            let y_t = c_t.unsqueeze(1)?.broadcast_mul(&h_new)?;

            outputs.push(y_t.unsqueeze(1)?);
            h = h_new;
        }

        let ssm_out = Tensor::cat(&outputs, 1)?;

        let d = self.ssm.d_linear().forward(&x_conv)?;
        let ssm_out = (&ssm_out + &d)?;

        let ssm_act = candle_nn::ops::silu(&ssm_out)?;
        let gated = z.broadcast_mul(&ssm_act)?;

        let output = self.output_proj.forward(&gated)?;
        let output = output.add(&residual)?;
        self.norm.forward(&output)
    }

    pub fn from_weights(
        d_model: usize,
        d_state: usize,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self, SSMError> {
        let config = SSMConfig::new(d_model).with_d_state(d_state);
        let d_inner = config.d_inner();

        let in_proj_w = weights
            .get(&format!("model.layers.{}.mamba.in_proj.weight", layer_idx))
            .cloned()
            .ok_or_else(|| {
                SSMError::Msg(format!("Missing in_proj weight for layer {}", layer_idx))
            })?;
        let input_proj = candle_nn::Linear::new(in_proj_w, None);

        let ssm = SSMLayer::from_weights(d_inner, d_state, layer_idx, weights)?;

        let out_proj_w = weights
            .get(&format!("model.layers.{}.mamba.out_proj.weight", layer_idx))
            .cloned()
            .ok_or_else(|| {
                SSMError::Msg(format!("Missing out_proj weight for layer {}", layer_idx))
            })?;
        let output_proj = candle_nn::Linear::new(out_proj_w, None);

        let norm_w = weights
            .get(&format!("model.layers.{}.mamba.norm.weight", layer_idx))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing norm weight for layer {}", layer_idx)))?;
        let norm_b = weights
            .get(&format!("model.layers.{}.mamba.norm.bias", layer_idx))
            .cloned()
            .unwrap_or_else(|| {
                Tensor::zeros(norm_w.shape().dims()[0], DType::F32, norm_w.device())
                    .expect("Failed to create zero bias")
            });
        let norm = candle_nn::LayerNorm::new(norm_w, norm_b, 1e-5);

        Ok(Self {
            input_proj,
            ssm,
            output_proj,
            norm,
        })
    }
}

#[allow(dead_code)]
pub struct SSMHarmonicSSMLayer {
    x_proj: Linear,
    in_proj_a: Linear,
    a_log: Tensor,
    #[allow(dead_code)]
    dt_bias: Tensor,
    conv: Conv1d,
    d_inner: usize,
    d_state: usize,
}

impl SSMHarmonicSSMLayer {
    pub fn new(
        d_inner: usize,
        d_state: usize,
        d_conv: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let x_proj = candle_nn::linear(d_inner * 3, d_inner * 3, vb.pp("x_proj"))?;
        let in_proj_a = candle_nn::linear(d_inner, d_state, vb.pp("in_proj_a"))?;
        let a_log = Tensor::zeros(d_state, DType::F32, vb.device())?;
        let dt_bias = Tensor::zeros(d_state, DType::F32, vb.device())?;
        let conv_cfg = candle_nn::Conv1dConfig {
            padding: d_conv - 1,
            ..Default::default()
        };
        let conv = conv1d(d_inner * 3, d_inner * 3, d_conv, conv_cfg, vb.pp("conv"))?;

        Ok(Self {
            x_proj,
            in_proj_a,
            a_log,
            dt_bias,
            conv,
            d_inner,
            d_state,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor, Tensor, Tensor)> {
        let x_conv = x.transpose(1, 2)?;
        let x_conv = self.conv.forward(&x_conv)?;
        let x_conv = x_conv.transpose(1, 2)?;
        let x_conv = candle_nn::ops::silu(&x_conv)?;

        let x_ssm = self.x_proj.forward(&x_conv)?;

        let parts = x_ssm.chunk(3, 2)?;
        let delta = &parts[0];
        let b = &parts[1];
        let c = &parts[2];

        let delta = candle_nn::ops::silu(delta)?;

        Ok((delta, b.clone(), c.clone(), x_conv))
    }

    pub fn forward_with_a(
        &self,
        x: &Tensor,
        a_input: &Tensor,
    ) -> CandleResult<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let x_conv = x.transpose(1, 2)?;
        let x_conv = self.conv.forward(&x_conv)?;
        let x_conv = x_conv.transpose(1, 2)?;
        let x_conv = candle_nn::ops::silu(&x_conv)?;

        let x_conv_len = x_conv.dims()[1];
        let x_len = x.dims()[1];
        let x_conv = if x_conv_len > x_len {
            x_conv.narrow(1, x_conv_len - x_len, x_len)?
        } else if x_conv_len < x_len {
            let pad = Tensor::zeros(
                (x_conv.dims()[0], x_len - x_conv_len, x_conv.dims()[2]),
                x_conv.dtype(),
                x.device(),
            )?;
            Tensor::cat(&[&x_conv, &pad], 1)?
        } else {
            x_conv
        };

        let x_ssm = self.x_proj.forward(&x_conv)?;

        let parts = x_ssm.chunk(3, 2)?;
        let delta = &parts[0];
        let b = &parts[1];
        let c = &parts[2];

        let a_proj_out = self.in_proj_a.forward(a_input)?;

        let delta = candle_nn::ops::silu(delta)?;

        Ok((delta, b.clone(), c.clone(), x_conv, a_proj_out))
    }

    pub fn d_inner(&self) -> usize {
        self.d_inner
    }

    pub fn d_state(&self) -> usize {
        self.d_state
    }

    pub fn a_log(&self) -> &Tensor {
        &self.a_log
    }

    pub fn from_weights(
        d_inner: usize,
        d_state: usize,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self, SSMError> {
        let x_proj_w = weights
            .get(&format!(
                "model.layers.{}.linear_attn.x_proj.weight",
                layer_idx
            ))
            .cloned()
            .ok_or_else(|| {
                SSMError::Msg(format!("Missing x_proj weight for layer {}", layer_idx))
            })?;
        let x_proj = candle_nn::Linear::new(x_proj_w, None);

        let in_proj_a_w = weights
            .get(&format!(
                "model.layers.{}.linear_attn.in_proj_a.weight",
                layer_idx
            ))
            .cloned()
            .ok_or_else(|| {
                SSMError::Msg(format!("Missing in_proj_a weight for layer {}", layer_idx))
            })?;
        let in_proj_a = candle_nn::Linear::new(in_proj_a_w, None);

        let a_log_w = weights
            .get(&format!("model.layers.{}.linear_attn.A_log", layer_idx))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing A_log for layer {}", layer_idx)))?;
        let dt_bias_w = weights
            .get(&format!("model.layers.{}.linear_attn.dt_bias", layer_idx))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing dt_bias for layer {}", layer_idx)))?;

        let conv_w = weights
            .get(&format!(
                "model.layers.{}.linear_attn.conv1d.weight",
                layer_idx
            ))
            .cloned()
            .ok_or_else(|| {
                SSMError::Msg(format!("Missing conv1d weight for layer {}", layer_idx))
            })?;
        let d_conv = 4;
        let conv_cfg = candle_nn::Conv1dConfig {
            padding: d_conv - 1,
            ..Default::default()
        };
        let conv = Conv1d::new(conv_w, None, conv_cfg);

        Ok(Self {
            x_proj,
            in_proj_a,
            a_log: a_log_w,
            dt_bias: dt_bias_w,
            conv,
            d_inner,
            d_state,
        })
    }
}

#[derive(Debug, Error)]
pub enum SSMError {
    #[error("{0}")]
    Msg(String),
}

impl From<std::convert::Infallible> for SSMError {
    fn from(_: std::convert::Infallible) -> Self {
        SSMError::Msg("Infallible error".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_ssm_config() {
        let config = SSMConfig::new(128);
        assert_eq!(config.d_inner(), 256);
        assert_eq!(config.d_state(), 16);
    }

    #[test]
    fn test_ssm_layer_creates() {
        let config = SSMConfig::new(128);
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let ssm = SSMLayer::new(&config, vb).unwrap();
        assert_eq!(ssm.d_inner(), 256);
        assert_eq!(ssm.d_state(), 16);
    }

    #[test]
    fn test_mamba_block_creates() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        MambaBlock::new(128, 16, vb).unwrap();
    }

    #[test]
    fn test_ssm_harmonic_creates() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        SSMHarmonicSSMLayer::new(256, 16, 4, vb).unwrap();
    }

    #[test]
    fn test_ssm_config_builder() {
        let config = SSMConfig::new(64)
            .with_d_state(32)
            .with_d_conv(3)
            .with_expand(3);

        assert_eq!(config.d_inner(), 192);
        assert_eq!(config.d_state(), 32);
        assert_eq!(config.d_conv(), 3);
    }
}
