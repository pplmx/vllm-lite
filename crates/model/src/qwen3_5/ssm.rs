#![allow(clippy::all, non_snake_case, dead_code)]

use std::collections::HashMap;

use candle_core::{DType, Module, Result as CandleResult, Tensor};
use candle_nn::{conv1d, Conv1d, LayerNorm, Linear, VarBuilder};
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
}

pub struct MambaBlock {
    input_proj: Linear,
    ssm: SSMLayer,
    output_proj: Linear,
    norm: LayerNorm,
}

impl MambaBlock {
    pub fn new(d_model: usize, d_state: usize, vb: VarBuilder) -> CandleResult<Self> {
        let config = SSMConfig {
            d_model,
            d_state,
            d_conv: 4,
            expand: 2,
        };

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

    #[allow(clippy::let_and_return)]
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

        // Pre-compute A_log
        let a_log = self
            .ssm
            .a_log()
            .forward(&x_conv)?
            .reshape((batch, seq_len, d_state, d_inner))?;

        // Pre-allocate hidden state
        let mut h = Tensor::zeros((batch, d_state, d_inner), DType::F32, x.device())?;

        // Pre-allocate outputs
        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);

        // Sequential processing (required for hidden state)
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
        let output = self.norm.forward(&output)?;

        Ok(output)
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

impl SSMLayer {
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

impl MambaBlock {
    pub fn from_weights(
        d_model: usize,
        d_state: usize,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self, SSMError> {
        let config = SSMConfig {
            d_model,
            d_state,
            d_conv: 4,
            expand: 2,
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_ssm_layer_creates_parameters() {
        let config = SSMConfig::new(128);
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let ssm = SSMLayer::new(&config, vb).unwrap();

        assert_eq!(ssm.d_inner(), 256);
        assert_eq!(ssm.d_state(), 16);
    }
}
