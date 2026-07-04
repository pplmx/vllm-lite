//! Harmonic SSM variant used in hybrid attention+SSM Qwen3.5 models.

use std::collections::HashMap;

use candle_core::{DType, Module, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, Linear, VarBuilder, conv1d};

use crate::components::ssm::error::SSMError;

#[derive(Debug)]
/// `SSMHarmonicSSMLayer`. See the type definition for fields and behavior.
pub struct SSMHarmonicSSMLayer {
    x_proj: Linear,
    in_proj_a: Linear,
    a_log: Tensor,
    dt_bias: Tensor,
    conv: Conv1d,
    d_inner: usize,
    d_state: usize,
}

impl SSMHarmonicSSMLayer {
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        d_inner: usize,
        d_state: usize,
        d_conv: usize,
        vb: VarBuilder<'_>,
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

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
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

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
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
        let x_conv = match x_conv_len.cmp(&x_len) {
            std::cmp::Ordering::Greater => x_conv.narrow(1, x_conv_len - x_len, x_len)?,
            std::cmp::Ordering::Less => {
                let pad = Tensor::zeros(
                    (x_conv.dims()[0], x_len - x_conv_len, x_conv.dims()[2]),
                    x_conv.dtype(),
                    x.device(),
                )?;
                Tensor::cat(&[&x_conv, &pad], 1)?
            }
            std::cmp::Ordering::Equal => x_conv,
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

    #[must_use]
    pub const fn d_inner(&self) -> usize {
        self.d_inner
    }

    #[must_use]
    pub const fn d_state(&self) -> usize {
        self.d_state
    }

    #[must_use]
    pub const fn a_log(&self) -> &Tensor {
        &self.a_log
    }

    #[must_use]
    pub const fn dt_bias(&self) -> &Tensor {
        &self.dt_bias
    }

    /// Build from weights.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    pub fn from_weights(
        d_inner: usize,
        d_state: usize,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self, SSMError> {
        let x_proj_w = weights
            .get(&format!(
                "model.layers.{layer_idx}.linear_attn.x_proj.weight"
            ))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing x_proj weight for layer {layer_idx}")))?;
        let x_proj = candle_nn::Linear::new(x_proj_w, None);

        let in_proj_a_w = weights
            .get(&format!(
                "model.layers.{layer_idx}.linear_attn.in_proj_a.weight"
            ))
            .cloned()
            .ok_or_else(|| {
                SSMError::Msg(format!("Missing in_proj_a weight for layer {layer_idx}"))
            })?;
        let in_proj_a = candle_nn::Linear::new(in_proj_a_w, None);

        let a_log_w = weights
            .get(&format!("model.layers.{layer_idx}.linear_attn.A_log"))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing A_log for layer {layer_idx}")))?;
        let dt_bias_w = weights
            .get(&format!("model.layers.{layer_idx}.linear_attn.dt_bias"))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing dt_bias for layer {layer_idx}")))?;

        let conv_w = weights
            .get(&format!(
                "model.layers.{layer_idx}.linear_attn.conv1d.weight"
            ))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing conv1d weight for layer {layer_idx}")))?;
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
