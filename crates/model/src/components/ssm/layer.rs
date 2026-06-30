//! Standard SSM layer and tensor-math helpers.

use std::collections::HashMap;

use candle_core::{Module, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, Linear, VarBuilder, conv1d};

use crate::components::ssm::config::SSMConfig;
use crate::components::ssm::error::SSMError;

/// # Errors
///
/// Returns `Err` if the operation fails.
/// Numerically stable softplus: log(1 + exp(x)).
pub fn softplus(xs: &Tensor) -> CandleResult<Tensor> {
    let exp_x = xs.exp()?;
    let one = Tensor::new(1.0f32, xs.device())?.to_dtype(xs.dtype())?;
    let one = one.broadcast_as(exp_x.dims())?;
    (exp_x + one)?.log()
}

#[derive(Debug)]
/// `SSMLayer`: ssm layer.
pub struct SSMLayer {
    x_proj: Linear,
    a_log: Linear,
    d: Linear,
    conv: Conv1d,
    d_inner: usize,
    d_state: usize,
}

impl SSMLayer {
    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(config: &SSMConfig, vb: VarBuilder<'_>) -> CandleResult<Self> {
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

    /// Runs the operation.
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

    #[must_use]
    pub const fn d_inner(&self) -> usize {
        self.d_inner
    }

    #[must_use]
    pub const fn d_state(&self) -> usize {
        self.d_state
    }

    #[must_use]
    pub const fn a_log(&self) -> &Linear {
        &self.a_log
    }

    #[must_use]
    pub const fn d_linear(&self) -> &Linear {
        &self.d
    }

    /// Runs the operation.
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
            .get(&format!("model.layers.{layer_idx}.mamba.x_proj.weight"))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing x_proj weight for layer {layer_idx}")))?;
        let x_proj = candle_nn::Linear::new(x_proj_w, None);

        let a_log_w = weights
            .get(&format!("model.layers.{layer_idx}.mamba.A_log.weight"))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing A_log weight for layer {layer_idx}")))?;
        let a_log = candle_nn::Linear::new(a_log_w, None);

        let d_w = weights
            .get(&format!("model.layers.{layer_idx}.mamba.D.weight"))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing D weight for layer {layer_idx}")))?;
        let d = candle_nn::Linear::new(d_w, None);

        let conv_w = weights
            .get(&format!("model.layers.{layer_idx}.mamba.conv1d.weight"))
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
            a_log,
            d,
            conv,
            d_inner,
            d_state,
        })
    }
}
