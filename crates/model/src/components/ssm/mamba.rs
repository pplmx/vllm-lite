//! Mamba block (input/SSM/output projections + residual `LayerNorm`).

use std::collections::HashMap;

use candle_core::{DType, Module, Result as CandleResult, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use crate::components::ssm::config::SSMConfig;
use crate::components::ssm::error::SSMError;
use crate::components::ssm::layer::SSMLayer;

#[derive(Debug)]
/// `MambaBlock`: mamba block.
pub struct MambaBlock {
    input_proj: Linear,
    ssm: SSMLayer,
    output_proj: Linear,
    norm: LayerNorm,
}

impl MambaBlock {
    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(d_model: usize, d_state: usize, vb: VarBuilder<'_>) -> CandleResult<Self> {
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

    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    #[allow(clippy::many_single_char_names)]
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

    /// Runs the operation.
    /// # Errors
    ///
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// Returns `Err` if reading or parsing the source fails.
    pub fn from_weights(
        d_model: usize,
        d_state: usize,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self, SSMError> {
        let config = SSMConfig::new(d_model).with_d_state(d_state);
        let d_inner = config.d_inner();

        let in_proj_w = weights
            .get(&format!("model.layers.{layer_idx}.mamba.in_proj.weight"))
            .cloned()
            .ok_or_else(|| {
                SSMError::Msg(format!("Missing in_proj weight for layer {layer_idx}"))
            })?;
        let input_proj = candle_nn::Linear::new(in_proj_w, None);

        let ssm = SSMLayer::from_weights(d_inner, d_state, layer_idx, weights)?;

        let out_proj_w = weights
            .get(&format!("model.layers.{layer_idx}.mamba.out_proj.weight"))
            .cloned()
            .ok_or_else(|| {
                SSMError::Msg(format!("Missing out_proj weight for layer {layer_idx}"))
            })?;
        let output_proj = candle_nn::Linear::new(out_proj_w, None);

        let norm_w = weights
            .get(&format!("model.layers.{layer_idx}.mamba.norm.weight"))
            .cloned()
            .ok_or_else(|| SSMError::Msg(format!("Missing norm weight for layer {layer_idx}")))?;
        let norm_b = weights
            .get(&format!("model.layers.{layer_idx}.mamba.norm.bias"))
            .cloned()
            .unwrap_or_else(|| {
                // invariant: tensor shape is derived from norm_w dimensions; allocation
                // of a fixed-size zero buffer on the same device as norm_w cannot fail.
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
