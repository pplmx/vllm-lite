//! SSM (State Space Model) / Mamba layer implementation.
//!
//! This module provides the core SSM layer used by Mamba-style models.
//! It can be extended to support different SSM variants.

use candle_core::{DType, Module, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, LayerNorm, Linear, VarBuilder, conv1d};

/// SSM Configuration
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

/// SSM Layer (selective scan)
///
/// This is the core component of Mamba-style models.
/// It performs:
/// 1. Convolution on input
/// 2. Projection to delta, B, C
/// 3. Discretization (delta = sigmoid(A))
/// 4. Selective scan over sequence
pub struct SSMLayer {
    x_proj: Linear,
    a_log: Linear,
    d: Linear,
    conv: Conv1d,
    d_inner: usize,
    d_state: usize,
}

impl SSMLayer {
    /// Create a new SSMLayer from config
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

    /// Forward pass returns (delta, B, C, x_conv)
    pub fn forward(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor, Tensor, Tensor)> {
        // Convolution + SiLU activation
        let x_conv_transposed = x.transpose(1, 2)?;
        let x_conv_transposed = self.conv.forward(&x_conv_transposed)?;
        let x_conv = x_conv_transposed.transpose(1, 2)?;
        let x_conv = candle_nn::ops::silu(&x_conv)?;

        // Project to delta, B, C
        let x_ssm = self.x_proj.forward(&x_conv)?;
        let parts = x_ssm.chunk(3, 2)?;
        let delta = candle_nn::ops::silu(&parts[0])?;

        Ok((delta, parts[1].clone(), parts[2].clone(), x_conv))
    }

    /// Simple forward without returning intermediate values
    pub fn forward_simple(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (_delta, _b, _c, x_conv) = self.forward(x)?;

        let batch = x.dims()[0];
        let seq_len = x_conv.dims()[1];

        // Get A matrix from a_log (unused in simple version)
        let _a_log = self.a_log().forward(&x_conv)?.reshape((
            batch,
            seq_len,
            self.d_state,
            self.d_inner(),
        ))?;

        // Selective scan (simplified - just return conv output for now)
        let d = self.d_linear().forward(&x_conv)?;
        let out = x_conv.add(&d)?;
        candle_nn::ops::silu(&out)
    }

    // Accessors
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

/// Mamba Block - combines SSM with gating
pub struct MambaBlock {
    input_proj: Linear,
    ssm: SSMLayer,
    output_proj: Linear,
    norm: LayerNorm,
}

impl MambaBlock {
    /// Create a new MambaBlock
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

    /// Full forward pass with selective scan
    pub fn forward(&mut self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();

        // Input projection and split
        let x_proj = self.input_proj.forward(x)?;
        let parts = x_proj.chunk(2, 2)?;
        let z = &parts[0];
        let x_inner = &parts[1];

        // SSM forward
        let (_delta, b, c, x_conv) = self.ssm.forward(x_inner)?;

        let batch = x.dims()[0];
        let seq_len = x_conv.dims()[1];

        // Get A from a_log and reshape
        let a_log = self.ssm.a_log().forward(&x_conv)?.reshape((
            batch,
            seq_len,
            self.ssm.d_state(),
            self.ssm.d_inner(),
        ))?;

        // Selective scan over sequence
        let mut outputs = Vec::with_capacity(seq_len);
        let mut h = Tensor::zeros(
            (batch, self.ssm.d_state(), self.ssm.d_inner()),
            DType::F32,
            x.device(),
        )?;

        for t in 0..seq_len {
            // Get time step t
            let a_t = a_log.narrow(1, t, 1)?.squeeze(1)?.exp()?;
            let b_t = b.narrow(1, t, 1)?.squeeze(1)?.unsqueeze(1)?.expand((
                batch,
                self.ssm.d_state(),
                self.ssm.d_inner(),
            ))?;
            let x_t = x_conv.narrow(1, t, 1)?.squeeze(1)?.unsqueeze(1)?.expand((
                batch,
                self.ssm.d_state(),
                self.ssm.d_inner(),
            ))?;

            // h = A * h + B * x
            let bx = b_t.broadcast_mul(&x_t)?;
            let h_new = a_t.broadcast_mul(&h)?.add(&bx)?;

            // y = C * h
            let c_t = c.narrow(1, t, 1)?.squeeze(1)?.unsqueeze(1)?.expand((
                batch,
                self.ssm.d_state(),
                self.ssm.d_inner(),
            ))?;
            let y_t = c_t.broadcast_mul(&h_new)?;

            outputs.push(y_t.unsqueeze(1)?);
            h = h_new;
        }

        // Concatenate outputs
        let ssm_out = Tensor::cat(&outputs, 1)?;

        // Add D term and apply SiLU gating
        let d = self.ssm.d_linear().forward(&x_conv)?;
        let ssm_out = ssm_out.add(&d)?;
        let ssm_act = candle_nn::ops::silu(&ssm_out)?;

        // Gated output
        let gated = z.broadcast_mul(&ssm_act)?;

        // Output projection + residual + norm
        let output = self.output_proj.forward(&gated)?.add(&residual)?;
        self.norm.forward(&output)
    }
}

/// Trait for SSM-based models
pub trait SSMModel: Module {
    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_ssm_config() {
        let config = SSMConfig::new(128);
        assert_eq!(config.d_inner(), 256);
        assert_eq!(config.d_state(), 16);
    }

    #[test]
    fn test_ssm_config_builder() {
        let config = SSMConfig::new(64)
            .with_d_state(32)
            .with_d_conv(3)
            .with_expand(3);

        assert_eq!(config.d_inner(), 192); // 64 * 3
        assert_eq!(config.d_state(), 32);
        assert_eq!(config.d_conv(), 3);
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
    fn test_ssm_layer_accessors() {
        let config = SSMConfig::new(128);
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let ssm = SSMLayer::new(&config, vb).unwrap();

        // Test accessors
        assert_eq!(ssm.d_inner(), 256);
        assert_eq!(ssm.d_state(), 16);
        assert!(ssm.a_log().weight().dims()[0] > 0);
        assert!(ssm.d_linear().weight().dims()[0] > 0);
    }

    #[test]
    fn test_mamba_block_creates() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        MambaBlock::new(128, 16, vb).unwrap();
    }

    #[test]
    fn test_mamba_block_different_configurations() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

        // Test different d_model and d_state
        for (d_model, d_state) in [(64, 8), (128, 16), (256, 32)] {
            MambaBlock::new(d_model, d_state, vb.clone()).unwrap();
        }
    }

    #[test]
    fn test_ssm_config_different_models() {
        // Test configs for different model sizes
        let small = SSMConfig::new(64);
        let medium = SSMConfig::new(128);
        let large = SSMConfig::new(256);

        assert_eq!(small.d_inner(), 128); // 64 * 2
        assert_eq!(medium.d_inner(), 256); // 128 * 2
        assert_eq!(large.d_inner(), 512); // 256 * 2
    }
}
