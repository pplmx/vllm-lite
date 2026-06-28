//! SSM components — see sub-modules for specific implementations.
//!
//! This module provides the core SSM layers used by Mamba-style models:
//! - SSMLayer / MambaBlock: Standard Mamba for Qwen3.5 Mamba-only models
//! - SSMHarmonicSSMLayer: Hybrid variant for Qwen3.5 attention+SSM models

mod config;
mod error;
mod harmonic;
mod layer;
mod mamba;

pub use config::SSMConfig;
pub use error::SSMError;
pub use harmonic::SSMHarmonicSSMLayer;
pub use layer::{SSMLayer, softplus};
pub use mamba::MambaBlock;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

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
    fn test_ssm_harmonic_exposes_dt_bias() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let ssm = SSMHarmonicSSMLayer::new(256, 16, 4, vb).unwrap();
        assert_eq!(ssm.dt_bias().dims(), &[16]);
    }

    #[test]
    fn test_softplus() {
        let device = Device::Cpu;
        let x = Tensor::new(&[-1.0f32, 0.0, 1.0], &device).unwrap();
        let y = softplus(&x).unwrap();
        let vals: Vec<f32> = y.to_vec1().unwrap();
        assert!(vals[0] > 0.0);
        assert!((vals[1] - std::f32::consts::LN_2).abs() < 1e-4);
        assert!(vals[2] > vals[1]);
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
