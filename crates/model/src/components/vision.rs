//! Vision encoder components for Sam/Vision models.

use candle_core::{Module, Result as CandleResult, Tensor};
use candle_nn::{Linear, VarBuilder};

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub depth: usize,
}

impl VisionConfig {
    pub fn new(image_size: usize, patch_size: usize) -> Self {
        Self {
            image_size,
            patch_size,
            embed_dim: 768,
            depth: 12,
        }
    }

    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }
}

pub struct PatchEmbed {
    proj: Linear,
}

impl PatchEmbed {
    pub fn new(config: &VisionConfig, vb: VarBuilder) -> CandleResult<Self> {
        let proj = candle_nn::linear(
            config.patch_size * config.patch_size * 3,
            config.embed_dim,
            vb.pp("proj"),
        )?;
        Ok(Self { proj })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.proj.forward(x)
    }
}

#[allow(dead_code)]
pub struct VisionEncoder {
    config: VisionConfig,
}

impl VisionEncoder {
    pub fn new(config: &VisionConfig, _vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        Ok(x.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_vision_config() {
        let config = VisionConfig::new(1024, 16);
        assert_eq!(config.num_patches(), 4096);
    }

    #[test]
    fn test_vision_config_different_sizes() {
        // Test different image sizes
        assert_eq!(VisionConfig::new(512, 16).num_patches(), 1024); // (512/16)^2
        assert_eq!(VisionConfig::new(768, 16).num_patches(), 2304); // (768/16)^2
        assert_eq!(VisionConfig::new(1024, 16).num_patches(), 4096); // (1024/16)^2
    }

    #[test]
    fn test_vision_config_different_patch_sizes() {
        // Test different patch sizes
        assert_eq!(VisionConfig::new(224, 16).num_patches(), 196); // (224/16)^2 = 14^2
        assert_eq!(VisionConfig::new(224, 14).num_patches(), 256); // (224/14)^2 = 16^2
        assert_eq!(VisionConfig::new(224, 7).num_patches(), 1024); // (224/7)^2 = 32^2
    }

    #[test]
    fn test_patch_embed_creates() {
        let config = VisionConfig::new(1024, 16);
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        PatchEmbed::new(&config, vb).unwrap();
    }

    #[test]
    fn test_vision_encoder_creates() {
        let config = VisionConfig::new(224, 16);
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        VisionEncoder::new(&config, vb).unwrap();
    }

    #[test]
    fn test_vision_encoder_empty_forward() {
        let config = VisionConfig::new(224, 16);
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let encoder = VisionEncoder::new(&config, vb).unwrap();

        // Currently forward just returns input (placeholder)
        let input = Tensor::zeros((1, 10, 768), DType::F32, &Device::Cpu).unwrap();
        let output = encoder.forward(&input).unwrap();
        assert_eq!(output.dims(), input.dims());
    }
}
