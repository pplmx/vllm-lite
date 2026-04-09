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
    use candle_core::Device;

    #[test]
    fn test_vision_config() {
        let config = VisionConfig::new(1024, 16);
        assert_eq!(config.num_patches(), 4096);
    }

    #[test]
    fn test_patch_embed() {
        let config = VisionConfig::new(1024, 16);
        let vb = VarBuilder::zeros(candle_core::DType::F32, &Device::Cpu);
        let _embed = PatchEmbed::new(&config, vb).unwrap();
    }
}
