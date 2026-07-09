//! Unit tests for the vision-encoder module.
//!
//! Locks in three contracts on the placeholder vision tower:
//!
//! 1. **`VisionConfig::num_patches`**: `(image_size / patch_size).pow(2)`
//!    for typical ViT input sizes (224 / 512 / 768 / 1024) and
//!    patch sizes (7 / 14 / 16).
//! 2. **`PatchEmbed::new`**: zero-init `VarBuilder` produces a
//!    working layer (the `Linear` projection allocates cleanly).
//! 3. **`VisionEncoder::forward`**: placeholder pass-through preserves
//!    shape (the future real ViT stack will replace this with
//!    actual patch → transformer → pooling, but the contract is
//!    "input shape == output shape" for now).
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
