//! Architecture abstraction for model loading.
//!
//! This module provides the Architecture trait for defining model architectures
//! and the Registry for dynamic registration.

use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

use crate::components::TransformerBlock;
use crate::config::ModelConfig;

pub mod registry;

pub use registry::{ARCHITECTURE_REGISTRY, ArchitectureRegistry, register_all_archs};

pub trait Architecture: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    fn detect(&self, config_json: &serde_json::Value) -> bool;

    fn create_block(
        &self,
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Box<dyn TransformerBlock>>;

    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Box<dyn ModelBackend>>;

    fn remap_weights(&self, weights: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_new_is_empty() {
        let registry = ArchitectureRegistry::new();
        let config = serde_json::json!({"model_type": "llama"});
        assert!(registry.detect(&config).is_none());
    }

    #[test]
    fn test_registry_register_returns_none_for_unknown() {
        let registry = ArchitectureRegistry::new();
        assert!(registry.get("unknown").is_none());
    }
}
