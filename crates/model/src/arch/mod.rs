#![allow(clippy::module_name_repetitions)]
//! Architecture abstraction for model loading.
//!
//! This module provides the Architecture trait for defining model architectures
//! and the Registry for dynamic registration.

use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;
use vllm_traits::ModelBackend;

use crate::components::TransformerBlock;
use crate::config::ModelConfig;

pub mod capabilities;
pub mod registry;

pub use capabilities::ArchCapabilities;
pub use registry::{ARCHITECTURE_REGISTRY, ArchitectureRegistry, register_all_archs};

/// Architecture: architecture trait.
pub trait Architecture: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    fn detect(&self, config_json: &serde_json::Value) -> bool;

    fn capabilities(&self) -> ArchCapabilities;

    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn create_block(
        &self,
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Box<dyn TransformerBlock>>;

    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> Result<Box<dyn ModelBackend>>;

    fn remap_weights(&self, weights: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        weights
    }
}

/// Default unknown `Architecture`.
///
/// Reports `name() == "unknown"` and never matches any config. `create_block`
/// and `create_model` always error. Used as a placeholder when no real
/// architecture is registered for a checkpoint.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct UnknownArchitecture;

impl Architecture for UnknownArchitecture {
    fn name(&self) -> &'static str {
        "unknown"
    }

    fn detect(&self, _config_json: &serde_json::Value) -> bool {
        false
    }

    fn capabilities(&self) -> ArchCapabilities {
        ArchCapabilities::STUB
    }

    fn create_block(
        &self,
        _config: &ModelConfig,
        _layer_idx: usize,
        _weights: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        Err(candle_core::Error::msg(
            "UnknownArchitecture: cannot create transformer block",
        ))
    }

    fn create_model(
        &self,
        _config: ModelConfig,
        _device: Device,
        _weights: HashMap<String, Tensor>,
        _num_kv_blocks: usize,
        _kv_quantization: bool,
    ) -> Result<Box<dyn ModelBackend>> {
        Err(candle_core::Error::msg(
            "UnknownArchitecture: cannot create model backend",
        ))
    }
}

impl dyn Architecture {
    /// Returns an `Arc<Self>` wrapping the unknown `UnknownArchitecture`.
    ///
    /// This is the closest equivalent to
    /// `Arc::<dyn Architecture>::default()`; Rust's orphan rule prevents
    /// a direct `impl Default for Arc<dyn ...>` because `Arc` is foreign and
    /// there is no local type appearing before the uncovered trait-object
    /// parameter.
    #[must_use]
    pub fn default_arc() -> Arc<Self> {
        Arc::new(UnknownArchitecture)
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

    #[test]
    fn architecture_default_arc_is_unknown() {
        let arch: Arc<dyn Architecture> = <dyn Architecture>::default_arc();
        assert_eq!(arch.name(), "unknown");
        assert!(!arch.detect(&serde_json::json!({"model_type": "llama"})));
        assert!(arch.capabilities().is_stub());
    }
}
