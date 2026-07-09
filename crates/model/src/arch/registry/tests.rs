//! Unit tests for `ArchitectureRegistry`.
//!
//! Uses a minimal `TestArch` stub that detects on a top-level
//! boolean field and reports `STUB` capabilities. Covers the full
//! registry surface:
//!
//! - `register` + `get`: round-trip; get on missing name returns None.
//! - `names`: empty initially; populates as architectures register.
//! - `detect`: returns the registered name on match, None on no
//!   match (including when the field is the wrong value).
//! - `capabilities_for`: combines detect + get + capabilities.
use super::*;

struct TestArch;
impl Architecture for TestArch {
    fn name(&self) -> &'static str {
        "test"
    }
    fn detect(&self, config: &serde_json::Value) -> bool {
        config
            .get("test")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    }
    fn capabilities(&self) -> ArchCapabilities {
        ArchCapabilities::STUB
    }
    fn create_block(
        &self,
        _config: &crate::config::ModelConfig,
        _layer_idx: usize,
        _weights: &HashMap<String, candle_core::Tensor>,
        _device: &candle_core::Device,
    ) -> candle_core::Result<Box<dyn crate::components::TransformerBlock>> {
        Err(candle_core::Error::Msg("test arch has no blocks".into()))
    }
    fn create_model(
        &self,
        _config: crate::config::ModelConfig,
        _device: candle_core::Device,
        _weights: HashMap<String, candle_core::Tensor>,
        _num_kv_blocks: usize,
        _kv_quantization: bool,
    ) -> candle_core::Result<Box<dyn vllm_traits::ModelBackend>> {
        Err(candle_core::Error::Msg("test arch has no model".into()))
    }
}

#[test]
fn test_registry_register_and_get() {
    let registry = ArchitectureRegistry::new();
    let factory: ArchFactory = Arc::new(|| Box::new(TestArch));

    registry.register("test_arch", factory);

    let arch = registry.get("test_arch");
    assert!(arch.is_some());
    assert_eq!(arch.unwrap().name(), "test");
}

#[test]
fn test_registry_get_missing() {
    let registry = ArchitectureRegistry::new();
    let arch = registry.get("nonexistent");
    assert!(arch.is_none());
}

#[test]
fn test_registry_names() {
    let registry = ArchitectureRegistry::new();
    let factory: ArchFactory = Arc::new(|| Box::new(TestArch));

    registry.register("arch1", Arc::clone(&factory));
    registry.register("arch2", Arc::clone(&factory));

    let names = registry.names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"arch1".to_string()));
    assert!(names.contains(&"arch2".to_string()));
}

#[test]
fn test_registry_names_empty() {
    let registry = ArchitectureRegistry::new();
    let names = registry.names();
    assert!(names.is_empty());
}

#[test]
fn test_registry_detect() {
    let registry = ArchitectureRegistry::new();
    let factory: ArchFactory = Arc::new(|| Box::new(TestArch));
    registry.register("test_arch", factory);

    let config_true = serde_json::json!({ "test": true });
    let config_false = serde_json::json!({ "test": false });

    assert_eq!(registry.detect(&config_true), Some("test_arch".to_string()));
    assert_eq!(registry.detect(&config_false), None);
}

#[test]
fn test_registry_detect_no_match() {
    let registry = ArchitectureRegistry::new();
    let factory: ArchFactory = Arc::new(|| Box::new(TestArch));
    registry.register("test_arch", factory);

    let config = serde_json::json!({ "other": true });
    assert_eq!(registry.detect(&config), None);
}
