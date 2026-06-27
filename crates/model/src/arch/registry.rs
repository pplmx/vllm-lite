//! Architecture registry for dynamic registration.

use once_cell::sync::Lazy;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::{ArchCapabilities, Architecture};

type ArchFactory = Arc<dyn Fn() -> Box<dyn Architecture> + Send + Sync>;

/// ArchitectureRegistry: architecture registry.
pub struct ArchitectureRegistry {
    architectures: RwLock<HashMap<String, ArchFactory>>,
}

impl Default for ArchitectureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchitectureRegistry {
    /// new: new.
    pub fn new() -> Self {
        Self {
            architectures: RwLock::new(HashMap::new()),
        }
    }

    /// register: register.
    pub fn register(&self, name: &'static str, factory: ArchFactory) {
        self.architectures
            .write()
            .expect("RwLock poisoned - this indicates a bug")
            .insert(name.to_string(), factory);
    }

    /// get: get.
    pub fn get(&self, name: &str) -> Option<Box<dyn Architecture>> {
        self.architectures
            .read()
            .ok()?
            .get(name)
            .map(|factory| factory())
    }

    /// detect: detect.
    pub fn detect(&self, config_json: &Value) -> Option<String> {
        let regs = self.architectures.read().ok()?;
        for (name, factory) in regs.iter() {
            let arch = factory();
            if arch.detect(config_json) {
                return Some(name.clone());
            }
        }
        None
    }

    /// names: names.
    pub fn names(&self) -> Vec<String> {
        self.architectures
            .read()
            .ok()
            .map(|guard| guard.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns capabilities for the architecture detected from `config_json`.
    pub fn capabilities_for(&self, config_json: &Value) -> Option<ArchCapabilities> {
        let name = self.detect(config_json)?;
        self.get(&name).map(|arch| arch.capabilities())
    }
}

/// ARCHITECTURE_REGISTRY: architecture registry static value.
pub static ARCHITECTURE_REGISTRY: Lazy<ArchitectureRegistry> = Lazy::new(ArchitectureRegistry::new);

/// Register all known architectures for config detection and model creation.
///
/// Stub architectures (Gemma3, Llama4, Phi4, MistralSmall) remain registered so
/// `detect()` works, but `ModelLoader` rejects them unless `--allow-stub` is set
/// (see Phase 4.4 Option C in `.planning/MODEL-ARCHITECTURE-REFACTOR.md`).
pub fn register_all_archs(registry: &ArchitectureRegistry) {
    crate::llama::register::register(registry);
    crate::mistral::register::register(registry);
    crate::qwen3::register::register(registry);
    crate::qwen3_5::register::register(registry);
    crate::gemma3::register::register(registry);
    crate::gemma4::register::register(registry);
    crate::llama4::register::register(registry);
    crate::mistral_small::register::register(registry);
    crate::phi4::register::register(registry);
    crate::mixtral::register::register(registry);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestArch;
    impl Architecture for TestArch {
        fn name(&self) -> &'static str {
            "test"
        }
        fn detect(&self, config: &serde_json::Value) -> bool {
            config
                .get("test")
                .and_then(|v| v.as_bool())
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
}
