//! Architecture registry for dynamic registration.

use once_cell::sync::Lazy;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::Architecture;

type ArchFactory = Arc<dyn Fn() -> Box<dyn Architecture> + Send + Sync>;

pub struct ArchitectureRegistry {
    architectures: RwLock<HashMap<String, ArchFactory>>,
}

impl Default for ArchitectureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchitectureRegistry {
    pub fn new() -> Self {
        Self {
            architectures: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&self, name: &'static str, factory: ArchFactory) {
        self.architectures
            .write()
            .unwrap()
            .insert(name.to_string(), factory);
    }

    pub fn get(&self, name: &str) -> Option<Box<dyn Architecture>> {
        self.architectures
            .read()
            .unwrap()
            .get(name)
            .map(|factory| factory())
    }

    pub fn detect(&self, config_json: &Value) -> Option<String> {
        let regs = self.architectures.read().unwrap();
        for (name, factory) in regs.iter() {
            let arch = factory();
            if arch.detect(config_json) {
                return Some(name.clone());
            }
        }
        None
    }
}

pub static ARCHITECTURE_REGISTRY: Lazy<ArchitectureRegistry> = Lazy::new(ArchitectureRegistry::new);

pub fn register_all_archs(registry: &ArchitectureRegistry) {
    crate::llama::register::register(registry);
    crate::mistral::register::register(registry);
    crate::qwen3::register::register(registry);
    crate::qwen3_5::register::register(registry);
    crate::gemma4::register::register(registry);
    crate::mixtral::register::register(registry);
}
