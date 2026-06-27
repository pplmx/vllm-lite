//! Llama 4 architecture registration.

use std::sync::Arc;

use crate::arch::{Architecture, ArchitectureRegistry};

use super::arch::Llama4Architecture;

pub fn register(registry: &ArchitectureRegistry) {
    let factory: Arc<dyn Fn() -> Box<dyn Architecture> + Send + Sync> =
        Arc::new(|| Box::new(Llama4Architecture::new()));
    registry.register("llama4", factory);
}
