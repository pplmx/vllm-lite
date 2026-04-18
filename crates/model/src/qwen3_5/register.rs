//! Qwen3.5 architecture registration.

use std::sync::Arc;

use crate::arch::{Architecture, ArchitectureRegistry};

use super::arch::Qwen35Architecture;

pub fn register(registry: &ArchitectureRegistry) {
    let factory: Arc<dyn Fn() -> Box<dyn Architecture> + Send + Sync> =
        Arc::new(|| Box::new(Qwen35Architecture::new()));
    registry.register("qwen3.5", factory);
}
