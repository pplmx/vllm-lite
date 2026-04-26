//! Gemma3 architecture registration.

use std::sync::Arc;

use crate::arch::{Architecture, ArchitectureRegistry};

use super::arch::Gemma3Architecture;

pub fn register(registry: &ArchitectureRegistry) {
    let factory: Arc<dyn Fn() -> Box<dyn Architecture> + Send + Sync> =
        Arc::new(|| Box::new(Gemma3Architecture::new()));
    registry.register("gemma3", factory);
}
