//! Phi-4 architecture registration.

use std::sync::Arc;

use crate::arch::{Architecture, ArchitectureRegistry};

use super::arch::Phi4Architecture;

pub fn register(registry: &ArchitectureRegistry) {
    let factory: Arc<dyn Fn() -> Box<dyn Architecture> + Send + Sync> =
        Arc::new(|| Box::new(Phi4Architecture::new()));
    registry.register("phi4", factory);
}
