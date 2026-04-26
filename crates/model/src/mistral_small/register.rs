//! Mistral Small architecture registration.

use std::sync::Arc;

use crate::arch::{Architecture, ArchitectureRegistry};

use super::arch::MistralSmallArchitecture;

pub fn register(registry: &ArchitectureRegistry) {
    let factory: Arc<dyn Fn() -> Box<dyn Architecture> + Send + Sync> =
        Arc::new(|| Box::new(MistralSmallArchitecture::new()));
    registry.register("mistral-small", factory);
}
