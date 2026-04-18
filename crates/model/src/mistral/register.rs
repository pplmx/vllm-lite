//! Mistral architecture registration.

use crate::arch::ARCHITECTURE_REGISTRY;
use std::sync::Arc;

use super::arch::MistralArchitecture;

pub fn register() {
    let factory: Arc<dyn Fn() -> Box<dyn crate::arch::Architecture> + Send + Sync> =
        Arc::new(move || Box::new(MistralArchitecture::new()));
    ARCHITECTURE_REGISTRY.register("mistral", factory);
}
