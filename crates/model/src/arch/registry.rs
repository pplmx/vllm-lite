#![allow(clippy::module_name_repetitions)]
//! Architecture registry for dynamic registration.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use super::{ArchCapabilities, Architecture, StubArchitecture};

type ArchFactory = Arc<dyn Fn() -> Box<dyn Architecture> + Send + Sync>;

/// Process-wide registry of [`Architecture`] implementations.
///
/// Architectures register themselves via [`register`](Self::register)
/// (typically once at startup, from [`register_all_archs`]) and are
/// later resolved by name through [`get`](Self::get) or by config
/// shape through [`detect`](Self::detect). The internal `RwLock`
/// makes concurrent reads cheap; the registry is not on the
/// inference hot path.
///
/// # Examples
///
/// ```ignore
/// let registry = ArchitectureRegistry::default();
/// let factory: ArchFactory = Arc::new(|| Box::new(MyArch));
/// registry.register("my_arch", factory);
/// assert!(registry.get("my_arch").is_some());
/// ```
pub struct ArchitectureRegistry {
    architectures: RwLock<HashMap<String, ArchFactory>>,
}

impl std::fmt::Debug for ArchitectureRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.architectures.read().map_or(0, |m| m.len());
        f.debug_struct("ArchitectureRegistry")
            .field("architectures_count", &count)
            .finish()
    }
}

impl Default for ArchitectureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchitectureRegistry {
    /// Construct an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            architectures: RwLock::new(HashMap::new()),
        }
    }

    /// Register an architecture factory under a stable name.
    ///
    /// If `name` is already registered, the factory is replaced.
    /// Typically called once at startup from [`register_all_archs`].
    ///
    /// # Panics
    ///
    /// Panics only if the internal `RwLock` is poisoned, which would
    /// indicate another thread panicked while holding the write lock.
    pub fn register(&self, name: &'static str, factory: ArchFactory) {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        self.architectures
            .write()
            // invariant: lock is only held for sync field access; poisoning only happens on panic during a critical section.
            .expect("RwLock poisoned - this indicates a bug")
            .insert(name.to_string(), factory);
    }

    /// Look up an architecture by name and instantiate it.
    ///
    /// Returns `None` if `name` is not registered. Each call
    /// invokes the registered factory, producing a fresh
    /// `Box<dyn Architecture>`; the registry itself is stateless
    /// with respect to the instances.
    pub fn get(&self, name: &str) -> Option<Box<dyn Architecture>> {
        self.architectures
            .read()
            .ok()?
            .get(name)
            .map(|factory| factory())
    }

    /// Iterate registered architectures and return the first one
    /// whose [`Architecture::detect`] returns `true` for
    /// `config_json`.
    ///
    /// Returns `Some(name)` (the registered name) on the first
    /// match, or `None` if no architecture claims the config.
    /// Iteration order is the underlying `HashMap` order, which is
    /// unspecified; callers that need a deterministic winner must
    /// order their registrations accordingly.
    pub fn detect(&self, config_json: &Value) -> Option<String> {
        let regs = self.architectures.read().ok()?;
        let result = {
            let mut detected = None;
            for (name, factory) in regs.iter() {
                let arch = factory();
                if arch.detect(config_json) {
                    detected = Some(name.clone());
                    break;
                }
            }
            detected
        };
        drop(regs);
        result
    }

    /// Snapshot of all currently-registered architecture names.
    ///
    /// Returns an empty `Vec` if the lock is poisoned; in normal
    /// operation this should always be non-empty after
    /// [`register_all_archs`] has run.
    pub fn names(&self) -> Vec<String> {
        self.architectures
            .read()
            .ok()
            .map(|guard| guard.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Resolve `config_json` to an architecture and return its
    /// capabilities.
    ///
    /// Returns `None` if no architecture claims the config.
    /// Combines [`detect`](Self::detect) + [`get`](Self::get) +
    /// [`Architecture::capabilities`] in one call.
    #[allow(dead_code)] // test-only helper; reachable under cfg(test) only
    pub(crate) fn capabilities_for(&self, config_json: &Value) -> Option<ArchCapabilities> {
        let name = self.detect(config_json)?;
        self.get(&name).map(|arch| arch.capabilities())
    }
}

/// Process-wide [`ArchitectureRegistry`] singleton.
///
/// Lazy-initialized via [`std::sync::LazyLock`] (Rust 1.80+), so
/// the first access triggers `ArchitectureRegistry::new`. Most
/// callers should also call [`register_all_archs`] once at
/// startup to populate it.
pub static ARCHITECTURE_REGISTRY: LazyLock<ArchitectureRegistry> =
    LazyLock::new(ArchitectureRegistry::new);

/// Register all known architectures for config detection and model creation.
///
/// Stub architectures (Gemma3, Llama4, Phi4, `MistralSmall`) are
/// registered as [`StubArchitecture`] instances (Phase 18 ARCH-05) —
/// `detect()` works, but `ModelLoader` rejects them unless
/// `--allow-stub` is set (see Phase 4.4 Option C in
/// `.planning/MODEL-ARCHITECTURE-REFACTOR.md`).
pub fn register_all_archs(registry: &ArchitectureRegistry) {
    crate::llama::register::register(registry);
    crate::mistral::register::register(registry);
    crate::qwen3::register::register(registry);
    crate::qwen3_5::register::register(registry);
    crate::gemma4::register::register(registry);
    crate::mixtral::register::register(registry);
    // Stubs (Phase 18 ARCH-05: shared `StubArchitecture`).
    register_stub(registry, StubArchitecture::gemma3());
    register_stub(registry, StubArchitecture::llama4());
    register_stub(registry, StubArchitecture::phi4());
    register_stub(registry, StubArchitecture::mistral_small());
}

/// Register a [`StubArchitecture`] under its `name()`.
fn register_stub(registry: &ArchitectureRegistry, stub: StubArchitecture) {
    let factory: ArchFactory = Arc::new(move || Box::new(stub));
    registry.register(stub.name(), factory);
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// registry module under the 800-line soft cap. They cover the
// register/get/names/detect/capabilities_for surface against a
// minimal `TestArch` stub.
#[cfg(test)]
mod tests;
