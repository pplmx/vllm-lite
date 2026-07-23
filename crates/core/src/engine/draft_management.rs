//! Engine-level draft registry accessors: resolve a draft model by id, register a new draft, list loaded drafts.
//!
//! Thin wrapper over the `DraftModelRegistry` from `vllm_core::speculative::registry`.
//! Kept separate from `mod.rs` so the Engine struct definition stays focused
//! on state, not behaviour.
//!
//! Methods that only exist to support unit tests live in a `[cfg(test)]`
//! impl block at the bottom of this file — they are not part of the
//! production Engine API surface.

use crate::engine::Engine;
use crate::speculative::AdaptiveSpeculativeDecoder;
use crate::speculative::draft_resolver::{DraftLoader, DraftResolver};
use crate::speculative::memory_budget::MemoryBudget;
use crate::speculative::registry::DraftModelRegistry;
use crate::types::AdaptiveDraftConfig;
use std::sync::Arc;

// Types used only by the test-only impl block below.
#[cfg(test)]
use crate::speculative::registry::{DraftId, DraftRegistryError, DraftSpec};
#[cfg(test)]
use vllm_traits::ModelBackend;

impl Engine {
    /// Replace the draft resolver's loader with a real implementation. The
    /// server calls this after constructing the Engine so that lazy-loaded
    /// drafts can actually be loaded from disk. Existing registrations are
    /// preserved.
    ///
    /// Returns `true` when a resolver was present and the new loader was
    /// installed. Returns `false` when the Engine was constructed without a
    /// resolver (e.g. via `new_boxed`) — the call is a no-op in that case
    /// and callers should log a warning to surface the misconfiguration.
    pub fn set_draft_loader(&mut self, loader: Arc<dyn DraftLoader>) -> bool {
        if let Some(resolver) = &self.draft_resolver {
            let registry = resolver.registry().clone();
            let self_spec = resolver.self_spec();
            let metrics = self.scheduler.metrics.clone();
            let new_resolver = Arc::new(DraftResolver::new(registry, self_spec, loader, metrics));
            self.draft_resolver = Some(new_resolver);
            true
        } else {
            false
        }
    }

    /// Access the shared memory budget.
    pub fn memory_budget(&self) -> Arc<MemoryBudget> {
        self.draft_registry.memory_budget().clone()
    }

    /// Access the draft registry for read-only inspection.
    pub fn draft_registry(&self) -> &DraftModelRegistry {
        &self.draft_registry
    }

    /// Switch the engine into speculative-decoding mode using the legacy
    /// single-`draft_model` path. Use [`Engine::enable_adaptive_speculative`]
    /// instead when you need per-request draft routing or adaptive acceptance.
    pub const fn enable_speculative(&mut self) {
        self.speculative_mode = true;
    }

    /// Enable adaptive speculative decoding.
    pub fn enable_adaptive_speculative(&mut self, config: AdaptiveDraftConfig) {
        self.adaptive_decoder = Some(AdaptiveSpeculativeDecoder::new(config));
        self.speculative_mode = true;
    }

    /// Disable adaptive speculative decoding.
    pub fn disable_adaptive_speculative(&mut self) {
        self.adaptive_decoder = None;
        self.speculative_mode = false;
    }

    /// Whether adaptive speculative decoding is enabled.
    pub const fn is_adaptive_speculative_enabled(&self) -> bool {
        self.adaptive_decoder.is_some()
    }
}

// ---------------------------------------------------------------------------
// Test-only accessors.
//
// These delegate straight through to the `DraftModelRegistry`; they exist so
// that `engine::tests` can drive the registry lifecycle (register → attach →
// unload → ref-count) without reaching into private fields. They are compiled
// out in non-test builds, which is why they don't need `#[allow(dead_code)]`.
// ---------------------------------------------------------------------------
#[cfg(test)]
impl Engine {
    /// Register a new draft at runtime.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a draft with the same id already exists.
    pub(crate) fn register_draft(&self, spec: DraftSpec) -> Result<(), DraftRegistryError> {
        self.draft_registry.register(spec)
    }

    /// Attach a loaded backend to a previously-registered draft id, promoting
    /// it from `Unloaded` to `Loaded`. Used by callers that drive the actual
    /// `ModelLoader` invocation. Does NOT reserve memory budget.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub(crate) fn attach_draft(
        &self,
        id: &DraftId,
        backend: Box<dyn ModelBackend>,
    ) -> Result<(), DraftRegistryError> {
        self.draft_registry.attach_loaded(id, backend)
    }

    /// Attach a loaded backend AND reserve the draft's estimated footprint in
    /// the memory budget.
    ///
    /// # Errors
    ///
    /// Returns `MemoryBudgetExceeded` if the load would exceed the configured
    /// budget, or another `DraftRegistryError` variant on failure.
    pub(crate) fn attach_draft_budgeted(
        &self,
        id: &DraftId,
        backend: Box<dyn ModelBackend>,
    ) -> Result<(), DraftRegistryError> {
        self.draft_registry.attach_loaded_budgeted(id, backend)
    }

    /// Unload a draft by id, releasing its backend and KV allocator.
    /// Returns `InUse(refcount)` if the draft is still referenced.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub(crate) fn unload_draft(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
        self.draft_registry.unload(id)
    }

    /// Force-unload a draft, bypassing refcount checks.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub(crate) fn force_unload_draft(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
        self.draft_registry.force_unload(id)
    }

    /// Increment the reference count for a draft. Driven by per-request
    /// routing logic since v18.3; see ADR-007.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub(crate) fn increment_draft_ref(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
        self.draft_registry.increment_ref(id)
    }

    /// Decrement the reference count for a draft. Auto-unloads when count
    /// reaches zero. Returns `true` if auto-unload was triggered.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub(crate) fn decrement_draft_ref(&self, id: &DraftId) -> Result<bool, DraftRegistryError> {
        self.draft_registry.decrement_ref(id)
    }
}
