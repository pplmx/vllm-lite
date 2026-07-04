//! Engine-level draft registry accessors: resolve a draft model by id, register a new draft, list loaded drafts.
//!
//! Thin wrapper over the `DraftModelRegistry` from `vllm_core::speculative::registry`.
//! Kept separate from `mod.rs` so the Engine struct definition stays focused
//! on state, not behaviour.

// Sub-module for draft registry, resolver, and speculative-mode management methods on Engine.
// See mod.rs for the Engine struct definition.

use crate::engine::Engine;
use crate::speculative::AdaptiveSpeculativeDecoder;
use crate::speculative::draft_resolver::{DraftLoader, DraftResolver};
use crate::speculative::memory_budget::MemoryBudget;
use crate::speculative::registry::{DraftId, DraftModelRegistry, DraftRegistryError, DraftSpec};
use crate::types::AdaptiveDraftConfig;
use std::sync::Arc;
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

    /// Register a new draft at runtime. Returns `DraftRegistryError::AlreadyLoaded`
    ///
    /// # Errors
    ///
    /// Returns `Err` if registration fails (e.g. duplicate name or invalid input).
    /// if a draft with the same id already exists.
    pub fn register_draft(&self, spec: DraftSpec) -> std::result::Result<(), DraftRegistryError> {
        self.draft_registry.register(spec)
    }

    /// Attach a loaded backend to a previously-registered draft id, promoting
    /// it from `Unloaded` to `Loaded`. Used by callers that drive the actual
    /// `ModelLoader` invocation. Does NOT reserve memory budget — use
    /// [`Self::attach_draft_budgeted`] when the registry was constructed with
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// a budget and you want VRAM enforcement.
    pub fn attach_draft(
        &self,
        id: &DraftId,
        backend: Box<dyn ModelBackend>,
    ) -> std::result::Result<(), DraftRegistryError> {
        self.draft_registry.attach_loaded(id, backend)
    }

    /// Attach a loaded backend AND reserve the draft's estimated footprint in
    /// the memory budget. Returns `MemoryBudgetExceeded` if the load would
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// exceed the configured budget.
    pub fn attach_draft_budgeted(
        &self,
        id: &DraftId,
        backend: Box<dyn ModelBackend>,
    ) -> std::result::Result<(), DraftRegistryError> {
        self.draft_registry.attach_loaded_budgeted(id, backend)
    }

    /// Unload a draft by id, releasing its backend and KV allocator.
    /// Returns `InUse(refcount)` if the draft is still referenced; use
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// [`Self::force_unload_draft`] to bypass.
    pub fn unload_draft(&self, id: &DraftId) -> std::result::Result<(), DraftRegistryError> {
        self.draft_registry.unload(id)
    }

    /// Force-unload a draft, bypassing refcount checks. Used by admin tooling
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// and tests.
    pub fn force_unload_draft(&self, id: &DraftId) -> std::result::Result<(), DraftRegistryError> {
        self.draft_registry.force_unload(id)
    }

    /// Increment the reference count for a draft. Driven by per-request
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// routing logic since v18.3; see ADR-007.
    pub fn increment_draft_ref(&self, id: &DraftId) -> std::result::Result<(), DraftRegistryError> {
        self.draft_registry.increment_ref(id)
    }

    /// Decrement the reference count for a draft. Auto-unloads when count
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// reaches zero. Returns `true` if auto-unload was triggered.
    pub fn decrement_draft_ref(
        &self,
        id: &DraftId,
    ) -> std::result::Result<bool, DraftRegistryError> {
        self.draft_registry.decrement_ref(id)
    }

    /// Switch the engine into speculative-decoding mode using the legacy
    /// single-`draft_model` path. Use [`Engine::enable_adaptive_speculative`]
    /// instead when you need per-request draft routing or adaptive acceptance.
    pub const fn enable_speculative(&mut self) {
        self.speculative_mode = true;
    }

    /// Enable adaptive speculative decoding
    pub fn enable_adaptive_speculative(&mut self, config: AdaptiveDraftConfig) {
        self.adaptive_decoder = Some(AdaptiveSpeculativeDecoder::new(config));
        self.speculative_mode = true;
    }

    /// Disable adaptive speculative decoding
    pub fn disable_adaptive_speculative(&mut self) {
        self.adaptive_decoder = None;
        self.speculative_mode = false;
    }

    /// Check if adaptive speculative is enabled
    pub const fn is_adaptive_speculative_enabled(&self) -> bool {
        self.adaptive_decoder.is_some()
    }
}
