//! Draft-aware engine constructors: `with_drafts_boxed`, `with_drafts`,
//! `with_budget_boxed`, plus the private `install_default_resolver` helper.
//!
//! These all build on top of `Engine::with_config_boxed` (defined in
//! [`super`]) by populating the draft model registry / resolver.

use std::sync::{Arc, Mutex};

use crate::engine::Engine;
use crate::speculative::draft_resolver::{DraftLoader, DraftResolver, NoopLoader};
use crate::speculative::memory_budget::MemoryBudget;
use crate::speculative::registry::{DraftModelRegistry, DraftSpec};
use crate::types::SchedulerConfig;
use vllm_traits::ModelBackend;

impl Engine {
    /// Construct an Engine pre-loaded with a set of draft specs.
    ///
    /// All specs are registered in the [`DraftModelRegistry`] as `Unloaded`.
    /// They will be loaded lazily on first use (or eagerly by the caller —
    /// see `DraftModelRegistry::attach_loaded`) — this method does NOT trigger
    /// any I/O. The Engine's `draft_resolver` is wired with a `NoopLoader`:
    /// any attempt to lazy-load falls back to self-spec via FALL-01. The
    /// server should construct a real `DraftLoader` via
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// [`Self::set_draft_loader`] before serving requests that name drafts.
    #[must_use]
    pub fn with_drafts_boxed(
        target_model: Box<dyn ModelBackend>,
        draft_model: Option<Box<dyn ModelBackend>>,
        draft_specs: Vec<DraftSpec>,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        let mut engine = Self::with_config_boxed(
            target_model,
            draft_model,
            config,
            max_draft_tokens,
            num_kv_blocks,
        );
        for spec in draft_specs {
            // Duplicate ids in the spec list are a programmer error — surface them.
            engine
                .draft_registry
                // invariant: caller (`with_drafts_boxed`) supplies a deduplicated spec list;
                // duplicates are a programmer error.
                .register(spec)
                // invariant: pre-check guarantees uniqueness; this path is a programmer error.
                .expect("with_drafts_boxed: duplicate draft id in spec list");
        }
        engine.install_default_resolver();
        engine
    }

    /// Construct an Engine pre-loaded with a set of draft specs (generic form).
    pub fn with_drafts<M: ModelBackend + 'static>(
        target_model: M,
        draft_model: Option<M>,
        draft_specs: Vec<DraftSpec>,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        Self::with_drafts_boxed(
            Box::new(target_model),
            draft_model.map(|m| Box::new(m) as Box<dyn ModelBackend>),
            draft_specs,
            config,
            max_draft_tokens,
            num_kv_blocks,
        )
    }

    /// Construct an Engine with a custom memory budget. The same budget is
    /// shared with the draft registry. `draft_resolver` is wired with a
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// `NoopLoader` (callers may replace via [`Self::set_draft_loader`]).
    pub fn with_budget_boxed(
        target_model: Box<dyn ModelBackend>,
        draft_model: Option<Box<dyn ModelBackend>>,
        draft_specs: Vec<DraftSpec>,
        budget: Arc<MemoryBudget>,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        let mut engine = Self::with_config_boxed(
            target_model,
            draft_model,
            config,
            max_draft_tokens,
            num_kv_blocks,
        );
        // Replace the default unlimited-budget registry with one bound to the
        // caller's budget.
        engine.draft_registry = Arc::new(DraftModelRegistry::with_budget(budget));
        for spec in draft_specs {
            engine
                .draft_registry
                // invariant: caller (`with_budget_boxed`) supplies a deduplicated spec list;
                // duplicates are a programmer error.
                .register(spec)
                // invariant: pre-check guarantees uniqueness; this path is a programmer error.
                .expect("with_budget_boxed: duplicate draft id in spec list");
        }
        engine.install_default_resolver();
        engine
    }

    /// Install a `DraftResolver` with a `NoopLoader` on this Engine. Called by
    /// `with_drafts_boxed` / `with_budget_boxed`. Idempotent — replaces any
    /// existing resolver. The resolver shares the same `Arc<DraftModelRegistry>`
    /// as `self.draft_registry`, so register/unload operations on the Engine
    /// are immediately visible to the resolver.
    fn install_default_resolver(&mut self) {
        let registry = self.draft_registry.clone();
        let metrics = self.scheduler.metrics.clone();
        let self_spec: Option<Arc<Mutex<Box<dyn ModelBackend>>>> = self.draft_model.clone();
        let loader: Arc<dyn DraftLoader> = Arc::new(NoopLoader);
        let resolver = Arc::new(DraftResolver::new(registry, self_spec, loader, metrics));
        self.draft_resolver = Some(resolver);
    }
}
