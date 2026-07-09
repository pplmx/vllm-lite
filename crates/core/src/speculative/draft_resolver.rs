//! `DraftResolver` — per-request draft selection with fallback semantics
//!
//! v18.0 Multi-Model Speculative Decoding phase 3 (RTE-01..03, FALL-01/02).
//!
//! Resolves a `Request::draft_model_id` to an actual `Box<dyn ModelBackend>` at
//! step time. When the requested draft is missing, unloaded, or fails to load,
//! silently falls back to the self-spec path (FALL-01) or non-spec decode.
//!
//! Runtime draft inference errors are handled at the engine level (FALL-02)
//! — the resolver itself only deals with load-time decisions.

use crate::metrics::{DraftResolutionKind, EnhancedMetricsCollector};
use crate::speculative::registry::{DraftId, DraftModelRegistry, DraftRegistryError};
use std::sync::{Arc, Mutex};
use vllm_traits::{ModelBackend, ModelError};

/// The outcome of resolving a draft for a single request.
#[derive(Clone)]
pub enum ResolvedDraft {
    /// Use an external draft from the registry.
    External(Arc<Mutex<Box<dyn ModelBackend>>>),
    /// Use the self-spec fallback (v17 baseline).
    SelfSpec(Arc<Mutex<Box<dyn ModelBackend>>>),
    /// No speculative decoding — pure target decode.
    None,
}

impl std::fmt::Debug for ResolvedDraft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::External(_) => f.debug_tuple("External").field(&"<backend>").finish(),
            Self::SelfSpec(_) => f.debug_tuple("SelfSpec").field(&"<backend>").finish(),
            Self::None => f.debug_tuple("None").finish(),
        }
    }
}

impl ResolvedDraft {
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::External(_) => "external",
            Self::SelfSpec(_) => "self_spec",
            Self::None => "none",
        }
    }

    #[must_use]
    pub const fn is_some(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Loader trait for resolving a draft id to a backend.
///
/// Implemented by the server (which has access to `vllm_model::loader`) or
/// by tests with stub loaders.
pub trait DraftLoader: Send + Sync {
    /// Run the loader and produce the target type (model, cache, etc.).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn load(&self, id: &DraftId) -> std::result::Result<Box<dyn ModelBackend>, DraftRegistryError>;
}

/// Loader that always returns `LoadFailed`. Used as a placeholder when an Engine
/// is constructed with `with_drafts_boxed` / `with_budget_boxed` but the server
/// hasn't yet wired a real loader.
///
/// The resolver treats every load failure as a FALL-01 fallback to self-spec —
/// so this loader effectively keeps all external drafts at `Unloaded` state and
/// the engine behaves like self-spec.
#[derive(Debug)]
pub struct NoopLoader;

impl DraftLoader for NoopLoader {
    fn load(&self, id: &DraftId) -> std::result::Result<Box<dyn ModelBackend>, DraftRegistryError> {
        // NoopLoader is a deliberate sentinel used when no real loader is
        // configured. Surface this as a typed `Model` failure (no I/O
        // happened — there is no path or source io::Error to attach).
        Err(DraftRegistryError::Model(
            id.clone(),
            ModelError::new(format!("NoopLoader: no loader wired for {id}")),
        ))
    }
}

impl dyn DraftLoader {
    /// Returns an `Arc<Self>` wrapping the default [`NoopLoader`] (always
    /// returns `LoadFailed` so the resolver falls back to self-spec).
    ///
    /// This is the closest equivalent to
    /// `Arc::<dyn DraftLoader>::default()`; Rust's orphan rule prevents
    /// a direct `impl Default for Arc<dyn ...>` because `Arc` is foreign and
    /// there is no local type appearing before the uncovered trait-object
    /// parameter.
    #[must_use]
    pub fn default_arc() -> Arc<Self> {
        Arc::new(NoopLoader)
    }
}
/// Per-request draft resolver.
pub struct DraftResolver {
    registry: Arc<DraftModelRegistry>,
    self_spec: Option<Arc<Mutex<Box<dyn ModelBackend>>>>,
    loader: Arc<dyn DraftLoader>,
    metrics: Arc<EnhancedMetricsCollector>,
}

impl std::fmt::Debug for DraftResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DraftResolver")
            .field("registry", &self.registry)
            .field(
                "self_spec",
                &self.self_spec.as_ref().map(|_| "<dyn ModelBackend>"),
            )
            .field("loader", &"<dyn DraftLoader>")
            .field("metrics", &self.metrics)
            .finish()
    }
}

impl DraftResolver {
    pub fn new(
        registry: Arc<DraftModelRegistry>,
        self_spec: Option<Arc<Mutex<Box<dyn ModelBackend>>>>,
        loader: Arc<dyn DraftLoader>,
        metrics: Arc<EnhancedMetricsCollector>,
    ) -> Self {
        Self {
            registry,
            self_spec,
            loader,
            metrics,
        }
    }

    /// Resolve a draft for a request. Returns SelfSpec/None on any failure
    /// (FALL-01 path). Records metrics for every call.
    pub fn resolve(&self, request_draft_id: Option<&DraftId>) -> ResolvedDraft {
        // Case 1: no external draft requested
        let Some(id) = request_draft_id else {
            self.metrics.inc_draft_resolution(DraftResolutionKind::None);
            return self.fallback_to_self_spec_or_none();
        };

        // Case 2: draft is already loaded → external
        if let Some(backend) = self.registry.get_loaded_backend(id) {
            self.metrics
                .inc_draft_resolution(DraftResolutionKind::External);
            return ResolvedDraft::External(backend);
        }

        // Case 3: draft is registered but unloaded → try to load
        if self.registry.contains(id) {
            match self.loader.load(id) {
                Ok(backend) => {
                    // Try to attach via the budgeted path (no-op if unlimited)
                    match self.registry.attach_loaded_budgeted(id, backend) {
                        Ok(()) => {
                            // Re-fetch from registry (now loaded)
                            if let Some(arc_backend) = self.registry.get_loaded_backend(id) {
                                self.metrics
                                    .inc_draft_resolution(DraftResolutionKind::External);
                                return ResolvedDraft::External(arc_backend);
                            }
                            // Shouldn't happen — attach succeeded but lookup failed
                            self.metrics.inc_draft_load_failure();
                            self.fallback_to_self_spec_or_none()
                        }
                        Err(e) => {
                            tracing::warn!(
                                draft_id = %id,
                                error = %e,
                                "draft attach failed; falling back to self-spec"
                            );
                            self.metrics.inc_draft_load_failure();
                            self.fallback_to_self_spec_or_none()
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        draft_id = %id,
                        error = %e,
                        "draft load failed; falling back to self-spec"
                    );
                    self.metrics.inc_draft_load_failure();
                    self.fallback_to_self_spec_or_none()
                }
            }
        } else {
            // Case 4: unknown draft id
            tracing::warn!(
                draft_id = %id,
                "unknown draft id; falling back to self-spec"
            );
            self.metrics.inc_draft_load_failure();
            self.fallback_to_self_spec_or_none()
        }
    }

    fn fallback_to_self_spec_or_none(&self) -> ResolvedDraft {
        self.self_spec.as_ref().map_or_else(
            || {
                self.metrics.inc_draft_resolution(DraftResolutionKind::None);
                ResolvedDraft::None
            },
            |s| {
                self.metrics
                    .inc_draft_resolution(DraftResolutionKind::SelfSpec);
                ResolvedDraft::SelfSpec(s.clone())
            },
        )
    }

    /// Access the underlying registry (for advanced callers).
    #[must_use]
    pub const fn registry(&self) -> &Arc<DraftModelRegistry> {
        &self.registry
    }

    /// Access the self-spec backend, if any. Returns the Arc clone so callers
    /// (e.g., the Engine) can keep a reference after the resolver is rebuilt.
    #[must_use]
    pub fn self_spec(&self) -> Option<Arc<Mutex<Box<dyn ModelBackend>>>> {
        self.self_spec.clone()
    }
}

// Unit tests are extracted to `tests.rs` to keep this file under the
// 800-line soft cap. See `tests.rs` for the test surface
// (resolve → SelfSpec / None / External, lazy-loader attach,
// loader-failure fallback, unknown-id fallback, resolution metrics,
// NoopLoader default).
#[cfg(test)]
mod tests;
