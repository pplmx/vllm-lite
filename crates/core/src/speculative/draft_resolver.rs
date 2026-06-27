//! DraftResolver — per-request draft selection with fallback semantics
//!
//! v18.0 Multi-Model Speculative Decoding phase 3 (RTE-01..03, FALL-01/02).
//!
//! Resolves a `Request::draft_model_id` to an actual `Box<dyn ModelBackend>` at
//! step time. When the requested draft is missing, unloaded, or fails to load,
//! silently falls back to the self-spec path (FALL-01) or non-spec decode.
//!
//! Runtime draft inference errors are handled at the engine level (FALL-02)
//! — the resolver itself only deals with load-time decisions.

use crate::metrics::EnhancedMetricsCollector;
use crate::speculative::draft_registry::{DraftId, DraftModelRegistry, DraftRegistryError};
use std::sync::{Arc, Mutex};
use vllm_traits::ModelBackend;

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
            ResolvedDraft::External(_) => f.debug_tuple("External").field(&"<backend>").finish(),
            ResolvedDraft::SelfSpec(_) => f.debug_tuple("SelfSpec").field(&"<backend>").finish(),
            ResolvedDraft::None => f.debug_tuple("None").finish(),
        }
    }
}

impl ResolvedDraft {
/// kind: kind.
    pub fn kind(&self) -> &'static str {
        match self {
            ResolvedDraft::External(_) => "external",
            ResolvedDraft::SelfSpec(_) => "self_spec",
            ResolvedDraft::None => "none",
        }
    }

/// is_some: is some.
    pub fn is_some(&self) -> bool {
        !matches!(self, ResolvedDraft::None)
    }
}

/// Loader trait for resolving a draft id to a backend.
///
/// Implemented by the server (which has access to `vllm_model::loader`) or
/// by tests with stub loaders.
pub trait DraftLoader: Send + Sync {
    fn load(&self, id: &DraftId) -> std::result::Result<Box<dyn ModelBackend>, DraftRegistryError>;
}

/// Loader that always returns `LoadFailed`. Used as a placeholder when an Engine
/// is constructed with `with_drafts_boxed` / `with_budget_boxed` but the server
/// hasn't yet wired a real loader. The resolver treats every load failure as
/// a FALL-01 fallback to self-spec — so this loader effectively keeps all
/// external drafts at `Unloaded` state and the engine behaves like self-spec.
pub struct NoopLoader;

impl DraftLoader for NoopLoader {
    fn load(&self, id: &DraftId) -> std::result::Result<Box<dyn ModelBackend>, DraftRegistryError> {
        Err(DraftRegistryError::LoadFailed(format!(
            "NoopLoader: no loader wired for {id}"
        )))
    }
}

/// Per-request draft resolver.
pub struct DraftResolver {
    registry: Arc<DraftModelRegistry>,
    self_spec: Option<Arc<Mutex<Box<dyn ModelBackend>>>>,
    loader: Arc<dyn DraftLoader>,
    metrics: Arc<EnhancedMetricsCollector>,
}

impl DraftResolver {
/// new: new.
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
        let id = match request_draft_id {
            None => {
                self.metrics.inc_draft_resolution("none");
                return self.fallback_to_self_spec_or_none();
            }
            Some(id) => id,
        };

        // Case 2: draft is already loaded → external
        if let Some(backend) = self.registry.get_loaded_backend(id) {
            self.metrics.inc_draft_resolution("external");
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
                                self.metrics.inc_draft_resolution("external");
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
        match &self.self_spec {
            Some(s) => {
                self.metrics.inc_draft_resolution("self_spec");
                ResolvedDraft::SelfSpec(s.clone())
            }
            None => {
                self.metrics.inc_draft_resolution("none");
                ResolvedDraft::None
            }
        }
    }

    /// Access the underlying registry (for advanced callers).
    pub fn registry(&self) -> &Arc<DraftModelRegistry> {
        &self.registry
    }

    /// Access the self-spec backend, if any. Returns the Arc clone so callers
    /// (e.g., the Engine) can keep a reference after the resolver is rebuilt.
    pub fn self_spec(&self) -> Option<Arc<Mutex<Box<dyn ModelBackend>>>> {
        self.self_spec.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::DraftSpec;
    use std::path::PathBuf;

    struct StubLoader {
        fail_on: Vec<DraftId>,
    }
    impl DraftLoader for StubLoader {
        fn load(
            &self,
            id: &DraftId,
        ) -> std::result::Result<Box<dyn ModelBackend>, DraftRegistryError> {
            if self.fail_on.iter().any(|f| f == id) {
                Err(DraftRegistryError::LoadFailed("stub failure".into()))
            } else {
                Ok(Box::new(StubBackend))
            }
        }
    }

    struct StubBackend;
    impl ModelBackend for StubBackend {
        fn forward(
            &mut self,
            _seq_ids: &[vllm_traits::SeqId],
            _input_tokens: &[Vec<vllm_traits::TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> vllm_traits::Result<vllm_traits::BatchOutput> {
            panic!("StubBackend::forward not used in resolver tests")
        }
        fn forward_logits(
            &mut self,
            _seq_ids: &[vllm_traits::SeqId],
            _input_tokens: &[Vec<vllm_traits::TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> vllm_traits::Result<Vec<Vec<f32>>> {
            panic!("StubBackend::forward_logits not used in resolver tests")
        }
        fn embed(
            &mut self,
            _input_tokens: &[Vec<vllm_traits::TokenId>],
            _positions: &[Vec<usize>],
        ) -> vllm_traits::Result<Vec<Vec<f32>>> {
            panic!("StubBackend::embed not used in resolver tests")
        }
        fn vocab_size(&self) -> usize {
            32000
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn num_heads(&self) -> usize {
            1
        }
    }

    fn make_resolver(
        specs: Vec<DraftSpec>,
        loaded: Vec<DraftId>,
        self_spec: Option<Arc<Mutex<Box<dyn ModelBackend>>>>,
        loader_failures: Vec<DraftId>,
    ) -> (Arc<DraftResolver>, Arc<EnhancedMetricsCollector>) {
        let registry = Arc::new(DraftModelRegistry::new());
        for spec in specs {
            registry.register(spec).unwrap();
        }
        for id in &loaded {
            registry.attach_loaded(id, Box::new(StubBackend)).unwrap();
        }
        let loader: Arc<dyn DraftLoader> = Arc::new(StubLoader {
            fail_on: loader_failures,
        });
        let metrics = Arc::new(EnhancedMetricsCollector::new());
        let resolver = Arc::new(DraftResolver::new(
            registry,
            self_spec,
            loader,
            metrics.clone(),
        ));
        (resolver, metrics)
    }

    #[test]
    fn test_resolve_none_returns_self_spec_when_available() {
        let self_spec: Arc<Mutex<Box<dyn ModelBackend>>> =
            Arc::new(Mutex::new(Box::new(StubBackend)));
        let (resolver, _) = make_resolver(vec![], vec![], Some(self_spec.clone()), vec![]);
        match resolver.resolve(None) {
            ResolvedDraft::SelfSpec(_) => {}
            other => panic!("expected SelfSpec, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_none_returns_none_when_no_self_spec() {
        let (resolver, _) = make_resolver(vec![], vec![], None, vec![]);
        match resolver.resolve(None) {
            ResolvedDraft::None => {}
            other => panic!("expected None, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_loaded_returns_external() {
        let spec = DraftSpec {
            id: DraftId("a".into()),
            model_dir: PathBuf::from("/nope"),
            arch_hint: None,
            kv_blocks: 4,
            weight_size_estimate_bytes: 0,
            ref_count: 0,
        };
        let self_spec: Arc<Mutex<Box<dyn ModelBackend>>> =
            Arc::new(Mutex::new(Box::new(StubBackend)));
        let (resolver, _) = make_resolver(
            vec![spec],
            vec![DraftId("a".into())],
            Some(self_spec),
            vec![],
        );
        match resolver.resolve(Some(&DraftId("a".into()))) {
            ResolvedDraft::External(_) => {}
            other => panic!("expected External, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_unloaded_triggers_loader_and_attaches() {
        let spec = DraftSpec {
            id: DraftId("a".into()),
            model_dir: PathBuf::from("/nope"),
            arch_hint: None,
            kv_blocks: 4,
            weight_size_estimate_bytes: 0,
            ref_count: 0,
        };
        let self_spec: Arc<Mutex<Box<dyn ModelBackend>>> =
            Arc::new(Mutex::new(Box::new(StubBackend)));
        let (resolver, _) = make_resolver(
            vec![spec],
            vec![], // not pre-loaded
            Some(self_spec),
            vec![], // loader succeeds
        );
        match resolver.resolve(Some(&DraftId("a".into()))) {
            ResolvedDraft::External(_) => {}
            other => panic!("expected External after lazy load, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_loader_failure_falls_back_to_self_spec() {
        let spec = DraftSpec {
            id: DraftId("bad".into()),
            model_dir: PathBuf::from("/nope"),
            arch_hint: None,
            kv_blocks: 4,
            weight_size_estimate_bytes: 0,
            ref_count: 0,
        };
        let self_spec: Arc<Mutex<Box<dyn ModelBackend>>> =
            Arc::new(Mutex::new(Box::new(StubBackend)));
        let (resolver, _) = make_resolver(
            vec![spec],
            vec![],
            Some(self_spec),
            vec![DraftId("bad".into())], // loader fails
        );
        match resolver.resolve(Some(&DraftId("bad".into()))) {
            ResolvedDraft::SelfSpec(_) => {}
            other => panic!("expected SelfSpec after load failure, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_unknown_id_with_no_loader_falls_back() {
        let self_spec: Arc<Mutex<Box<dyn ModelBackend>>> =
            Arc::new(Mutex::new(Box::new(StubBackend)));
        let (resolver, _) = make_resolver(vec![], vec![], Some(self_spec), vec![]);
        // "ghost" was never registered
        match resolver.resolve(Some(&DraftId("ghost".into()))) {
            ResolvedDraft::SelfSpec(_) => {}
            other => panic!("expected SelfSpec for unknown id, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_records_metrics() {
        let self_spec: Arc<Mutex<Box<dyn ModelBackend>>> =
            Arc::new(Mutex::new(Box::new(StubBackend)));
        let (resolver, metrics) = make_resolver(vec![], vec![], Some(self_spec), vec![]);
        resolver.resolve(None);
        resolver.resolve(Some(&DraftId("nope".into())));
        let snap = metrics.draft_metrics_snapshot();
        assert!(
            snap.resolutions_self_spec_total > 0
                || snap.resolutions_external_total > 0
                || snap.resolutions_none_total > 0
        );
    }
}
