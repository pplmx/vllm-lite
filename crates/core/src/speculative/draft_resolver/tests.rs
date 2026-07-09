//! Unit tests for the `DraftResolver` lazy-loading fallback path:
//!
//! - `ResolvedDraft::SelfSpec` when self-spec is available
//! - `ResolvedDraft::None` when no self-spec and no draft requested
//! - `ResolvedDraft::External` for a pre-loaded draft id
//! - `ResolvedDraft::External` after lazy loader + attach
//! - Fallback to `SelfSpec` on loader failure
//! - Fallback to `SelfSpec` for unknown draft id
//! - Resolution metrics counters (`self_spec` / `external` / `none`)
//! - `DraftLoader::default_arc()` returns a `NoopLoader` that always errors

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
            return Err(DraftRegistryError::Model(
                id.clone(),
                ModelError::new("stub failure"),
            ));
        }
        Ok(Box::new(StubBackend))
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
    loaded: &[DraftId],
    self_spec: Option<Arc<Mutex<Box<dyn ModelBackend>>>>,
    loader_failures: Vec<DraftId>,
) -> (Arc<DraftResolver>, Arc<EnhancedMetricsCollector>) {
    let registry = Arc::new(DraftModelRegistry::new());
    for spec in specs {
        registry.register(spec).unwrap();
    }
    for id in loaded {
        registry.attach_loaded(id, Box::new(StubBackend)).unwrap();
    }
    let stub_loader: Arc<dyn DraftLoader> = Arc::new(StubLoader {
        fail_on: loader_failures,
    });
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    let resolver = Arc::new(DraftResolver::new(
        registry,
        self_spec,
        stub_loader,
        metrics.clone(),
    ));
    (resolver, metrics)
}

#[test]
fn test_resolve_none_returns_self_spec_when_available() {
    let self_spec: Arc<Mutex<Box<dyn ModelBackend>>> =
        Arc::new(Mutex::new(Box::new(StubBackend)));
    let (resolver, _) = make_resolver(vec![], &[], Some(self_spec.clone()), vec![]);
    match resolver.resolve(None) {
        ResolvedDraft::SelfSpec(_) => {}
        other => panic!("expected SelfSpec, got {other:?}"),
    }
}

#[test]
fn test_resolve_none_returns_none_when_no_self_spec() {
    let (resolver, _) = make_resolver(vec![], &[], None, vec![]);
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
    let (resolver, _) =
        make_resolver(vec![spec], &[DraftId("a".into())], Some(self_spec), vec![]);
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
        &[], // not pre-loaded
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
        &[],
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
    let (resolver, _) = make_resolver(vec![], &[], Some(self_spec), vec![]);
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
    let (resolver, metrics) = make_resolver(vec![], &[], Some(self_spec), vec![]);
    resolver.resolve(None);
    resolver.resolve(Some(&DraftId("nope".into())));
    let snap = metrics.draft_metrics_snapshot();
    assert!(
        snap.resolutions_self_spec_total > 0
            || snap.resolutions_external_total > 0
            || snap.resolutions_none_total > 0
    );
}

#[test]
fn draft_loader_default_arc_is_noop() {
    let loader: Arc<dyn DraftLoader> = <dyn DraftLoader>::default_arc();
    let id = DraftId("any".into());
    let result = loader.load(&id);
    assert!(result.is_err(), "NoopLoader must always error");
}
