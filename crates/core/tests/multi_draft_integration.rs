//! Multi-Model Speculative Decoding (v18.0) — integration test harness
//!
//! Composes `DraftModelRegistry` + `DraftResolver` + `MemoryBudget` into a
//! single testable surface. Uses stub backends that record calls and support
//! deterministic outputs and configurable failure modes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::speculative::{
    DraftId, DraftLoader, DraftModelRegistry, DraftRegistryError, DraftResolver, MemoryBudget,
    ResolvedDraft,
};
use vllm_traits::{BatchOutput, ModelBackend, ModelError, Result as ModelResult, SeqId, TokenId};

// ─────────────────────────── Stub Backend ────────────────────────────

/// Configurable stub backend. Tracks `forward()` call count + last input,
#[derive(Debug)]
/// supports per-instance "fail next N calls" for runtime-error testing.
pub struct StubBackend {
    pub id: String,
    pub fail_next_n: AtomicU32,
    pub call_count: AtomicU64,
}

impl StubBackend {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fail_next_n: AtomicU32::new(0),
            call_count: AtomicU64::new(0),
        }
    }

    pub fn calls(&self) -> u64 {
        self.call_count.load(Ordering::Relaxed)
    }

    pub fn fail_next(&self, n: u32) {
        self.fail_next_n.store(n, Ordering::Relaxed);
    }
}

impl ModelBackend for StubBackend {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<BatchOutput> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        let n = self.fail_next_n.load(Ordering::Relaxed);
        if n > 0 {
            self.fail_next_n.fetch_sub(1, Ordering::Relaxed);
            return Err(ModelError::new(format!(
                "simulated draft runtime error from {}",
                self.id
            )));
        }
        // Deterministic output: one token per seq, value = self.id hash.
        let token: u32 = self.id.bytes().map(u32::from).sum();
        let next_tokens = seq_ids.iter().map(|_| token).collect();
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }

    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<Vec<Vec<f32>>> {
        Ok(vec![])
    }

    fn embed(
        &mut self,
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> ModelResult<Vec<Vec<f32>>> {
        Ok(vec![])
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

// ─────────────────────────── Map Loader ──────────────────────────────

/// Stub loader that returns the requested backend from a map, or
/// `LoadFailed` if the id isn't in the map.
pub struct MapLoader {
    pub backends: Mutex<HashMap<DraftId, Box<dyn ModelBackend>>>,
    pub load_count: AtomicU64,
}

impl std::fmt::Debug for MapLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.backends.lock().map_or(0, |m| m.len());
        f.debug_struct("MapLoader")
            .field("backends_count", &count)
            .field("load_count", &self.load_count)
            .finish()
    }
}

impl MapLoader {
    #[must_use]
    pub fn new() -> Self {
        Self {
            backends: Mutex::new(HashMap::new()),
            load_count: AtomicU64::new(0),
        }
    }

    /// Runs the operation.
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    pub fn insert(&self, id: DraftId, backend: Box<dyn ModelBackend>) {
        self.backends.lock().unwrap().insert(id, backend);
    }
}

impl Default for MapLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DraftLoader for MapLoader {
    fn load(&self, id: &DraftId) -> std::result::Result<Box<dyn ModelBackend>, DraftRegistryError> {
        self.load_count.fetch_add(1, Ordering::Relaxed);
        self.backends
            .lock()
            .unwrap()
            .remove(id)
            .ok_or_else(|| DraftRegistryError::LoadFailed(format!("no stub for {id}")))
    }
}

// ─────────────────────────── Harness ─────────────────────────────────

/// Bundle for tests: all the pieces wired together.
pub struct Harness {
    pub registry: Arc<DraftModelRegistry>,
    pub budget: Arc<MemoryBudget>,
    pub resolver: Arc<DraftResolver>,
    pub metrics: Arc<EnhancedMetricsCollector>,
    /// The loader used by the resolver. Tests must use this one to populate
    /// backends, not a separate `MapLoader` instance.
    pub loader: Arc<MapLoader>,
    pub self_spec: Arc<Mutex<Box<dyn ModelBackend>>>,
}

impl std::fmt::Debug for Harness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Harness")
            .field("registry", &self.registry)
            .field("budget", &self.budget)
            .field("resolver", &self.resolver)
            .field("metrics", &self.metrics)
            .field("loader", &self.loader)
            .field("self_spec", &"<dyn ModelBackend>")
            .finish()
    }
}

#[must_use]
pub fn harness_unlimited() -> Harness {
    harness_with_budget(u64::MAX)
}

#[must_use]
/// Runs the operation.
/// # Panics
///
/// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
pub fn harness_with_budget(total_bytes: u64) -> Harness {
    let budget = Arc::new(MemoryBudget::new(total_bytes).unwrap());
    let registry = Arc::new(DraftModelRegistry::with_budget(budget.clone()));
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    let loader = Arc::new(MapLoader::new());
    let loader_dyn: Arc<dyn DraftLoader> = loader.clone();
    let self_spec: Arc<Mutex<Box<dyn ModelBackend>>> =
        Arc::new(Mutex::new(Box::new(StubBackend::new("self-spec"))));
    let resolver = Arc::new(DraftResolver::new(
        registry.clone(),
        Some(self_spec.clone()),
        loader_dyn,
        metrics.clone(),
    ));
    Harness {
        registry,
        budget,
        resolver,
        metrics,
        loader,
        self_spec,
    }
}

// ─────────────────────────── Tests ───────────────────────────────────

#[test]
fn test_lifecycle_register_lazy_load_route_unload() {
    let h = harness_unlimited();
    let spec = vllm_core::speculative::DraftSpec::new("a", "/nope", 4).with_weight_size(1024);
    h.registry.register(spec).unwrap();
    assert_eq!(h.loader.load_count.load(Ordering::Relaxed), 0);
    // Loaded via resolver (lazy)
    h.loader
        .insert(DraftId("a".into()), Box::new(StubBackend::new("a")));
    let resolved = h.resolver.resolve(Some(&DraftId("a".into())));
    assert!(matches!(resolved, ResolvedDraft::External(_)));
    assert_eq!(h.loader.load_count.load(Ordering::Relaxed), 1);
    // Now unloaded
    h.registry.force_unload(&DraftId("a".into())).unwrap();
    assert!(!h.registry.is_loaded(&DraftId("a".into())));
}

#[test]
fn test_budget_accept_just_fits() {
    let h = harness_with_budget(100_000_000_000);
    let spec = vllm_core::speculative::DraftSpec::new("a", "/nope", 4).with_weight_size(1000);
    h.registry.register(spec.clone()).unwrap();
    h.loader
        .insert(DraftId("a".into()), Box::new(StubBackend::new("a")));
    let _ = h.resolver.resolve(Some(&DraftId("a".into())));
    let reserved = h.budget.snapshot().reserved_drafts_bytes;
    assert_eq!(reserved, spec.estimated_total_bytes());
}

#[test]
fn test_budget_refuse_just_over() {
    let h = harness_with_budget(100); // very tiny
    let spec = vllm_core::speculative::DraftSpec::new("huge", "/nope", 4).with_weight_size(1000);
    h.registry.register(spec).unwrap();
    h.loader
        .insert(DraftId("huge".into()), Box::new(StubBackend::new("huge")));
    let resolved = h.resolver.resolve(Some(&DraftId("huge".into())));
    // Budget too small → fallback to SelfSpec
    assert!(matches!(resolved, ResolvedDraft::SelfSpec(_)));
    // And the budget didn't grow
    assert_eq!(h.budget.snapshot().reserved_drafts_bytes, 0);
}

#[test]
fn test_budget_accepts_n_drafts_within_budget() {
    // 5 GB budget; each draft is ~16 MiB → all fit comfortably.
    let h = harness_with_budget(5_000_000_000);
    for name in ["a", "b", "c", "d", "e"] {
        let spec = vllm_core::speculative::DraftSpec::new(name, "/nope", 1);
        h.registry.register(spec).unwrap();
        h.loader
            .insert(DraftId(name.into()), Box::new(StubBackend::new(name)));
        let _ = h.resolver.resolve(Some(&DraftId(name.into())));
    }
    assert_eq!(
        h.budget.snapshot().reserved_drafts_bytes,
        5 * 16 * 1024 * 1024
    );
    let snap = h.metrics.draft_metrics_snapshot();
    assert_eq!(snap.resolutions_external_total, 5);
    assert_eq!(snap.resolutions_self_spec_total, 0);
}

#[test]
fn test_budget_refuses_n_plus_one_drafts_over_budget() {
    // 32 MiB budget; each draft's kv_blocks=2 → 2 * 16MiB = 32 MiB footprint.
    // 1 fits, 2 doesn't.
    let h = harness_with_budget(32 * 1024 * 1024);
    let spec_a = vllm_core::speculative::DraftSpec::new("a", "/nope", 2).with_weight_size(0);
    let spec_b = vllm_core::speculative::DraftSpec::new("b", "/nope", 2).with_weight_size(0);
    h.registry.register(spec_a).unwrap();
    h.registry.register(spec_b).unwrap();
    h.loader
        .insert(DraftId("a".into()), Box::new(StubBackend::new("a")));
    h.loader
        .insert(DraftId("b".into()), Box::new(StubBackend::new("b")));
    let _ = h.resolver.resolve(Some(&DraftId("a".into()))); // fits
    let r = h.resolver.resolve(Some(&DraftId("b".into()))); // over budget → fallback
    assert!(matches!(r, ResolvedDraft::SelfSpec(_)));
    assert!(!h.registry.is_loaded(&DraftId("b".into())));
    let snap = h.metrics.draft_metrics_snapshot();
    assert_eq!(snap.resolutions_external_total, 1);
    assert_eq!(snap.resolutions_self_spec_total, 1);
    assert!(snap.load_failures_total >= 1);
}

#[test]
fn test_mixed_draft_routing_two_different_drafts_one_batch() {
    let h = harness_unlimited();
    for name in ["a", "b"] {
        let spec = vllm_core::speculative::DraftSpec::new(name, "/nope", 4);
        h.registry.register(spec).unwrap();
        h.loader
            .insert(DraftId(name.into()), Box::new(StubBackend::new(name)));
    }
    let r_a = h.resolver.resolve(Some(&DraftId("a".into())));
    let r_b = h.resolver.resolve(Some(&DraftId("b".into())));
    // Both resolved, distinct backends
    assert!(matches!(r_a, ResolvedDraft::External(_)));
    assert!(matches!(r_b, ResolvedDraft::External(_)));
    let snap = h.metrics.draft_metrics_snapshot();
    assert_eq!(snap.resolutions_external_total, 2);
}

#[test]
fn test_fallback_unknown_id_returns_self_spec() {
    let h = harness_unlimited();
    // No draft registered with this id
    let r = h.resolver.resolve(Some(&DraftId("ghost".into())));
    assert!(matches!(r, ResolvedDraft::SelfSpec(_)));
    let snap = h.metrics.draft_metrics_snapshot();
    assert!(snap.load_failures_total >= 1);
}

#[test]
fn test_fallback_loader_error_returns_self_spec() {
    let h = harness_unlimited();
    let spec = vllm_core::speculative::DraftSpec::new("bad", "/nope", 4);
    h.registry.register(spec).unwrap();
    // Don't insert the backend in the loader → load returns error
    let r = h.resolver.resolve(Some(&DraftId("bad".into())));
    assert!(matches!(r, ResolvedDraft::SelfSpec(_)));
}

#[test]
fn test_stub_backend_fail_next_n() {
    let h = harness_unlimited();
    let spec = vllm_core::speculative::DraftSpec::new("flaky", "/nope", 4);
    h.registry.register(spec).unwrap();
    // Insert a backend that fails its first call
    let backend = StubBackend::new("flaky");
    backend.fail_next(1);
    h.loader.insert(DraftId("flaky".into()), Box::new(backend));
    let r = h.resolver.resolve(Some(&DraftId("flaky".into())));
    let ResolvedDraft::External(backend) = r else {
        panic!("expected External after resolve");
    };
    // First forward() should error → engine marks degraded_draft=true
    // (simulated here; in real engine, the step loop catches and sets the flag)
    let mut guard = backend.lock().unwrap();
    let result = guard.forward(&[1], &[vec![10]], &[vec![0]], &[vec![0]], &[0], &[false]);
    assert!(result.is_err(), "first call should fail");
    // Subsequent calls succeed
    let result = guard.forward(&[1], &[vec![10]], &[vec![0]], &[vec![0]], &[0], &[false]);
    assert!(result.is_ok(), "subsequent calls should succeed");
    drop(guard);
}

#[test]
fn test_unload_with_nonzero_refcount_returns_inuse() {
    let h = harness_unlimited();
    let spec = vllm_core::speculative::DraftSpec::new("a", "/nope", 4);
    h.registry.register(spec).unwrap();
    h.loader
        .insert(DraftId("a".into()), Box::new(StubBackend::new("a")));
    let _ = h.resolver.resolve(Some(&DraftId("a".into())));
    h.registry.increment_ref(&DraftId("a".into())).unwrap();
    let err = h.registry.unload(&DraftId("a".into())).unwrap_err();
    assert!(matches!(err, DraftRegistryError::InUse(1)));
    assert!(h.registry.is_loaded(&DraftId("a".into())));
}

#[test]
fn test_force_unload_overrides_refcount() {
    let h = harness_unlimited();
    let spec = vllm_core::speculative::DraftSpec::new("a", "/nope", 4);
    h.registry.register(spec).unwrap();
    h.loader
        .insert(DraftId("a".into()), Box::new(StubBackend::new("a")));
    let _ = h.resolver.resolve(Some(&DraftId("a".into())));
    h.registry.increment_ref(&DraftId("a".into())).unwrap();
    h.registry.increment_ref(&DraftId("a".into())).unwrap();
    h.registry.force_unload(&DraftId("a".into())).unwrap();
    assert!(!h.registry.is_loaded(&DraftId("a".into())));
}

#[test]
fn test_decrement_ref_auto_unloads_at_zero() {
    let h = harness_unlimited();
    let spec = vllm_core::speculative::DraftSpec::new("a", "/nope", 4);
    h.registry.register(spec).unwrap();
    h.loader
        .insert(DraftId("a".into()), Box::new(StubBackend::new("a")));
    let _ = h.resolver.resolve(Some(&DraftId("a".into())));
    h.registry.increment_ref(&DraftId("a".into())).unwrap();
    let auto_unloaded = h.registry.decrement_ref(&DraftId("a".into())).unwrap();
    assert!(auto_unloaded);
    assert!(!h.registry.is_loaded(&DraftId("a".into())));
}

#[test]
fn test_get_loaded_backend_returns_arc_for_loaded_drafts() {
    let h = harness_unlimited();
    let spec = vllm_core::speculative::DraftSpec::new("a", "/nope", 4);
    h.registry.register(spec).unwrap();
    h.loader
        .insert(DraftId("a".into()), Box::new(StubBackend::new("a")));
    let _ = h.resolver.resolve(Some(&DraftId("a".into())));
    let arc1 = h
        .registry
        .get_loaded_backend(&DraftId("a".into()))
        .expect("should be loaded");
    let arc2 = h
        .registry
        .get_loaded_backend(&DraftId("a".into()))
        .expect("should still be loaded");
    assert!(
        Arc::ptr_eq(&arc1, &arc2),
        "get_loaded_backend returns same Arc"
    );
}

#[test]
fn test_metrics_snapshot_reflects_all_counters() {
    let h = harness_unlimited();
    // Trigger various resolutions
    h.resolver.resolve(None);
    h.resolver.resolve(Some(&DraftId("ghost".into())));
    let snap = h.metrics.draft_metrics_snapshot();
    assert!(snap.resolutions_none_total >= 1);
    assert!(snap.load_failures_total >= 1);
    // Verify metric accessors return non-zero for the right counters
    h.metrics.inc_draft_load_failure();
    h.metrics.inc_draft_runtime_error();
    let snap2 = h.metrics.draft_metrics_snapshot();
    assert_eq!(snap2.load_failures_total, snap.load_failures_total + 1);
    assert_eq!(snap2.runtime_errors_total, 1);
}
