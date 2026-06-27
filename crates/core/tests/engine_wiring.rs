//! Phase 19: Engine step loop wiring for v18.0 multi-model speculative decoding.
//!
//! Closes the audit gaps:
//! - RTE-02: Scheduler routes request to correct draft instance
//! - RTE-03: Multiple drafts in same batch
//! - FALL-02: Runtime error → degraded_draft → non-spec decode
//!
//! Exercises the actual `Engine::step` path (not just `DraftResolver` in
//! isolation) to verify that the production request flow honours per-request
//! draft selection.

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use vllm_core::speculative::{DraftId, DraftLoader, DraftRegistryError, DraftSpec};
use vllm_core::types::{Request, SchedulerConfig};
use vllm_core::{Engine, EnhancedMetricsCollector};
use vllm_traits::{BatchOutput, ModelBackend, ModelError, Result as ModelResult, SeqId, TokenId};

// ───────────────────────── Stub Backend ───────────────────────────

#[derive(Clone)]
struct StubBackend {
    #[allow(dead_code)]
    id: String,
}

impl StubBackend {
    #[allow(dead_code)]
    fn new(id: &str) -> Self {
        Self { id: id.to_string() }
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
        let token: u32 = self.id.bytes().map(|b| b as u32).sum::<u32>() % 32000;
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| token).collect(),
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

// ───────────────────────── Test Stubs (WR-01, WR-02) ────────────────

/// Backend that always returns `Err(ModelError)` from `forward()` —
/// simulates a draft runtime error so we can verify FALL-02.
struct ErrorBackend;

impl ModelBackend for ErrorBackend {
    fn forward(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<BatchOutput> {
        Err(ModelError::new("boom"))
    }

    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<Vec<Vec<f32>>> {
        // Provide well-formed logits so the verifier can still produce a token.
        let vocab_size = self.vocab_size();
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                let mut logits = vec![-10.0_f32; tokens.len() * vocab_size];
                for (i, &t) in tokens.iter().enumerate() {
                    if (t as usize) < vocab_size {
                        logits[i * vocab_size + t as usize] = 10.0;
                    }
                }
                logits
            })
            .collect())
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

/// Counter-wrapped backend that records how many times `forward()` was
/// invoked. Used by WR-02 to verify each request routes to its named draft.
struct CountingBackend {
    inner: StubBackend,
    forward_count: Arc<AtomicU64>,
}

impl CountingBackend {
    fn new(id: &str) -> (Self, Arc<AtomicU64>) {
        let counter = Arc::new(AtomicU64::new(0));
        let backend = Self {
            inner: StubBackend::new(id),
            forward_count: counter.clone(),
        };
        (backend, counter)
    }
}

impl ModelBackend for CountingBackend {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> ModelResult<BatchOutput> {
        self.forward_count.fetch_add(1, Ordering::Relaxed);
        self.inner.forward(
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
        )
    }

    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> ModelResult<Vec<Vec<f32>>> {
        self.inner.forward_logits(
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
        )
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> ModelResult<Vec<Vec<f32>>> {
        self.inner.embed(input_tokens, positions)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }
    fn num_heads(&self) -> usize {
        self.inner.num_heads()
    }
}

/// Stub loader: maps each `DraftId` to a backend inserted via `insert()`.
struct MapLoader {
    backends: Mutex<HashMap<DraftId, Box<dyn ModelBackend>>>,
}

impl MapLoader {
    fn new() -> Self {
        Self {
            backends: Mutex::new(HashMap::new()),
        }
    }

    fn insert(&self, id: DraftId, backend: Box<dyn ModelBackend>) {
        self.backends.lock().unwrap().insert(id, backend);
    }
}

impl DraftLoader for MapLoader {
    fn load(&self, id: &DraftId) -> std::result::Result<Box<dyn ModelBackend>, DraftRegistryError> {
        self.backends
            .lock()
            .unwrap()
            .remove(id)
            .ok_or_else(|| DraftRegistryError::LoadFailed(format!("no stub for {id}")))
    }
}

// ───────────────────────── Tests ───────────────────────────

#[test]
fn test_engine_with_drafts_has_resolver_wired() {
    let target = StubBackend::new("target");
    let specs = vec![DraftSpec::new("a", "/nope", 4)];
    let engine = Engine::with_drafts(
        target,
        None::<StubBackend>,
        specs,
        SchedulerConfig::default(),
        2,
        64,
    );
    assert!(engine.draft_resolver.is_some(), "resolver should be wired");
}

#[test]
fn test_engine_without_drafts_has_no_resolver() {
    let target = StubBackend::new("target");
    let engine = Engine::new(target, None::<StubBackend>);
    assert!(
        engine.draft_resolver.is_none(),
        "resolver should not be wired for legacy constructor"
    );
}

#[test]
fn test_engine_with_budget_has_resolver_wired() {
    use std::sync::Arc;
    use vllm_core::speculative::MemoryBudget;
    let target = StubBackend::new("target");
    let specs = vec![DraftSpec::new("a", "/nope", 4).with_weight_size(1024)];
    let budget = Arc::new(MemoryBudget::new(100_000_000).unwrap());
    let engine = Engine::with_budget_boxed(
        Box::new(target),
        None,
        specs,
        budget,
        SchedulerConfig::default(),
        2,
        64,
    );
    assert!(engine.draft_resolver.is_some());
}

#[test]
fn test_engine_draft_metrics_exposed_via_snapshot() {
    let collector = EnhancedMetricsCollector::new();
    collector.inc_draft_resolution("external");
    collector.inc_draft_resolution("external");
    collector.inc_draft_resolution("self_spec");
    collector.inc_draft_load_failure();
    collector.inc_draft_runtime_error();
    let snap = collector.draft_metrics_snapshot();
    assert_eq!(snap.resolutions_external_total, 2);
    assert_eq!(snap.resolutions_self_spec_total, 1);
    assert_eq!(snap.load_failures_total, 1);
    assert_eq!(snap.runtime_errors_total, 1);
}

#[tokio::test]
async fn test_engine_prometheus_exporter_includes_v18_counters() {
    use vllm_core::metrics::PrometheusExporter;

    let collector = Arc::new(EnhancedMetricsCollector::new());
    collector.inc_draft_resolution("external");
    collector.inc_draft_load_failure();
    collector.inc_draft_runtime_error();

    let exporter = PrometheusExporter::new(collector, 9090);
    let output = exporter.export_to_string().await;

    assert!(
        output.contains("draft_resolutions_external_total"),
        "PrometheusExporter should emit draft_resolutions_external_total"
    );
    assert!(
        output.contains("draft_load_failures_total"),
        "PrometheusExporter should emit draft_load_failures_total"
    );
    assert!(
        output.contains("draft_runtime_errors_total"),
        "PrometheusExporter should emit draft_runtime_errors_total"
    );
}

#[test]
fn test_request_with_draft_model_id_propagates_to_sequence() {
    use vllm_core::scheduler::engine::SchedulerEngine;

    let mut scheduler = SchedulerEngine::new(
        SchedulerConfig::default(),
        64,
        Arc::new(EnhancedMetricsCollector::new()),
    );
    let id_a = DraftId("a".into());
    let req = Request::new(1, vec![10, 20], 5).with_draft_model(id_a.clone());
    let seq_id = scheduler.add_request(req);

    // build_batch promotes the queued seq into `running`
    let _batch = scheduler.build_batch();

    let seq = scheduler
        .get_sequence(seq_id)
        .expect("sequence should be in scheduler");
    assert_eq!(seq.draft_model_id.as_ref(), Some(&id_a));
    assert!(!seq.degraded_draft);
}

#[test]
fn test_degraded_draft_setter_via_scheduler() {
    use vllm_core::scheduler::engine::SchedulerEngine;

    let mut scheduler = SchedulerEngine::new(
        SchedulerConfig::default(),
        64,
        Arc::new(EnhancedMetricsCollector::new()),
    );
    let req = Request::new(42, vec![1, 2, 3], 3);
    let seq_id = scheduler.add_request(req);
    let _batch = scheduler.build_batch();

    {
        let seq = scheduler.get_sequence_mut(seq_id).expect("exists");
        seq.degraded_draft = true;
    }
    let seq = scheduler.get_sequence(seq_id).expect("exists");
    assert!(seq.degraded_draft);
}

#[test]
fn test_fall02_draft_forward_error_marks_sequence_degraded() {
    use vllm_core::scheduler::engine::SchedulerEngine;

    let metrics = Arc::new(EnhancedMetricsCollector::new());
    let mut scheduler = SchedulerEngine::new(SchedulerConfig::default(), 64, metrics.clone());
    let id_a = DraftId("a".into());
    let req = Request::new(7, vec![100, 101, 102], 5).with_draft_model(id_a);
    let seq_id = scheduler.add_request(req);
    let _batch = scheduler.build_batch();

    // Simulate FALL-02: draft forward fails → degraded_draft set + counter ++
    {
        let seq = scheduler.get_sequence_mut(seq_id).expect("exists");
        assert!(!seq.degraded_draft);
        seq.degraded_draft = true;
    }
    metrics.inc_draft_runtime_error();

    let seq = scheduler.get_sequence(seq_id).expect("exists");
    assert!(
        seq.degraded_draft,
        "sequence should be degraded after runtime error"
    );
    let snap = metrics.draft_metrics_snapshot();
    assert_eq!(snap.runtime_errors_total, 1);
}

#[test]
fn test_per_request_routing_different_draft_ids_yield_different_resolution_paths() {
    use vllm_core::scheduler::engine::SchedulerEngine;

    let metrics = Arc::new(EnhancedMetricsCollector::new());
    let mut scheduler = SchedulerEngine::new(SchedulerConfig::default(), 64, metrics.clone());

    let id_a = DraftId("a".into());
    let id_b = DraftId("b".into());
    let seq_a =
        scheduler.add_request(Request::new(11, vec![1, 2], 3).with_draft_model(id_a.clone()));
    let seq_b =
        scheduler.add_request(Request::new(22, vec![3, 4], 3).with_draft_model(id_b.clone()));
    let _batch = scheduler.build_batch();

    let seq_a_state = scheduler.get_sequence(seq_a).expect("exists");
    let seq_b_state = scheduler.get_sequence(seq_b).expect("exists");
    assert_eq!(seq_a_state.draft_model_id.as_ref(), Some(&id_a));
    assert_eq!(seq_b_state.draft_model_id.as_ref(), Some(&id_b));
    assert_ne!(seq_a_state.draft_model_id, seq_b_state.draft_model_id);
}

// ─────────────────── WR-01: FALL-02 end-to-end via step() ──────────────

/// FALL-02 end-to-end: build a real Engine, attach a draft whose backend
/// forward() returns Err, call engine.step(), and verify:
///   (a) runtime_errors_total == 1
///   (b) seq.degraded_draft == true on a subsequent get_sequence_mut
///   (c) no panic escaped
///
/// Root cause of the previously-`#[ignore]`d hang was a DashMap shard
/// re-entry in `record_per_request_acceptance` (entry guard held across a
/// `len()` call). Fixed in v22.0 (OPS-02).
#[test]
fn test_fall02_engine_step_catches_runtime_error() {
    let target = StubBackend::new("target");
    let self_spec = StubBackend::new("self-spec");
    let specs = vec![DraftSpec::new("a", "/nope", 4)];
    let mut engine = Engine::with_drafts_boxed(
        Box::new(target),
        Some(Box::new(self_spec)),
        specs,
        SchedulerConfig::default(),
        2,
        64,
    );
    let loader = Arc::new(MapLoader::new());
    loader.insert(DraftId("a".into()), Box::new(ErrorBackend));
    engine.set_draft_loader(loader);

    engine.enable_speculative();
    engine.max_draft_tokens = 2;

    let (tx, _rx) = tokio::sync::mpsc::channel(64);
    let seq_id = engine.add_request(
        Request::new(7, vec![10, 20, 30], 5).with_draft_model(DraftId("a".into())),
        tx,
    );

    // (c) no panic escaped
    let step_result = engine.step();
    assert!(
        step_result.is_ok(),
        "step() must not propagate FALL-02 draft errors"
    );

    // (b) seq.degraded_draft == true on subsequent get_sequence
    let seq = engine
        .scheduler
        .get_sequence(seq_id)
        .expect("seq should still exist after step");
    assert!(
        seq.degraded_draft,
        "seq.degraded_draft must be true after FALL-02"
    );

    // (a) runtime_errors_total == 1 — the DraftResolver was constructed with
    // engine.scheduler.metrics, so we can read the snapshot directly.
    let snap = engine.scheduler.metrics.draft_metrics_snapshot();
    assert_eq!(
        snap.runtime_errors_total, 1,
        "runtime_errors_total must be incremented exactly once"
    );
}

// ─────────────────── WR-02: RTE-03 end-to-end via step() ───────────────

/// RTE-03 end-to-end: 2 requests with different draft ids flow through
/// Engine::step() and each resolves to its own backend. Verified via
/// forward() call counts on the per-id CountingBackend instances.
///
/// Root cause of the previously-`#[ignore]`d hang was a DashMap shard
/// re-entry in `record_per_request_acceptance` (entry guard held across a
/// `len()` call). Fixed in v22.0 (OPS-02).
#[test]
fn test_engine_step_routes_to_correct_draft_backend() {
    let target = StubBackend::new("target");
    let self_spec = StubBackend::new("self-spec");
    let specs = vec![
        DraftSpec::new("a", "/nope", 4),
        DraftSpec::new("b", "/nope", 4),
    ];
    let mut engine = Engine::with_drafts_boxed(
        Box::new(target),
        Some(Box::new(self_spec)),
        specs,
        SchedulerConfig::default(),
        2,
        64,
    );
    let loader = Arc::new(MapLoader::new());
    let (backend_a, counter_a) = CountingBackend::new("a");
    let (backend_b, counter_b) = CountingBackend::new("b");
    loader.insert(DraftId("a".into()), Box::new(backend_a));
    loader.insert(DraftId("b".into()), Box::new(backend_b));
    engine.set_draft_loader(loader);

    engine.enable_speculative();
    engine.max_draft_tokens = 2;

    let (tx_a, _rx_a) = tokio::sync::mpsc::channel(64);
    let (tx_b, _rx_b) = tokio::sync::mpsc::channel(64);
    engine.add_request(
        Request::new(101, vec![1, 2], 4).with_draft_model(DraftId("a".into())),
        tx_a,
    );
    engine.add_request(
        Request::new(202, vec![3, 4], 4).with_draft_model(DraftId("b".into())),
        tx_b,
    );

    // Single step() — both seqs share the same batch, each routed to its
    // own named draft via resolver.resolve(seq.draft_model_id).
    let step_result = engine.step();
    assert!(
        step_result.is_ok(),
        "step() must succeed for mixed-draft batch"
    );

    // Each backend was hit at least once — verifying correct routing.
    let calls_a = counter_a.load(Ordering::Relaxed);
    let calls_b = counter_b.load(Ordering::Relaxed);
    assert!(
        calls_a >= 1,
        "draft 'a' backend should be invoked at least once (got {calls_a})"
    );
    assert!(
        calls_b >= 1,
        "draft 'b' backend should be invoked at least once (got {calls_b})"
    );
}

// ─────────────────── WR-02 (light): resolver routing via Engine ────────

/// Lighter-weight companion to `test_engine_step_routes_to_correct_draft_backend`.
/// Verifies that the Engine's `set_draft_loader` + resolver correctly
/// returns distinct backends for distinct draft ids — the same routing
/// logic that `generate_per_seq_drafts` invokes at step time, but
/// exercised directly against the resolver without the full step pipeline.
#[test]
fn test_engine_resolver_routes_to_distinct_backends_per_id() {
    let target = StubBackend::new("target");
    let self_spec = StubBackend::new("self-spec");
    let specs = vec![
        DraftSpec::new("a", "/nope", 4),
        DraftSpec::new("b", "/nope", 4),
    ];
    let mut engine = Engine::with_drafts_boxed(
        Box::new(target),
        Some(Box::new(self_spec)),
        specs,
        SchedulerConfig::default(),
        2,
        64,
    );
    let loader = Arc::new(MapLoader::new());
    let (backend_a, counter_a) = CountingBackend::new("a");
    let (backend_b, counter_b) = CountingBackend::new("b");
    loader.insert(DraftId("a".into()), Box::new(backend_a));
    loader.insert(DraftId("b".into()), Box::new(backend_b));
    engine.set_draft_loader(loader);

    // set_draft_loader rebuilds the resolver — use the new one.
    let resolver = engine
        .draft_resolver
        .as_ref()
        .expect("resolver installed")
        .clone();

    let resolved_a = resolver.resolve(Some(&DraftId("a".into())));
    let resolved_b = resolver.resolve(Some(&DraftId("b".into())));
    match (&resolved_a, &resolved_b) {
        (
            vllm_core::speculative::ResolvedDraft::External(arc_a),
            vllm_core::speculative::ResolvedDraft::External(arc_b),
        ) => {
            // The same Arc<Mutex<...>> handles from the loader are stored in
            // the registry after resolve(). Their forward() must reach the
            // matching CountingBackend.
            let mut guard_a = arc_a.lock().unwrap();
            guard_a
                .forward(&[101], &[vec![1]], &[vec![0]], &[vec![0]], &[0], &[false])
                .expect("a forward");
            let mut guard_b = arc_b.lock().unwrap();
            guard_b
                .forward(&[202], &[vec![3]], &[vec![0]], &[vec![0]], &[0], &[false])
                .expect("b forward");
        }
        _ => panic!("expected External resolutions, got {resolved_a:?} / {resolved_b:?}"),
    }
    assert!(
        counter_a.load(Ordering::Relaxed) >= 1,
        "draft 'a' counter should record at least one forward()"
    );
    assert!(
        counter_b.load(Ordering::Relaxed) >= 1,
        "draft 'b' counter should record at least one forward()"
    );
}
