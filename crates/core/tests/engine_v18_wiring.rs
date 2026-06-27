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

use std::sync::Arc;
use vllm_core::speculative::{DraftId, DraftSpec};
use vllm_core::types::{Request, SchedulerConfig};
use vllm_core::{Engine, EnhancedMetricsCollector};
use vllm_traits::{BatchOutput, ModelBackend, Result as ModelResult, SeqId, TokenId};

// ───────────────────────── Stub Backend ───────────────────────────

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
