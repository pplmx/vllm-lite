//! Unit tests for `EnhancedMetricsCollector` record / snapshot /
//! `DraftResolutionKind` parsing.
//!
//! Extracted from `sampler.rs` to keep the implementation file under
//! the project's 800-line soft cap. Exercises:
//!
//! - Counter / gauge recorders (`cuda_graph_hit`, `packing_efficiency`,
//!   `speculative_acceptance`, `inference_latency`,
//!   `throughput_speedup`)
//! - Draft-resolution metric counters
//!   (`DraftResolutionKind::{External, SelfSpec, None}`)
//! - `DraftResolutionKind::parse` round-trip + aliases + invalid input
//! - `DraftResolutionKind::Display` formatting

use super::*;

#[test]
fn test_collector_records_cuda_graph_hit() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_cuda_graph_hit();
    let hits = collector.get_counter("cuda_graph_hits_total");
    assert_eq!(hits, 1);
}

/// RIL ISS-074 / TASK-089: the engine must make token-channel drops
/// observable. The token-send loop `try_send`s each sampled token into the
/// bounded response channel and previously ignored the result — on
/// `TrySendError::Full` the token vanished with no log, no counter, and no
/// trace (a permanent gap in the client's stream). The counter lets
/// operators detect silent loss on `/metrics`.
#[test]
fn test_collector_records_dropped_token() {
    let collector = EnhancedMetricsCollector::new();
    assert_eq!(collector.get_counter("dropped_tokens_total"), 0);
    collector.record_dropped_token();
    collector.record_dropped_token();
    assert_eq!(collector.get_counter("dropped_tokens_total"), 2);
}

/// RIL ISS-090: `errors_total` in /debug/metrics was backed by the
/// collector's `_ => 0` fallback — pinned at 0 no matter how many real engine
/// step errors occurred. The engine's `error_count` is now surfaced through
/// `record_engine_error` at the single increment site (run.rs).
#[test]
fn test_collector_records_engine_error() {
    let collector = EnhancedMetricsCollector::new();
    assert_eq!(collector.get_counter("errors_total"), 0);
    collector.record_engine_error();
    collector.record_engine_error();
    assert_eq!(collector.get_counter("errors_total"), 2);
}

#[test]
fn test_collector_records_packing_efficiency() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_packing_efficiency(0.85);
    let efficiency = collector.get_gauge("packing_efficiency");
    assert_eq!(efficiency, 85000);
}

#[test]
fn test_collector_records_speculative_acceptance() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_speculative_acceptance(8, 10);
    let rate = collector.get_gauge("speculative_acceptance_rate");
    assert_eq!(rate, 80000);
}

// ---- Plan 17.4-H: Metrics Tests ----

/// RIL ISS-108: `speculative_acceptance_rate` and `speculative_efficiency`
/// were duplicate gauges — the producer fed BOTH from the identical
/// accepted/drafted ratio (dispatch.rs), so /metrics emitted two
/// differently-named gauges with the same value and the same HELP text.
/// The duplicate is removed from the collector: the acceptance-rate gauge
/// is the single wire name for accepted/drafted.
#[test]
fn test_speculative_gauge_is_deduplicated() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_speculative_acceptance(8, 10);
    // The surviving gauge reports the accepted/drafted ratio (× 100_000).
    assert_eq!(collector.get_gauge("speculative_acceptance_rate"), 80000);
}

#[test]
fn test_throughput_speedup_set_get() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_throughput_speedup(1.5);
    let gauge = collector.get_gauge("throughput_speedup_ratio");
    assert_eq!(gauge, 150_000);
}

#[test]
fn test_throughput_speedup_default() {
    let collector = EnhancedMetricsCollector::new();
    let gauge = collector.get_gauge("throughput_speedup_ratio");
    assert_eq!(gauge, 0);
}

#[test]
fn test_collector_records_draft_resolution_metrics() {
    let collector = EnhancedMetricsCollector::new();
    collector.inc_draft_resolution(DraftResolutionKind::External);
    collector.inc_draft_resolution(DraftResolutionKind::External);
    collector.inc_draft_resolution(DraftResolutionKind::SelfSpec);
    collector.inc_draft_resolution(DraftResolutionKind::None);
    let snap = collector.draft_metrics_snapshot();
    assert_eq!(snap.resolutions_external_total, 2);
    assert_eq!(snap.resolutions_self_spec_total, 1);
    assert_eq!(snap.resolutions_none_total, 1);
}

#[test]
fn draft_resolution_kind_parse_roundtrip() {
    for kind in [
        DraftResolutionKind::External,
        DraftResolutionKind::SelfSpec,
        DraftResolutionKind::None,
    ] {
        assert_eq!(DraftResolutionKind::parse(kind.as_str()), Some(kind));
    }
}

#[test]
fn draft_resolution_kind_parse_aliases() {
    assert_eq!(
        DraftResolutionKind::parse("self-spec"),
        Some(DraftResolutionKind::SelfSpec)
    );
    assert_eq!(
        DraftResolutionKind::parse("selfspec"),
        Some(DraftResolutionKind::SelfSpec)
    );
    assert_eq!(
        DraftResolutionKind::parse("SELF_SPEC"),
        Some(DraftResolutionKind::SelfSpec)
    );
    assert_eq!(
        DraftResolutionKind::parse("External"),
        Some(DraftResolutionKind::External)
    );
    assert_eq!(
        DraftResolutionKind::parse("NONE"),
        Some(DraftResolutionKind::None)
    );
}

#[test]
fn draft_resolution_kind_parse_invalid() {
    assert_eq!(DraftResolutionKind::parse("invalid"), None);
    assert_eq!(DraftResolutionKind::parse(""), None);
    assert_eq!(DraftResolutionKind::parse("shared"), None);
    assert_eq!(DraftResolutionKind::parse("per_request"), None);
}

#[test]
fn draft_resolution_kind_display() {
    assert_eq!(DraftResolutionKind::External.to_string(), "external");
    assert_eq!(DraftResolutionKind::SelfSpec.to_string(), "self_spec");
    assert_eq!(DraftResolutionKind::None.to_string(), "none");
}

#[test]
fn test_collector_records_draft_failures() {
    let collector = EnhancedMetricsCollector::new();
    collector.inc_draft_load_failure();
    collector.inc_draft_load_failure();
    collector.inc_draft_runtime_error();
    let snap = collector.draft_metrics_snapshot();
    assert_eq!(snap.load_failures_total, 2);
    assert_eq!(snap.runtime_errors_total, 1);
}

/// RIL ISS-095: `record_batch_phase_tokens` must split a scheduler
/// `Batch` into the prefill (input positions processed) vs decode
/// (1 token per decode seq) counters that feed `prefill_throughput` /
/// `decode_throughput`. A pure-decode batch must move only the decode
/// gauge; a pure-prefill batch only the prefill one.
#[test]
fn test_record_batch_phase_tokens_splits_prefill_and_decode() {
    use vllm_traits::Batch;

    // Pure-decode batch: two decode sequences, one emitted token each.
    let mut decode_batch = Batch::empty();
    decode_batch.seq_ids = vec![1, 2];
    decode_batch.is_prefill = vec![false, false];
    let decode_collector = EnhancedMetricsCollector::new();
    decode_collector.record_batch_phase_tokens(&decode_batch);
    let decode_snap = decode_collector.snapshot();
    assert!(
        decode_snap.decode_throughput > 0.0,
        "decode-only batch must move decode_throughput; got {}",
        decode_snap.decode_throughput
    );
    assert!(
        decode_snap.prefill_throughput.abs() < f64::EPSILON,
        "decode-only batch must NOT move prefill_throughput; got {}",
        decode_snap.prefill_throughput
    );

    // Pure-prefill batch: one 3-token prompt processed in the prefill phase.
    let mut prefill_batch = Batch::empty();
    prefill_batch.seq_ids = vec![3];
    prefill_batch.input_tokens = vec![vec![1, 2, 3]];
    prefill_batch.is_prefill = vec![true];
    let prefill_collector = EnhancedMetricsCollector::new();
    prefill_collector.record_batch_phase_tokens(&prefill_batch);
    let prefill_snap = prefill_collector.snapshot();
    assert!(
        prefill_snap.prefill_throughput > 0.0,
        "prefill-only batch (3 input positions) must move prefill_throughput; got {}",
        prefill_snap.prefill_throughput
    );
    assert!(
        prefill_snap.decode_throughput.abs() < f64::EPSILON,
        "prefill-only batch must NOT move decode_throughput; got {}",
        prefill_snap.decode_throughput
    );

    // Mixed batch: 2 prefill (3 + 1 input positions) + 1 decode seq.
    // Both gauges must move.
    let mut mixed = Batch::empty();
    mixed.seq_ids = vec![4, 5, 6];
    mixed.input_tokens = vec![vec![7, 8, 9], vec![10], vec![11]];
    mixed.is_prefill = vec![true, true, false];
    let mixed_collector = EnhancedMetricsCollector::new();
    mixed_collector.record_batch_phase_tokens(&mixed);
    let mixed_snap = mixed_collector.snapshot();
    assert!(mixed_snap.prefill_throughput > 0.0);
    assert!(mixed_snap.decode_throughput > 0.0);
}
