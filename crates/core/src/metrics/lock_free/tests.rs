//! Unit tests for the metrics subsystem (`MetricsCollector` +
//! `LockFreeMetrics`).
//!
//! Two concerns are exercised:
//!
//! 1. **`MetricsCollector` snapshot accuracy (3 tests)**: new fields
//!    on the snapshot populate correctly under a hand-rolled
//!    record-session (request_start → kv_cache → prefix_cache →
//!    prefill/decode tokens → wait_time → end), `kv_cache_usage`
//!    with `total=0` returns `0.0` (no division-by-zero), and
//!    `prefix_cache_hit_rate` is `0.0` when no requests have been
//!    recorded.
//! 2. **`LockFreeMetrics` ring-buffer behavior (3 tests)**: single
//!    record populates the avg, 100-record burst hits the expected
//!    avg + p50, and ring-buffer overflow (capacity=10, records=100)
//!    does not panic — it just wraps.
//!
//! Both collectors are exercised on the same thread to avoid
//! scheduling noise; concurrency invariants for `LockFreeMetrics`
//! are covered by property tests in other modules.
use super::*;

#[test]
fn test_metrics_snapshot_new_fields() {
    let collector = MetricsCollector::new();

    collector.record_request_start();
    collector.record_kv_cache_usage(50, 100);
    collector.record_prefix_cache_hit();
    collector.record_prefix_cache_request();
    collector.record_prefill_tokens(100);
    collector.record_decode_tokens(50);
    collector.record_scheduler_wait_time(10.0);
    collector.record_request_end();

    let snapshot = collector.snapshot();

    assert_eq!(snapshot.requests_in_flight, 0);
    assert!((snapshot.kv_cache_usage_percent - 50.0).abs() < 0.01);
    assert!((snapshot.prefix_cache_hit_rate - 100.0).abs() < 0.01);
}

#[test]
fn test_metrics_kv_cache_zero_total() {
    let collector = MetricsCollector::new();
    collector.record_kv_cache_usage(10, 0);

    let snapshot = collector.snapshot();
    assert!(snapshot.kv_cache_usage_percent.abs() < 1e-6);
}

#[test]
fn test_metrics_prefix_cache_no_requests() {
    let collector = MetricsCollector::new();

    let snapshot = collector.snapshot();
    assert!(snapshot.prefix_cache_hit_rate.abs() < 1e-6);
}

#[test]
fn test_lock_free_metrics_single_record() {
    let collector = LockFreeMetrics::with_capacity(1024);
    collector.record_latency(10.5);

    let snapshot = collector.snapshot();
    assert!((snapshot.avg_latency_ms - 10.5).abs() < 0.01);
}

#[test]
fn test_lock_free_metrics_burst_records() {
    let collector = LockFreeMetrics::with_capacity(1024);

    for i in 1..=100 {
        collector.record_latency(f64::from(i));
    }

    let snapshot = collector.snapshot();
    assert!((snapshot.avg_latency_ms - 50.5).abs() < 0.01);
    assert!((snapshot.p50_latency_ms - 50.0).abs() < 1.0);
}

#[test]
fn test_lock_free_metrics_buffer_overflow() {
    let collector = LockFreeMetrics::with_capacity(10);

    for i in 0..100 {
        collector.record_latency(f64::from(i));
    }

    let snapshot = collector.snapshot();
    assert!(snapshot.avg_latency_ms > 0.0);
}
