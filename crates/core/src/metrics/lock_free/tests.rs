//! Unit tests for the metrics subsystem (`MetricsCollector` +
//! `LockFreeMetrics`).
//!
//! Two concerns are exercised:
//!
//! 1. **`MetricsCollector` snapshot accuracy (3 tests)**: new fields
//!    on the snapshot populate correctly under a hand-rolled
//!    record-session (`request_start` → `kv_cache` → `prefix_cache` →
//!    `prefill` / `decode_tokens` → `wait_time` → `end`), `kv_cache_usage`
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

/// `snapshot()` latency percentiles must use the documented index formula.
///
/// The percentiles are `xs[floor(p * (n - 1))]` over the sorted window. No test
/// compared them against a value, so `cargo mutants` could turn the `- 1` into
/// `+` or `/`, or the index into a modulo, without a failure. Samples `1..=100`
/// make every rank unambiguous.
#[test]
fn test_snapshot_percentiles_match_the_index_formula() {
    let collector = MetricsCollector::new();
    for ms in 1..=100 {
        #[allow(clippy::cast_precision_loss)]
        collector.record_latency(f64::from(ms));
    }

    let snapshot = collector.snapshot();

    // floor(p * 99) -> p50 = 49, p90 = 89, p99 = 98 over 1..=100.
    assert!(
        (snapshot.p50_latency_ms - 50.0).abs() < 1e-6,
        "p50 of 1..=100 must be 50; got {}",
        snapshot.p50_latency_ms
    );
    assert!(
        (snapshot.p90_latency_ms - 90.0).abs() < 1e-6,
        "p90 of 1..=100 must be 90; got {}",
        snapshot.p90_latency_ms
    );
    assert!(
        (snapshot.p99_latency_ms - 99.0).abs() < 1e-6,
        "p99 of 1..=100 must be 99; got {}",
        snapshot.p99_latency_ms
    );
    assert!(
        (snapshot.avg_latency_ms - 50.5).abs() < 1e-6,
        "mean of 1..=100 must be 50.5; got {}",
        snapshot.avg_latency_ms
    );
}

/// Throughput is phase tokens divided by uptime, and must be non-zero once
/// tokens have been recorded.
///
/// `prefill_throughput` / `decode_throughput` divide by `uptime` behind a
/// `uptime > 0.0` guard. `cargo mutants` flipped that guard and the divisions
/// with no failure, because no test ever recorded phase tokens and then read
/// the throughput back.
#[test]
fn test_snapshot_throughput_is_positive_after_recording_phase_tokens() {
    let collector = MetricsCollector::new();
    collector.record_prefill_tokens(1_000);
    collector.record_decode_tokens(500);

    let snapshot = collector.snapshot();

    assert!(
        snapshot.prefill_throughput > 0.0,
        "1000 prefill tokens over a non-zero uptime must yield positive \
         throughput; got {}",
        snapshot.prefill_throughput
    );
    assert!(
        snapshot.decode_throughput > 0.0,
        "500 decode tokens over a non-zero uptime must yield positive \
         throughput; got {}",
        snapshot.decode_throughput
    );
    // Decode recorded half as many tokens, so its rate must be strictly lower.
    assert!(
        snapshot.decode_throughput < snapshot.prefill_throughput,
        "decode (500) must be slower than prefill (1000); got {} vs {}",
        snapshot.decode_throughput,
        snapshot.prefill_throughput
    );

    // A collector with no phase tokens reports exactly zero, not NaN.
    let idle = MetricsCollector::new();
    let snapshot = idle.snapshot();
    assert!(
        snapshot.prefill_throughput.abs() < 1e-9 && snapshot.decode_throughput.abs() < 1e-9,
        "an idle collector must report 0 throughput"
    );
}

#[test]
fn test_metrics_snapshot_new_fields() {
    let collector = MetricsCollector::new();

    collector.record_request_start();
    collector.record_kv_cache_usage(50, 100);
    collector.record_prefix_cache_hit();
    collector.record_prefix_cache_request();
    collector.record_prefix_cache_nodes(7);
    collector.record_prefill_tokens(100);
    collector.record_decode_tokens(50);
    collector.record_scheduler_wait_time(10.0);
    collector.record_request_end();

    let snapshot = collector.snapshot();

    assert_eq!(snapshot.requests_in_flight, 0);
    assert!((snapshot.kv_cache_usage_percent - 50.0).abs() < 0.01);
    assert!((snapshot.prefix_cache_hit_rate - 100.0).abs() < 0.01);
    // raw block counts must ride on the snapshot so debug
    // endpoints can report REAL totals instead of fabricating 1024.
    assert_eq!(snapshot.kv_cache_blocks_used, 50);
    assert_eq!(snapshot.kv_cache_blocks_total, 100);
    // real prefix-cache entry count rides on the snapshot so
    // kv_cache_dump stops fabricating `prefix_cache_nodes: 0`.
    assert_eq!(snapshot.prefix_cache_nodes, 7);
    // the production scheduler-wait recorder's sample must
    // ride through to the snapshot driving `avg_scheduler_wait_time_ms`.
    assert!((snapshot.avg_scheduler_wait_time_ms - 10.0).abs() < 0.01);
}

#[test]
fn test_scheduler_wait_time_average() {
    let collector = MetricsCollector::new();

    collector.record_scheduler_wait_time(4.0);
    collector.record_scheduler_wait_time(10.0);
    collector.record_scheduler_wait_time(16.0);

    let snapshot = collector.snapshot();
    assert!((snapshot.avg_scheduler_wait_time_ms - 10.0).abs() < 0.01);
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

/// `snapshot()` must be **non-destructive**. Pre-fix it
/// drained the bounded latency/batch/wait ring channels with `try_recv`,
/// so a *second* consumer of the same `LockFreeMetrics` (the engine's
/// `GetMetrics` round-trip serving `/health/details`, a concurrent
/// `/metrics` scrape, or a future OTLP exporter) silently stole the
/// samples the first consumer's aggregate was computed from — recording
/// one latency then snapshotting again returned `0.0` on the second read.
#[test]
fn test_snapshot_is_non_destructive() {
    let collector = LockFreeMetrics::with_capacity(1024);
    collector.record_latency(4.0);
    collector.record_latency(10.0);
    collector.record_latency(16.0);
    collector.record_batch_size(2);
    collector.record_scheduler_wait_time(5.0);

    let first = collector.snapshot();
    assert!((first.avg_latency_ms - 10.0).abs() < 0.01);
    assert_eq!(first.current_batch_size, 2);
    assert!((first.avg_scheduler_wait_time_ms - 5.0).abs() < 0.01);

    // A second snapshot (any other consumer) must see the SAME samples —
    // pre-fix this read the already-drained channels and returned zeros.
    let second = collector.snapshot();
    assert!(
        (second.avg_latency_ms - 10.0).abs() < 0.01,
        "second snapshot must preserve latency samples, got {}",
        second.avg_latency_ms
    );
    assert!(
        (second.avg_scheduler_wait_time_ms - 5.0).abs() < 0.01,
        "second snapshot must preserve wait samples, got {}",
        second.avg_scheduler_wait_time_ms
    );
    assert_eq!(
        second.current_batch_size, 2,
        "second snapshot must preserve batch samples"
    );
}

/// `avg_batch_size` must be the mean of the window, and the hit rate a
/// percentage of requests.
///
/// Both are quotients in `snapshot()`. The existing suite records KV usage and
/// one cache hit (so the rate is trivially 100%) but never checks a MEAN, so
/// `cargo mutants` could turn the divide into a multiply or a modulo
/// unnoticed.
#[test]
fn test_snapshot_averages_match_the_recorded_samples() {
    let collector = MetricsCollector::new();

    // Mean of [2, 4, 6] is 4.0; the last sample is the "current" batch.
    for size in [2, 4, 6] {
        collector.record_batch_size(size);
    }
    let snapshot = collector.snapshot();
    assert!(
        (snapshot.avg_batch_size - 4.0).abs() < 1e-9,
        "mean of [2, 4, 6] must be 4.0; got {}",
        snapshot.avg_batch_size
    );
    assert_eq!(
        snapshot.current_batch_size, 6,
        "current batch size must be the most recent sample"
    );

    // Three hits out of four requests is 75%, not 100% and not 3.
    for _ in 0..4 {
        collector.record_prefix_cache_request();
    }
    for _ in 0..3 {
        collector.record_prefix_cache_hit();
    }
    let snapshot = collector.snapshot();
    assert!(
        (snapshot.prefix_cache_hit_rate - 75.0).abs() < 1e-9,
        "3 hits of 4 requests is 75%; got {}",
        snapshot.prefix_cache_hit_rate
    );
}

/// The throughput guards must be strict `> 0.0`, and the division must be a
/// division.
///
/// `snapshot()` computes `prefill_throughput` / `decode_throughput` as
/// `tokens / uptime` behind `if uptime > 0.0`. `cargo mutants` relaxed the two
/// guards to `>=` and the divide to `*` / `%`; all four survived because the
/// existing throughput test only asserts "positive" and "decode < prefill",
/// which the broken versions also satisfy (a ratio check cannot see `*` at all:
/// both rates scale identically).
///
/// The bound is absolute. The collector's uptime starts when it is built, so
/// after a 20 ms sleep the internal uptime lies in `[0.018, measured_total]`,
/// which puts the true rate in `[1000/measured_total, 1000/0.018]`. A multiply
/// yields ~`1000 * 0.02 = 20` and a modulo a few units — both far below the
/// floor; a `>=` guard would divide by an uptime of exactly zero, which only
/// happens for an idle collector created in the same instant (covered by the
/// existing zero-rate assertion).
#[test]
fn throughput_is_tokens_divided_by_measured_uptime() {
    // Taken BEFORE construction, so the measured span is never shorter than
    // the collector's own uptime and `1000 / measured` is a true lower bound.
    let created = std::time::Instant::now();
    let collector = MetricsCollector::new();
    collector.record_prefill_tokens(1_000);
    collector.record_decode_tokens(250);

    std::thread::sleep(std::time::Duration::from_millis(20));

    let snapshot = collector.snapshot();
    let measured_total = created.elapsed().as_secs_f64();
    assert!(measured_total >= 0.018, "fixture: the sleep was observed");

    let floor = 1_000.0 / measured_total;
    // The collector's uptime is at least the sleep, so the rate is at most
    // `1000 / sleep`; 0.015 s leaves room for clock granularity.
    let ceiling = 1_000.0 / 0.015;
    assert!(
        (floor..=ceiling).contains(&snapshot.prefill_throughput),
        "1000 prefill tokens over ~{measured_total}s must be {floor}..={ceiling}; \
         got {} (a `*` mutant lands near 20, a `%` mutant near 0)",
        snapshot.prefill_throughput
    );
    assert!(
        (250.0 / measured_total..=250.0 / 0.015).contains(&snapshot.decode_throughput),
        "250 decode tokens over ~{measured_total}s must be a quarter of the \
         prefill rate; got {}",
        snapshot.decode_throughput
    );

    // A collector with no phase tokens reports exactly zero, not NaN.
    let idle = MetricsCollector::new();
    let snapshot = idle.snapshot();
    assert!(
        snapshot.prefill_throughput.abs() < f64::EPSILON,
        "an idle collector must report exactly 0 throughput, got {}",
        snapshot.prefill_throughput
    );
}
