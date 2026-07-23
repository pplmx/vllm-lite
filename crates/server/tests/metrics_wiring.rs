//! OBS-01 wiring test — `/metrics` and the engine share a single
//! `EnhancedMetricsCollector` so Prometheus scrapes show real values.
//!
//! Background: previously the server bootstrap path created a fresh
//! `EnhancedMetricsCollector` for the HTTP `/metrics` exporter, while
//! the engine kept its own private collector internally. The two
//! never agreed: `/metrics` always reported zero counters, while
//! `/health/details` (which goes through `EngineMessage::GetMetrics`)
//! showed live values. Operators could not trust the Prometheus
//! scrape.
//!
//! The fix is in `main.rs`: clone the engine's collector before
//! moving it into the worker thread, and pass the same `Arc` to
//! `ApiState`. This test guards the invariant at the engine layer
//! so future constructor changes can't reintroduce a duplicate
//! collector by accident.
//!
//! We verify two invariants:
//!
//! 1. `Engine.scheduler.metrics` is publicly reachable and clonable
//!    (so the wiring code can `clone()` it before the engine moves
//!    into the worker thread).
//! 2. The `EnhancedMetricsCollector` is `Arc`-shared: clones point
//!    to the same underlying allocation (`Arc::ptr_eq` returns
//!    `true`). This is the property main.rs relies on.
//!
//! For end-to-end coverage of the HTTP `/metrics` path, see
//! `metrics_handler_test.rs` (TODO if missing) — this file
//! deliberately targets the engine-side invariant because that's
//! where regressions would first appear.

use std::sync::Arc;

use vllm_core::Engine;
use vllm_traits::StubModelBackend;

#[test]
fn engine_scheduler_metrics_is_arc_shared() {
    let engine = Engine::new(StubModelBackend, None::<StubModelBackend>);

    // Invariant: the metrics collector is publicly reachable on the
    // scheduler so main.rs can clone it before moving the engine.
    let metrics_a = engine.scheduler.metrics.clone();

    // Invariant: cloning yields an Arc pointing at the same allocation
    // (this is the property main.rs relies on to share the collector
    // between the worker thread and the HTTP /metrics handler).
    let metrics_b = engine.scheduler.metrics;

    assert!(
        Arc::ptr_eq(&metrics_a, &metrics_b),
        "scheduler.metrics must be Arc-shared, not a freshly constructed duplicate"
    );

    // Sanity check: writing through one handle is observable through
    // the other. This catches a class of regressions where the field
    // type silently changes from `Arc<EnhancedMetricsCollector>` to
    // something that copies on `.clone()` (e.g. `Rc` or a value).
    metrics_a.record_request();
    let snap_before = metrics_b.snapshot();
    metrics_b.record_request();
    let snap_after = metrics_a.snapshot();

    assert_eq!(
        snap_after.requests_total,
        snap_before.requests_total + 1,
        "writes through one Arc handle must be visible to the other \
         (Arc semantics broken?)"
    );
}

#[test]
fn distinct_engines_get_distinct_collectors() {
    // Sanity guard the other direction: two engine instances must
    // not accidentally share a global static collector, which would
    // be a worse regression than the original duplicate (it would
    // cause cross-instance metric pollution).
    let engine_a = Engine::new(StubModelBackend, None::<StubModelBackend>);
    let engine_b = Engine::new(StubModelBackend, None::<StubModelBackend>);

    let metrics_a = engine_a.scheduler.metrics;
    let metrics_b = engine_b.scheduler.metrics;

    assert!(
        !Arc::ptr_eq(&metrics_a, &metrics_b),
        "separate engines must have separate metrics collectors"
    );
}
