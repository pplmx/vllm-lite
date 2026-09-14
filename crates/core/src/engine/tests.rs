//! Unit tests for the top-level `Engine` orchestration surface —
//! `Engine::{new, with_config, with_drafts, with_budget_boxed, step,
//! add_request, has_pending}` and the `EngineBuilder` / `SleepPolicy`
//! value semantics. Tests for the deeper sub-modules (`ctor`,
//! `cuda_graph`, `draft_management`, `graph_step`, `lifecycle`,
//! `run`, `spec_dispatch`, `beam`) live alongside their respective
//! modules.
//!
//! Extracted from `engine/mod.rs` to keep the implementation file
//! under the project's 800-line soft cap.

use super::*;
use crate::metrics::EnhancedMetricsCollector;
use crate::speculative::AdaptiveSpeculativeDecoder;
use crate::speculative::draft_resolver::{DraftLoader, DraftResolver, NoopLoader};
use crate::speculative::registry::{DraftId, DraftModelRegistry, DraftRegistryError, DraftSpec};
use crate::types::{AdaptiveDraftConfig, Request, SchedulerConfig};
use std::sync::Arc;
use tokio::sync::mpsc;
use vllm_testing::StubModel;

#[test]
fn test_engine_streaming() {
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx, mut rx) = mpsc::channel(64);

    engine.add_request(Request::new(1, vec![10, 20], 5), tx);

    // First step: prefill, should return at least 1 output (the generated token)
    let out = engine.step().unwrap();
    assert!(!out.is_empty());
    assert_eq!(rx.try_recv().unwrap().token, 42);

    // Keep stepping until done
    let mut steps = 0;
    while engine.has_pending() && steps < 10 {
        let out = engine.step().unwrap();
        if !out.is_empty() {
            assert_eq!(out[0].1.token, 42);
            assert_eq!(rx.try_recv().unwrap().token, 42);
        }
        steps += 1;
    }

    assert!(
        !engine.has_pending(),
        "Sequence should finish after max_tokens"
    );
}

#[test]
fn test_engine_multi_request() {
    let stub = StubModel::returning(10);
    // Disable PD separation so every step processes both sequences,
    // making the step count deterministic in tests.
    let config = SchedulerConfig {
        enable_pd_separation: false,
        enable_dynamic_batching: false,
        ..SchedulerConfig::default()
    };
    let mut engine = Engine::with_config(stub, None, config, 4, 1024);
    let (tx1, mut rx1) = mpsc::channel(64);
    let (tx2, mut rx2) = mpsc::channel(64);

    engine.add_request(Request::new(1, vec![10], 2), tx1);
    engine.add_request(Request::new(2, vec![20], 2), tx2);

    engine.step().unwrap();
    assert_eq!(rx1.try_recv().unwrap().token, 10);
    assert_eq!(rx2.try_recv().unwrap().token, 10);

    engine.step().unwrap();
    assert_eq!(rx1.try_recv().unwrap().token, 10);
    assert_eq!(rx2.try_recv().unwrap().token, 10);

    assert!(!engine.has_pending());
}

#[test]
fn test_engine_no_requests() {
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let out = engine.step().unwrap();
    assert!(out.is_empty());
}

#[test]
fn test_engine_max_draft_tokens_config() {
    let stub = StubModel::returning(42);
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: true,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };
    let engine = Engine::with_config(stub, None, config, 8, 1024);
    assert_eq!(engine.max_draft_tokens, 8);
}

#[test]
fn test_engine_error_tracking() {
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10], 3), tx);

    let _ = engine.step();

    assert_eq!(engine.error_count, 0);
}

#[test]
fn test_engine_response_channel_cleanup() {
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx1, _rx1) = mpsc::channel(64);
    let (tx2, _rx2) = mpsc::channel(64);

    engine.add_request(Request::new(1, vec![10], 1), tx1);
    engine.add_request(Request::new(2, vec![20], 1), tx2);

    for _ in 0..3 {
        let _ = engine.step();
    }

    assert!(!engine.has_pending());
}

/// RIL ISS-110: when the client drops the response channel mid-generation
/// (`TrySendError::Closed` — the ONLY disconnect signal on non-streaming
/// paths, which build no `CancelOnDrop` guard), the engine must cancel the
/// sequence instead of silently generating into a closed channel for the
/// rest of its `max_tokens` budget. Pre-fix `try_send_token` treated
/// `Closed` as a silent no-op, so every aborted request burned
/// CPU/tokens/KV up to `max_tokens` (a resource-exhaustion vector under
/// client aborts). Post-fix `Closed` cancels the sequence (releases KV,
/// balances in-flight, removes the response-channel entry).
#[test]
fn test_engine_cancels_sequence_on_closed_response_channel() {
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx, rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(1, vec![10, 20], 5), tx);

    // Step 1: prefill emits the first token into the still-open channel.
    let _ = engine.step().unwrap();
    assert!(
        engine.response_txs.contains_key(&seq_id),
        "sequence must still be tracked before the disconnect"
    );

    // Client disconnects: dropping the receiver makes the next `try_send`
    // return `Closed`.
    drop(rx);

    // Step 2: the token hits the closed channel → the engine must cancel
    // the sequence (drop the response-channel entry, stop generating).
    let _ = engine.step().unwrap();
    assert!(
        !engine.response_txs.contains_key(&seq_id),
        "engine must cancel a sequence whose response channel closed (client gone)"
    );

    // The cancelled sequence must not keep burning the remaining max_tokens.
    for _ in 0..3 {
        let _ = engine.step().unwrap();
    }
    assert!(
        !engine.has_pending(),
        "cancelled sequence must not keep generating to max_tokens"
    );
}

/// RIL ISS-074 / TASK-089: a full token response channel must not silently
/// lose the token. Pre-fix `send_and_collect_results` did `let _ =
/// tx.try_send(...)` and ignored `TrySendError::Full`, so a handler that
/// drained slower than generation got a permanent stream gap with zero
/// observability. Post-fix the Full drop is logged AND counted in
/// `dropped_tokens_total` on the metrics collector.
#[test]
fn test_engine_full_response_channel_records_dropped_token() {
    use vllm_traits::SampledToken;

    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);

    // Capacity-1 channel, pre-filled and NOT drained: the receiver stays
    // alive (so try_send gives Full, not Closed) but is never read.
    let (tx, _rx) = mpsc::channel::<SampledToken>(1);
    let _ = tx.try_send(SampledToken {
        token: 999,
        logprob: 0.0,
        top_logprobs: Vec::new(),
    });

    engine.add_request(Request::new(1, vec![10], 1), tx);
    // Drive requests to completion so `send_and_collect_results` fires with
    // a generated token against the full channel.
    for _ in 0..10 {
        let _ = engine.step();
        if !engine.has_pending() {
            break;
        }
    }
    assert!(!engine.has_pending());
    assert_eq!(
        engine.scheduler.metrics.get_counter("dropped_tokens_total"),
        1,
        "a Full try_send must be counted as a dropped token (RIL ISS-074)"
    );
}

#[test]
fn test_tokens_total_counts_generated_not_input() {
    // RIL ISS-083: `tokens_total` ("Total tokens generated") must count
    // the emitted OUTPUT tokens, not the sum of input (prompt) token
    // lengths. Pre-fix the regular path (`scheduler/batch.rs`) summed
    // `input_tokens.len()` per step — the full 40-token prompt on the
    // prefill step — and the speculative path (`spec_dispatch`) did the
    // same; only the CUDA-graph path (`graph_step.rs`) counted emitted
    // results. A 40-token prompt with `max_tokens = 5` therefore
    // accumulated ~44 pre-fix (40 + 4 decode steps) instead of exactly 5
    // generated tokens.
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![7; 40], 5), tx);
    for _ in 0..40 {
        let _ = engine.step();
        if !engine.has_pending() {
            break;
        }
    }
    assert!(!engine.has_pending());
    let generated = engine.scheduler.metrics.runtime_snapshot().tokens_total;
    assert_eq!(
        generated, 5,
        "tokens_total must count the 5 generated tokens (max_tokens budget), \
         not the ~44 input-token length sum the pre-fix code recorded \
         (got {generated})"
    );
}

#[test]
fn test_prefill_and_decode_throughput_are_live() {
    // RIL ISS-095: `prefill_throughput` / `decode_throughput` (exported
    // on /metrics as prefill_throughput_tps / decode_throughput_tps) were
    // pinned at 0.000 forever because the only recorders
    // (`record_prefill_tokens` / `record_decode_tokens`) lived under
    // `#[cfg(test)]` — no production step path called them. Every step
    // path (regular, speculative, CUDA-graph) now records the phase split
    // via `record_batch_phase_tokens`. A 40-token prompt with
    // `max_tokens = 5` runs one prefill step (40 input positions) followed
    // by four decode steps (1 token each), so both gauges must be positive
    // after completion.
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![7; 40], 5), tx);
    for _ in 0..40 {
        let _ = engine.step();
        if !engine.has_pending() {
            break;
        }
    }
    assert!(!engine.has_pending());
    let snap = engine.scheduler.metrics.runtime_snapshot();
    assert!(
        snap.prefill_throughput > 0.0,
        "prefill_throughput must be live after a 40-token prefill step \
         (was pinned at 0 pre-fix); got {}",
        snap.prefill_throughput
    );
    assert!(
        snap.decode_throughput > 0.0,
        "decode_throughput must be live after 4 decode steps (was pinned \
         at 0 pre-fix); got {}",
        snap.decode_throughput
    );
}

#[test]
fn test_scheduler_wait_time_is_live() {
    // RIL TASK-119: `avg_scheduler_wait_time_ms` was pinned at 0.000 in
    // production because the only recorder (`record_scheduler_wait_time`)
    // lived under `#[cfg(test)]` — no admission path measured queue→drain
    // delay. Both scheduler batch builders now record each drained
    // sequence's `arrival_time`→drain elapsed via
    // `RequestQueue::drain_by_phase`. We sleep after enqueue so the drain
    // wait is deterministically > 0 (no reliance on clock tick granularity).
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![7; 16], 2), tx);
    std::thread::sleep(std::time::Duration::from_millis(10));

    let _ = engine.step();

    let snap = engine.scheduler.metrics.runtime_snapshot();
    assert!(
        snap.avg_scheduler_wait_time_ms >= 5.0,
        "avg_scheduler_wait_time_ms must be live after admitting a queued \
         request that waited 10ms (was pinned at 0 pre-fix); got {}",
        snap.avg_scheduler_wait_time_ms
    );
}

#[test]
fn test_kv_cache_usage_is_live_without_get_metrics() {
    // RIL ISS-100: `kv_cache_usage_percent` (exported on /metrics) was
    // pinned at 0.000 whenever an operator scraped only /metrics: the sole
    // production writer (`record_kv_cache_usage`) lived inside the
    // `EngineMessage::GetMetrics` arm of `run()`, but the /metrics handler
    // renders the runtime snapshot directly with no round-trip. So the
    // gauge was only ever refreshed by /health/details or /debug/kv-cache
    // probes. Every `Engine::step` now records allocator usage, so a step
    // that allocates KV blocks must leave the snapshot non-zero with no
    // GetMetrics involved.
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx, _rx) = mpsc::channel(64);

    // Nothing admitted yet → no blocks allocated → usage is 0.
    assert!(
        engine
            .scheduler
            .metrics
            .runtime_snapshot()
            .kv_cache_usage_percent
            .abs()
            < f64::EPSILON,
        "pre-step: no blocks should be allocated"
    );

    engine.add_request(Request::new(1, vec![7; 16], 2), tx);
    // One step: prefill a 16-token prompt (allocates 1 block) + pull the
    // first sampled token. No `GetMetrics` message is sent anywhere.
    let _ = engine.step();

    let snap = engine.scheduler.metrics.runtime_snapshot();
    assert!(
        snap.kv_cache_usage_percent > 0.0,
        "kv_cache_usage_percent must be live after a step that allocated KV \
         blocks (was pinned at 0 pre-fix — only GetMetrics refreshed it); got {}",
        snap.kv_cache_usage_percent
    );
}

#[test]
fn test_prefix_cache_nodes_is_live_without_get_metrics() {
    // RIL ISS-120: `prefix_cache_nodes` (exposed on /debug/metrics) had
    // the same GetMetrics-only writer as `kv_cache_usage_percent`, which
    // ISS-100 fixed — a /metrics- or /debug-metrics-only deployment never
    // round-trips, so the gauge pinned at 0 forever. A prefill step that
    // installs a prefix-cache block must leave `snapshot().prefix_cache_nodes
    // > 0` with no GetMetrics message sent.
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx, _rx) = mpsc::channel(64);

    assert_eq!(
        engine
            .scheduler
            .metrics
            .runtime_snapshot()
            .prefix_cache_nodes,
        0,
        "pre-step: no prefix blocks should exist"
    );

    // prefix_cache inserts only when a sequence reaches Finished (its
    // prompt-covering blocks are cached then), so max_tokens=1 lets the
    // prefill + first decode finish the sequence.
    engine.add_request(Request::new(1, vec![7; 16], 1), tx);
    let _ = engine.step();
    let _ = engine.step();

    let snap = engine.scheduler.metrics.runtime_snapshot();
    assert!(
        snap.prefix_cache_nodes > 0,
        "prefix_cache_nodes must be live after a step that finished a sequence \
         and inserted its prefix blocks (was pinned at 0 pre-fix — only \
         GetMetrics refreshed it); got {}",
        snap.prefix_cache_nodes
    );
}

#[test]
fn test_requests_in_flight_live_and_balanced() {
    // RIL ISS-082: `requests_in_flight` (Prometheus gauge / OTLP
    // `inflight_requests`) must be a live signal. Pre-fix the only
    // writers lived under `#[cfg(test)]`, so the gauge was pinned at 0
    // under any load. Now `add_request` (admitted) increments and
    // `finalize_finished` decrements.
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx, _rx) = mpsc::channel(64);
    assert_eq!(
        engine
            .scheduler
            .metrics
            .runtime_snapshot()
            .requests_in_flight,
        0
    );
    engine.add_request(Request::new(1, vec![10, 20], 3), tx);
    assert_eq!(
        engine
            .scheduler
            .metrics
            .runtime_snapshot()
            .requests_in_flight,
        1,
        "admitted request must increment requests_in_flight"
    );
    for _ in 0..10 {
        let _ = engine.step();
        if !engine.has_pending() {
            break;
        }
    }
    assert!(!engine.has_pending());
    assert_eq!(
        engine
            .scheduler
            .metrics
            .runtime_snapshot()
            .requests_in_flight,
        0,
        "completion must decrement requests_in_flight back to 0"
    );
}

#[test]
fn test_requests_in_flight_stop_completion_decrements_once() {
    // RIL ISS-173: a stop-sequence (or EOS) completion must decrement
    // `requests_in_flight` exactly ONCE. Pre-fix the Length pass
    // re-finalized every stop-completed sequence: `finalize_stop_sequences`
    // already called `finalize_finished(Stop)` (removing the response_tx
    // and decrementing), but the seq was moved into `finished` and the
    // second `finished_sequences()` pass called `finalize_finished(Length)`
    // again — decrementing the gauge a second time. With two requests in
    // flight where one finishes by stop, the correct in-flight value is 1
    // (the other request is still running); the double-decrement printed 0.
    let stub = StubModel::returning(42);
    let config = SchedulerConfig {
        enable_pd_separation: false,
        enable_dynamic_batching: false,
        ..SchedulerConfig::default()
    };
    let mut engine = Engine::with_config(stub, None, config, 4, 1024);

    // Request A (id 1): generates 42 on the first decode step, which
    // matches the stop sequence [42] — completes by stop after one step.
    let stop_params = vllm_traits::SamplingParams::builder()
        .with_stop_token_sequences(vec![vec![42]])
        .build();
    let req_a = Request {
        id: 1,
        prompt: vec![10],
        max_tokens: 100,
        sampling_params: stop_params,
        priority: crate::types::Priority::default(),
        draft_model_id: None,
    };
    // Request B (id 2): no stop sequence, long budget — stays in flight.
    let req_b = Request {
        id: 2,
        prompt: vec![10],
        max_tokens: 100,
        sampling_params: vllm_traits::SamplingParams::default(),
        priority: crate::types::Priority::default(),
        draft_model_id: None,
    };
    let (tx_a, _rx_a) = mpsc::channel(64);
    let (tx_b, _rx_b) = mpsc::channel(64);
    engine.add_request(req_a, tx_a);
    engine.add_request(req_b, tx_b);
    assert_eq!(
        engine
            .scheduler
            .metrics
            .runtime_snapshot()
            .requests_in_flight,
        2,
        "two admitted requests must be in flight"
    );

    // One step: A prefills, generates 42, stop-matches, finishes. B also
    // generates 42 (no stop) and stays running.
    let out = engine.step().unwrap();
    assert!(!out.is_empty());

    // A is done; B is still pending.
    assert!(
        engine.has_pending(),
        "B must still be pending after A's stop completion"
    );
    assert_eq!(engine.scheduler.running_count(), 1, "only B should remain");
    assert_eq!(
        engine
            .scheduler
            .metrics
            .runtime_snapshot()
            .requests_in_flight,
        1,
        "stop completion must decrement in-flight exactly once (the other request is still running)"
    );
}

#[test]
fn test_requests_in_flight_skips_rejected_admission() {
    // RIL ISS-082: a rejected admission (empty prompt → seq_id 0) must NOT
    // increment the in-flight counter — it never reaches finalize_finished,
    // so a start without an end would leave the gauge stuck at 1.
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx, _rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(1, vec![], 3), tx);
    assert_eq!(seq_id, 0, "empty prompt must be rejected");
    assert_eq!(
        engine
            .scheduler
            .metrics
            .runtime_snapshot()
            .requests_in_flight,
        0
    );
}

#[test]
fn test_requests_in_flight_balances_on_cancel() {
    // RIL ISS-082: cancel_request finalizes with FinishReason::Cancelled,
    // which must decrement an admitted request's in-flight count.
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 3), tx);
    assert_eq!(
        engine
            .scheduler
            .metrics
            .runtime_snapshot()
            .requests_in_flight,
        1
    );
    assert!(engine.cancel_request(1));
    assert_eq!(
        engine
            .scheduler
            .metrics
            .runtime_snapshot()
            .requests_in_flight,
        0,
        "cancel must balance the start"
    );
}

#[test]
fn test_requests_in_flight_end_saturates_at_zero() {
    // RIL ISS-082: an unbalanced `record_request_end` (double-finalize,
    // or cancel without a start) must saturate at 0 — the pre-fix raw
    // `fetch_sub(1)` wrapped to u64::MAX (~1.8e19), poisoning dashboards.
    let metrics = crate::metrics::EnhancedMetricsCollector::default();
    metrics.record_request_end();
    metrics.record_request_end();
    assert_eq!(
        metrics.runtime_snapshot().requests_in_flight,
        0,
        "saturating decrement must not wrap below 0"
    );
}

#[test]
fn test_sleep_policy_immediate_work() {
    let mut policy = SleepPolicy::default();
    let interval = policy.next_interval(true);
    assert_eq!(interval, 1);
    assert_eq!(policy.consecutive_idle, 0);
}

#[test]
fn test_sleep_policy_exponential_backoff() {
    let mut policy = SleepPolicy::default();

    let _ = policy.next_interval(false);
    assert_eq!(policy.consecutive_idle, 1);

    let interval2 = policy.next_interval(false);
    assert_eq!(policy.consecutive_idle, 2);

    let interval3 = policy.next_interval(false);
    assert!(interval3 >= interval2);

    let interval4 = policy.next_interval(true);
    assert_eq!(interval4, 1);
}

#[test]
fn test_sleep_policy_max_interval() {
    let mut policy = SleepPolicy::default();

    for _ in 0..100 {
        policy.next_interval(false);
    }

    let interval = policy.next_interval(false);
    assert!(interval <= policy.max_interval);
}

#[test]
fn test_engine_default_has_empty_draft_registry() {
    let stub = StubModel::returning(42);
    let engine = Engine::new(stub, None);
    assert!(engine.draft_registry().is_empty());
    assert_eq!(engine.draft_registry().len(), 0);
}

#[test]
fn test_engine_with_drafts_registers_all_specs_as_unloaded() {
    let stub = StubModel::returning(42);
    let drafts = vec![
        DraftSpec::new("a", "/tmp/model-a", 64),
        DraftSpec::new("b", "/tmp/model-b", 32),
    ];
    let engine = Engine::with_drafts(stub, None, drafts, SchedulerConfig::default(), 4, 1024);
    assert_eq!(engine.draft_registry().len(), 2);
    assert!(engine.draft_registry().contains(&DraftId("a".into())));
    assert!(engine.draft_registry().contains(&DraftId("b".into())));
    assert!(!engine.draft_registry().is_loaded(&DraftId("a".into())));
    assert!(!engine.draft_registry().is_loaded(&DraftId("b".into())));
}

#[test]
fn test_engine_runtime_register_unload_draft() {
    let stub = StubModel::returning(42);
    let engine = Engine::new(stub, None);
    engine
        .register_draft(DraftSpec::new("late", "/tmp/late", 16))
        .unwrap();
    assert!(engine.draft_registry().contains(&DraftId("late".into())));

    // Unload on already-unloaded draft is a no-op
    engine.unload_draft(&DraftId("late".into())).unwrap();

    // Unload of unknown id errors
    let err = engine.unload_draft(&DraftId("nope".into())).unwrap_err();
    assert!(matches!(err, DraftRegistryError::UnknownDraftId(_)));
}

#[test]
fn test_engine_default_has_unlimited_budget() {
    let stub = StubModel::returning(42);
    let engine = Engine::new(stub, None);
    assert_eq!(
        engine.memory_budget().total_bytes(),
        u64::MAX,
        "default Engine memory budget should be unlimited"
    );
}

#[test]
fn test_engine_with_budget_shares_with_registry() {
    use crate::speculative::memory_budget::MemoryBudget;
    let stub = StubModel::returning(42);
    let budget = Arc::new(MemoryBudget::new(1024).unwrap());
    let engine = Engine::with_budget_boxed(
        Box::new(stub),
        None,
        vec![DraftSpec::new("a", "/tmp", 0).with_weight_size(100)],
        budget,
        SchedulerConfig::default(),
        4,
        1024,
    );
    assert_eq!(engine.memory_budget().total_bytes(), 1024);
    assert_eq!(engine.draft_registry().memory_budget().total_bytes(), 1024);
}

#[test]
fn test_engine_attach_draft_budgeted_refuses_oversized() {
    use crate::speculative::memory_budget::MemoryBudget;
    let stub = StubModel::returning(42);
    let budget = Arc::new(MemoryBudget::new(100).unwrap());
    let engine = Engine::with_budget_boxed(
        Box::new(stub),
        None,
        vec![DraftSpec::new("huge", "/tmp", 4).with_weight_size(1000)],
        budget,
        SchedulerConfig::default(),
        4,
        1024,
    );
    let backend: Box<dyn ModelBackend> = Box::new(StubModel::returning(1));
    let err = engine
        .attach_draft_budgeted(&DraftId("huge".into()), backend)
        .unwrap_err();
    assert!(matches!(err, DraftRegistryError::MemoryBudgetExceeded(_)));
    assert!(!engine.draft_registry().is_loaded(&DraftId("huge".into())));
}

#[test]
fn test_engine_increment_decrement_ref_auto_unloads() {
    let stub = StubModel::returning(42);
    let engine = Engine::new(stub, None);
    engine
        .register_draft(DraftSpec::new("a", "/tmp", 4))
        .unwrap();
    let backend: Box<dyn ModelBackend> = Box::new(StubModel::returning(1));
    engine.attach_draft(&DraftId("a".into()), backend).unwrap();
    engine.increment_draft_ref(&DraftId("a".into())).unwrap();
    engine.increment_draft_ref(&DraftId("a".into())).unwrap();

    // First decrement: still in use
    let auto_unloaded = engine.decrement_draft_ref(&DraftId("a".into())).unwrap();
    assert!(!auto_unloaded);
    assert!(engine.draft_registry().is_loaded(&DraftId("a".into())));

    // Second decrement: count -> 0, auto-unload
    let auto_unloaded = engine.decrement_draft_ref(&DraftId("a".into())).unwrap();
    assert!(auto_unloaded);
    assert!(!engine.draft_registry().is_loaded(&DraftId("a".into())));
}

#[test]
fn test_engine_unload_draft_with_refcount_errors_in_use() {
    let stub = StubModel::returning(42);
    let engine = Engine::new(stub, None);
    engine
        .register_draft(DraftSpec::new("a", "/tmp", 4))
        .unwrap();
    let backend: Box<dyn ModelBackend> = Box::new(StubModel::returning(1));
    engine.attach_draft(&DraftId("a".into()), backend).unwrap();
    engine.increment_draft_ref(&DraftId("a".into())).unwrap();
    let err = engine.unload_draft(&DraftId("a".into())).unwrap_err();
    assert!(matches!(err, DraftRegistryError::InUse(1)));

    // force_unload_draft bypasses
    engine.force_unload_draft(&DraftId("a".into())).unwrap();
    assert!(!engine.draft_registry().is_loaded(&DraftId("a".into())));
}

#[test]
fn test_engine_builder_minimal() {
    let target: Box<dyn ModelBackend> = Box::new(StubModel::default());
    let engine = EngineBuilder::new(target).build();
    assert_eq!(engine.max_draft_tokens, 4);
    assert_eq!(engine.error_count, 0);
    assert!(engine.draft_model.is_none());
    assert!(engine.adaptive_decoder.is_none());
    assert!(engine.draft_resolver.is_none());
}

#[test]
fn test_engine_builder_with_all_options() {
    let target: Box<dyn ModelBackend> = Box::new(StubModel::default());
    let draft: Box<dyn ModelBackend> = Box::new(StubModel::default());
    let registry = Arc::new(DraftModelRegistry::new());
    let loader: Arc<dyn DraftLoader> = Arc::new(NoopLoader);
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    let resolver = Arc::new(DraftResolver::new(registry, None, loader, metrics));
    let decoder = AdaptiveSpeculativeDecoder::new(AdaptiveDraftConfig::default());

    let engine = EngineBuilder::new(target)
        .with_draft_model(draft)
        .with_max_draft_tokens(8)
        .with_num_kv_blocks(2048)
        .with_adaptive_decoder(decoder)
        .with_draft_resolver(resolver)
        .build();

    assert_eq!(engine.max_draft_tokens, 8);
    assert!(engine.draft_model.is_some());
    assert!(engine.adaptive_decoder.is_some());
    assert!(engine.draft_resolver.is_some());
}

#[test]
fn test_engine_builder_sleep_policy_override() {
    let target: Box<dyn ModelBackend> = Box::new(StubModel::default());
    let policy = SleepPolicy {
        base_interval: 0,
        max_interval: 0,
        backoff_factor: 1.0,
        consecutive_idle: 0,
    };
    let engine = EngineBuilder::new(target).with_sleep_policy(policy).build();
    assert_eq!(engine.sleep_policy.base_interval, 0);
    assert_eq!(engine.sleep_policy.max_interval, 0);
}

// ──────────────────────────────────────────────────────────────────────
// `EngineBuilder::with_paged_kv_cache` tests (P41 T4). The engine-
// side plumbing lives in `engine/paged_kv_cache.rs` and the builder
// method in `engine/ctor/builder.rs`.
// ──────────────────────────────────────────────────────────────────────

#[cfg(feature = "multi-node")]
#[test]
fn engine_builder_with_paged_kv_cache_wires_wrapper_to_memory_manager() {
    use parking_lot::Mutex;
    use vllm_dist::BlockDataSource;
    use vllm_model::paged_tensor::PagedKvCache;
    let target: Box<dyn ModelBackend> = Box::new(StubModel::default());
    // `set_paged_kv_cache` takes a pre-wrapped `Arc<Mutex<PagedKvCache>>`
    // (the shape yielded by `ModelLoader::paged_kv_cache_clone` / the
    // model layer's `create_model`). The engine and the wrapper both
    // hold an `Arc` to the same `Mutex`, so `strong_count` >= 2.
    let cache = Arc::new(Mutex::new(
        PagedKvCache::new(2, 2, 4, 4, candle_core::Device::Cpu, false).expect("small cache"),
    ));
    let mut engine = EngineBuilder::new(target)
        .with_paged_kv_cache(cache)
        .build();

    // The wrapper getter should now produce a BlockDataSource.
    let wrapper = engine
        .paged_kv_cache_wrapper()
        .expect("paged_kv_cache_wrapper must be Some when wired in");
    // Confirm the wrapper is wired all the way through to MemoryManager.
    let memory_source = engine
        .scheduler
        .memory_mut()
        .block_data_source()
        .expect("MemoryManager must hold the wired BlockDataSource");
    assert!(
        Arc::ptr_eq(&wrapper, &memory_source),
        "Engine's stored wrapper must match the one threaded to MemoryManager"
    );
    // The wrapper should be usable as a BlockDataSource trait object.
    let _trait_obj: Arc<dyn BlockDataSource + Send + Sync> = wrapper;
    // P42: `paged_kv_cache()` returns `Arc<Mutex<PagedKvCache>>`. The
    // returned Arc should be the same one held by the wrapper's
    // `inner` — both the engine and the wrapper hold an Arc to the
    // same Mutex, so `Arc::strong_count` should be >= 2.
    let stored_cache = engine
        .paged_kv_cache()
        .expect("paged_kv_cache() must be Some when wired in");
    assert!(
        Arc::strong_count(&stored_cache) >= 2,
        "engine + wrapper should both hold an Arc to the cache"
    );
}

#[cfg(feature = "multi-node")]
#[test]
fn engine_without_with_paged_kv_cache_has_no_wrapper() {
    let target: Box<dyn ModelBackend> = Box::new(StubModel::default());
    let mut engine = EngineBuilder::new(target).build();
    assert!(engine.paged_kv_cache_wrapper().is_none());
    assert!(engine.paged_kv_cache().is_none());
    assert!(engine.scheduler.memory_mut().block_data_source().is_none());
}

#[test]
fn test_finalize_stop_sequences_tolerates_missing_params() {
    // Synthetic batches may carry an empty `sampling_params` (the
    // Batch docs call this "equivalent to greedy decoding"). The
    // stop-sequence pass must not panic on the seq/params index
    // mismatch — mirroring the defensive `.get(i)` used by the
    // spec-verifier path.
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let batch = vllm_traits::Batch {
        seq_ids: vec![1, 2],
        sampling_params: vec![],
        ..vllm_traits::Batch::empty()
    };
    let stopped = engine.finalize_stop_sequences(&batch);
    assert!(stopped.is_empty());
}

#[test]
fn test_finalize_stop_sequences_matches_and_releases() {
    // Positive control: with params present and a matching stop
    // sequence, the sequence is finished (KV blocks released) even
    // though it is nowhere near max_tokens.
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    let params = vllm_traits::SamplingParams::builder()
        .with_stop_token_sequences(vec![vec![42]])
        .build();
    let req = Request {
        id: 1,
        prompt: vec![10, 20],
        max_tokens: 100,
        sampling_params: params,
        priority: crate::types::Priority::default(),
        draft_model_id: None,
    };
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(req, tx);

    // One step: prefill + first token (42) → matches the stop seq.
    let out = engine.step().unwrap();
    assert!(!out.is_empty());
    assert!(
        !engine.has_pending(),
        "stop-match must finish the sequence long before max_tokens"
    );
    assert_eq!(engine.scheduler.running_count(), 0);
}

#[test]
fn test_finalize_stop_sequences_ignores_prompt_only_match() {
    // The stop check must only consider the GENERATED region
    // (`seq.tokens[seq.prompt_len..]`): a stop sequence that is a suffix
    // of the PROMPT must not finish the sequence while it is still
    // generating. Guards the slice-borrow contract of the no-copy
    // suffix check (the perf fix in `finalize_stop_sequences` — the
    // check borrows the generated region instead of `to_vec()`-copying
    // it, but a regression that compared against raw `seq.tokens`
    // would wrongly match prompt content).
    let stub = StubModel::returning(1);
    let mut engine = Engine::new(stub, None);

    // The prompt ends with [1, 2] — the SAME suffix as the stop
    // sequence — but every generated token is 1, so the generated
    // region never ends in [1, 2].
    let params = vllm_traits::SamplingParams::builder()
        .with_stop_token_sequences(vec![vec![1, 2]])
        .build();
    let req = Request {
        id: 1,
        prompt: vec![7, 1, 2],
        max_tokens: 100,
        sampling_params: params,
        priority: crate::types::Priority::default(),
        draft_model_id: None,
    };
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(req, tx);

    // Several decode steps: generated tokens are all 1, so the stop
    // sequence [1, 2] must never match even though the PROMPT [7,1,2]
    // ends in it.
    for _ in 0..5 {
        let out = engine.step().unwrap();
        assert!(!out.is_empty());
    }
    assert!(
        engine.has_pending(),
        "a stop sequence matching only the prompt suffix must not finish the sequence"
    );
    assert_eq!(engine.scheduler.running_count(), 1);
}

/// RIL ISS-075: when the engine knows the model's EOS token id and the
/// generated token equals it, the sequence must finish with
/// `FinishReason::Stop` long before `max_tokens` — a 'short' answer must not
/// burn the whole budget or report `Length`.
#[test]
fn test_eos_stop_finalizes_with_stop_reason() {
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    engine.set_eos_token_id(Some(42));

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 100), tx);
    // Direct callers of `add_request` get no finish-reason channel (the HTTP
    // mailbox message carries one); inject a capturing oneshot ourselves.
    let (fr_tx, mut fr_rx) = tokio::sync::oneshot::channel();
    engine.finish_reason_txs.insert(1, fr_tx);

    let out = engine.step().unwrap();
    assert!(!out.is_empty());
    assert!(
        !engine.has_pending(),
        "EOS-stop must finish the sequence long before max_tokens (100)"
    );
    assert_eq!(engine.scheduler.running_count(), 0);
    let reason = fr_rx.try_recv().expect("finish reason must be delivered");
    assert_eq!(
        reason,
        FinishReason::Stop,
        "an EOS token must be reported as Stop, not Length"
    );
}

/// RIL ISS-075: with no EOS id configured (the engine default, e.g. a mock
/// or a checkpoint that declares none), a token equal to what WOULD be the
/// EOS id must not stop the sequence — legacy run-to-`max_tokens` behavior
/// is preserved.
#[test]
fn test_eos_stop_disabled_when_eos_unset() {
    let stub = StubModel::returning(42);
    let mut engine = Engine::new(stub, None);
    // engine.eos_token_id is None by default.

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 5), tx);

    let out = engine.step().unwrap();
    assert!(!out.is_empty());
    assert!(
        engine.has_pending(),
        "without an EOS id the sequence must keep running to max_tokens"
    );
    assert_eq!(engine.scheduler.running_count(), 1);
}
