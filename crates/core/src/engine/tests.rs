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

/// `is_healthy` must turn false once the engine accumulates the documented 10
/// errors.
///
/// `cargo mutants` found that replacing `is_healthy` with a constant `true`
/// survived the entire suite: every existing assertion only checks the healthy
/// direction, so the threshold itself had no coverage. `error_count` advances
/// only inside `Engine::run` (a direct `step()` error does NOT count), so the
/// test drives a real engine loop over a backend whose forward always fails and
/// reads the count back after the loop shuts down.
#[test]
fn test_is_healthy_turns_false_after_error_threshold() {
    use crate::types::EngineMessage;
    use vllm_testing::FailingModel;

    let mut engine = Engine::new_boxed(Box::new(FailingModel), None::<Box<dyn ModelBackend>>);
    assert!(engine.is_healthy(), "a new engine starts healthy");

    // The threshold is `error_count < 10`: 9 stays healthy, the 10th flips it.
    // This is the branch a constant-`true` mutant hides.
    engine.error_count = 9;
    assert!(
        engine.is_healthy(),
        "9 errors must still be healthy (threshold is 10)"
    );
    engine.error_count = 10;
    assert!(!engine.is_healthy(), "10 errors must report unhealthy");

    // And the counter really rises through `run` with a failing backend.
    let (msg_tx, msg_rx) = mpsc::channel::<EngineMessage>(16);
    let (count_tx, count_rx) = std::sync::mpsc::channel::<usize>();
    let loop_handle = std::thread::spawn(move || {
        engine.run(msg_rx);
        count_tx.send(engine.error_count).ok();
    });

    let (token_tx, _token_rx) = mpsc::channel(64);
    msg_tx
        .try_send(EngineMessage::AddRequest {
            request: Box::new(Request::new(1, vec![10, 20], 50)),
            response_tx: token_tx,
            seq_id_tx: None,
            finish_reason_tx: None,
            request_id: None,
        })
        .expect("the engine mailbox has room");
    // The failing forward fails fast, so a short wait is enough; `Shutdown`
    // ends the loop deterministically.
    std::thread::sleep(std::time::Duration::from_millis(50));
    msg_tx
        .try_send(EngineMessage::Shutdown)
        .expect("the engine mailbox has room");
    loop_handle
        .join()
        .expect("the engine loop must exit cleanly");

    let errors = count_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("the loop must report its error count");
    assert!(
        errors > 0,
        "a failing backend must accumulate step errors through `run`; got {errors}"
    );
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

/// when the client drops the response channel mid-generation
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

/// a full token response channel must not silently
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
        "a Full try_send must be counted as a dropped token"
    );
}

#[test]
fn test_tokens_total_counts_generated_not_input() {
    // `tokens_total` ("Total tokens generated") must count
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
    // `prefill_throughput` / `decode_throughput` (exported
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
    // `avg_scheduler_wait_time_ms` was pinned at 0.000 in
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
    // `kv_cache_usage_percent` (exported on /metrics) was
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
    // `prefix_cache_nodes` (exposed on /debug/metrics) had
    // the same GetMetrics-only writer as `kv_cache_usage_percent`, which
    // fixed — a /metrics- or /debug-metrics-only deployment never
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
    // `requests_in_flight` (Prometheus gauge / OTLP
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
    // a stop-sequence (or EOS) completion must decrement
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
    // a rejected admission (empty prompt → seq_id 0) must NOT
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
    // cancel_request finalizes with FinishReason::Cancelled,
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
    // an unbalanced `record_request_end` (double-finalize,
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

/// when the engine knows the model's EOS token id and the
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

/// with no EOS id configured (the engine default, e.g. a mock
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

/// `stop_hold_back` must reserve `longest_stop - 1` trailing tokens, and 0 when
/// there is nothing to look ahead for.
///
/// `cargo mutants` replaced the whole function with the constants 0 and 1. The
/// count decides how many generated tokens are WITHHELD from the client: too few
/// and a multi-token stop sequence leaks its prefix before the match is
/// recognised; too many and ordinary output is delayed for no reason.
#[test]
fn stop_hold_back_reserves_the_longest_stop_minus_one() {
    let engine = Engine::new(StubModel::returning(1), None);

    // No params / no stops: nothing to hold back.
    assert_eq!(
        engine.stop_hold_back(None),
        0,
        "a sequence without params must hold nothing back"
    );
    let plain = vllm_traits::SamplingParams::default();
    assert_eq!(
        engine.stop_hold_back(Some(&plain)),
        0,
        "a sequence without stop sequences must hold nothing back"
    );

    // A single-token stop needs no lookahead.
    let single = vllm_traits::SamplingParams::builder()
        .with_stop_token_sequences(vec![vec![9]])
        .build();
    assert_eq!(
        engine.stop_hold_back(Some(&single)),
        0,
        "a one-token stop is recognised on the token itself"
    );

    // The LONGEST stop governs, not the first or the shortest.
    let mixed = vllm_traits::SamplingParams::builder()
        .with_stop_token_sequences(vec![vec![1], vec![1, 2, 3, 4], vec![1, 2]])
        .build();
    assert_eq!(
        engine.stop_hold_back(Some(&mixed)),
        3,
        "the longest stop (4 tokens) requires reserving 3"
    );
}

/// `token_completes_stop_sequence` must report a match ONLY when the generated
/// tail really ends with a configured stop.
///
/// `cargo mutants` replaced the whole function with `true`, which would finalize
/// EVERY sequence with `FinishReason::Stop` on its first token — truncating every
/// response. The function also ignores the PROMPT: only tokens after `prompt_len`
/// count, so a prompt containing the stop pair must not trip it.
#[test]
fn token_completes_stop_sequence_checks_only_the_generated_tail() {
    let stops = vllm_traits::SamplingParams::builder()
        .with_stop_token_sequences(vec![vec![77, 78]])
        .build();
    let no_stops = vllm_traits::SamplingParams::default();

    let mut engine = Engine::new(StubModel::returning(1), None);
    // A prompt that itself ENDS with the stop pair.
    let mut req = Request::new(1, vec![77, 78], 10);
    req.sampling_params = stops.clone();
    engine.add_request(req, mpsc::channel(8).0);
    let _ = engine.scheduler.build_batch();

    // Prompt-only: no match, because the generated tail is empty.
    assert!(
        !engine.token_completes_stop_sequence(1, Some(&stops)),
        "a stop pair inside the PROMPT must not complete the sequence"
    );

    // Append the stop pair to the generated tail: now it matches.
    engine
        .scheduler
        .get_sequence_mut(1)
        .expect("seq 1 is running")
        .tokens
        .extend([5, 77, 78]);
    assert!(
        engine.token_completes_stop_sequence(1, Some(&stops)),
        "a generated tail ending with the stop pair must complete the sequence"
    );

    // A non-matching tail stays false.
    engine
        .scheduler
        .get_sequence_mut(1)
        .expect("seq 1 is running")
        .tokens
        .push(6);
    assert!(
        !engine.token_completes_stop_sequence(1, Some(&stops)),
        "a tail without the stop pair must not complete"
    );

    // No stops / no params / unknown id: never a match.
    assert!(!engine.token_completes_stop_sequence(1, Some(&no_stops)));
    assert!(!engine.token_completes_stop_sequence(1, None));
    assert!(!engine.token_completes_stop_sequence(9999, Some(&stops)));
}

/// A mid-chunk prefill must NOT emit its predicted token to the client.
///
/// `step_regular` builds `stale_mask` as
/// `is_prefill[i] && num_computed + input_len < prompt_len` and passes it to
/// `send_and_collect_results`, which skips those entries. `cargo mutants` relaxed
/// that conjunction to `||` — marking EVERY prefill stale, so a completed
/// prefill's real first token would be swallowed — and relaxed the comparison to
/// `<=`/`>`/`==`. Emitting a mid-chunk prediction would put a token in the client
/// stream that the model never really produced, and one that `update` later
/// trims from `seq.tokens`.
#[test]
fn mid_chunk_prefill_does_not_emit_a_prediction() {
    let config = SchedulerConfig {
        // A budget AND chunk far smaller than the prompt force several
        // prefill rounds: a fresh prompt that FITS the budget is deliberately
        // not split, so both knobs must be tight.
        prefill_chunk_size: 4,
        max_num_batched_tokens: 4,
        ..Default::default()
    };
    let mut engine = Engine::with_config(StubModel::returning(42), None, config, 8, 1024);
    let (tx, mut rx) = mpsc::channel(64);

    // A 16-token prompt against a 4-token chunk: the first step is mid-chunk.
    engine.add_request(Request::new(1, vec![10; 16], 4), tx);

    let mut steps = 0;
    let mut saw_any = false;
    while engine.has_pending() && steps < 20 {
        let out = engine.step().unwrap();
        if !out.is_empty() {
            saw_any = true;
            for (seq_id, tok) in &out {
                assert_eq!(*seq_id, 1);
                assert_eq!(tok.token, 42, "only real sampled tokens may be emitted");
            }
        }
        steps += 1;
    }

    // The request must still complete and deliver real tokens — proving the
    // assertion above cannot pass by nothing ever being emitted.
    assert!(saw_any, "a completed request must emit at least one token");
    assert!(
        steps > 1,
        "a 16-token prompt over a 4-token chunk needs several steps"
    );
    let mut received = 0;
    while rx.try_recv().is_ok() {
        received += 1;
    }
    assert!(
        received > 0,
        "the client channel must receive the real tokens"
    );
}

/// A multi-token stop sequence must HOLD BACK `stop_len - 1` trailing tokens
/// while releasing everything before them.
///
/// `send_and_collect_results` releases with `while pending.len() > hold`. `cargo
/// mutants` relaxed that to `==` and `<`: with either, tokens are held that
/// should have been released (delaying ordinary output) or released that could
/// still complete a stop (leaking the stop's prefix to the client). The bound is
/// what keeps ordinary output flowing while a stop match is still possible.
#[test]
fn multi_token_stop_hold_releases_all_but_the_lookahead_window() {
    // A 3-token stop needs 2 tokens held back as lookahead.
    let stops = vllm_traits::SamplingParams::builder()
        .with_stop_token_sequences(vec![vec![90, 91, 92]])
        .build();
    assert_eq!(
        Engine::new(StubModel::returning(1), None).stop_hold_back(Some(&stops)),
        2,
        "setup: a 3-token stop reserves a 2-token window"
    );

    let config = SchedulerConfig {
        max_num_batched_tokens: 4096,
        ..Default::default()
    };
    let mut engine = Engine::with_config(StubModel::returning(7), None, config, 8, 1024);
    let (tx, mut rx) = mpsc::channel(64);

    let mut req = Request::new(1, vec![10], 6);
    req.sampling_params = stops;
    engine.add_request(req, tx);

    for _ in 0..20 {
        let _ = engine.step();
        if !engine.has_pending() {
            break;
        }
    }
    assert!(!engine.has_pending(), "the request must complete");

    // Every token the model produced is 7, which never matches [90,91,92], so
    // the lookahead must have been released at the end rather than swallowed.
    let mut received = Vec::new();
    while let Ok(tok) = rx.try_recv() {
        received.push(tok.token);
    }
    assert!(
        !received.is_empty(),
        "ordinary output must reach the client despite the lookahead window"
    );
    assert!(
        received.iter().all(|&t| t == 7),
        "only real model tokens may reach the client; got {received:?}"
    );
}

/// A completed step must record its latency, so the p50/p90/p99 gauges are not
/// stuck at zero.
///
/// `finalize_and_record` guards with `if elapsed > 0.0 { record_latency }`.
/// `cargo mutants` relaxed that to `==`, `<` and `>=`: with `<` a real elapsed
/// time records NOTHING (the branch becomes unreachable), and with `>=` the
/// zero-elapsed case would also record. A step always takes some time, so the
/// latency window must receive a sample.
#[test]
fn a_completed_step_records_its_latency() {
    let mut engine = Engine::new(StubModel::returning(42), None);

    // No samples before any step runs.
    assert!(
        engine.scheduler.metrics.snapshot().p50_latency_ms.abs() < f64::EPSILON,
        "setup: no latency has been recorded yet"
    );

    let (tx, _rx) = mpsc::channel(8);
    engine.add_request(Request::new(1, vec![10, 20], 2), tx);
    let _ = engine.step().expect("the step must succeed");

    let snapshot = engine.scheduler.metrics.snapshot();
    assert!(
        snapshot.p50_latency_ms >= 0.0,
        "the median latency must be a real sample"
    );
    assert!(
        snapshot.avg_latency_ms > 0.0,
        "a completed step must have recorded a positive latency; got {}",
        snapshot.avg_latency_ms
    );
}

/// `chain_cursors_mut` must expose the engine's OWN cursor map, so a write
/// through it is visible to the next block-hash computation.
///
/// `cargo mutants` replaced it with `Box::leak(Box::new(HashMap::new()))` and
/// with one-entry maps — a fresh detached map every call. Seeding a cursor
/// would then be silently discarded, and `record_block_tokens` would chain from
/// parent hash 0 instead of the real predecessor: every block hash after the
/// first becomes wrong, so the distributed prefix cache can never hit.
#[cfg(feature = "multi-node")]
#[test]
fn chain_cursors_mut_exposes_the_engines_own_map() {
    let mut engine = Engine::new(StubModel::returning(1), None);

    engine.scheduler.chain_cursors_mut().insert(7, 4242);

    // A second borrow must observe the write: a leaked detached map would not.
    assert_eq!(
        engine.scheduler.chain_cursors_mut().get(&7).copied(),
        Some(4242),
        "a cursor written through the accessor must be readable through it"
    );
    assert_eq!(
        engine.scheduler.chain_cursors_mut().len(),
        1,
        "the accessor must not inject extra placeholder entries"
    );

    // Overwriting through the same accessor replaces, not appends.
    engine.scheduler.chain_cursors_mut().insert(7, 99);
    assert_eq!(
        engine.scheduler.chain_cursors_mut().get(&7).copied(),
        Some(99)
    );
}

/// `set_block_data_source` must install the source on the memory manager so the
/// engine-level accessor reports it.
///
/// `cargo mutants` replaced the forwarder with `()`: the gRPC `TransferKVBlock`
/// path would then resolve to `None` and fall back to the placeholder data
/// source, silently serving empty K/V bytes for real transfers.
#[cfg(feature = "multi-node")]
#[test]
fn set_block_data_source_reaches_the_memory_manager() {
    use vllm_dist::BlockDataSource;

    #[derive(Debug)]
    struct StubSource;
    #[async_trait::async_trait]
    impl BlockDataSource for StubSource {
        async fn fetch_block(&self, _block_id: u64) -> Result<Vec<u8>, vllm_dist::FetchError> {
            Err(vllm_dist::FetchError::NotFound(_block_id))
        }
    }

    let mut engine = Engine::new(StubModel::returning(1), None);
    assert!(
        engine.scheduler.memory_mut().block_data_source().is_none(),
        "setup: nothing is wired yet"
    );

    engine.scheduler.set_block_data_source(Arc::new(StubSource));

    assert!(
        engine.scheduler.memory_mut().block_data_source().is_some(),
        "set_block_data_source must reach the MemoryManager"
    );
}

/// A speculative-mode flag of `false` must take the REGULAR path even when a
/// draft model is attached.
///
/// `Engine::step` dispatches on
/// `self.speculative_mode && (self.draft_model.is_some() || self.draft_resolver.is_some())`.
/// `cargo mutants` relaxed that `&&` to `||` and it survived: no test ran a
/// step with a draft model attached but speculative mode OFF. With `||` the
/// engine drafts and verifies anyway, so a caller who turned speculation off
/// still pays for draft generation — and, as pinned below, reserves KV blocks
/// for a verification span it will never use.
#[test]
fn step_takes_the_regular_path_when_speculative_mode_is_off() {
    let mut engine = EngineBuilder::new(Box::new(StubModel::default()))
        .with_draft_model(Box::new(StubModel::default()))
        .with_max_draft_tokens(20)
        .with_num_kv_blocks(256)
        .build();
    assert!(!engine.speculative_mode, "setup: speculative mode is off");
    assert!(
        engine.draft_model.is_some(),
        "setup: a draft model IS attached"
    );

    // 8-token prompt: the regular path holds one KV block for it plus the
    // token it generates, while the speculative path reserves
    // `input + max_draft = 28` tokens => two blocks.
    let (tx, _rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(0, vec![7; 8], 8), tx);
    let _ = engine.step();

    let blocks = engine
        .scheduler
        .get_sequence(seq_id)
        .map(|s| s.kv_blocks.len());
    assert_eq!(
        blocks,
        Some(1),
        "a regular step must not reserve a verification span; two blocks means \
         the speculative path ran with speculative_mode off"
    );
}

/// `SleepPolicy::next_interval` must implement the documented backoff curve.
///
/// `cargo mutants` flipped `consecutive_idle == 1` to `!=`, changed the
/// multiplier to `/` and `+`, and the exponent's `- 1` to `+` / `/` — six
/// survivors. The idle loop is the ONE place the engine sleeps, so a wrong
/// curve is either a busy-spin (100% CPU on an idle server) or a stall (a
/// request waits seconds after an idle period). The fixture pins every
/// transition: base interval on work and on the FIRST idle tick, then the
/// geometric growth, then the cap.
#[test]
fn sleep_policy_backoff_curve_is_pinned() {
    let mut policy = SleepPolicy {
        base_interval: 10,
        max_interval: 100,
        backoff_factor: 2.0,
        consecutive_idle: 0,
    };

    // Work resets the counter and returns the base interval.
    assert_eq!(policy.next_interval(true), 10);
    assert_eq!(policy.next_interval(true), 10);
    assert_eq!(policy.consecutive_idle, 0, "work must reset the idle count");

    // The FIRST idle tick returns the base interval (a 1-tick idle gap must
    // not sleep longer than the scheduler tick).
    assert_eq!(policy.next_interval(false), 10);
    assert_eq!(policy.consecutive_idle, 1);

    // Then geometric growth: base * factor^(n-1).
    assert_eq!(policy.next_interval(false), 20, "10 * 2^1");
    assert_eq!(policy.next_interval(false), 40, "10 * 2^2");
    assert_eq!(policy.next_interval(false), 80, "10 * 2^3");

    // And the cap.
    assert_eq!(policy.next_interval(false), 100, "clamped to max_interval");
    assert_eq!(policy.next_interval(false), 100, "stays clamped");

    // Work returns the base interval immediately, even after a long idle run.
    assert_eq!(policy.next_interval(true), 10);
    assert_eq!(policy.consecutive_idle, 0);
}

/// `get_last_error` must report the recorded error, not a constant.
///
/// Replaced by `None`, `Some("")` and `Some("xyzzy")` — three survivors. The
/// accessor feeds the server's health probe: a constant `None` says "healthy"
/// no matter what happened, and a fabricated string says the opposite.
#[test]
fn last_error_accessor_reports_what_was_recorded() {
    let mut engine = Engine::new(StubModel::returning(42), None);
    assert_eq!(engine.get_last_error(), None, "a fresh engine has no error");

    engine.last_error = Some("boom: kernel fault".to_string());
    assert_eq!(engine.get_last_error(), Some("boom: kernel fault"));

    // Overwrite, as the run loop does on each new failure.
    engine.last_error = Some("second failure".to_string());
    assert_eq!(
        engine.get_last_error(),
        Some("second failure"),
        "the accessor must reflect the LATEST error"
    );
}

/// The adaptive-speculative switch must be observable from both sides.
///
/// `enable_adaptive_speculative` and `disable_adaptive_speculative` replaced by
/// `()`, plus `is_adaptive_speculative_enabled` replaced by the constants
/// `true` and `false` — four survivors. The switch routes requests into the
/// adaptive draft path and back out again; a no-op disable leaves the engine
/// paying for drafts it was told to stop generating.
#[test]
fn adaptive_speculative_toggle_is_observable() {
    let mut engine = Engine::new(StubModel::returning(42), None);
    assert!(!engine.is_adaptive_speculative_enabled());
    assert!(!engine.speculative_mode);

    engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());
    assert!(
        engine.is_adaptive_speculative_enabled(),
        "enable must install the decoder"
    );
    assert!(engine.speculative_mode, "enable must switch the mode on");

    engine.disable_adaptive_speculative();
    assert!(!engine.is_adaptive_speculative_enabled());
    assert!(!engine.speculative_mode, "disable must switch the mode off");

    // Enabling again after a disable must work (the decoder is re-created).
    engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());
    assert!(engine.is_adaptive_speculative_enabled());
}

/// `set_draft_loader` must report whether a resolver was present.
///
/// Replaced by the constants `true` and `false` — two survivors. The server
/// calls this once at startup and logs a warning on `false`; a constant `false`
/// reports a misconfigured engine that is actually fine, and a constant `true`
/// hides the case where lazy draft loading is silently unavailable.
#[test]
fn set_draft_loader_reports_resolver_presence() {
    let loader: Arc<dyn DraftLoader> = Arc::new(NoopLoader);

    let mut without = Engine::new(StubModel::returning(42), None);
    assert!(
        !without.set_draft_loader(Arc::clone(&loader)),
        "an engine built without a resolver must report false"
    );

    let drafts = vec![DraftSpec::new(
        "d",
        std::path::PathBuf::from("/tmp/d"),
        1024,
    )];
    let mut with = Engine::with_drafts(
        StubModel::returning(42),
        None,
        drafts,
        SchedulerConfig::default(),
        4,
        1024,
    );
    assert!(
        with.set_draft_loader(loader),
        "an engine with a resolver must report true"
    );
}

/// `Engine`'s and `EngineBuilder`'s `Debug` impls must describe their fields.
///
/// Both were replaced by `Ok(())` and survived; they are what an operator sees
/// when a startup failure is logged with `{engine:?}`.
#[test]
fn engine_and_builder_debug_name_their_fields() {
    let engine = Engine::new(StubModel::returning(42), None);
    let rendered = format!("{engine:?}");
    assert!(
        rendered.contains("Engine"),
        "Debug must name the type; got {rendered}"
    );
    assert!(
        rendered.contains("scheduler") || rendered.contains("error_count"),
        "Debug must include engine state; got {rendered}"
    );

    let builder = EngineBuilder::new(Box::new(StubModel::returning(42)));
    let rendered = format!("{builder:?}");
    assert!(
        rendered.contains("EngineBuilder"),
        "Debug must name the type; got {rendered}"
    );
}

/// `cuda_graph_enabled` and `capture_cuda_graphs` must reflect the installed
/// executor, not a constant.
///
/// `cargo mutants` replaced both with `Ok(())` / `false` / `true` — five
/// survivors in `cuda_graph.rs`. `cuda_graph_enabled` is what the run loop
/// consults on EVERY step to choose between `step_with_graph` and `step`, so a
/// constant `false` silently disables the fast path for every deployment and a
/// constant `true` sends unmapped shapes into graph replay.
#[test]
#[cfg(all(feature = "cuda-graph", feature = "multi-node"))]
fn cuda_graph_enabled_follows_the_executor() {
    use vllm_traits::{BatchOutput, GraphExecutionError};

    /// Executor whose `is_enabled` is a field, so the accessor can be observed.
    #[derive(Debug)]
    struct ToggleExecutor {
        enabled: bool,
        captures: Arc<std::sync::atomic::AtomicUsize>,
        fail_capture: bool,
    }

    impl vllm_traits::CudaGraphExecutor for ToggleExecutor {
        fn is_enabled(&self) -> bool {
            self.enabled
        }
        fn execute(
            &self,
            _batch: &vllm_traits::Batch,
        ) -> std::result::Result<BatchOutput, GraphExecutionError> {
            Err(GraphExecutionError::GraphNotFound(0))
        }
        fn capture_all_graphs(&mut self) -> std::result::Result<(), GraphExecutionError> {
            self.captures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if self.fail_capture {
                Err(GraphExecutionError::GraphCaptureFailed("synthetic".into()))
            } else {
                Ok(())
            }
        }
    }

    let captures = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // No executor installed: disabled, and capture is a harmless success.
    let mut bare = Engine::new(StubModel::returning(42), None);
    assert!(!bare.cuda_graph_enabled(), "no executor means not enabled");
    bare.capture_cuda_graphs()
        .expect("no executor is not an error");

    // An executor reporting "not captured yet": still disabled.
    let mut pending = EngineBuilder::new(Box::new(StubModel::returning(42)))
        .with_cuda_graph_executor(Box::new(ToggleExecutor {
            enabled: false,
            captures: Arc::clone(&captures),
            fail_capture: false,
        }))
        .build();
    assert!(
        !pending.cuda_graph_enabled(),
        "an executor that has not captured must report false"
    );
    pending.capture_cuda_graphs().expect("capture succeeds");
    assert_eq!(
        captures.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "capture_cuda_graphs must reach the executor"
    );

    // An executor reporting "ready": enabled.
    let ready = EngineBuilder::new(Box::new(StubModel::returning(42)))
        .with_cuda_graph_executor(Box::new(ToggleExecutor {
            enabled: true,
            captures: Arc::clone(&captures),
            fail_capture: false,
        }))
        .build();
    assert!(
        ready.cuda_graph_enabled(),
        "a ready executor must report true, not a constant"
    );

    // A capture failure must surface as an engine error.
    let mut failing = EngineBuilder::new(Box::new(StubModel::returning(42)))
        .with_cuda_graph_executor(Box::new(ToggleExecutor {
            enabled: false,
            captures,
            fail_capture: true,
        }))
        .build();
    assert!(
        failing.capture_cuda_graphs().is_err(),
        "a failed capture must not be swallowed"
    );
}

/// `distributed_kv_enabled` must report presence, not a constant.
///
/// `cargo mutants` replaced the `multi-node` body with the constant `true` and
/// the stub with `true`/`false` — three survivors. The server uses this before
/// calling `distributed_kv_stats`, so a constant `true` makes a single-node
/// engine claim a cache it does not have.
#[test]
#[cfg(feature = "multi-node")]
fn distributed_kv_enabled_reports_presence() {
    let without = Engine::new(StubModel::returning(42), None);
    assert!(
        !without.distributed_kv_enabled(),
        "an engine with no cache must report false, not a constant true"
    );
    assert!(without.distributed_kv_stats().is_none());
    assert!(without.distributed_kv_cache().is_none());

    // With a cache installed, all three accessors must report it. The
    // `cargo mutants` variants that return a constant `None`/`()` survive
    // against an engine that has nothing wired; only a positive case shows
    // that `set_distributed_kv` actually stores what it is given.
    let cache = Arc::new(vllm_dist::DistributedKVCache::new(
        vllm_dist::CacheConfig::new(vllm_dist::distributed_kv::protocol::NodeId(0), 4),
    ));
    let mut with = Engine::new(StubModel::returning(42), None);
    with.set_distributed_kv(Arc::clone(&cache));
    assert!(
        with.distributed_kv_enabled(),
        "set_distributed_kv must flip the flag"
    );
    assert!(
        with.distributed_kv_stats().is_some(),
        "the stats accessor must report the installed cache"
    );
    assert!(
        with.distributed_kv_cache().is_some(),
        "the cache accessor must return the installed cache, not None"
    );
}

#[test]
#[cfg(not(feature = "multi-node"))]
fn distributed_kv_enabled_stub_is_false() {
    let engine = Engine::new(StubModel::returning(42), None);
    assert!(
        !engine.distributed_kv_enabled(),
        "the non-multi-node stub must report false"
    );
}
