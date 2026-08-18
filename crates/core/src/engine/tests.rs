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
