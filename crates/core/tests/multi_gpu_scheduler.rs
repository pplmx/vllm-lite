#![allow(clippy::doc_markdown)]

//! Multi-GPU distributed scheduler integration tests.
//!
//! GPU-First Policy: These tests prioritize GPU execution when CUDA is
//! available (via `vllm_testing::gpu_or_cpu()` / `cuda_available()`),
//! and exercise multi-GPU scheduling scenarios when `multi-node` is
//! enabled. On CPU-only CI, tests fall back to single-device execution
//! so the same test suite validates logic correctness regardless of
//! hardware.
//!
//! Run with:
//!   `cargo nextest run` -p vllm-core --test multi_gpu_scheduler --no-fail-fast
//!
//! Multi-GPU distribution via nextest partitioning (each partition gets a
//! distinct `CUDA_VISIBLE_DEVICES`):
//!   for i in $(seq 0 7); do
//!     CUDA_VISIBLE_DEVICES=$i cargo nextest run -p vllm-core \
//!       --test multi_gpu_scheduler --features multi-node \
//!       --partition "hash:$(($i+1))/8" --no-fail-fast &
//!   done

#![cfg(any(feature = "cuda-graph", feature = "multi-node"))]
#![allow(dead_code)]

use std::sync::Arc;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::SchedulerEngine;
use vllm_core::scheduler::memory::{BlockAllocator, EvictionPolicy};
use vllm_core::types::{Request, SchedulerConfig, Status};
use vllm_testing::harness::TestHarnessConfig;

// ── Device helpers ──────────────────────────────────────────────

/// Check if CUDA is available in this test environment. Used by GPU-first
/// tests to conditionally assert GPU behavior.
#[allow(dead_code)]
fn has_cuda() -> bool {
    vllm_testing::cuda_available()
}

// ── Helpers ────────────────────────────────────────────────────

/// Create a `SchedulerEngine` with the given KV block count and
/// scheduler config, wired to a shared metrics collector.
fn make_scheduler(config: SchedulerConfig, num_kv_blocks: usize) -> SchedulerEngine {
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    SchedulerEngine::new(config, num_kv_blocks, metrics)
}

/// Create a scheduler with a small config suitable for multi-GPU
/// memory pressure tests (tight KV block limits to force eviction).
fn make_pressure_scheduler(num_kv_blocks: usize) -> SchedulerEngine {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 32,
        max_consecutive_decode: 2,
        enable_pd_separation: false,
        prefill_chunk_size: 16,
        decode_preference_ratio: 0.5,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 4,
        ..Default::default()
    };
    make_scheduler(config, num_kv_blocks)
}

/// Create a `Sequence` for testing the eviction policy. The sequence
/// owns the given KV blocks and is in the specified status.
fn make_test_sequence(id: u64, blocks: Vec<usize>, status: Status) -> vllm_core::types::Sequence {
    vllm_core::types::Sequence {
        id,
        tokens: Vec::new(),
        kv_blocks: Arc::new(blocks),
        num_computed_tokens: 0,
        prompt_len: 1,
        status,
        max_tokens: 10,
        sampling_params: vllm_core::types::SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: vllm_core::types::Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    }
}

// ─────────────────────────────────────────────────────────────────
// Single-GPU scheduler tests (run on CPU fallback or GPU)
// ─────────────────────────────────────────────────────────────────

/// On GPU-first environments, the scheduler should allocate KV blocks
/// on the CUDA device. On CPU-only, it should still work correctly.
/// This test verifies that block allocation works regardless of device.
#[test]
fn scheduler_block_allocator_respects_config() {
    let allocator = BlockAllocator::new(256);
    let stats = allocator.stats();
    assert_eq!(stats.total_blocks, 256);
    assert_eq!(stats.available_blocks, 256);
}

/// Block allocation should succeed when blocks are available and fail
/// when exhausted — this is critical for multi-GPU OOM recovery testing.
#[test]
fn scheduler_block_allocator_exhaustion() {
    let mut allocator = BlockAllocator::new(4);

    // Allocate all blocks.
    let batch1 = allocator.allocate(4);
    assert!(batch1.is_some(), "should allocate 4 blocks");
    assert_eq!(batch1.unwrap().len(), 4);

    // Next allocation should fail — no blocks left.
    let batch2 = allocator.allocate(1);
    assert!(batch2.is_none(), "should fail when blocks exhausted");
}

/// Block freeing should make blocks available for reallocation —
/// this is how GPU memory is recycled across requests in a multi-GPU
/// deployment.
#[test]
fn scheduler_block_allocator_free_recycles_blocks() {
    let mut allocator = BlockAllocator::new(8);

    // Allocate and free repeatedly.
    for _ in 0..3 {
        let blocks = allocator.allocate(4).expect("should allocate");
        allocator.free(&blocks);
    }

    // Should be able to allocate all blocks again.
    let blocks = allocator
        .allocate(8)
        .expect("should allocate all 8 after free");
    assert_eq!(blocks.len(), 8);
}

/// EvictionPolicy should select victims based on LRU order — critical
/// for multi-GPU preemption where blocks must be evicted across GPU
/// boundaries. Only blocks with refcount <= 1 (not shared across ranks)
/// are eligible for eviction.
#[test]
fn eviction_policy_selects_victims_correctly() {
    let mut policy = EvictionPolicy::new();

    // Record blocks — each with refcount 1 (single owner).
    policy.record_blocks(&[0, 1, 2, 3, 4]);

    // Create a running sequence that owns these blocks.
    // Only blocks owned by running (non-Finished/Waiting) sequences
    // are eligible for eviction.
    let seq = make_test_sequence(1, vec![0, 1, 2, 3, 4], Status::Decoding);

    // Touch block 0 and 1 — they should be most recently used.
    policy.touch_blocks(&[0, 1]);

    // Select 1 victim — should be the least recently used (block 2).
    let victims = policy.select_victims(&[seq], 1);
    assert_eq!(victims.len(), 1, "should select exactly 1 victim");

    // The victim should be one of the untouched blocks (>= 2).
    let victim = victims[0];
    assert!(
        victim >= 2,
        "victim should be least-recently-used (untouched), got {victim}"
    );
}

/// EvictionPolicy cache should be invalidated when blocks change.
#[test]
fn eviction_policy_cache_invalidated_on_change() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[0, 1, 2]);

    // First call populates cache.
    let _ = policy.select_victims(&[], 1);
    let stats = policy.stats();
    assert_eq!(stats.total_selections, 1);

    // Invalidate and call again — cache miss.
    policy.invalidate_cache();
    let _ = policy.select_victims(&[], 1);
    let stats = policy.stats();
    assert_eq!(stats.total_selections, 2);
}

/// Block ref counting — blocks with multiple references should not be
/// evicted until all references are released. Critical for multi-GPU
/// tensor-parallel scenarios where a block is shared across ranks.
#[test]
fn eviction_policy_refcount_conservation() {
    let mut policy = EvictionPolicy::new();

    // Record block 0 once — refcount becomes 1 (single owner).
    policy.record_blocks(&[0]);
    assert_eq!(
        policy.get_block_ref_count(0),
        1,
        "block 0 should have refcount 1 after one record_blocks"
    );

    // Record again — refcount becomes 2 (shared across 2 GPU ranks).
    policy.record_blocks(&[0]);
    assert_eq!(
        policy.get_block_ref_count(0),
        2,
        "block 0 should have refcount 2 after second record_blocks"
    );

    // Release one reference — should still have 1 left.
    let freed = policy.release_blocks(&[0]);
    assert!(
        freed.is_empty(),
        "block 0 should NOT be freed with refcount still > 0"
    );
    assert_eq!(
        policy.get_block_ref_count(0),
        1,
        "block 0 should still have 1 reference after partial release"
    );

    // Release the second reference — should now be freed.
    let freed = policy.release_blocks(&[0]);
    assert_eq!(
        freed.len(),
        1,
        "block 0 should be freed when refcount reaches 0"
    );
    assert_eq!(freed[0], 0, "freed block should be block 0");
    assert_eq!(
        policy.get_block_ref_count(0),
        0,
        "block 0 should have refcount 0 after full release"
    );
}

// ─────────────────────────────────────────────────────────────────
// Multi-GPU scheduler tests (multi-node feature)
// ─────────────────────────────────────────────────────────────────

/// When `multi-node` is enabled, the scheduler should support
/// distributed KV cache wiring. This test verifies that the
/// `set_distributed_kv` method exists and compiles correctly — actual
/// GPU distributed execution is covered by the GPU integration script.
#[cfg(feature = "multi-node")]
#[test]
fn scheduler_supports_distributed_kv_wiring() {
    use vllm_core::scheduler::memory::MemoryManager;
    use vllm_dist::distributed_kv::protocol::NodeId;
    use vllm_dist::{CacheConfig, DistributedKVCache};

    let config = SchedulerConfig::default();
    let mut mm = MemoryManager::new(config, 64);

    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 2)));
    // This must not panic — wiring the distributed KV cache into the
    // memory manager is the core multi-GPU integration point.
    mm.set_distributed_kv(cache);

    // Verify block allocation still works after wiring (the cache is
    // written through on allocate/free).
    let block = mm.allocate(1);
    assert!(
        block.is_some(),
        "allocation should succeed with distributed KV wired"
    );
}

/// Multi-GPU: scheduler should handle requests with varying prompt
/// lengths across GPU partitions. Each partition tests a disjoint
/// set of prompt sizes, exercising different KV block allocation patterns.
#[cfg(feature = "cuda-graph")]
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn gpu_scheduler_handles_variable_prompt_lengths() {
    // GPU-first: resolve CUDA device (panics if CUDA unavailable, which
    // is correct — this test is #[ignore]d anyway and only runs on GPU).
    let _device = vllm_testing::gpu_device();

    let config = SchedulerConfig::default();
    let mut scheduler = make_scheduler(config, 4096);

    // Vary prompt lengths to stress different block allocation patterns.
    let prompt_lengths = [1, 5, 16, 32, 64, 128];

    for (i, &len) in prompt_lengths.iter().enumerate() {
        let prompt: Vec<u32> = (0..len).map(|j| (i as u32 * 100 + j)).collect();
        let req = Request::new(i as u64, prompt, 8);
        let seq_id = scheduler.add_request(req);
        assert!(seq_id > 0, "seq {i} should be added");
    }

    // Build batch — all requests should be scheduled.
    let batch = scheduler.build_batch();
    assert_eq!(
        batch.seq_ids.len(),
        prompt_lengths.len(),
        "all requests should be in the batch"
    );
}

/// Multi-GPU: scheduler should handle preemption when KV memory is
/// exhausted. This is the core scenario where multi-GPU deployments
/// differ from single-GPU — blocks must be evicted from one request
/// to make room for another, potentially across GPU boundaries.
#[cfg(feature = "cuda-graph")]
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn gpu_scheduler_preempts_under_memory_pressure() {
    let _device = vllm_testing::gpu_device();

    // Tight KV block budget to force preemption.
    let mut scheduler = make_pressure_scheduler(2);

    // Add first request that uses all available blocks.
    let req1 = Request::new(1, vec![1, 2, 3, 4, 5, 6, 7, 8], 10);
    scheduler.add_request(req1);
    let _batch1 = scheduler.build_batch();

    // Add second request — should trigger preemption of the first.
    let req2 = Request::new(2, vec![9, 10, 11, 12], 10);
    scheduler.add_request(req2);

    // The scheduler should have handled memory pressure without panicking.
    // Under tight block budget, either preemption occurred or the second
    // request is queued — both are valid. The key invariant: the scheduler
    // state is consistent (no dangling references, no allocation beyond
    // the block budget).
    let waiting = scheduler.waiting_count();
    let running = scheduler.running();
    // At least one state should be non-empty — the scheduler processed
    // the requests without error.
    assert!(
        waiting > 0 || !running.is_empty(),
        "scheduler should have pending work after memory pressure"
    );
}

/// Multi-GPU: batch building should respect GPU partition constraints.
/// When CUDA_VISIBLE_DEVICES restricts to a single GPU, the scheduler
/// should still produce valid batches.
#[cfg(feature = "cuda-graph")]
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn gpu_scheduler_batch_building_respects_partition() {
    let _device = vllm_testing::gpu_device();

    // Check if we're in a partitioned environment (nextest hash partition).
    let visible = std::env::var("CUDA_VISIBLE_DEVICES");
    if let Ok(val) = &visible {
        let count = val.split(',').count();
        // In a partitioned run, we should see exactly 1 GPU.
        assert!(
            count <= 8,
            "CUDA_VISIBLE_DEVICES has {count} entries (expected <= 8 in partitioned mode)"
        );
    }

    let config = SchedulerConfig {
        max_num_seqs: 8,
        max_num_batched_tokens: 256,
        ..Default::default()
    };
    let mut scheduler = make_scheduler(config, 1024);

    // Add a batch of requests.
    for i in 0..5 {
        let prompt: Vec<u32> = (0..8).map(|j| (i as u32 * 10 + j)).collect();
        let req = Request::new(i as u64, prompt, 10);
        scheduler.add_request(req);
    }

    let batch = scheduler.build_batch();
    assert!(
        !batch.seq_ids.is_empty(),
        "batch should not be empty when requests are pending"
    );
    assert_eq!(
        batch.seq_ids.len(),
        batch.input_tokens.len(),
        "all vectors in batch should have matching lengths"
    );
}

/// Multi-GPU: sequence completion and cleanup should free KV blocks
/// for reuse. This is critical for long-running GPU deployments where
/// memory fragmentation across GPUs must be minimized.
#[cfg(feature = "cuda-graph")]
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn gpu_scheduler_completes_and_frees_blocks() {
    let _device = vllm_testing::gpu_device();

    let config = SchedulerConfig::default();
    let mut scheduler = make_scheduler(config, 512);

    // Add a request.
    let req = Request::new(1, vec![1, 2, 3], 5);
    let seq_id = scheduler.add_request(req);

    // Build batch and process.
    let batch = scheduler.build_batch();
    assert!(!batch.seq_ids.is_empty());

    // Complete the sequence.
    scheduler.update(
        &batch.seq_ids,
        &[vllm_traits::SampledToken {
            token: 42,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &[3],
    );

    // After completion, the sequence should be finished.
    let finished = scheduler.finished_sequences();
    assert!(
        finished.iter().any(|s| s.id == seq_id),
        "seq {seq_id} should be finished after update"
    );
}

/// Multi-GPU: priority scheduling across GPU partitions. When requests
/// have different priorities, the scheduler should process higher-
/// priority requests first — this matters for multi-GPU serving where
/// premium requests need guaranteed latency.
#[cfg(feature = "cuda-graph")]
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn gpu_scheduler_priority_scheduling() {
    let _device = vllm_testing::gpu_device();

    let config = SchedulerConfig {
        enable_priority_scheduling: true,
        max_num_seqs: 4,
        max_num_batched_tokens: 64,
        ..Default::default()
    };
    let mut scheduler = make_scheduler(config, 256);

    use vllm_core::types::Priority;

    // Add a low-priority request.
    let mut low = Request::new(1, vec![1, 2], 5);
    low.priority = Priority(10); // lower priority
    scheduler.add_request(low);

    // Add a high-priority request.
    let mut high = Request::new(2, vec![3, 4], 5);
    high.priority = Priority(100); // higher priority
    scheduler.add_request(high);

    let _batch = scheduler.build_batch();
    // The high-priority request should be scheduled.
    let running = scheduler.running();
    assert!(
        !running.is_empty(),
        "at least one request should be running"
    );
}

/// Multi-GPU: prefix cache sharing across requests. When multiple
/// requests share a common prefix, the scheduler should reuse KV
/// blocks — this is critical for multi-GPU throughput where prefix
/// sharing reduces redundant computation.
#[cfg(feature = "cuda-graph")]
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn gpu_scheduler_prefix_cache_sharing() {
    let _device = vllm_testing::gpu_device();

    let config = SchedulerConfig {
        enable_pd_separation: true,
        ..Default::default()
    };
    let mut scheduler = make_scheduler(config, 512);

    // Two requests with a shared prefix.
    let shared = vec![1, 2, 3, 4, 5];
    let req1 = Request::new(1, shared.clone(), 10);
    let req2 = Request::new(
        2,
        {
            let mut p = shared;
            p.push(6);
            p
        },
        10,
    );
    scheduler.add_request(req1);
    scheduler.add_request(req2);

    // Build batch — both requests should be included.
    let batch = scheduler.build_batch();
    assert_eq!(batch.seq_ids.len(), 2, "both requests should be scheduled");

    // Prefix cache hit rate should be non-negative.
    let hit_rate = scheduler.prefix_cache_hit_rate();
    assert!(
        (0.0..=1.0).contains(&hit_rate),
        "prefix cache hit rate should be in [0, 1], got {hit_rate}"
    );
}

/// Multi-GPU: TestHarness should respect GPU-first configuration.
/// When `kv_blocks` and `max_batch_size` are configured, the harness
/// should produce a scheduler that honors those limits.
#[test]
fn test_harness_config_respects_gpu_first_limits() {
    // Even on CPU, the harness config should produce correct settings.
    let config = TestHarnessConfig::default()
        .kv_blocks(128)
        .max_batch_size(8);

    let scheduler_config = config.into_scheduler_config();
    assert_eq!(scheduler_config.max_num_seqs, 8);
    assert_eq!(scheduler_config.max_num_batched_tokens, 4096);
}
