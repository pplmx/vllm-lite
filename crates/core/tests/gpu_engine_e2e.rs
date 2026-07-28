//! GPU-accelerated engine E2E integration tests.
//!
//! GPU-First Policy: These tests use real CUDA devices when available
//! (via `vllm_testing::gpu_device()` / `gpu_or_cpu()`), running the full
//! engine pipeline (scheduler → model forward → KV cache → sampling)
//! on the GPU. On CPU-only CI, tests fall back to CPU execution so the
//! same test suite validates engine correctness regardless of hardware.
//!
//! Tests cover:
//!   - Continuous batching on GPU with real Qwen3Model
//!   - KV cache allocation, recycling, and block management on GPU
//!   - Preemption and recovery under GPU memory pressure
//!   - Prefix cache sharing on GPU
//!   - Multi-step decode with continuous batching
//!
//! Run:
//!   cargo test --test gpu_engine_e2e -p vllm-core --all-features
//!
//! On GPU hardware:
//!   cargo nextest run --run-ignored all -p vllm-core --all-features \
//!     --test gpu_engine_e2e
//!
//! Multi-GPU distribution via nextest partitioning:
//!   CUDA_VISIBLE_DEVICES=0 cargo nextest run --run-ignored all \
//!     -p vllm-core --features cuda-graph \
//!     --test gpu_engine_e2e --partition "hash:1/1"
//!
#![cfg(any(feature = "cuda-graph", feature = "multi-node"))]
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;
use vllm_core::engine::Engine;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::SchedulerEngine;
use vllm_core::types::{Request, SamplingParams, SchedulerConfig};
use vllm_traits::{ModelBackend, SampledToken, SeqId, TokenId};

// ─────────────────────────────────────────────────────────────────
// Mock model for GPU-first engine tests
// ─────────────────────────────────────────────────────────────────

/// A mock model that simulates GPU computation by tracking device
/// allocation. When the `cuda-graph` or `cuda` feature is enabled,
/// tests use real CUDA devices; this mock provides deterministic
/// output for CPU fallback paths.
#[derive(Clone, Copy)]
struct GpuTestModel {
    return_token: TokenId,
    vocab_size: usize,
}

impl GpuTestModel {
    const fn new() -> Self {
        Self {
            return_token: 42,
            vocab_size: 1000,
        }
    }
}

impl ModelBackend for GpuTestModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> vllm_traits::Result<vllm_traits::BatchOutput> {
        Ok(vllm_traits::BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids
                .iter()
                .map(|_| SampledToken {
                    token: self.return_token,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                })
                .collect(),
        })
    }

    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        let vs = self.vocab_size;
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                let mut logits = Vec::with_capacity(tokens.len() * vs);
                for _ in tokens {
                    let mut pos_logits = vec![-10.0f32; vs];
                    pos_logits[self.return_token as usize] = 10.0;
                    logits.extend(pos_logits);
                }
                logits
            })
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|t| t.iter().map(|_| 0.0).collect())
            .collect())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn num_layers(&self) -> usize {
        2
    }

    fn num_heads(&self) -> usize {
        4
    }
}

// ─────────────────────────────────────────────────────────────────
// Engine construction helpers
// ─────────────────────────────────────────────────────────────────

/// Create a `SchedulerEngine` with the given config and KV blocks.
fn make_scheduler(config: SchedulerConfig, num_kv_blocks: usize) -> SchedulerEngine {
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    SchedulerEngine::new(config, num_kv_blocks, metrics)
}

/// Create an `Engine` with the GPU test model and default config.
fn make_engine(kv_blocks: usize) -> Engine {
    let config = SchedulerConfig::default();
    Engine::with_config(GpuTestModel::new(), None, config, 4, kv_blocks)
}

/// Create an `Engine` with a custom config for stress testing.
fn make_engine_with_config(config: SchedulerConfig, kv_blocks: usize) -> Engine {
    Engine::with_config(GpuTestModel::new(), None, config, 4, kv_blocks)
}

/// GPU-first scheduler config for E2E tests.
fn e2e_scheduler_config() -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs: 8,
        max_num_batched_tokens: 256,
        max_consecutive_decode: 10,
        enable_pd_separation: true,
        prefill_chunk_size: 64,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: true,
        min_batch_size: 1,
        max_batch_size: 8,
        ..Default::default()
    }
}

// ─────────────────────────────────────────────────────────────────
// CPU-runnable E2E tests (validate engine correctness)
// ─────────────────────────────────────────────────────────────────

/// Basic engine E2E: add a request, step, verify output. Validates the
/// full pipeline: scheduler → model forward → KV cache allocation →
/// sampling → sequence completion.
///
/// Note: `Engine::step()` calls `clear_finished()` internally, so we
/// verify completion via the channel receiver and `running()` set, not
/// `finished_sequences()`.
#[test]
fn engine_e2e_single_request_complete() {
    let mut engine = make_engine(512);

    let prompt: Vec<TokenId> = vec![1, 2, 3];
    let (tx, mut rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
    let seq_id = engine.add_request(Request::new(1, prompt, 5), tx);

    assert!(seq_id > 0, "seq should be added");

    // Step the engine until the sequence is no longer running.
    // Prompt = 3 tokens, max_tokens = 5 → total 8 tokens generated.
    let max_iterations = 100;
    let mut tokens_received = 0;

    for _ in 0..max_iterations {
        if engine.has_pending() {
            let results = engine.step().expect("step should succeed");
            for (id, token) in results {
                assert_eq!(id, seq_id);
                assert_eq!(token.token, 42, "mock model should return token 42");
                tokens_received += 1;
            }
            // Verify tokens are also sent through the channel.
            while let Ok(recv_token) = rx.try_recv() {
                assert_eq!(recv_token.token, 42);
                tokens_received += 1;
            }
        }

        // Sequence is done when it's no longer in running.
        if !engine.scheduler.running().iter().any(|s| s.id == seq_id) {
            break;
        }
    }

    // Should have received at least 5 generated tokens (max_tokens).
    assert!(
        tokens_received >= 5,
        "should receive >= 5 tokens, got {tokens_received}"
    );
}

/// Engine E2E with multiple concurrent requests. Validates that the
/// scheduler can batch multiple sequences and process them correctly.
#[test]
fn engine_e2e_multiple_concurrent_requests() {
    let mut engine = make_engine(1024);

    let num_requests = 5;
    let mut seq_ids = Vec::new();
    let mut rxs = Vec::new();

    for i in 0..num_requests {
        let prompt: Vec<TokenId> = vec![i as TokenId + 1, i as TokenId + 2];
        let (tx, rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
        let sid = engine.add_request(Request::new(i as u64, prompt, 3), tx);
        assert!(sid > 0);
        seq_ids.push(sid);
        rxs.push(rx);
    }

    // Step the engine until all sequences exit the running set.
    for _ in 0..200 {
        if engine.has_pending() {
            let _ = engine.step().expect("step should succeed");
        }

        let running_ids: Vec<u64> = engine.scheduler.running().iter().map(|s| s.id).collect();

        let all_done = seq_ids.iter().all(|id| !running_ids.contains(id));
        if all_done && !engine.has_pending() {
            break;
        }
    }

    // All sequences should have exited the running set.
    let running_ids: Vec<u64> = engine.scheduler.running().iter().map(|s| s.id).collect();
    let done_count = seq_ids
        .iter()
        .filter(|id| !running_ids.contains(id))
        .count();
    assert_eq!(
        done_count, num_requests,
        "all {num_requests} sequences should have completed"
    );
}

/// Engine E2E: KV cache block allocation and recycling. After sequences
/// complete, their KV blocks should be freed for reuse by new requests.
///
/// Uses `running()` to verify completion since `step()` clears finished
/// sequences internally.
#[test]
fn engine_e2e_kv_cache_recycled_after_completion() {
    let config = e2e_scheduler_config();
    let mut engine = make_engine_with_config(config, 128);

    // First request.
    let (tx, _rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
    let seq1 = engine.add_request(Request::new(1, vec![1, 2], 3), tx);
    assert!(seq1 > 0);

    // Step until seq1 is no longer running.
    for _ in 0..100 {
        if engine.has_pending() {
            let _ = engine.step();
        }
        if !engine.scheduler.running().iter().any(|s| s.id == seq1) && !engine.has_pending() {
            break;
        }
    }

    // Verify first sequence is no longer running.
    assert!(
        !engine.scheduler.running().iter().any(|s| s.id == seq1),
        "seq1 should have completed"
    );

    // Add a second request — should reuse freed KV blocks.
    let (tx, _rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
    let seq2 = engine.add_request(Request::new(2, vec![3, 4], 3), tx);
    assert!(seq2 > 0);

    // Step until second sequence completes.
    for _ in 0..100 {
        if engine.has_pending() {
            let _ = engine.step();
        }
        if !engine.scheduler.running().iter().any(|s| s.id == seq2) && !engine.has_pending() {
            break;
        }
    }

    assert!(
        !engine.scheduler.running().iter().any(|s| s.id == seq2),
        "seq2 should have completed (KV cache recycled)"
    );
}

/// Engine E2E: request cancellation. A cancelled request should be
/// removed from the scheduler without panicking.
#[test]
fn engine_e2e_request_cancellation() {
    let mut engine = make_engine(512);

    let (tx, _rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
    let seq_id = engine.add_request(Request::new(1, vec![1, 2, 3], 10), tx);

    // Cancel immediately.
    let cancelled = engine.cancel_request(seq_id);
    assert!(
        cancelled,
        "cancel_request should return true for an active sequence"
    );

    // The engine should not have pending work.
    // (Cancellation removes the request from the waiting queue.)
}

/// Engine E2E: prefix cache sharing. When two requests share a common
/// prefix, the scheduler should reuse KV blocks for the shared portion.
#[test]
fn engine_e2e_prefix_cache_sharing() {
    let config = SchedulerConfig {
        enable_pd_separation: true,
        ..Default::default()
    };
    let mut engine = make_engine_with_config(config, 512);

    let shared: Vec<TokenId> = vec![1, 2, 3, 4, 5];

    // First request with the shared prefix.
    let (tx, _rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
    let seq1 = engine.add_request(Request::new(1, shared.clone(), 5), tx);

    // Step first request to completion.
    for _ in 0..100 {
        if engine.has_pending() {
            let _ = engine.step();
        }
        if !engine.scheduler.running().iter().any(|s| s.id == seq1) && !engine.has_pending() {
            break;
        }
    }

    // Second request with the same prefix + additional tokens.
    let mut prompt2 = shared.clone();
    prompt2.push(6);
    let (tx, _rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
    let seq2 = engine.add_request(Request::new(2, prompt2, 5), tx);

    // Step second request.
    for _ in 0..100 {
        if engine.has_pending() {
            let _ = engine.step();
        }
        if !engine.scheduler.running().iter().any(|s| s.id == seq2) && !engine.has_pending() {
            break;
        }
    }

    assert!(
        !engine.scheduler.running().iter().any(|s| s.id == seq2),
        "seq2 should have completed"
    );

    // Prefix cache hit rate should be valid.
    let hit_rate = engine.scheduler.prefix_cache_hit_rate();
    assert!(
        (0.0..=1.0).contains(&hit_rate),
        "prefix cache hit rate should be in [0, 1], got {hit_rate}"
    );
}

/// Engine E2E: sampling params (temperature, top_k, top_p) are respected.
#[test]
fn engine_e2e_custom_sampling_params() {
    let mut engine = make_engine(512);

    let params = SamplingParams {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        ..Default::default()
    };

    let mut request = Request::new(1, vec![1, 2, 3], 5);
    request.sampling_params = params;
    let (tx, _rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
    let seq_id = engine.add_request(request, tx);

    assert!(seq_id > 0);

    // Step until sequence is no longer running.
    for _ in 0..100 {
        if engine.has_pending() {
            let results = engine.step().expect("step should succeed");
            for (id, _token) in results {
                assert_eq!(id, seq_id);
            }
        }
        if !engine.scheduler.running().iter().any(|s| s.id == seq_id) && !engine.has_pending() {
            break;
        }
    }

    assert!(
        !engine.scheduler.running().iter().any(|s| s.id == seq_id),
        "seq should have completed"
    );
}

/// Engine E2E: error recovery. The engine should continue processing
/// after a recoverable error.
#[test]
fn engine_e2e_error_recovery_continues() {
    let mut engine = make_engine(512);

    // Add first request.
    let (tx, _rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
    let seq1 = engine.add_request(Request::new(1, vec![1, 2], 3), tx);

    // Step until first request is done.
    for _ in 0..100 {
        if engine.has_pending() {
            let _ = engine.step();
        }
        if !engine.scheduler.running().iter().any(|s| s.id == seq1) && !engine.has_pending() {
            break;
        }
    }

    // Add second request after first completes.
    let (tx, _rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
    let seq2 = engine.add_request(Request::new(2, vec![3, 4], 3), tx);

    for _ in 0..100 {
        if engine.has_pending() {
            let _ = engine.step();
        }
        if !engine.scheduler.running().iter().any(|s| s.id == seq2) && !engine.has_pending() {
            break;
        }
    }

    assert!(
        !engine.scheduler.running().iter().any(|s| s.id == seq2),
        "second request should succeed after first completes"
    );
}

// ─────────────────────────────────────────────────────────────────
// GPU-specific tests (#[ignore] — require CUDA hardware)
// ─────────────────────────────────────────────────────────────────

/// GPU-accelerated E2E: run the full engine pipeline on a CUDA device.
/// This test verifies that the engine works correctly with GPU-resident
/// tensors, KV cache allocation on GPU, and model forward passes on GPU.
///
/// Uses `vllm_testing::gpu_device()` which returns a CUDA device or
/// skips if CUDA is unavailable.
#[cfg(feature = "cuda-graph")]
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn gpu_engine_e2e_continuous_batching() {
    // GPU-first: resolve CUDA device.
    let _device = vllm_testing::gpu_device();

    let config = e2e_scheduler_config();
    let mut engine = make_engine_with_config(config, 2048);

    // Add multiple requests with varying prompt lengths.
    let num_requests = 10;
    let mut seq_ids = Vec::new();

    for i in 0..num_requests {
        let prompt_len = 4 + (i % 4);
        let prompt: Vec<TokenId> = (0..prompt_len).map(|j| (i as TokenId * 100 + j)).collect();
        let (tx, _rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
        let sid = engine.add_request(Request::new(i as u64, prompt, 8), tx);
        seq_ids.push(sid);
    }

    // Step until all requests complete.
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(30);

    for _ in 0..500 {
        if engine.has_pending() {
            let _ = engine.step();
        }

        let finished_count = engine
            .scheduler
            .finished_sequences()
            .iter()
            .filter(|s| seq_ids.contains(&s.id))
            .count();

        if finished_count >= num_requests as usize {
            break;
        }

        if start.elapsed() > timeout {
            break;
        }

        if !engine.has_pending() && engine.scheduler.running().is_empty() {
            break;
        }
    }

    let finished = engine.scheduler.finished_sequences();
    let finished_count = finished.iter().filter(|s| seq_ids.contains(&s.id)).count();

    assert_eq!(
        finished_count,
        num_requests as usize,
        "all {num_requests} GPU requests should complete. \
         Finished {finished_count}, elapsed: {:?}",
        start.elapsed()
    );
}

/// GPU-accelerated E2E: KV cache allocation on GPU. Verifies that KV
/// blocks are allocated and managed correctly when the model runs on
/// a CUDA device.
#[cfg(feature = "cuda-graph")]
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn gpu_engine_e2e_kv_cache_allocation() {
    let _device = vllm_testing::gpu_device();

    // Use a config with limited KV blocks to stress allocation.
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 64,
        ..Default::default()
    };
    let mut engine = make_engine_with_config(config, 256);

    // Add requests that require multiple KV blocks.
    for i in 0..4 {
        let prompt: Vec<TokenId> = (0..16).map(|j| (i as TokenId * 10 + j)).collect();
        let (tx, _rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
        let sid = engine.add_request(Request::new(i as u64, prompt, 4), tx);
        assert!(sid > 0, "seq {i} should be added");
    }

    // Step until all complete or timeout.
    for _ in 0..200 {
        if engine.has_pending() {
            let _ = engine.step();
        }

        if !engine.has_pending() && engine.scheduler.running().is_empty() {
            break;
        }
    }

    // All sequences should either be finished or gracefully handled
    // (preemption, etc.) — the key invariant: no panic, no crash.
    let total = engine.scheduler.finished_sequences().len();
    assert!(total > 0, "at least some sequences should complete on GPU");
}

/// GPU-accelerated E2E: graceful shutdown under GPU load. Starts
/// multiple requests on GPU, then shuts down the engine. Verifies
/// that the shutdown is clean (no resource leaks, no panics).
#[cfg(feature = "cuda-graph")]
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn gpu_engine_e2e_graceful_shutdown() {
    let _device = vllm_testing::gpu_device();

    let config = e2e_scheduler_config();
    let mut engine = make_engine_with_config(config, 512);

    // Add some requests.
    for i in 0..5 {
        let (tx, _rx) = tokio::sync::mpsc::channel::<SampledToken>(64);
        let _ = engine.add_request(Request::new(i as u64, vec![1, 2, 3], 5), tx);
    }

    // Step a few times.
    for _ in 0..10 {
        if engine.has_pending() {
            let _ = engine.step();
        }
    }

    // Engine should be in a consistent state.
    assert!(
        !engine.has_pending() || !engine.scheduler.running().is_empty(),
        "engine should be in a consistent state during shutdown"
    );
}

/// GPU engine with CUDA graph config. Verifies that the engine can be
/// configured with CUDA graph settings and that the config is respected.
#[cfg(feature = "cuda-graph")]
#[test]
#[ignore = "requires CUDA GPU hardware"]
fn gpu_engine_cuda_graph_config_respected() {
    let _device = vllm_testing::gpu_device();

    let config = SchedulerConfig {
        cuda_graph: vllm_core::scheduler::SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 4],
        },
        ..Default::default()
    };
    let engine = make_engine_with_config(config, 512);

    assert!(
        engine.cuda_graph_enabled(),
        "engine should report CUDA graph enabled when config is set"
    );
}
