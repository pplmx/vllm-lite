//! Unit tests for `BatchCudaGraphExecutor`.
//!
//! Locks in the executor's lifecycle contract on CPU:
//!
//! 1. **Construction & flags**: `is_enabled()` mirrors `config.enabled`;
//!    `config()` accessor round-trips; default is disabled.
//! 2. **Capture / lookup**: `has_graph(n)` is `false` until
//!    `capture_graph_for_batch_size(n)`; after capture
//!    `lookup_graph(n, n)` returns `Ok`. `lookup_graph` for a
//!    missing key returns `Err(GraphNotFound(n))` (typed error,
//!    not a panic).
//! 3. **Execution gate**: `execute(&batch)` for an unknown batch
//!    size surfaces `Err(GraphNotFound(batch_size))` instead of
//!    dispatching to a real CUDA kernel.
//! 4. **Pool & stats**: `stats()` initializes to all-zero; the
//!    `cache_hit_rate()` accessor returns ~0.
//! 5. **Mutation**: `clear()` drops all captured graphs;
//!    `invalidate(n)` drops a single one; `warmup(&[sizes])`
//!    captures the listed sizes.
//!
//! All tests run on CPU — they only exercise the executor's
//! metadata + lookup paths, never an actual CUDA graph launch.
use super::*;
use vllm_traits::{Batch, BatchPhase, SamplingParams};

fn create_mock_batch(batch_size: usize) -> Batch {
    Batch {
        seq_ids: (0..batch_size as u64).collect(),
        input_tokens: vec![vec![1u32]; batch_size],
        positions: vec![vec![0usize]; batch_size],
        kv_block_ids: vec![vec![]; batch_size],
        num_computed_tokens: vec![0; batch_size],
        is_prefill: vec![false; batch_size],
        sampling_params: vec![SamplingParams::default(); batch_size],
        phase: BatchPhase::Decode,
        total_tokens: batch_size,
        max_seq_len: 1,
    }
}

#[test]
fn test_config_accessor() {
    let config = CudaGraphConfig {
        enabled: true,
        batch_sizes: vec![1, 2, 4],
        ..Default::default()
    };
    let executor = BatchCudaGraphExecutor::new(config.clone()).unwrap();
    assert_eq!(executor.config().batch_sizes, config.batch_sizes);
}

#[test]
fn test_executor_disabled_when_config_disabled() {
    let config = CudaGraphConfig {
        enabled: false,
        ..Default::default()
    };
    let executor = BatchCudaGraphExecutor::new(config).unwrap();
    assert!(!executor.is_enabled());
}

#[test]
fn test_executor_enabled_when_config_enabled() {
    let config = CudaGraphConfig {
        enabled: true,
        ..Default::default()
    };
    let executor = BatchCudaGraphExecutor::new(config).unwrap();
    assert!(executor.is_enabled());
}

#[test]
fn test_has_graph_returns_false_for_empty_executor() {
    let config = CudaGraphConfig::default();
    let executor = BatchCudaGraphExecutor::new(config).unwrap();
    assert!(!executor.has_graph(1));
    assert!(!executor.has_graph(4));
}

#[test]
fn test_capture_graph_increases_graph_count() {
    let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig::default()).unwrap();
    assert_eq!(executor.graph_count(), 0);

    executor.capture_graph_for_batch_size(1).unwrap();
    assert_eq!(executor.graph_count(), 1);
    assert!(executor.has_graph(1));
}

#[test]
fn test_execute_returns_error_for_unknown_batch_size() {
    // Create config with enabled=true so we can test graph not found
    let config = CudaGraphConfig {
        enabled: true,
        ..Default::default()
    };
    let executor = BatchCudaGraphExecutor::new(config).unwrap();
    let batch = create_mock_batch(2);
    let result = executor.execute(&batch);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        GraphExecutionError::GraphNotFound(2)
    ));
}

#[test]
fn test_lookup_graph_returns_not_found_for_missing_key() {
    // Verifies the unwrap-to-error conversion at the post-find_graph_key
    // lookup site. In normal operation this branch is unreachable because
    // find_graph_key checks contains_key before returning Some, but the
    // typed error protects against future refactors that weaken that
    // invariant or against logic errors where a graph is invalidated
    // between the key lookup and the actual retrieval.
    let config = CudaGraphConfig {
        enabled: true,
        ..Default::default()
    };
    let executor = BatchCudaGraphExecutor::new(config).unwrap();
    let result = executor.lookup_graph(999, 999);
    assert!(matches!(
        result,
        Err(GraphExecutionError::GraphNotFound(999))
    ));
}

#[test]
fn test_lookup_graph_returns_graph_for_captured_key() {
    // Positive control: verifies lookup_graph returns the captured graph
    // when the key is present, so the typed-error test above cannot be
    // passing because of an unrelated always-Err implementation.
    let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig::default()).unwrap();
    executor.capture_graph_for_batch_size(4).unwrap();
    let result = executor.lookup_graph(4, 4);
    assert!(result.is_ok());
}

#[test]
fn test_graph_pooling() {
    let config = CudaGraphConfig {
        enabled: true,
        batch_sizes: vec![1, 2, 3, 4, 8],
        enable_graph_pooling: Some(true),
        ..Default::default()
    };
    let executor = BatchCudaGraphExecutor::new(config).unwrap();
    let stats = executor.stats();

    assert_eq!(stats.total_executions, 0);
    assert_eq!(stats.cached_graphs, 0);
}

#[test]
fn test_stats_initialization() {
    let config = CudaGraphConfig::default();
    let executor = BatchCudaGraphExecutor::new(config).unwrap();
    let stats = executor.stats();

    assert_eq!(stats.total_executions, 0);
    assert_eq!(stats.cache_hits, 0);
    assert_eq!(stats.cached_graphs, 0);
    assert!(stats.cache_hit_rate().abs() < 1e-6);
}

#[test]
fn test_clear_graphs() {
    let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig::default()).unwrap();
    executor.capture_graph_for_batch_size(1).unwrap();
    executor.capture_graph_for_batch_size(2).unwrap();
    assert_eq!(executor.graph_count(), 2);

    executor.clear();
    assert_eq!(executor.graph_count(), 0);
}

#[test]
fn test_invalidate_graph() {
    let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig::default()).unwrap();
    executor.capture_graph_for_batch_size(4).unwrap();
    assert!(executor.has_graph(4));

    executor.invalidate(4);
    assert!(!executor.has_graph(4));
}

#[test]
fn test_warmup() {
    let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig {
        enabled: true,
        batch_sizes: vec![1, 2, 4, 8],
        ..Default::default()
    })
    .unwrap();

    executor.warmup(&[1, 4, 8]).unwrap();

    assert!(executor.has_graph(1));
    assert!(executor.has_graph(4));
    assert!(executor.has_graph(8));
}

#[test]
fn test_trait_dispatch_via_cuda_graph_executor() {
    // The engine stores `Box<dyn CudaGraphExecutor + Send>`, so every
    // production call goes through the trait, not the inherent methods.
    // This test exercises the trait surface end-to-end to make sure the
    // dispatch wiring is correct.
    use vllm_traits::CudaGraphExecutor;

    let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig {
        enabled: true,
        batch_sizes: vec![1, 2],
        ..Default::default()
    })
    .unwrap();
    let boxed: Box<dyn CudaGraphExecutor + Send> = Box::new(executor);

    // `is_enabled` reflects the config flag.
    assert!(boxed.is_enabled());

    // `capture_all_graphs` succeeds and is idempotent.
    let mut owned = boxed;
    owned.capture_all_graphs().expect("capture should succeed");
    owned
        .capture_all_graphs()
        .expect("re-capture should also succeed");

    // `execute` for an uncaptured batch size returns the typed error.
    let batch = create_mock_batch(16);
    let err = owned
        .execute(&batch)
        .expect_err("execute for size 16 must miss the captured pool");
    assert!(matches!(
        err,
        vllm_traits::GraphExecutionError::GraphNotFound(16)
    ));
}
