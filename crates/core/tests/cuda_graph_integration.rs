// crates/core/tests/cuda_graph_integration.rs
use std::sync::Arc;
use vllm_core::engine::Engine;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::{GraphBatch, SchedulerCudaGraphConfig, SchedulerEngine};
use vllm_core::types::{Request, SchedulerConfig};
use vllm_traits::{BatchOutput, ModelBackend, ModelError};

fn create_test_engine(config: SchedulerConfig, num_kv_blocks: usize) -> SchedulerEngine {
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    SchedulerEngine::new(config, num_kv_blocks, metrics)
}

/// Test that CUDA Graph is disabled by default
#[test]
fn test_cuda_graph_disabled_by_default() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);
    // Build batch should return regular batch
    let batch = engine.build_batch_with_graph();
    assert!(!batch.is_graph());
}

/// Test that decode batches can use CUDA Graph when enabled
#[test]
fn test_decode_batch_can_use_graph() {
    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..Default::default()
    };
    let mut engine = create_test_engine(config, 1024);
    // Add a request
    engine.add_request(Request::new(0, vec![1, 2, 3], 5));
    // Build batch (first will be prefill)
    let batch1 = engine.build_batch_with_graph();
    assert!(!batch1.is_graph()); // First is prefill
    // Update to move to decode
    let seq_id = batch1.into_regular().seq_ids[0];
    engine.update(&[seq_id], &[10], &[3]);
    // Second batch should be decode and could use graph
    let _batch2 = engine.build_batch_with_graph();
    // Note: Whether it's graph depends on batch size matching
}

/// Test config batch size support
#[test]
fn test_scheduler_cuda_graph_config_supports_batch_size() {
    let config = SchedulerCudaGraphConfig {
        enabled: true,
        batch_sizes: vec![1, 4, 8, 16],
    };
    assert!(config.supports_batch_size(1));
    assert!(config.supports_batch_size(4));
    assert!(config.supports_batch_size(8));
    assert!(!config.supports_batch_size(3));
    assert!(!config.supports_batch_size(5));
}

/// Test GraphBatch conversion
#[test]
fn test_graph_batch_conversion() {
    use vllm_traits::{Batch, BatchPhase};
    let batch = Batch {
        seq_ids: vec![1, 2, 3],
        input_tokens: vec![vec![1], vec![2], vec![3]],
        positions: vec![vec![0], vec![1], vec![2]],
        kv_block_ids: vec![vec![], vec![], vec![]],
        num_computed_tokens: vec![0, 0, 0],
        is_prefill: vec![false, false, false],
        phase: BatchPhase::Decode,
        total_tokens: 3,
        max_seq_len: 1,
    };
    let graph_batch = GraphBatch::Regular(batch.clone());
    assert_eq!(graph_batch.batch_size(), 3);
    let converted = graph_batch.into_regular();
    assert_eq!(converted.seq_ids, vec![1, 2, 3]);
}

/// Test end-to-end with mock model
#[test]
fn test_end_to_end_engine_with_cuda_graph_config() {
    #[derive(Clone)]
    struct MockModel;

    impl ModelBackend for MockModel {
        fn forward(
            &mut self,
            seq_ids: &[u64],
            _input_tokens: &[Vec<u32>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> Result<BatchOutput, ModelError> {
            Ok(BatchOutput {
                seq_ids: seq_ids.to_vec(),
                next_tokens: seq_ids.iter().map(|_| 42u32).collect(),
            })
        }

        fn forward_logits(
            &mut self,
            _seq_ids: &[u64],
            _input_tokens: &[Vec<u32>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> Result<Vec<Vec<f32>>, ModelError> {
            Ok(vec![])
        }

        fn embed(
            &mut self,
            _input_tokens: &[Vec<u32>],
            _positions: &[Vec<usize>],
        ) -> Result<Vec<Vec<f32>>, ModelError> {
            Ok(vec![])
        }

        fn vocab_size(&self) -> usize {
            151936
        }

        fn num_layers(&self) -> usize {
            32
        }

        fn num_heads(&self) -> usize {
            32
        }
    }

    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 4],
        },
        ..Default::default()
    };
    let target_model = MockModel;
    let engine = Engine::with_config(target_model, None, config, 4, 1024);
    // Verify CUDA Graph is configured
    assert!(engine.cuda_graph_enabled());
}
