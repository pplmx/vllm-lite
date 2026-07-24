//! Multi-GPU tensor-parallel integration tests.
//!
//! Tests the tensor-parallel subsystem (`DeviceMesh`, `NodeMesh`,
//! `AllReduce`, `ColumnParallelLinear`, `RowParallelLinear`,
//! `TensorParallelManager`) at the public API surface — covering GPU
//! counts 1, 2, 4, and 8 to match the 8xA100 target hardware.
//!
//! These tests are pure Rust (no CUDA required): they exercise the
//! sharding, validation, and reduction logic that would run on real
//! GPUs, without launching server processes or making HTTP requests.
//! They complement the shell-based `gpu_integration_test.sh` end-to-end
//! suite with fast, deterministic, distributable Rust coverage.
//!
//! Run with:
//!   `cargo nextest run` -p vllm-dist --test `multi_gpu_tensor_parallel`
//!
//! When distributed across GPUs via nextest partitioning, each test
//! partition validates a disjoint subset of the GPU-count matrix.
//! The `tensor_parallel` module is always compiled in vllm-dist (not
//! feature-gated), so these tests run without --features.

use std::sync::Arc;
use vllm_dist::{
    AllReduce, DeviceMesh, LocalSumAllReduce, NodeMesh, ReduceOp, TensorParallelError,
    TensorParallelManager,
};

// ─────────────────────────────────────────────────────────────────
// DeviceMesh — 1, 2, 4, 8 GPU configurations
// ─────────────────────────────────────────────────────────────────

#[test]
fn device_mesh_single_gpu() -> Result<(), TensorParallelError> {
    let mesh = DeviceMesh::new(1, 0, vec![0])?;
    assert_eq!(mesh.world_size, 1);
    assert_eq!(mesh.rank, 0);
    assert!(mesh.is_first_rank());
    assert!(mesh.is_last_rank());
    assert_eq!(mesh.local_device_id(), 0);
    Ok(())
}

#[test]
fn device_mesh_two_gpu() -> Result<(), TensorParallelError> {
    let mesh = DeviceMesh::new(2, 0, vec![0, 1])?;
    assert_eq!(mesh.world_size, 2);
    assert!(mesh.is_first_rank());
    assert!(!mesh.is_last_rank());
    assert_eq!(mesh.local_device_id(), 0);

    let mesh1 = DeviceMesh::new(2, 1, vec![0, 1])?;
    assert!(!mesh1.is_first_rank());
    assert!(mesh1.is_last_rank());
    assert_eq!(mesh1.local_device_id(), 1);
    Ok(())
}

#[test]
fn device_mesh_four_gpu() -> Result<(), TensorParallelError> {
    let mesh = DeviceMesh::new(4, 2, vec![0, 1, 2, 3])?;
    assert_eq!(mesh.world_size, 4);
    assert_eq!(mesh.rank, 2);
    assert!(!mesh.is_first_rank());
    assert!(!mesh.is_last_rank());
    assert_eq!(mesh.local_device_id(), 2);
    Ok(())
}

#[test]
fn device_mesh_eight_gpu() -> Result<(), TensorParallelError> {
    // The full 8xA100 configuration.
    let ids: Vec<usize> = (0..8).collect();
    let mesh = DeviceMesh::new(8, 4, ids.clone())?;
    assert_eq!(mesh.world_size, 8);
    assert_eq!(mesh.rank, 4);
    assert!(!mesh.is_first_rank());
    assert!(!mesh.is_last_rank());
    assert_eq!(mesh.local_device_id(), 4);

    let last = DeviceMesh::new(8, 7, ids)?;
    assert!(last.is_last_rank());
    assert_eq!(last.local_device_id(), 7);
    Ok(())
}

#[test]
fn device_mesh_validation_errors() {
    // world_size == 0
    let result = DeviceMesh::new(0, 0, vec![]);
    assert!(matches!(result, Err(TensorParallelError::InvalidWorldSize)));

    // rank >= world_size
    let result = DeviceMesh::new(4, 5, vec![0, 1, 2, 3]);
    assert!(matches!(result, Err(TensorParallelError::InvalidRank)));

    // device_ids length mismatch
    let result = DeviceMesh::new(4, 0, vec![0, 1, 2]);
    assert!(matches!(result, Err(TensorParallelError::DeviceMismatch)));

    // empty device_ids for world_size=1
    let result = DeviceMesh::new(1, 0, vec![]);
    assert!(matches!(result, Err(TensorParallelError::DeviceMismatch)));
}

#[test]
fn device_mesh_arbitrary_device_ids() -> Result<(), TensorParallelError> {
    // Non-contiguous GPU IDs (e.g., MIG slices or topology-aware assignment).
    let mesh = DeviceMesh::new(4, 1, vec![0, 2, 4, 6])?;
    assert_eq!(mesh.local_device_id(), 2);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// NodeMesh — multi-node scenarios
// ─────────────────────────────────────────────────────────────────

#[test]
fn node_mesh_two_nodes_four_gpus() -> Result<(), TensorParallelError> {
    // 2 nodes, 4 GPUs per node = 8 total ranks.
    let mesh = NodeMesh::new(2, 0, 4, 0, 8)?;
    assert_eq!(mesh.num_nodes, 2);
    assert_eq!(mesh.local_world_size, 4);
    assert_eq!(mesh.global_world_size, 8);
    assert_eq!(mesh.global_rank, 0);
    assert!(mesh.is_first_node());
    assert!(!mesh.is_last_node());

    let peers = mesh.peers();
    assert_eq!(peers.len(), 1);
    assert!(peers[0].contains("vllm-lite-peer-1"));
    Ok(())
}

#[test]
fn node_mesh_last_node() -> Result<(), TensorParallelError> {
    let mesh = NodeMesh::new(4, 3, 2, 7, 8)?;
    assert!(mesh.is_last_node());
    assert_eq!(mesh.peers().len(), 3);
    Ok(())
}

#[test]
fn node_mesh_validation_errors() {
    // num_nodes == 0
    let result = NodeMesh::new(0, 0, 2, 0, 0);
    assert!(matches!(result, Err(TensorParallelError::InvalidWorldSize)));

    // node_rank >= num_nodes
    let result = NodeMesh::new(2, 5, 4, 0, 8);
    assert!(matches!(result, Err(TensorParallelError::InvalidRank)));

    // global_world_size != num_nodes * gpus_per_node
    let result = NodeMesh::new(2, 0, 4, 0, 7);
    assert!(matches!(result, Err(TensorParallelError::DeviceMismatch)));
}

// ─────────────────────────────────────────────────────────────────
// AllReduce — Sum, Avg, Max across various world sizes
// ─────────────────────────────────────────────────────────────────

fn make_reducer(world_size: usize) -> (LocalSumAllReduce, Arc<DeviceMesh>) {
    let mesh = Arc::new(DeviceMesh::new(world_size, 0, (0..world_size).collect()).unwrap());
    let reducer = LocalSumAllReduce::new(mesh.clone());
    (reducer, mesh)
}

#[test]
fn all_reduce_sum() -> Result<(), TensorParallelError> {
    let (reducer, _mesh) = make_reducer(4);
    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let result = reducer.all_reduce(&input, ReduceOp::Sum)?;

    // Sum reduction: every element becomes the total sum.
    let total: f32 = input.iter().sum();
    for v in &result {
        assert!((v - total).abs() < 1e-6, "expected {total}, got {v}");
    }
    Ok(())
}

#[test]
fn all_reduce_avg() -> Result<(), TensorParallelError> {
    let (reducer, _mesh) = make_reducer(8);
    let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = reducer.all_reduce(&input, ReduceOp::Avg)?;

    // Avg: sum / world_size.
    let sum: f32 = input.iter().sum();
    let expected = sum / 8.0;
    for v in &result {
        assert!((v - expected).abs() < 1e-6, "expected {expected}, got {v}");
    }
    Ok(())
}

#[test]
fn all_reduce_max() -> Result<(), TensorParallelError> {
    let (reducer, _mesh) = make_reducer(2);
    let input = vec![-1.5_f32, 3.0, 0.0, -42.0];
    let result = reducer.all_reduce(&input, ReduceOp::Max)?;

    for v in &result {
        assert!((v - 3.0).abs() < 1e-6, "expected 3.0, got {v}");
    }
    Ok(())
}

#[test]
fn all_reduce_inplace_sum() -> Result<(), TensorParallelError> {
    let (reducer, _mesh) = make_reducer(4);
    let mut input = vec![10.0_f32, 20.0, 30.0];
    reducer.all_reduce_inplace(&mut input, ReduceOp::Sum)?;

    let expected: f32 = 10.0 + 20.0 + 30.0;
    for v in &input {
        assert!((v - expected).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn all_reduce_single_gpu_sum_broadcasts_total() -> Result<(), TensorParallelError> {
    // world_size=1: Sum reduction still broadcasts the total sum to every
    // element (LocalSumAllReduce replaces all elements with the sum).
    let (reducer, _mesh) = make_reducer(1);
    let input = vec![5.0_f32, 10.0, 15.0];
    let result = reducer.all_reduce(&input, ReduceOp::Sum)?;
    let total: f32 = input.iter().sum(); // 30.0
    for v in &result {
        assert!((v - total).abs() < 1e-6, "expected {total}, got {v}");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// ColumnParallelLinear — output sharded across ranks
// ─────────────────────────────────────────────────────────────────

#[test]
fn column_parallel_output_size_per_rank() -> Result<(), TensorParallelError> {
    // output_size 1024 split across 8 ranks → 128 per rank.
    let manager = TensorParallelManager::new(8, 0, (0..8).collect())?;
    let layer = manager.create_column_parallel(512, 1024);
    assert_eq!(layer.output_size_per_rank(), 128);
    Ok(())
}

#[test]
fn column_parallel_forward_2gpu() -> Result<(), TensorParallelError> {
    let manager = TensorParallelManager::new(2, 0, vec![0, 1])?;
    let layer = manager.create_column_parallel(64, 128);

    let input = vec![1.0_f32; 64];
    let output = layer.forward(&input)?;

    // Output is sharded: 128 / 2 = 64 elements per rank.
    assert_eq!(output.len(), 64);

    // Each output element is non-zero (input is all 1s, weights are weight_idx * 0.1).
    for v in &output {
        assert!(*v != 0.0, "output should be non-zero");
    }
    Ok(())
}

#[test]
fn column_parallel_forward_8gpu() -> Result<(), TensorParallelError> {
    let manager = TensorParallelManager::new(8, 0, (0..8).collect())?;
    let layer = manager.create_column_parallel(128, 512);

    let input = vec![1.0_f32; 128];
    let output = layer.forward(&input)?;
    assert_eq!(output.len(), 64); // 512 / 8

    // column_parallel_forward applies all_reduce_inplace(Sum) after the
    // local compute, which broadcasts the sum of ALL local outputs to
    // every element. Verify the broadcasted value matches.
    //
    // local_output[i] = sum_j(1.0 * (i*128 + j) * 0.1) for j in 0..128
    //                 = 0.1 * (i*128*128 + sum(0..127))
    //                 = 0.1 * (i*16384 + 8128)
    //
    // total = sum_i(local_output[i]) for i in 0..63
    //       = 0.1 * (16384 * sum(0..63) + 8128 * 64)
    //       = 0.1 * (16384 * 2016 + 520192)
    //       = 0.1 * 33550336 = 3355033.6
    let expected_total: f32 = 0.1 * (16384.0 * 2016.0 + 520192.0);
    for v in &output {
        assert!(
            (v - expected_total).abs() < 1.0,
            "expected {expected_total}, got {v}"
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// RowParallelLinear — input sharded across ranks, output replicated
// ─────────────────────────────────────────────────────────────────

#[test]
fn row_parallel_input_size_per_rank() -> Result<(), TensorParallelError> {
    let manager = TensorParallelManager::new(8, 0, (0..8).collect())?;
    let layer = manager.create_row_parallel(1024, 512);
    assert_eq!(layer.input_size_per_rank(), 128); // 1024 / 8
    Ok(())
}

#[test]
fn row_parallel_forward_2gpu() -> Result<(), TensorParallelError> {
    let manager = TensorParallelManager::new(2, 0, vec![0, 1])?;
    let layer = manager.create_row_parallel(128, 64);

    // input_size_per_rank = 64, so input len must be 64.
    let input = vec![1.0_f32; 64];
    let output = layer.forward(&input)?;
    assert_eq!(output.len(), 64);

    for v in &output {
        assert!(*v != 0.0);
    }
    Ok(())
}

#[test]
fn row_parallel_input_size_mismatch() -> Result<(), TensorParallelError> {
    let manager = TensorParallelManager::new(4, 0, (0..4).collect())?;
    let layer = manager.create_row_parallel(256, 64);

    // input_size_per_rank = 256/4 = 64, so providing 32 should fail.
    let input = vec![1.0_f32; 32];
    let result = layer.forward(&input);
    assert!(matches!(
        result,
        Err(TensorParallelError::InputSizeMismatch)
    ));
    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// TensorParallelManager — end-to-end configuration
// ─────────────────────────────────────────────────────────────────

#[test]
fn manager_creates_column_and_row_parallel() -> Result<(), TensorParallelError> {
    let manager = TensorParallelManager::new(4, 1, vec![0, 1, 2, 3])?;

    let col = manager.create_column_parallel(256, 512);
    let row = manager.create_row_parallel(512, 256);

    assert_eq!(col.output_size_per_rank(), 128); // 512/4
    assert_eq!(row.input_size_per_rank(), 128); // 512/4
    Ok(())
}

#[test]
fn manager_rejects_invalid_config() {
    let result = TensorParallelManager::new(0, 0, vec![]);
    assert!(matches!(result, Err(TensorParallelError::InvalidWorldSize)));
}

#[test]
fn manager_shares_mesh_and_all_reduce() -> Result<(), TensorParallelError> {
    // Both layers created from the same manager should share the same
    // DeviceMesh and AllReduce instances, ensuring consistent rank topology.
    let manager = TensorParallelManager::new(8, 2, (0..8).collect())?;

    let col = manager.create_column_parallel(128, 1024);
    let row = manager.create_row_parallel(1024, 128);

    // col: output 1024/8 = 128 per rank
    assert_eq!(col.output_size_per_rank(), 128);
    // row: input 1024/8 = 128 per rank
    assert_eq!(row.input_size_per_rank(), 128);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// GPU distribution awareness — tests that adapt to CUDA_VISIBLE_DEVICES
// ─────────────────────────────────────────────────────────────────

/// Detect the number of visible GPUs from the `CUDA_VISIBLE_DEVICES`
/// environment variable, falling back to 0 if unset or empty.
fn visible_gpu_count() -> usize {
    match std::env::var("CUDA_VISIBLE_DEVICES") {
        Ok(val) if !val.is_empty() => val.split(',').count(),
        _ => 0,
    }
}

#[test]
fn gpu_distribution_matches_visible_devices() -> Result<(), TensorParallelError> {
    // When nextest distributes tests across GPUs via CUDA_VISIBLE_DEVICES=$i,
    // this test verifies that exactly 1 GPU is visible to the current process
    // (the one it was assigned). On CPU-only or unpartitioned runs, the
    // env var may be unset, in which case we skip the assertion.
    let visible = visible_gpu_count();

    // The partition-per-GPU approach sets CUDA_VISIBLE_DEVICES to a single
    // device. If set, assert it's a singleton partition.
    if visible > 0 {
        assert_eq!(visible, 1, "partition-per-GPU should expose exactly 1 GPU");
    }

    // Either way, a 1-GPU mesh should always work.
    let mesh = DeviceMesh::new(1, 0, vec![0])?;
    assert_eq!(mesh.world_size, 1);
    Ok(())
}
