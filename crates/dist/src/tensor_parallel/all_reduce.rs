//! `AllReduce` primitives for tensor-parallel collectives.
//!
//! Wraps the underlying ring/tree algorithms behind a trait object so the
//! tensor-parallel linear layers can swap between a real cross-device
//! backend (future NCCL / gRPC), a stub no-op implementation in tests,
//! and a single-process reference reducer.
//!
//! # Honest naming
//!
//! The previous name `NcclAllReduce` was misleading: despite the
//! `Nccl` prefix, the struct did **not** call NVIDIA NCCL and did
//! **not** communicate across devices. It computed a local reduction
//! over a single tensor using [`DeviceMesh::world_size`] only as the
//! divisor for [`ReduceOp::Avg`]. The misleading alias was removed in
//! v0.1; this type is the honest name (see the
//! v31.0 P4 follow-up batch + `docs/technical-due-diligence/architecture-performance.md`
//! §6 "分布式"). A real cross-device NCCL backend is intentionally out
//! of scope for v0.x — see `roadmap.md §6` ("暂缓事项").
#![allow(clippy::module_name_repetitions)]
use super::device_mesh::DeviceMesh;
use std::sync::Arc;
use vllm_traits::TensorParallelError;

/// `ReduceOp`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Avg,
    Max,
}

/// `AllReduce`. See the type definition for fields and behavior.
pub trait AllReduce: Send + Sync + std::fmt::Debug {
    /// Sum a tensor across all ranks in the device mesh.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn all_reduce(&self, input: &[f32], op: ReduceOp) -> Result<Vec<f32>, TensorParallelError>;
    /// Sum a tensor in place across all ranks in the device mesh.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn all_reduce_inplace(
        &self,
        input: &mut [f32],
        op: ReduceOp,
    ) -> Result<(), TensorParallelError>;
}

/// Honest name for what this reducer actually does: a single-process
/// reduction over a local tensor, using `world_size` only as the
/// divisor for [`ReduceOp::Avg`].
///
/// This is the type new code should use. (A previous misleading
/// `NcclAllReduce` alias existed for the v0.x transition window and
/// was removed once no internal callers remained.)
#[derive(Debug)]
pub struct LocalSumAllReduce {
    mesh: Arc<DeviceMesh>,
}
impl LocalSumAllReduce {
    /// Create a new all-reduce backed by the given device mesh.
    #[must_use]
    pub const fn new(mesh: Arc<DeviceMesh>) -> Self {
        Self { mesh }
    }
}

impl AllReduce for LocalSumAllReduce {
    fn all_reduce(&self, input: &[f32], op: ReduceOp) -> Result<Vec<f32>, TensorParallelError> {
        let mut result = input.to_vec();
        self.all_reduce_inplace(&mut result, op)?;
        Ok(result)
    }

    fn all_reduce_inplace(
        &self,
        input: &mut [f32],
        op: ReduceOp,
    ) -> Result<(), TensorParallelError> {
        local_reduce_inplace(input, op, self.mesh.world_size);
        Ok(())
    }
}

/// In-place reduction over a single tensor.
fn local_reduce_inplace(input: &mut [f32], op: ReduceOp, world_size: usize) {
    // invariant: world_size is a small mesh size (typ. < GPU count < 1k);
    // f32 precision loss is acceptable for the divisor.
    #[allow(clippy::cast_precision_loss)]
    let world_size_f = world_size as f32;

    match op {
        ReduceOp::Sum => {
            let sum: f32 = input.iter().sum();
            for v in input.iter_mut() {
                *v = sum;
            }
        }
        ReduceOp::Avg => {
            let sum: f32 = input.iter().sum();
            let avg = sum / world_size_f;
            for v in input.iter_mut() {
                *v = avg;
            }
        }
        ReduceOp::Max => {
            let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for v in input.iter_mut() {
                *v = max_val;
            }
        }
    }
}

/// Default single-node `AllReduce`.
///
/// Passes inputs through unchanged. Used in single-node deployments where
/// there is nothing to reduce across ranks.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NoopAllReduce;

impl AllReduce for NoopAllReduce {
    fn all_reduce(&self, input: &[f32], _op: ReduceOp) -> Result<Vec<f32>, TensorParallelError> {
        Ok(input.to_vec())
    }

    fn all_reduce_inplace(
        &self,
        _input: &mut [f32],
        _op: ReduceOp,
    ) -> Result<(), TensorParallelError> {
        Ok(())
    }
}

impl dyn AllReduce {
    /// Returns an `Arc<Self>` wrapping the single-node `NoopAllReduce`.
    ///
    /// This is the closest equivalent to
    /// `Arc::<dyn AllReduce>::default()`; Rust's orphan rule prevents
    /// a direct `impl Default for Arc<dyn ...>` because `Arc` is foreign and
    /// there is no local type appearing before the uncovered trait-object
    /// parameter.
    #[must_use]
    pub fn default_arc() -> Arc<Self> {
        Arc::new(NoopAllReduce)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_reduce_sum() -> Result<(), TensorParallelError> {
        let mesh = DeviceMesh::new(2, 0, vec![0, 1])?;
        let all_reduce = LocalSumAllReduce::new(mesh.into());

        let input = vec![1.0, 2.0, 3.0];
        let result = all_reduce.all_reduce(&input, ReduceOp::Sum)?;

        let expected: f32 = input.iter().sum();
        for v in &result {
            assert!((*v - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_all_reduce_avg() -> Result<(), TensorParallelError> {
        let mesh = DeviceMesh::new(2, 0, vec![0, 1])?;
        let all_reduce = LocalSumAllReduce::new(mesh.into());

        let input = vec![2.0, 4.0, 6.0];
        let result = all_reduce.all_reduce(&input, ReduceOp::Avg)?;

        let expected: f32 = input.iter().sum::<f32>() / 2.0;
        for v in &result {
            assert!((*v - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_all_reduce_max() -> Result<(), TensorParallelError> {
        let mesh = DeviceMesh::new(2, 0, vec![0, 1])?;
        let all_reduce = LocalSumAllReduce::new(mesh.into());

        let input = vec![1.0, 5.0, 3.0];
        let result = all_reduce.all_reduce(&input, ReduceOp::Max)?;

        let expected = *input
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        for v in &result {
            assert!((*v - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_all_reduce_inplace() -> Result<(), TensorParallelError> {
        let mesh = DeviceMesh::new(2, 0, vec![0, 1])?;
        let all_reduce = LocalSumAllReduce::new(mesh.into());

        let mut input = vec![1.0, 2.0, 3.0];
        all_reduce.all_reduce_inplace(&mut input, ReduceOp::Sum)?;

        let sum: f32 = 6.0;
        for v in &input {
            assert!((*v - sum).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_all_reduce_different_world_sizes() -> Result<(), TensorParallelError> {
        let sizes = vec![1, 2, 4, 8];

        for size in sizes {
            let mesh = DeviceMesh::new(size, 0, (0..size).collect())?;
            let all_reduce = LocalSumAllReduce::new(mesh.into());

            let input = vec![1.0, 2.0, 3.0];
            let result = all_reduce.all_reduce(&input, ReduceOp::Sum)?;

            let expected: f32 = input.iter().sum();
            for v in &result {
                assert!((*v - expected).abs() < 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn all_reduce_default_arc_is_noop() -> Result<(), TensorParallelError> {
        let ar: Arc<dyn AllReduce> = <dyn AllReduce>::default_arc();
        let input = vec![1.0, 2.0, 3.0];
        let result = ar.all_reduce(&input, ReduceOp::Sum)?;
        assert_eq!(result, input);
        Ok(())
    }
}
