use super::device_mesh::DeviceMesh;
use std::sync::Arc;
use vllm_traits::TensorParallelError;

/// ReduceOp: reduce op enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Avg,
    Max,
}

/// AllReduce: all reduce trait.
pub trait AllReduce: Send + Sync {
    fn all_reduce(&self, input: &[f32], op: ReduceOp) -> Result<Vec<f32>, TensorParallelError>;
    fn all_reduce_inplace(
        &self,
        input: &mut [f32],
        op: ReduceOp,
    ) -> Result<(), TensorParallelError>;
}

/// NcclAllReduce: nccl all reduce.
pub struct NcclAllReduce {
    mesh: Arc<DeviceMesh>,
}

impl NcclAllReduce {
    pub fn new(mesh: Arc<DeviceMesh>) -> Self {
        Self { mesh }
    }
}

impl AllReduce for NcclAllReduce {
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
        let world_size = self.mesh.world_size as f32;

        match op {
            ReduceOp::Sum => {
                let sum: f32 = input.iter().sum();
                for v in input.iter_mut() {
                    *v = sum;
                }
            }
            ReduceOp::Avg => {
                let sum: f32 = input.iter().sum();
                let avg = sum / world_size;
                for v in input.iter_mut() {
                    *v = avg;
                }
            }
            ReduceOp::Max => {
                let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                for v in input.iter_mut() {
                    *v = max_val;
                }
            }
        }

        Ok(())
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
        let all_reduce = NcclAllReduce::new(mesh.into());

        let input = vec![1.0, 2.0, 3.0];
        let result = all_reduce.all_reduce(&input, ReduceOp::Sum)?;

        let expected: f32 = input.iter().sum();
        for v in result.iter() {
            assert_eq!(*v, expected);
        }
        Ok(())
    }

    #[test]
    fn test_all_reduce_avg() -> Result<(), TensorParallelError> {
        let mesh = DeviceMesh::new(2, 0, vec![0, 1])?;
        let all_reduce = NcclAllReduce::new(mesh.into());

        let input = vec![2.0, 4.0, 6.0];
        let result = all_reduce.all_reduce(&input, ReduceOp::Avg)?;

        let expected: f32 = input.iter().sum::<f32>() / 2.0;
        for v in result.iter() {
            assert_eq!(*v, expected);
        }
        Ok(())
    }

    #[test]
    fn test_all_reduce_max() -> Result<(), TensorParallelError> {
        let mesh = DeviceMesh::new(2, 0, vec![0, 1])?;
        let all_reduce = NcclAllReduce::new(mesh.into());

        let input = vec![1.0, 5.0, 3.0];
        let result = all_reduce.all_reduce(&input, ReduceOp::Max)?;

        let expected = *input
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        for v in result.iter() {
            assert_eq!(*v, expected);
        }
        Ok(())
    }

    #[test]
    fn test_all_reduce_inplace() -> Result<(), TensorParallelError> {
        let mesh = DeviceMesh::new(2, 0, vec![0, 1])?;
        let all_reduce = NcclAllReduce::new(mesh.into());

        let mut input = vec![1.0, 2.0, 3.0];
        all_reduce.all_reduce_inplace(&mut input, ReduceOp::Sum)?;

        let sum: f32 = 6.0;
        for v in input.iter() {
            assert_eq!(*v, sum);
        }
        Ok(())
    }

    #[test]
    fn test_all_reduce_different_world_sizes() -> Result<(), TensorParallelError> {
        let sizes = vec![1, 2, 4, 8];

        for size in sizes {
            let mesh = DeviceMesh::new(size, 0, (0..size).collect())?;
            let all_reduce = NcclAllReduce::new(mesh.into());

            let input = vec![1.0, 2.0, 3.0];
            let result = all_reduce.all_reduce(&input, ReduceOp::Sum)?;

            let expected: f32 = input.iter().sum();
            for v in result.iter() {
                assert_eq!(*v, expected);
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
