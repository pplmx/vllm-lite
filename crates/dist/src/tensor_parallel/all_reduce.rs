use super::device_mesh::{DeviceMesh, TensorParallelError};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Avg,
    Max,
}

pub trait AllReduce: Send + Sync {
    fn all_reduce(&self, input: &[f32], op: ReduceOp) -> Result<Vec<f32>, TensorParallelError>;
    fn all_reduce_inplace(
        &self,
        input: &mut [f32],
        op: ReduceOp,
    ) -> Result<(), TensorParallelError>;
}

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
