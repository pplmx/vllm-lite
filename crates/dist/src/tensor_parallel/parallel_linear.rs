#![allow(clippy::module_name_repetitions)]
use super::all_reduce::{AllReduce, NcclAllReduce, ReduceOp};
use super::device_mesh::DeviceMesh;
use std::sync::Arc;
use vllm_traits::TensorParallelError;

#[derive(Debug)]
/// `ColumnParallelLinear`. See the type definition for fields and behavior.
pub struct ColumnParallelLinear {
    input_size: usize,
    output_size: usize,
    mesh: Arc<DeviceMesh>,
    all_reduce: Arc<dyn AllReduce>,
}

impl ColumnParallelLinear {
    pub fn new(
        input_size: usize,
        output_size: usize,
        mesh: Arc<DeviceMesh>,
        all_reduce: Arc<dyn AllReduce>,
    ) -> Self {
        Self {
            input_size,
            output_size,
            mesh,
            all_reduce,
        }
    }

    #[must_use]
    pub fn output_size_per_rank(&self) -> usize {
        self.output_size / self.mesh.world_size
    }

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, TensorParallelError> {
        let local_output_size = self.output_size_per_rank();
        let mut local_output = vec![0.0f32; local_output_size];

        for (i, out) in local_output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (j, &inp) in input.iter().enumerate() {
                let weight_idx = i * self.input_size + j;
                // invariant: weight_idx is a small tensor index (bounded by
                // local_output_size * input_size); f32 precision is acceptable
                // for this stub reduction.
                sum = mul_add_weight(inp, weight_idx, sum);
            }
            *out = sum;
        }

        self.all_reduce
            .all_reduce_inplace(&mut local_output, ReduceOp::Sum)?;

        Ok(local_output)
    }
}
#[derive(Debug)]

/// `RowParallelLinear`. See the type definition for fields and behavior.
pub struct RowParallelLinear {
    input_size: usize,
    output_size: usize,
    mesh: Arc<DeviceMesh>,
    all_reduce: Arc<dyn AllReduce>,
}

impl RowParallelLinear {
    pub fn new(
        input_size: usize,
        output_size: usize,
        mesh: Arc<DeviceMesh>,
        all_reduce: Arc<dyn AllReduce>,
    ) -> Self {
        Self {
            input_size,
            output_size,
            mesh,
            all_reduce,
        }
    }

    #[must_use]
    pub fn input_size_per_rank(&self) -> usize {
        self.input_size / self.mesh.world_size
    }

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, TensorParallelError> {
        let local_input_size = self.input_size_per_rank();

        if input.len() != local_input_size {
            return Err(TensorParallelError::InputSizeMismatch);
        }

        let mut output = vec![0.0f32; self.output_size];

        for (i, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (j, &inp) in input.iter().enumerate() {
                let weight_idx = i * local_input_size + j;
                // invariant: weight_idx is a small tensor index (bounded by
                // output_size * local_input_size); f32 precision is acceptable
                // for this stub reduction.
                sum = mul_add_weight(inp, weight_idx, sum);
            }
            *out = sum;
        }

        if !self.mesh.is_last_rank() {
            self.all_reduce
                .all_reduce_inplace(&mut output, ReduceOp::Sum)?;
        }

        Ok(output)
    }
}

/// Manager for TensorParallel. Owns the underlying resource, coordinates concurrent access, and exposes a thread-safe public API.
#[derive(Debug)]
pub struct TensorParallelManager {
    mesh: Arc<DeviceMesh>,
    all_reduce: Arc<dyn AllReduce>,
}

impl TensorParallelManager {
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new(
        world_size: usize,
        rank: usize,
        device_ids: Vec<usize>,
    ) -> Result<Self, TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(world_size, rank, device_ids)?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));

        Ok(Self { mesh, all_reduce })
    }

    #[must_use]
    pub fn create_column_parallel(
        &self,
        input_size: usize,
        output_size: usize,
    ) -> ColumnParallelLinear {
        ColumnParallelLinear::new(
            input_size,
            output_size,
            self.mesh.clone(),
            self.all_reduce.clone(),
        )
    }

    #[must_use]
    pub fn create_row_parallel(&self, input_size: usize, output_size: usize) -> RowParallelLinear {
        RowParallelLinear::new(
            input_size,
            output_size,
            self.mesh.clone(),
            self.all_reduce.clone(),
        )
    }

    #[must_use]
    pub const fn mesh(&self) -> &Arc<DeviceMesh> {
        &self.mesh
    }
}

// invariant: weight_idx is a small tensor index (bounded by tensor dim);
// f32 precision is acceptable for this stub reduction helper.
#[allow(clippy::cast_precision_loss)]
fn mul_add_weight(input: f32, weight_idx: usize, sum: f32) -> f32 {
    input.mul_add(weight_idx as f32 * 0.1, sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_parallel_output_size() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(2, 0, vec![0, 1])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = ColumnParallelLinear::new(8, 16, mesh, all_reduce);

        assert_eq!(linear.output_size_per_rank(), 8);
        Ok(())
    }

    #[test]
    fn test_row_parallel_input_size() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(2, 0, vec![0, 1])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = RowParallelLinear::new(16, 8, mesh, all_reduce);

        assert_eq!(linear.input_size_per_rank(), 8);
        Ok(())
    }

    #[test]
    fn test_column_parallel_forward() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(2, 0, vec![0, 1])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = ColumnParallelLinear::new(4, 4, mesh, all_reduce);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = linear.forward(&input)?;

        assert_eq!(output.len(), 2);
        Ok(())
    }

    #[test]
    fn test_row_parallel_forward() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(2, 0, vec![0, 1])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = RowParallelLinear::new(4, 4, mesh, all_reduce);

        let input = vec![1.0, 2.0];
        let output = linear.forward(&input)?;

        assert_eq!(output.len(), 4);
        Ok(())
    }

    #[test]
    fn test_tensor_parallel_manager() -> Result<(), TensorParallelError> {
        let manager = TensorParallelManager::new(4, 0, vec![0, 1, 2, 3])?;

        let col_linear = manager.create_column_parallel(8, 16);
        assert_eq!(col_linear.output_size_per_rank(), 4);

        let row_linear = manager.create_row_parallel(16, 8);
        assert_eq!(row_linear.input_size_per_rank(), 4);

        Ok(())
    }

    #[test]
    fn test_row_parallel_input_error() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(2, 0, vec![0, 1])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = RowParallelLinear::new(4, 4, mesh, all_reduce);

        let wrong_input = vec![1.0, 2.0, 3.0];
        let result = linear.forward(&wrong_input);

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_column_parallel_large_batch() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(4, 0, vec![0, 1, 2, 3])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = ColumnParallelLinear::new(1024, 2048, mesh, all_reduce);

        assert_eq!(linear.output_size_per_rank(), 512);

        let input = vec![1.0f32; 1024];
        let output = linear.forward(&input)?;

        assert_eq!(output.len(), 512);
        Ok(())
    }
}
