use std::sync::Arc;

use super::all_reduce::{AllReduce, NcclAllReduce, ReduceOp};
use super::device_mesh::{DeviceMesh, TensorParallelError};

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

    pub fn output_size_per_rank(&self) -> usize {
        self.output_size / self.mesh.world_size
    }

    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, TensorParallelError> {
        let local_output_size = self.output_size_per_rank();
        let mut local_output = vec![0.0f32; local_output_size];

        for (i, out) in local_output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (j, &inp) in input.iter().enumerate() {
                let weight_idx = i * self.input_size + j;
                sum += inp * (weight_idx as f32 * 0.1);
            }
            *out = sum;
        }

        self.all_reduce
            .all_reduce_inplace(&mut local_output, ReduceOp::Sum)?;

        Ok(local_output)
    }
}

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

    pub fn input_size_per_rank(&self) -> usize {
        self.input_size / self.mesh.world_size
    }

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
                sum += inp * (weight_idx as f32 * 0.1);
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

pub struct TensorParallelManager {
    mesh: Arc<DeviceMesh>,
    all_reduce: Arc<dyn AllReduce>,
}

impl TensorParallelManager {
    pub fn new(
        world_size: usize,
        rank: usize,
        device_ids: Vec<usize>,
    ) -> Result<Self, TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(world_size, rank, device_ids)?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));

        Ok(Self { mesh, all_reduce })
    }

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

    pub fn create_row_parallel(&self, input_size: usize, output_size: usize) -> RowParallelLinear {
        RowParallelLinear::new(
            input_size,
            output_size,
            self.mesh.clone(),
            self.all_reduce.clone(),
        )
    }

    pub fn mesh(&self) -> &Arc<DeviceMesh> {
        &self.mesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_mesh_creation() -> Result<(), TensorParallelError> {
        let mesh = DeviceMesh::new(4, 0, vec![0, 1, 2, 3])?;

        assert_eq!(mesh.world_size, 4);
        assert_eq!(mesh.rank, 0);
        assert!(mesh.is_first_rank());
        assert!(!mesh.is_last_rank());

        Ok(())
    }

    #[test]
    fn test_device_mesh_errors() {
        let result = DeviceMesh::new(0, 0, vec![]);
        assert!(result.is_err());

        let result = DeviceMesh::new(4, 5, vec![0, 1, 2, 3]);
        assert!(result.is_err());

        let result = DeviceMesh::new(4, 0, vec![0, 1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_parallel_output_size() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(2, 0, vec![0, 1])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = ColumnParallelLinear::new(8, 16, mesh.clone(), all_reduce);

        assert_eq!(linear.output_size_per_rank(), 8);

        Ok(())
    }

    #[test]
    fn test_row_parallel_input_size() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(2, 0, vec![0, 1])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = RowParallelLinear::new(16, 8, mesh.clone(), all_reduce);

        assert_eq!(linear.input_size_per_rank(), 8);

        Ok(())
    }

    #[test]
    fn test_column_parallel_forward() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(2, 0, vec![0, 1])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = ColumnParallelLinear::new(4, 4, mesh.clone(), all_reduce);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = linear.forward(&input)?;

        assert_eq!(output.len(), 2);

        Ok(())
    }

    #[test]
    fn test_row_parallel_forward() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(2, 0, vec![0, 1])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = RowParallelLinear::new(4, 4, mesh.clone(), all_reduce);

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
    fn test_device_mesh_first_last_rank() -> Result<(), TensorParallelError> {
        let mesh0 = DeviceMesh::new(4, 0, vec![0, 1, 2, 3])?;
        assert!(mesh0.is_first_rank());
        assert!(!mesh0.is_last_rank());

        let mesh3 = DeviceMesh::new(4, 3, vec![0, 1, 2, 3])?;
        assert!(!mesh3.is_first_rank());
        assert!(mesh3.is_last_rank());

        let mesh1 = DeviceMesh::new(4, 1, vec![0, 1, 2, 3])?;
        assert!(!mesh1.is_first_rank());
        assert!(!mesh1.is_last_rank());

        Ok(())
    }

    #[test]
    fn test_device_mesh_local_device_id() -> Result<(), TensorParallelError> {
        let mesh = DeviceMesh::new(4, 2, vec![10, 11, 12, 13])?;
        assert_eq!(mesh.local_device_id(), 12);

        Ok(())
    }

    #[test]
    fn test_column_parallel_large_batch() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(4, 0, vec![0, 1, 2, 3])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = ColumnParallelLinear::new(1024, 2048, mesh.clone(), all_reduce);

        assert_eq!(linear.output_size_per_rank(), 512);

        let input = vec![1.0f32; 1024];
        let output = linear.forward(&input)?;

        assert_eq!(output.len(), 512);

        Ok(())
    }

    #[test]
    fn test_row_parallel_large_batch() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(4, 0, vec![0, 1, 2, 3])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = RowParallelLinear::new(2048, 1024, mesh.clone(), all_reduce);

        assert_eq!(linear.input_size_per_rank(), 512);

        let input = vec![1.0f32; 512];
        let output = linear.forward(&input)?;

        assert_eq!(output.len(), 1024);

        Ok(())
    }

    #[test]
    fn test_column_parallel_single_gpu() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(1, 0, vec![0])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = ColumnParallelLinear::new(8, 16, mesh.clone(), all_reduce);

        assert_eq!(linear.output_size_per_rank(), 16);

        let input = vec![1.0; 8];
        let output = linear.forward(&input)?;

        assert_eq!(output.len(), 16);

        Ok(())
    }

    #[test]
    fn test_column_parallel_output_values() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(2, 0, vec![0, 1])?);

        struct IdentityAllReduce;

        impl AllReduce for IdentityAllReduce {
            fn all_reduce(
                &self,
                input: &[f32],
                _op: ReduceOp,
            ) -> Result<Vec<f32>, TensorParallelError> {
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

        let identity_all_reduce = Arc::new(IdentityAllReduce);
        let linear = ColumnParallelLinear::new(2, 2, mesh.clone(), identity_all_reduce);

        let input = vec![1.0, 2.0];
        let output = linear.forward(&input)?;

        for v in output.iter() {
            assert!(*v > 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_row_parallel_input_error() -> Result<(), TensorParallelError> {
        let mesh = Arc::new(DeviceMesh::new(2, 0, vec![0, 1])?);
        let all_reduce = Arc::new(NcclAllReduce::new(mesh.clone()));
        let linear = RowParallelLinear::new(4, 4, mesh.clone(), all_reduce);

        let wrong_input = vec![1.0, 2.0, 3.0];
        let result = linear.forward(&wrong_input);

        assert!(result.is_err());

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
    fn test_tensor_parallel_manager_mesh_access() -> Result<(), TensorParallelError> {
        let manager = TensorParallelManager::new(8, 3, vec![0, 1, 2, 3, 4, 5, 6, 7])?;

        let mesh = manager.mesh();
        assert_eq!(mesh.world_size, 8);
        assert_eq!(mesh.rank, 3);
        assert_eq!(mesh.local_device_id(), 3);

        Ok(())
    }

    #[test]
    fn test_tensor_parallel_4_gpu() -> Result<(), TensorParallelError> {
        let manager = TensorParallelManager::new(4, 1, vec![0, 1, 2, 3])?;

        let col_linear = manager.create_column_parallel(16, 16);
        assert_eq!(col_linear.output_size_per_rank(), 4);

        let row_linear = manager.create_row_parallel(16, 16);
        assert_eq!(row_linear.input_size_per_rank(), 4);

        Ok(())
    }

    #[test]
    fn test_column_parallel_8_gpu() -> Result<(), TensorParallelError> {
        let manager = TensorParallelManager::new(8, 5, vec![0, 1, 2, 3, 4, 5, 6, 7])?;

        let col_linear = manager.create_column_parallel(64, 128);
        assert_eq!(col_linear.output_size_per_rank(), 16);

        assert_eq!(manager.mesh().rank, 5);
        assert_eq!(manager.mesh().local_device_id(), 5);

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
}
