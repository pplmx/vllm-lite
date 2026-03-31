use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DeviceMesh {
    pub world_size: usize,
    pub rank: usize,
    pub device_ids: Vec<usize>,
}

impl DeviceMesh {
    pub fn new(
        world_size: usize,
        rank: usize,
        device_ids: Vec<usize>,
    ) -> Result<Self, TensorParallelError> {
        if world_size == 0 {
            return Err(TensorParallelError::InvalidWorldSize);
        }
        if rank >= world_size {
            return Err(TensorParallelError::InvalidRank);
        }
        if device_ids.len() != world_size {
            return Err(TensorParallelError::DeviceMismatch);
        }

        Ok(Self {
            world_size,
            rank,
            device_ids,
        })
    }

    pub fn is_first_rank(&self) -> bool {
        self.rank == 0
    }

    pub fn is_last_rank(&self) -> bool {
        self.rank == self.world_size - 1
    }

    pub fn local_device_id(&self) -> usize {
        self.device_ids[self.rank]
    }
}

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

#[derive(Debug, Clone)]
pub enum TensorParallelError {
    InvalidWorldSize,
    InvalidRank,
    DeviceMismatch,
    InputSizeMismatch,
    AllReduceFailed(String),
    CudaError(String),
}

impl std::fmt::Display for TensorParallelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWorldSize => write!(f, "World size must be > 0"),
            Self::InvalidRank => write!(f, "Rank must be < world size"),
            Self::DeviceMismatch => write!(f, "Number of device IDs must match world size"),
            Self::InputSizeMismatch => {
                write!(f, "Input size does not match expected size per rank")
            }
            Self::AllReduceFailed(msg) => write!(f, "All-reduce failed: {}", msg),
            Self::CudaError(msg) => write!(f, "CUDA error: {}", msg),
        }
    }
}

impl std::error::Error for TensorParallelError {}

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
}
