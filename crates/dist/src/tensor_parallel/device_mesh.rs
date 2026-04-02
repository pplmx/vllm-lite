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
