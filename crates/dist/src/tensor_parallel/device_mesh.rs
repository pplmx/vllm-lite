use vllm_traits::TensorParallelError;

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
}
