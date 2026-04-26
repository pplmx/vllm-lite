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

#[derive(Debug, Clone)]
pub struct NodeMesh {
    pub node_mesh: Vec<DeviceMesh>,
    pub num_nodes: usize,
    pub node_rank: usize,
    pub local_world_size: usize,
    pub global_rank: usize,
    pub global_world_size: usize,
}

impl NodeMesh {
    pub fn new(
        num_nodes: usize,
        node_rank: usize,
        gpus_per_node: usize,
        global_rank: usize,
        global_world_size: usize,
    ) -> Result<Self, TensorParallelError> {
        if num_nodes == 0 {
            return Err(TensorParallelError::InvalidWorldSize);
        }
        if node_rank >= num_nodes {
            return Err(TensorParallelError::InvalidRank);
        }
        if global_world_size != num_nodes * gpus_per_node {
            return Err(TensorParallelError::DeviceMismatch);
        }

        let local_rank = global_rank % gpus_per_node;
        let device_ids: Vec<usize> = (0..gpus_per_node).collect();
        let local_device_ids: Vec<usize> = device_ids
            .into_iter()
            .map(|i| i + node_rank * gpus_per_node)
            .collect();

        let device_mesh = DeviceMesh::new(gpus_per_node, local_rank, local_device_ids)?;

        Ok(Self {
            node_mesh: vec![device_mesh],
            num_nodes,
            node_rank,
            local_world_size: gpus_per_node,
            global_rank,
            global_world_size,
        })
    }

    pub fn is_first_node(&self) -> bool {
        self.node_rank == 0
    }

    pub fn is_last_node(&self) -> bool {
        self.node_rank == self.num_nodes - 1
    }

    pub fn local_mesh(&self) -> &DeviceMesh {
        &self.node_mesh[0]
    }

    pub fn peers(&self) -> Vec<String> {
        let mut peers = Vec::new();
        for i in 0..self.num_nodes {
            if i != self.node_rank {
                peers.push(format!(
                    "vllm-lite-peer-{}.vllm-lite.svc.cluster.local:50051",
                    i
                ));
            }
        }
        peers
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

    #[test]
    fn test_node_mesh_creation() -> Result<(), TensorParallelError> {
        let mesh = NodeMesh::new(2, 0, 2, 0, 4)?;

        assert_eq!(mesh.num_nodes, 2);
        assert_eq!(mesh.node_rank, 0);
        assert!(mesh.is_first_node());
        assert!(!mesh.is_last_node());
        assert_eq!(mesh.local_world_size, 2);
        assert_eq!(mesh.global_world_size, 4);

        Ok(())
    }

    #[test]
    fn test_node_mesh_peers() -> Result<(), TensorParallelError> {
        let mesh = NodeMesh::new(2, 0, 2, 0, 4)?;
        let peers = mesh.peers();

        assert_eq!(peers.len(), 1);
        assert!(peers[0].contains("vllm-lite-peer-1"));

        Ok(())
    }
}
