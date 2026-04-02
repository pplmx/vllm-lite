use candle_core::Device;

#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    pub world_size: usize,
    pub rank: usize,
    pub device_ids: Vec<usize>,
}

impl TensorParallelConfig {
    pub fn new(world_size: usize, rank: usize, device_ids: Vec<usize>) -> Option<Self> {
        if world_size == 0 || rank >= world_size || device_ids.len() != world_size {
            return None;
        }
        Some(Self {
            world_size,
            rank,
            device_ids,
        })
    }

    pub fn local_device(&self) -> Device {
        let gpu_id = self.device_ids[self.rank];
        Device::new_cuda(gpu_id).unwrap_or(Device::Cpu)
    }

    pub fn is_first_rank(&self) -> bool {
        self.rank == 0
    }
}

pub fn compute_vocab_shard(vocab_size: usize, world_size: usize, rank: usize) -> (usize, usize) {
    let vocab_per_rank = vocab_size / world_size;
    let remainder = vocab_size % world_size;

    let my_vocab_size = if rank < remainder {
        vocab_per_rank + 1
    } else {
        vocab_per_rank
    };

    let offset = if rank < remainder {
        rank * (vocab_per_rank + 1)
    } else {
        remainder * (vocab_per_rank + 1) + (rank - remainder) * vocab_per_rank
    };

    (my_vocab_size, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_shard_even() {
        let (size, offset) = compute_vocab_shard(100, 4, 0);
        assert_eq!(size, 25);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_vocab_shard_remainder() {
        let (size, offset) = compute_vocab_shard(100, 3, 0);
        assert_eq!(size, 34);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_vocab_shard_remainder_rank2() {
        let (size, offset) = compute_vocab_shard(100, 3, 2);
        assert_eq!(size, 33);
        assert_eq!(offset, 67);
    }

    #[test]
    fn test_tensor_parallel_config_creation() {
        let config = TensorParallelConfig::new(2, 0, vec![0, 1]);
        assert!(config.is_some());

        let config = config.unwrap();
        assert_eq!(config.world_size, 2);
        assert_eq!(config.rank, 0);
        assert!(config.is_first_rank());
    }

    #[test]
    fn test_tensor_parallel_config_invalid() {
        let config = TensorParallelConfig::new(0, 0, vec![]);
        assert!(config.is_none());

        let config = TensorParallelConfig::new(2, 3, vec![0, 1]);
        assert!(config.is_none());

        let config = TensorParallelConfig::new(2, 0, vec![0]);
        assert!(config.is_none());
    }
}
