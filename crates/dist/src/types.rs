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
