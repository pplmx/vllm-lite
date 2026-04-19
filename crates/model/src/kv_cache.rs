use candle_core::{Device, Result, Tensor};

#[deprecated(since = "0.2.0", note = "Use paged_tensor module instead")]
pub use crate::paged_tensor::{
    dequantize, quantization::QuantizedTensor, quantize, tensor_store::PagedKvCache,
};

pub struct MlaKvCache {
    kv_lora_rank: usize,
    block_size: usize,
    num_blocks: usize,
    device: Device,
    cache: Vec<Tensor>,
}

impl MlaKvCache {
    pub fn new(
        num_layers: usize,
        kv_lora_rank: usize,
        block_size: usize,
        num_blocks: usize,
        device: Device,
    ) -> Self {
        let cache: Vec<Tensor> = (0..num_layers)
            .map(|_| {
                Tensor::zeros(
                    (num_blocks, block_size, kv_lora_rank),
                    candle_core::DType::F32,
                    &device,
                )
                .unwrap()
            })
            .collect();

        Self {
            kv_lora_rank,
            block_size,
            num_blocks,
            device,
            cache,
        }
    }

    pub fn write_compressed(
        &mut self,
        layer: usize,
        block_id: usize,
        offset: usize,
        kv: &Tensor,
    ) -> Result<()> {
        let _block = &mut self.cache[layer];
        let seq_len = kv.dims()[1];

        for i in 0..seq_len {
            let token_idx = block_id * self.block_size + offset + i;
            let block_idx = token_idx / self.block_size;
            let block_offset = token_idx % self.block_size;

            if block_idx < self.num_blocks {
                let src_data = kv.narrow(1, i, 1)?.flatten_all()?.to_vec1()?;
                let num_blocks = self.num_blocks;
                let kv_lora_rank = self.kv_lora_rank;
                let block_flat = self.cache[layer].to_vec3()?;
                let flat: Vec<f32> = block_flat
                    .into_iter()
                    .flat_map(|block_2d| block_2d.into_iter().flat_map(|row| row))
                    .collect();

                let mut mutable_flat = flat;
                let write_start = (block_idx * self.block_size + block_offset) * kv_lora_rank;
                for (j, &val) in src_data.iter().enumerate() {
                    mutable_flat[write_start + j] = val;
                }

                let shape = (num_blocks, self.block_size, kv_lora_rank);
                self.cache[layer] = Tensor::from_slice(&mutable_flat, shape, &self.device)?;
            }
        }
        Ok(())
    }

    pub fn read_compressed(
        &self,
        layer: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let mut parts = Vec::new();
        let mut remaining = seq_len;
        let mut current_pos = start_pos;

        while remaining > 0 {
            let block_idx = current_pos / self.block_size;
            let block_offset = current_pos % self.block_size;
            let block_remaining = self.block_size - block_offset;
            let to_read = remaining.min(block_remaining);

            if block_idx < self.num_blocks {
                let tensor = self.cache[layer]
                    .narrow(0, block_idx, 1)?
                    .narrow(1, block_offset, to_read)?
                    .squeeze(0)?;
                parts.push(tensor);
            } else {
                parts.push(Tensor::zeros(
                    (to_read, self.kv_lora_rank),
                    candle_core::DType::F32,
                    &self.device,
                )?);
            }

            remaining -= to_read;
            current_pos += to_read;
        }

        Tensor::cat(&parts, 0)?.unsqueeze(0)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mla_kv_cache_basic() {
        let device = Device::Cpu;
        let mut cache = MlaKvCache::new(1, 512, 8, 16, device.clone());

        let kv_compressed =
            Tensor::randn(0.0f32, 1.0, (1, 1, 512), &device).unwrap();
        cache
            .write_compressed(0, 0, 0, &kv_compressed)
            .unwrap();

        let retrieved = cache.read_compressed(0, 0, 1).unwrap();
        assert_eq!(retrieved.dims(), &[1, 1, 512]);
    }
}
