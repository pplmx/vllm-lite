use candle_core::{DType, Device, Result, Tensor};

pub const BLOCK_SIZE: usize = 16;

#[allow(dead_code)]
pub struct PagedKvCache {
    key_cache: Vec<Tensor>,
    value_cache: Vec<Tensor>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    device: Device,
}

impl PagedKvCache {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        num_blocks: usize,
        device: Device,
    ) -> Result<Self> {
        let mut key_cache = Vec::with_capacity(num_layers);
        let mut value_cache = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let shape = (num_blocks, num_heads, BLOCK_SIZE, head_dim);
            let key = Tensor::zeros(shape, DType::F32, &device)?;
            let value = Tensor::zeros(shape, DType::F32, &device)?;
            key_cache.push(key);
            value_cache.push(value);
        }

        Ok(Self {
            key_cache,
            value_cache,
            num_layers,
            num_heads,
            head_dim,
            block_size: BLOCK_SIZE,
            device,
        })
    }

    pub fn num_blocks(&self) -> usize {
        self.key_cache
            .first()
            .map(|t| t.shape().dims()[0])
            .unwrap_or(0)
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn write_kv(
        &mut self,
        layer_idx: usize,
        block_id: usize,
        token_offset: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        let num_kv_heads = self.num_heads;
        let head_dim = self.head_dim;
        let block = self.block_size;

        let layer_key = self.key_cache[layer_idx].clone();
        let layer_value = self.value_cache[layer_idx].clone();

        let mut new_blocks = Vec::new();
        for b in 0..self.num_blocks() {
            if b == block_id {
                let mut new_heads = Vec::new();
                for h in 0..num_kv_heads {
                    let k_head: Vec<f32> = k.squeeze(0)?.narrow(0, h, 1)?.squeeze(0)?.to_vec1()?;
                    let v_head: Vec<f32> = v.squeeze(0)?.narrow(0, h, 1)?.squeeze(0)?.to_vec1()?;

                    let mut k_data = vec![0.0f32; block * head_dim];
                    let mut v_data = vec![0.0f32; block * head_dim];

                    for i in 0..block {
                        for d in 0..head_dim {
                            let existing: f32 = layer_key
                                .narrow(0, block_id, 1)?
                                .narrow(1, h, 1)?
                                .narrow(2, i, 1)?
                                .narrow(3, d, 1)?
                                .squeeze(0)?
                                .squeeze(0)?
                                .squeeze(0)?
                                .squeeze(0)?
                                .to_vec0()?;
                            k_data[i * head_dim + d] = existing;
                            let existing: f32 = layer_value
                                .narrow(0, block_id, 1)?
                                .narrow(1, h, 1)?
                                .narrow(2, i, 1)?
                                .narrow(3, d, 1)?
                                .squeeze(0)?
                                .squeeze(0)?
                                .squeeze(0)?
                                .squeeze(0)?
                                .to_vec0()?;
                            v_data[i * head_dim + d] = existing;
                        }
                    }

                    if token_offset < block {
                        for d in 0..head_dim {
                            k_data[token_offset * head_dim + d] = k_head[d];
                            v_data[token_offset * head_dim + d] = v_head[d];
                        }
                    }

                    let k_tensor =
                        Tensor::from_vec(k_data.clone(), (block, head_dim), &self.device)?;
                    let v_tensor =
                        Tensor::from_vec(v_data.clone(), (block, head_dim), &self.device)?;

                    new_heads.push((k_tensor, v_tensor));
                }

                let mut k_parts = Vec::new();
                let mut v_parts = Vec::new();
                for (k_t, v_t) in new_heads {
                    k_parts.push(k_t);
                    v_parts.push(v_t);
                }
                let key_block = Tensor::stack(&k_parts, 0)?;
                let value_block = Tensor::stack(&v_parts, 0)?;
                new_blocks.push((key_block, value_block));
            } else {
                let key_block = layer_key.narrow(0, b, 1)?.squeeze(0)?;
                let value_block = layer_value.narrow(0, b, 1)?.squeeze(0)?;
                new_blocks.push((key_block, value_block));
            }
        }

        let mut new_key_parts = Vec::new();
        let mut new_value_parts = Vec::new();
        for (kb, vb) in new_blocks {
            new_key_parts.push(kb.unsqueeze(0)?);
            new_value_parts.push(vb.unsqueeze(0)?);
        }

        self.key_cache[layer_idx] = Tensor::cat(&new_key_parts, 0)?;
        self.value_cache[layer_idx] = Tensor::cat(&new_value_parts, 0)?;

        Ok(())
    }

    pub fn read_kv(
        &self,
        layer_idx: usize,
        block_ids: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let mut k_parts = Vec::new();
        let mut v_parts = Vec::new();

        for (block_idx, &block_id) in block_ids.iter().enumerate() {
            let start_token = block_idx * self.block_size;
            let end_token = std::cmp::min(start_token + self.block_size, seq_len);
            let block_len = end_token - start_token;

            let k_block = self.key_cache[layer_idx]
                .narrow(0, block_id, 1)?
                .narrow(1, 0, self.num_heads)?
                .narrow(2, 0, block_len)?
                .squeeze(0)?;

            let v_block = self.value_cache[layer_idx]
                .narrow(0, block_id, 1)?
                .narrow(1, 0, self.num_heads)?
                .narrow(2, 0, block_len)?
                .squeeze(0)?;

            k_parts.push(k_block);
            v_parts.push(v_block);
        }

        let k = Tensor::cat(&k_parts, 1)?.transpose(0, 1)?;
        let v = Tensor::cat(&v_parts, 1)?.transpose(0, 1)?;

        Ok((k, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_kv_cache_creation() -> Result<()> {
        let device = Device::Cpu;
        let cache = PagedKvCache::new(2, 4, 32, 10, device)?;

        assert_eq!(cache.num_layers(), 2);
        assert_eq!(cache.num_blocks(), 10);
        Ok(())
    }

    #[test]
    fn test_paged_kv_cache_single_layer() -> Result<()> {
        let device = Device::Cpu;
        let cache = PagedKvCache::new(1, 8, 64, 5, device)?;

        assert_eq!(cache.num_layers(), 1);
        assert_eq!(cache.num_blocks(), 5);
        Ok(())
    }

    #[test]
    fn test_paged_kv_cache_tensor_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cache = PagedKvCache::new(2, 4, 32, 10, device)?;

        let key_shape = cache.key_cache[0].dims();
        assert_eq!(key_shape, &[10, 4, 16, 32]);

        let value_shape = cache.value_cache[0].dims();
        assert_eq!(value_shape, &[10, 4, 16, 32]);
        Ok(())
    }

    #[test]
    fn test_write_kv_basic() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 8, 4, device.clone())?;

        let k = Tensor::ones((1, 2, 8), DType::F32, &device)?;
        let v = Tensor::ones((1, 2, 8), DType::F32, &device)?;

        cache.write_kv(0, 0, 0, &k, &v)?;

        let (k_out, v_out) = cache.read_kv(0, &[0], 1)?;
        assert_eq!(k_out.dims(), &[1, 2, 8]);
        assert_eq!(v_out.dims(), &[1, 2, 8]);

        Ok(())
    }

    #[test]
    fn test_write_kv_and_read_kv() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone())?;

        let k_data = vec![1.0f32; 8];
        let k = Tensor::from_slice(&k_data, (1, 2, 4), &device)?;
        let v_data = vec![2.0f32; 8];
        let v = Tensor::from_slice(&v_data, (1, 2, 4), &device)?;

        cache.write_kv(0, 0, 0, &k, &v)?;

        let k_data2 = vec![3.0f32; 8];
        let k2 = Tensor::from_slice(&k_data2, (1, 2, 4), &device)?;
        let v_data2 = vec![4.0f32; 8];
        let v2 = Tensor::from_slice(&v_data2, (1, 2, 4), &device)?;

        cache.write_kv(0, 0, 1, &k2, &v2)?;

        let (k_out, _v_out) = cache.read_kv(0, &[0], 2)?;
        assert_eq!(k_out.dims(), &[2, 2, 4]);

        Ok(())
    }

    #[test]
    fn test_read_kv_multiple_blocks() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone())?;

        let k_data = vec![1.0f32; 8];
        let k = Tensor::from_slice(&k_data, (1, 2, 4), &device)?;
        let v_data = vec![2.0f32; 8];
        let v = Tensor::from_slice(&v_data, (1, 2, 4), &device)?;

        cache.write_kv(0, 0, 0, &k, &v)?;
        cache.write_kv(0, 1, 0, &k, &v)?;

        let (k_out, _v_out) = cache.read_kv(0, &[0, 1], 32)?;
        assert_eq!(k_out.dims(), &[32, 2, 4]);

        Ok(())
    }
}
