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
}
