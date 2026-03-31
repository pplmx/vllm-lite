use candle_core::{DType, Device, Result, Tensor};

pub const BLOCK_SIZE: usize = 16;

#[allow(dead_code)]
fn quantize_block(data: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if max_abs == 0.0 {
        return (vec![0; data.len()], 1.0);
    }
    let scale = max_abs / 127.0;
    let quantized: Vec<i8> = data.iter().map(|v| (v / scale).round() as i8).collect();
    (quantized, scale)
}

#[allow(dead_code)]
fn dequantize_block(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&v| v as f32 * scale).collect()
}

#[allow(dead_code)]
pub struct PagedKvCache {
    key_cache: Vec<Tensor>,
    value_cache: Vec<Tensor>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    device: Device,
    pub quantized: bool,
    pub scales: Vec<f32>,
}

impl PagedKvCache {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        num_blocks: usize,
        device: Device,
        quantized: bool,
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

        let scales = vec![1.0f32; num_layers];

        Ok(Self {
            key_cache,
            value_cache,
            num_layers,
            num_heads,
            head_dim,
            block_size: BLOCK_SIZE,
            device,
            quantized,
            scales,
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

    pub fn get_scale(&self, layer_idx: usize) -> f32 {
        self.scales.get(layer_idx).copied().unwrap_or(1.0)
    }

    fn update_scale(&mut self, layer_idx: usize, new_scale: f32) {
        if layer_idx < self.scales.len() {
            self.scales[layer_idx] = new_scale;
        }
    }

    pub fn write_kv(
        &mut self,
        layer_idx: usize,
        block_id: usize,
        token_offset: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        if layer_idx >= self.num_layers {
            return Err(candle_core::Error::msg(format!(
                "layer_idx {} out of bounds for {} layers",
                layer_idx, self.num_layers
            )));
        }

        let num_blocks = self.num_blocks();
        if block_id >= num_blocks {
            return Err(candle_core::Error::msg(format!(
                "block_id {} out of bounds for {} blocks",
                block_id, num_blocks
            )));
        }

        if token_offset >= self.block_size {
            return Err(candle_core::Error::msg(format!(
                "token_offset {} out of bounds for block size {}",
                token_offset, self.block_size
            )));
        }

        let k_dims = k.dims();
        if k_dims.len() != 3 {
            return Err(candle_core::Error::msg(format!(
                "Expected k to have 3 dimensions, got {}",
                k_dims.len()
            )));
        }
        if k_dims[0] != 1 || k_dims[1] != self.num_heads || k_dims[2] != self.head_dim {
            return Err(candle_core::Error::msg(format!(
                "Expected k shape [1, {}, {}], got {:?}",
                self.num_heads, self.head_dim, k_dims
            )));
        }

        let v_dims = v.dims();
        if v_dims.len() != 3 {
            return Err(candle_core::Error::msg(format!(
                "Expected v to have 3 dimensions, got {}",
                v_dims.len()
            )));
        }
        if v_dims[0] != 1 || v_dims[1] != self.num_heads || v_dims[2] != self.head_dim {
            return Err(candle_core::Error::msg(format!(
                "Expected v shape [1, {}, {}], got {:?}",
                self.num_heads, self.head_dim, v_dims
            )));
        }

        let key_block = self.key_cache[layer_idx]
            .narrow(0, block_id, 1)?
            .squeeze(0)?;
        let value_block = self.value_cache[layer_idx]
            .narrow(0, block_id, 1)?
            .squeeze(0)?;

        let mut k_block_3d: Vec<Vec<Vec<f32>>> = key_block.to_vec3()?;
        let mut v_block_3d: Vec<Vec<Vec<f32>>> = value_block.to_vec3()?;

        let k_squeezed = k.squeeze(0)?;
        let v_squeezed = v.squeeze(0)?;

        for h in 0..self.num_heads {
            let k_head = k_squeezed.narrow(0, h, 1)?.squeeze(0)?.to_vec1()?;
            let v_head = v_squeezed.narrow(0, h, 1)?.squeeze(0)?.to_vec1()?;

            k_block_3d[h][token_offset][..self.head_dim].copy_from_slice(&k_head);
            v_block_3d[h][token_offset][..self.head_dim].copy_from_slice(&v_head);
        }

        let k_flat: Vec<f32> = k_block_3d
            .into_iter()
            .flat_map(|inner| inner.into_iter().flatten())
            .collect();
        let v_flat: Vec<f32> = v_block_3d
            .into_iter()
            .flat_map(|inner| inner.into_iter().flatten())
            .collect();

        let (k_final, v_final, _scale) = if self.quantized {
            let k_max = k_flat.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let v_max = v_flat.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let max_val = k_max.max(v_max);
            let scale = if max_val > 0.0 { max_val / 127.0 } else { 1.0 };

            let k_quant: Vec<f32> = k_flat.iter().map(|v| (*v / scale).round()).collect();
            let v_quant: Vec<f32> = v_flat.iter().map(|v| (*v / scale).round()).collect();

            self.update_scale(layer_idx, scale);
            (k_quant, v_quant, scale)
        } else {
            (k_flat, v_flat, 1.0)
        };

        let updated_key_block = Tensor::from_slice(
            &k_final,
            (self.num_heads, self.block_size, self.head_dim),
            &self.device,
        )?;
        let updated_value_block = Tensor::from_slice(
            &v_final,
            (self.num_heads, self.block_size, self.head_dim),
            &self.device,
        )?;

        let mut key_parts = Vec::new();
        let mut value_parts = Vec::new();
        for b in 0..num_blocks {
            if b == block_id {
                key_parts.push(updated_key_block.unsqueeze(0)?);
                value_parts.push(updated_value_block.unsqueeze(0)?);
            } else {
                let kb = self.key_cache[layer_idx]
                    .narrow(0, b, 1)?
                    .squeeze(0)?
                    .unsqueeze(0)?;
                let vb = self.value_cache[layer_idx]
                    .narrow(0, b, 1)?
                    .squeeze(0)?
                    .unsqueeze(0)?;
                key_parts.push(kb);
                value_parts.push(vb);
            }
        }

        self.key_cache[layer_idx] = Tensor::cat(&key_parts, 0)?;
        self.value_cache[layer_idx] = Tensor::cat(&value_parts, 0)?;

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

        if self.quantized {
            let scale = self.get_scale(layer_idx);
            let k_data: Vec<f32> = k.flatten_all()?.to_vec1()?;
            let v_data: Vec<f32> = v.flatten_all()?.to_vec1()?;

            let k_dequant: Vec<f32> = k_data.iter().map(|x| x * scale).collect();
            let v_dequant: Vec<f32> = v_data.iter().map(|x| x * scale).collect();

            let k_shape = k.dims();
            let v_shape = v.dims();

            let k = Tensor::from_slice(&k_dequant, k_shape, &self.device)?;
            let v = Tensor::from_slice(&v_dequant, v_shape, &self.device)?;

            Ok((k, v))
        } else {
            Ok((k, v))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_kv_cache_creation() -> Result<()> {
        let device = Device::Cpu;
        let cache = PagedKvCache::new(2, 4, 32, 10, device, false)?;

        assert_eq!(cache.num_layers(), 2);
        assert_eq!(cache.num_blocks(), 10);
        Ok(())
    }

    #[test]
    fn test_paged_kv_cache_single_layer() -> Result<()> {
        let device = Device::Cpu;
        let cache = PagedKvCache::new(1, 8, 64, 5, device, false)?;

        assert_eq!(cache.num_layers(), 1);
        assert_eq!(cache.num_blocks(), 5);
        Ok(())
    }

    #[test]
    fn test_paged_kv_cache_tensor_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cache = PagedKvCache::new(2, 4, 32, 10, device, false)?;

        let key_shape = cache.key_cache[0].dims();
        assert_eq!(key_shape, &[10, 4, 16, 32]);

        let value_shape = cache.value_cache[0].dims();
        assert_eq!(value_shape, &[10, 4, 16, 32]);
        Ok(())
    }

    #[test]
    fn test_write_kv_basic() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 8, 4, device.clone(), false)?;

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
        let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

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
        let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

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

    #[test]
    fn test_write_kv_invalid_layer_idx() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

        let k = Tensor::ones((1, 2, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 2, 4), DType::F32, &device)?;

        let result = cache.write_kv(1, 0, 0, &k, &v);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_write_kv_invalid_block_id() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

        let k = Tensor::ones((1, 2, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 2, 4), DType::F32, &device)?;

        let result = cache.write_kv(0, 4, 0, &k, &v);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_write_kv_invalid_token_offset() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

        let k = Tensor::ones((1, 2, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 2, 4), DType::F32, &device)?;

        let result = cache.write_kv(0, 0, 16, &k, &v);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_write_kv_invalid_k_shape() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

        let k = Tensor::ones((1, 3, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 2, 4), DType::F32, &device)?;

        let result = cache.write_kv(0, 0, 0, &k, &v);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_write_kv_invalid_v_shape() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

        let k = Tensor::ones((1, 2, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 2, 5), DType::F32, &device)?;

        let result = cache.write_kv(0, 0, 0, &k, &v);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_quantized_kv_cache() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), true)?;

        let k_data = vec![10.0f32; 8];
        let k = Tensor::from_slice(&k_data, (1, 2, 4), &device)?;
        let v_data = vec![20.0f32; 8];
        let v = Tensor::from_slice(&v_data, (1, 2, 4), &device)?;

        cache.write_kv(0, 0, 0, &k, &v)?;

        let (k_out, v_out) = cache.read_kv(0, &[0], 1)?;

        let k_out_data: Vec<f32> = k_out.flatten_all()?.to_vec1()?;
        let v_out_data: Vec<f32> = v_out.flatten_all()?.to_vec1()?;

        for val in k_out_data.iter() {
            assert!((val - 10.0).abs() < 0.1, "Expected ~10.0, got {}", val);
        }
        for val in v_out_data.iter() {
            assert!((val - 20.0).abs() < 0.1, "Expected ~20.0, got {}", val);
        }

        Ok(())
    }
}
