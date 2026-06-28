// crates/model/src/paged_tensor/tensor_store/buffer.rs
//
// PagedKvCache write/read operations on the K/V buffer:
// `write_kv`, `write_kv_batch`, `read_kv`.

use super::super::quantization::dequantize;
use super::PagedKvCache;
use candle_core::{DType, Result, Tensor};

impl PagedKvCache {
    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn write_kv_batch(
        &mut self,
        layer_idx: usize,
        block_id: usize,
        token_offset: usize,
        k_batch: &Tensor,
        v_batch: &Tensor,
    ) -> Result<()> {
        if layer_idx >= self.num_layers {
            return Err(candle_core::Error::msg(format!(
                "layer_idx {} out of bounds for {} layers",
                layer_idx, self.num_layers
            )));
        }

        let k_dims = k_batch.dims();
        let v_dims = v_batch.dims();

        if k_dims.len() != 4 || v_dims.len() != 4 {
            return Err(candle_core::Error::msg("Expected 4D tensors for batch"));
        }

        if k_dims != v_dims {
            return Err(candle_core::Error::msg(format!(
                "k_batch and v_batch must have same dimensions, got {k_dims:?} vs {v_dims:?}"
            )));
        }

        if k_dims[2] != self.num_heads || k_dims[3] != self.head_dim {
            return Err(candle_core::Error::msg(format!(
                "Expected k_batch shape [1, *, {}, {}], got {:?}",
                self.num_heads, self.head_dim, k_dims
            )));
        }

        let batch_size = k_dims[0];
        let num_tokens = k_dims[1];

        if batch_size != 1 {
            return Err(candle_core::Error::msg("Batch size must be 1 for now"));
        }

        let num_blocks = self.num_blocks();
        let max_possible_tokens = (num_blocks - block_id) * self.block_size;
        if token_offset + num_tokens > max_possible_tokens {
            return Err(candle_core::Error::msg(format!(
                "Token offset {token_offset} + num_tokens {num_tokens} exceeds available space {max_possible_tokens}"
            )));
        }

        let mut current_block = block_id;
        let mut current_offset = token_offset;

        for i in 0..num_tokens {
            let k_slice = k_batch.narrow(1, i, 1)?.squeeze(1)?;
            let v_slice = v_batch.narrow(1, i, 1)?.squeeze(1)?;
            let k_slice = k_slice.reshape((1, self.num_heads, self.head_dim))?;
            let v_slice = v_slice.reshape((1, self.num_heads, self.head_dim))?;
            self.write_kv(layer_idx, current_block, current_offset, &k_slice, &v_slice)?;

            current_offset += 1;
            if current_offset >= self.block_size {
                current_block += 1;
                current_offset = 0;
            }
        }

        Ok(())
    }

    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn write_kv(
        &mut self,
        layer_idx: usize,
        block_id: usize,
        token_offset: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        tracing::trace!(
            layer_idx = layer_idx,
            block_id = block_id,
            token_offset = token_offset,
            "KV cache write"
        );

        if layer_idx >= self.num_layers {
            return Err(candle_core::Error::msg(format!(
                "layer_idx {} out of bounds for {} layers",
                layer_idx, self.num_layers
            )));
        }

        let num_blocks = self.num_blocks();
        if block_id >= num_blocks {
            return Err(candle_core::Error::msg(format!(
                "block_id {block_id} out of bounds for {num_blocks} blocks"
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
            (1, self.num_heads, self.block_size, self.head_dim),
            &self.device,
        )?;
        let updated_value_block = Tensor::from_slice(
            &v_final,
            (1, self.num_heads, self.block_size, self.head_dim),
            &self.device,
        )?;

        self.key_cache[layer_idx] = self.key_cache[layer_idx].slice_assign(
            &[
                block_id..block_id + 1,
                0..self.num_heads,
                0..self.block_size,
                0..self.head_dim,
            ],
            &updated_key_block,
        )?;
        self.value_cache[layer_idx] = self.value_cache[layer_idx].slice_assign(
            &[
                block_id..block_id + 1,
                0..self.num_heads,
                0..self.block_size,
                0..self.head_dim,
            ],
            &updated_value_block,
        )?;

        let hash = Self::compute_block_hash_from_slice(&k_final);
        self.block_hashes[layer_idx].insert(hash, block_id);

        Ok(())
    }

    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    pub fn read_kv(
        &self,
        layer_idx: usize,
        block_ids: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        if block_ids.is_empty() || seq_len == 0 {
            return Ok((
                Tensor::zeros((0, self.num_heads, self.head_dim), DType::F32, &self.device)?,
                Tensor::zeros((0, self.num_heads, self.head_dim), DType::F32, &self.device)?,
            ));
        }

        tracing::trace!(
            layer_idx = layer_idx,
            block_ids = ?block_ids,
            seq_len = seq_len,
            "KV cache read"
        );

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

            let k_dequant = dequantize(&k_data, scale);
            let v_dequant = dequantize(&v_data, scale);

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
    use candle_core::Device;

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

        for val in &k_out_data {
            assert!((val - 10.0).abs() < 0.1, "Expected ~10.0, got {val}");
        }
        for val in &v_out_data {
            assert!((val - 20.0).abs() < 0.1, "Expected ~20.0, got {val}");
        }

        Ok(())
    }

    #[test]
    fn test_quantized_vs_non_quantized() -> Result<()> {
        let device = Device::Cpu;

        let mut cache_no_quant = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;
        let k_data = vec![5.0f32; 8];
        let k = Tensor::from_slice(&k_data, (1, 2, 4), &device)?;
        let v_data = vec![15.0f32; 8];
        let v = Tensor::from_slice(&v_data, (1, 2, 4), &device)?;

        cache_no_quant.write_kv(0, 0, 0, &k, &v)?;
        let (k_no_q, _v_no_q) = cache_no_quant.read_kv(0, &[0], 1)?;

        let mut cache_quant = PagedKvCache::new(1, 2, 4, 4, device, true)?;
        cache_quant.write_kv(0, 0, 0, &k, &v)?;
        let (k_q, _v_q) = cache_quant.read_kv(0, &[0], 1)?;

        let k_no_q_data: Vec<f32> = k_no_q.flatten_all()?.to_vec1()?;
        let k_q_data: Vec<f32> = k_q.flatten_all()?.to_vec1()?;

        for (a, b) in k_no_q_data.iter().zip(k_q_data.iter()) {
            assert!(
                (a - b).abs() < 1.0,
                "Quantized should be close to non-quantized"
            );
        }

        Ok(())
    }

    #[test]
    fn test_quantized_scale_tracking() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(2, 2, 4, 4, device.clone(), true)?;

        let k0 = Tensor::from_slice(&[100.0f32; 8], (1, 2, 4), &device)?;
        let v0 = Tensor::from_slice(&[100.0f32; 8], (1, 2, 4), &device)?;
        cache.write_kv(0, 0, 0, &k0, &v0)?;

        let k1 = Tensor::from_slice(&[50.0f32; 8], (1, 2, 4), &device)?;
        let v1 = Tensor::from_slice(&[50.0f32; 8], (1, 2, 4), &device)?;
        cache.write_kv(1, 0, 0, &k1, &v1)?;

        let scale0 = cache.get_scale(0);
        let scale1 = cache.get_scale(1);

        assert!(scale0 > 0.0);
        assert!(scale1 > 0.0);

        Ok(())
    }

    #[test]
    fn test_write_kv_batch_single_token() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 4, 16, 4, device.clone(), false)?;

        let k = Tensor::randn(0.0f32, 1.0, (1, 1, 4, 16), &device)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, 1, 4, 16), &device)?;

        cache.write_kv_batch(0, 0, 0, &k, &v)?;

        let (k_out, _v_out) = cache.read_kv(0, &[0], 1)?;
        assert_eq!(k_out.dims(), &[1, 4, 16]);

        Ok(())
    }

    #[test]
    fn test_write_kv_batch_multi_token() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 4, 16, 4, device.clone(), false)?;

        let k = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 16), &device)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 16), &device)?;

        cache.write_kv_batch(0, 0, 0, &k, &v)?;

        let (k_out, _v_out) = cache.read_kv(0, &[0], 4)?;
        assert_eq!(k_out.dims(), &[4, 4, 16]);

        Ok(())
    }

    #[test]
    fn test_write_kv_batch_crosses_block_boundary() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 8, 4, device.clone(), false)?;

        let k = Tensor::randn(0.0f32, 1.0, (1, 20, 2, 8), &device)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, 20, 2, 8), &device)?;

        cache.write_kv_batch(0, 0, 0, &k, &v)?;

        let (k_out, _v_out) = cache.read_kv(0, &[0, 1], 20)?;
        assert_eq!(k_out.dims(), &[20, 2, 8]);

        Ok(())
    }

    #[test]
    fn test_write_kv_batch_invalid_layer_idx() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(2, 2, 8, 4, device.clone(), false)?;

        let k = Tensor::ones((1, 4, 2, 8), DType::F32, &device)?;
        let v = Tensor::ones((1, 4, 2, 8), DType::F32, &device)?;

        let result = cache.write_kv_batch(2, 0, 0, &k, &v);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_write_kv_batch_invalid_kv_dims() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 8, 4, device.clone(), false)?;

        let k = Tensor::ones((1, 4, 2, 8), DType::F32, &device)?;
        let v = Tensor::ones((1, 4, 3, 8), DType::F32, &device)?;

        let result = cache.write_kv_batch(0, 0, 0, &k, &v);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_write_kv_batch_wrong_num_heads() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 4, 8, 4, device.clone(), false)?;

        let k = Tensor::ones((1, 4, 2, 8), DType::F32, &device)?;
        let v = Tensor::ones((1, 4, 2, 8), DType::F32, &device)?;

        let result = cache.write_kv_batch(0, 0, 0, &k, &v);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_gqa_expanded_heads_cache() -> Result<()> {
        let device = Device::Cpu;
        let num_heads = 14;
        let _num_kv_heads = 2;
        let head_dim = 64;

        let mut cache = PagedKvCache::new(1, num_heads, head_dim, 4, device.clone(), false)?;

        let k = Tensor::randn(0.0f32, 1.0, (1, 1, num_heads, head_dim), &device)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, 1, num_heads, head_dim), &device)?;

        cache.write_kv_batch(0, 0, 0, &k, &v)?;

        let (k_out, _v_out) = cache.read_kv(0, &[0], 1)?;
        assert_eq!(k_out.dims(), &[1, num_heads, head_dim]);

        Ok(())
    }

    #[test]
    fn test_block_size_getter() -> Result<()> {
        let device = Device::Cpu;
        let cache = PagedKvCache::new(1, 4, 16, 10, device, false)?;

        assert_eq!(cache.block_size(), 16);

        Ok(())
    }

    #[test]
    fn test_write_kv_at_large_num_blocks_slice_assign() -> Result<()> {
        // H-13 (PERF-01): regression test for the Tensor::cat →
        // Tensor::slice_assign rewrite. Verifies that writes at a
        // production-like num_blocks (64) correctly update the targeted
        // block without disturbing other blocks, and that reads see the
        // correct data.
        let device = Device::Cpu;
        let num_blocks = 64;
        let num_heads = 2;
        let head_dim = 64;
        let block_size = 16;
        let mut cache =
            PagedKvCache::new(1, num_heads, head_dim, num_blocks, device.clone(), false)?;

        // Seed block 5 with a unique value.
        let seed_block_id = 5usize;
        let seed_token_offset = 3usize;
        let seed_k = Tensor::full(7.5f32, (1, num_heads, head_dim), &device)?;
        let seed_v = Tensor::full(11.25f32, (1, num_heads, head_dim), &device)?;
        cache.write_kv(0, seed_block_id, seed_token_offset, &seed_k, &seed_v)?;

        // Seed block 7 with a different unique value.
        let other_block_id = 7usize;
        let other_token_offset = 0usize;
        let other_k = Tensor::full(-3.0f32, (1, num_heads, head_dim), &device)?;
        let other_v = Tensor::full(2.5f32, (1, num_heads, head_dim), &device)?;
        cache.write_kv(0, other_block_id, other_token_offset, &other_k, &other_v)?;

        // Read back block 5 only — must see the seeded value at the
        // written offset and zeros at the other offsets in that block.
        let (k_block5, v_block5) = cache.read_kv(0, &[seed_block_id], block_size)?;
        assert_eq!(k_block5.dims(), &[block_size, num_heads, head_dim]);
        let k_data: Vec<f32> = k_block5.flatten_all()?.to_vec1()?;
        let v_data: Vec<f32> = v_block5.flatten_all()?.to_vec1()?;
        let stride = num_heads * head_dim;
        for token in 0..block_size {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let idx = token * stride + h * head_dim + d;
                    let expected_k = if token == seed_token_offset { 7.5 } else { 0.0 };
                    let expected_v = if token == seed_token_offset {
                        11.25
                    } else {
                        0.0
                    };
                    assert!(
                        (k_data[idx] - expected_k).abs() < 1e-5,
                        "block 5 k mismatch at token={token} h={h} d={d}: got {}, expected {}",
                        k_data[idx],
                        expected_k
                    );
                    assert!(
                        (v_data[idx] - expected_v).abs() < 1e-5,
                        "block 5 v mismatch at token={token} h={h} d={d}: got {}, expected {}",
                        v_data[idx],
                        expected_v
                    );
                }
            }
        }

        // Read back block 7 separately to verify cross-block isolation.
        let (k_block7, _) = cache.read_kv(0, &[other_block_id], block_size)?;
        let k7_data: Vec<f32> = k_block7.flatten_all()?.to_vec1()?;
        let idx0 = other_token_offset * stride + 0 * head_dim + 0;
        assert!(
            (k7_data[idx0] - -3.0).abs() < 1e-5,
            "block 7 slot must be -3.0, got {}",
            k7_data[idx0]
        );

        Ok(())
    }

    #[test]
    fn test_write_kv_overwrite_preserves_other_tokens_in_block() -> Result<()> {
        // H-13 (PERF-01): regression test ensuring that two writes to
        // the SAME block at different token offsets do not clobber each
        // other (the slice_assign path writes just the targeted slot).
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

        let k_a = Tensor::full(1.0f32, (1, 2, 4), &device)?;
        let v_a = Tensor::full(2.0f32, (1, 2, 4), &device)?;
        cache.write_kv(0, 0, 0, &k_a, &v_a)?;

        let k_b = Tensor::full(3.0f32, (1, 2, 4), &device)?;
        let v_b = Tensor::full(4.0f32, (1, 2, 4), &device)?;
        cache.write_kv(0, 0, 5, &k_b, &v_b)?;

        let (k_out, v_out) = cache.read_kv(0, &[0], 16)?;
        let k_data: Vec<f32> = k_out.flatten_all()?.to_vec1()?;
        let v_data: Vec<f32> = v_out.flatten_all()?.to_vec1()?;
        let stride = 2 * 4;
        for d in 0..stride {
            assert!(
                (k_data[d] - 1.0).abs() < 1e-5,
                "slot 0 clobbered: {}",
                k_data[d]
            );
            assert!(
                (v_data[d] - 2.0).abs() < 1e-5,
                "slot 0 v clobbered: {}",
                v_data[d]
            );
        }
        let off5 = 5 * stride;
        for d in 0..stride {
            assert!(
                (k_data[off5 + d] - 3.0).abs() < 1e-5,
                "slot 5 k wrong: {}",
                k_data[off5 + d]
            );
            assert!(
                (v_data[off5 + d] - 4.0).abs() < 1e-5,
                "slot 5 v wrong: {}",
                v_data[off5 + d]
            );
        }

        Ok(())
    }
}
