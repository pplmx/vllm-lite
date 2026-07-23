//! `PagedKV` buffer operations: `write_kv`, `write_kv_batch`, `read_kv`, and the layer-aware batched variants.
//!
//! These are the only entry points attention layers should call to
//! mutate or read the `KV` cache. Block-level accounting is handled by
//! the pool (`pool.rs`); this file is pure read/write logic.
//!
//! Tests for these methods live in `tests.rs` (sibling file) to keep this
//! module under the 800-line soft cap.

// crates/model/src/paged_tensor/tensor_store/buffer.rs
//
// PagedKvCache write/read operations on the K/V buffer:
// `write_kv`, `write_kv_batch`, `read_kv`.

use super::super::quantization::dequantize;
use super::PagedKvCache;
use candle_core::{DType, Result, Tensor};

#[cfg(test)]
mod tests;

impl PagedKvCache {
    /// Batch-write K/V across multiple blocks.
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

    /// Write K/V for the current decode step into the paged cache.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    #[allow(clippy::too_many_lines)] // KV-cache write path: bound checks + reshape + slice_assign in one linear sequence
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

    /// Read cached K/V from the paged `KV` cache for a given sequence.
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
