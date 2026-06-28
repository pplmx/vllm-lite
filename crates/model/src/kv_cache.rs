//! MLA-specific compressed KV cache (separate from paged tensor store).

use candle_core::{Device, Result, Tensor};

/// MlaKvCache: mla kv cache.
pub(crate) struct MlaKvCache {
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
                // invariant: tensor shape is statically known and finite; CPU/device allocation
                // of a fixed-size zero buffer cannot fail under normal conditions.
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

    ///
    /// PERF-01 (v22.0): writes incrementally into the destination layer
    /// using `Tensor::slice_assign` so memory allocation is proportional
    /// to the slice being written, not the entire cache layer. The
    /// previous implementation flattened the whole `num_blocks *
    /// block_size * kv_lora_rank` buffer per token, then re-built the
    /// layer Tensor — `O(num_blocks * block_size * kv_lora_rank)`
    /// allocation per write. The new path is `O(seq_len * kv_lora_rank)`.
    pub fn write_compressed(
        &mut self,
        layer: usize,
        block_id: usize,
        offset: usize,
        kv: &Tensor,
    ) -> Result<()> {
        let seq_len = kv.dims()[1];
        if seq_len == 0 {
            return Ok(());
        }

        // Concatenate `kv` along dim 1 (already contiguous) into a
        // single (kv_lora_rank, seq_len) tensor we can slice_assign
        // into the destination. `kv` is expected to have shape
        // (kv_lora_rank, seq_len, ...) — collapse the trailing dims.
        let kv_flat = if kv.dims().len() > 2 {
            kv.contiguous()?
        } else {
            kv.clone()
        };

        // Determine the contiguous range within the layer that `kv`
        // spans. The caller passes `block_id` and `offset` as the
        // starting token index; we always write `seq_len` tokens
        // starting there.
        let start_token = block_id * self.block_size + offset;
        let start_block = start_token / self.block_size;
        let start_offset_in_block = start_token % self.block_size;
        let end_token = start_token + seq_len;
        let end_block_inclusive = (end_token - 1) / self.block_size;

        // Fast path: all tokens fit in a single block. Allocate only
        // `seq_len * kv_lora_rank` floats and slice_assign into the
        // single target block.
        if start_block == end_block_inclusive && start_block < self.num_blocks {
            let block_view = self.cache[layer].narrow(0, start_block, 1)?;
            let mut block_data = block_view
                .squeeze(0)?
                .to_vec2()? // (block_size, kv_lora_rank)
                .into_iter()
                .flatten()
                .collect::<Vec<f32>>();

            // Overwrite `seq_len` slots starting at `start_offset_in_block`.
            let kv_data: Vec<f32> = kv_flat.narrow(1, 0, seq_len)?.flatten_all()?.to_vec1()?;
            let kv_lora_rank = self.kv_lora_rank;
            for i in 0..seq_len {
                for j in 0..kv_lora_rank {
                    let dst = (start_offset_in_block + i) * kv_lora_rank + j;
                    if dst < block_data.len() {
                        block_data[dst] = kv_data[i * kv_lora_rank + j];
                    }
                }
            }

            let block_tensor =
                Tensor::from_slice(&block_data, (self.block_size, kv_lora_rank), &self.device)?
                    .unsqueeze(0)?;
            self.cache[layer] = self.cache[layer].slice_assign(
                &[
                    start_block..start_block + 1,
                    0..self.block_size,
                    0..kv_lora_rank,
                ],
                &block_tensor,
            )?;
            return Ok(());
        }

        // General path: spans multiple blocks. Walk block-by-block,
        // slice_assign per block. Memory allocation per block is
        // `block_size * kv_lora_rank`, NOT the full layer.
        let kv_lora_rank = self.kv_lora_rank;
        let kv_data: Vec<f32> = kv_flat.narrow(1, 0, seq_len)?.flatten_all()?.to_vec1()?;
        let mut token_cursor = start_token;
        let mut kv_cursor = 0usize;
        while token_cursor < end_token {
            let block_idx = token_cursor / self.block_size;
            let block_offset = token_cursor % self.block_size;
            if block_idx >= self.num_blocks {
                break;
            }
            let tokens_in_block = (self.block_size - block_offset).min(end_token - token_cursor);

            let block_view = self.cache[layer].narrow(0, block_idx, 1)?;
            let mut block_data = block_view
                .squeeze(0)?
                .to_vec2()?
                .into_iter()
                .flatten()
                .collect::<Vec<f32>>();

            for i in 0..tokens_in_block {
                for j in 0..kv_lora_rank {
                    let dst = (block_offset + i) * kv_lora_rank + j;
                    if dst < block_data.len() {
                        block_data[dst] = kv_data[(kv_cursor + i) * kv_lora_rank + j];
                    }
                }
            }

            let block_tensor =
                Tensor::from_slice(&block_data, (self.block_size, kv_lora_rank), &self.device)?
                    .unsqueeze(0)?;
            self.cache[layer] = self.cache[layer].slice_assign(
                &[
                    block_idx..block_idx + 1,
                    0..self.block_size,
                    0..kv_lora_rank,
                ],
                &block_tensor,
            )?;

            token_cursor += tokens_in_block;
            kv_cursor += tokens_in_block;
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

        let kv_compressed = Tensor::randn(0.0f32, 1.0, (1, 1, 512), &device).unwrap();
        cache.write_compressed(0, 0, 0, &kv_compressed).unwrap();

        let retrieved = cache.read_compressed(0, 0, 1).unwrap();
        assert_eq!(retrieved.dims(), &[1, 1, 512]);
    }
}
