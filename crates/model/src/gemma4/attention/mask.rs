//! Sliding-window causal mask helpers for Gemma4 attention.
//!
//! The mask is `-inf` at positions where either (a) the key violates
//! causal ordering (`k_pos > q_pos`) or (b) the key is older than the
//! configured `sliding_window`. Otherwise the entry is `0`.

use candle_core::{Device, Result, Tensor};

use super::Gemma4Attention;

impl Gemma4Attention {
    fn key_position(key_idx: usize, kv_seq: usize, q_seq: usize, positions: &[usize]) -> usize {
        if kv_seq == q_seq {
            positions.get(key_idx).copied().unwrap_or(key_idx)
        } else {
            key_idx
        }
    }

    pub(super) fn sliding_causal_mask(
        &self,
        q_seq: usize,
        kv_seq: usize,
        query_positions: &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        let mut mask_data = vec![0f32; q_seq * kv_seq];
        for qi in 0..q_seq {
            let q_pos = query_positions.get(qi).copied().unwrap_or(qi);
            for kj in 0..kv_seq {
                let k_pos = Self::key_position(kj, kv_seq, q_seq, query_positions);
                let in_window = q_pos.saturating_sub(k_pos) < self.sliding_window;
                let causal = k_pos <= q_pos;
                if !(causal && in_window) {
                    mask_data[qi * kv_seq + kj] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_slice(&mask_data, (1, 1, q_seq, kv_seq), device)
    }
}
