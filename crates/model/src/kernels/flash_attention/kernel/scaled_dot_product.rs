//! `ScaledDotProductAttention` (tiled + sliding-window variants).
//!
//! Split out of `kernel.rs` to keep the facade file under the project's
//! 800-line soft cap. This is the CPU reference attention implementation
//! and the dispatcher used by [`super::FlashAttentionKernel`] when the
//! variant is `Tiled`, `Flash`, or `Standard`.

use super::super::util::softmax_last_dim;
use super::FlashAttention;
use candle_core::{Result, Tensor};

/// `ScaledDotProductAttention`. See the type definition for fields and behavior.
#[derive(Debug)]
pub struct ScaledDotProductAttention {
    pub(super) scale: f32,
    pub(super) tile_size: usize,
}

impl ScaledDotProductAttention {
    #[must_use]
    pub fn new(head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let optimal_tile = if head_dim <= 64 { 32 } else { 64 };
        Self {
            scale,
            tile_size: optimal_tile,
        }
    }

    #[must_use]
    pub const fn with_tile_size(mut self, tile_size: usize) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Compute tiled.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn compute_tiled(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        tile_size: usize,
    ) -> Result<Tensor> {
        let q_shape = q.dims();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_len = q_shape[2];
        let _ = q_shape[3];

        if seq_len <= 32 {
            return self.forward(q, k, v);
        }

        // H-12 #2: removed `let scale_tensor = Tensor::new(self.scale, q.device())?;`.
        // Per-tile savings compound: tiled forward at seq_len=2048 / tile_size=64
        // saves ~32 per-tile `broadcast_mul` calls (per head, per batch).
        let mut all_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);

        for _b in 0..batch_size {
            let mut head_outputs: Vec<Tensor> = Vec::with_capacity(num_heads);

            for h in 0..num_heads {
                let q_bh = q.narrow(1, h, 1)?.squeeze(1)?;
                let k_bh = k.narrow(1, h, 1)?.squeeze(1)?;
                let v_bh = v.narrow(1, h, 1)?.squeeze(1)?;

                let mut tile_outputs: Vec<Tensor> = Vec::new();

                for start in (0..seq_len).step_by(tile_size) {
                    let end = (start + tile_size).min(seq_len);
                    let actual_tile_size = end - start;
                    let q_tile = q_bh.narrow(1, start, actual_tile_size)?;

                    let k_start = 0;
                    let k_len = end.min(seq_len);
                    let k_tile = k_bh.narrow(1, k_start, k_len)?;
                    let v_tile = v_bh.narrow(1, k_start, k_len)?;

                    let qk = q_tile.matmul(&k_tile.t()?)?;
                    let qk_scaled = qk.affine(f64::from(self.scale), 0.0)?;
                    let attn = softmax_last_dim(&qk_scaled)?;
                    let out_tile = attn.matmul(&v_tile)?;

                    tile_outputs.push(out_tile);
                }

                let head_output = Tensor::cat(&tile_outputs, 0)?;
                head_outputs.push(head_output);
            }

            let batch_output = Tensor::stack(&head_outputs, 0)?;
            all_outputs.push(batch_output);
        }

        let result = Tensor::stack(&all_outputs, 0)?;
        Ok(result)
    }

    /// Compute sliding window.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    #[allow(dead_code)] // test-only helper; reachable under cfg(test) only
    pub(crate) fn compute_sliding_window(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window_size: usize,
    ) -> Result<Tensor> {
        let k_len = k.dims()[2];

        if k_len <= window_size {
            return self.forward(q, k, v);
        }

        let k_window = k.narrow(2, k_len.saturating_sub(window_size), window_size)?;
        let v_window = v.narrow(2, k_len.saturating_sub(window_size), window_size)?;

        self.forward(q, &k_window, &v_window)
    }
}

impl FlashAttention for ScaledDotProductAttention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let qk = q.matmul(&k.t()?)?;
        // H-12 #2: replaced `qk.broadcast_mul(scale_tensor)` with `qk.affine(scale, 0.0)`.
        let qk_scaled = qk.affine(f64::from(self.scale), 0.0)?;
        let attn = softmax_last_dim(&qk_scaled)?;
        attn.matmul(v)
    }

    fn forward_with_mask(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _mask: &Tensor,
    ) -> Result<Tensor> {
        self.forward(q, k, v)
    }

    fn forward_tiled(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        tile_size: usize,
    ) -> Result<Tensor> {
        self.compute_tiled(q, k, v, tile_size)
    }
}
