//! `FlashAttentionV2` (tiling + online softmax) implementation.
//!
//! Split out of `kernel.rs` to keep the facade file under the project's
//! 800-line soft cap. The flash-V2 algorithm processes K/V blocks one at a
//! time and rescales the running output via the online-softmax recurrence
//! (running max + running denominator), which avoids materialising the full
//! attention matrix.

use super::super::util::softmax_last_dim;
use super::FlashAttention;
use candle_core::{Result, Tensor};

/// `FlashAttentionV2`. See the type definition for fields and behavior.
#[derive(Debug)]
pub struct FlashAttentionV2 {
    pub(super) scale: f32,
    pub(super) block_size: usize,
    pub(super) num_heads: usize,
    pub(super) head_dim: usize,
}

impl FlashAttentionV2 {
    #[must_use]
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            scale,
            block_size: 64,
            num_heads,
            head_dim,
        }
    }

    #[must_use]
    pub const fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (_batch_size, _num_heads_q, _seq_len_q, _head_dim) = q.dims4()?;
        let (_, _, seq_len_k, _) = k.dims4()?;

        if seq_len_k <= 128 {
            return self.forward_standard(q, k, v);
        }

        self.forward_flash_v2(q, k, v)
    }

    fn forward_standard(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let qk = q.matmul(&k.t()?)?;
        // H-12 #2: replaced `qk.broadcast_mul(scale_tensor)` with `qk.affine(scale, 0.0)`.
        // Per H-9 profile (MED #5): eliminates per-call 0-D `Tensor::new(scale, device)`
        // allocation. `affine` fuses the scaling into a single kernel without materializing
        // a scalar tensor. Same fix as H-11 #2 (GQA/util) and H-12 #1 (MLA).
        let qk_scaled = qk.affine(f64::from(self.scale), 0.0)?;
        let attn = softmax_last_dim(&qk_scaled)?;
        attn.matmul(v)
    }

    fn forward_flash_v2(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (batch_size, num_heads_q, _seq_len_q, _head_dim) = q.dims4()?;
        let (_, num_heads_k, _seq_len_k, _) = k.dims4()?;
        debug_assert_eq!(num_heads_q, self.num_heads);

        let mut outputs = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let q_b = q.narrow(0, b, 1)?.squeeze(0)?;
            let k_b = k.narrow(0, b, 1)?.squeeze(0)?;
            let v_b = v.narrow(0, b, 1)?.squeeze(0)?;

            let mut head_outputs = Vec::with_capacity(num_heads_q);

            for h in 0..num_heads_q {
                let q_h = q_b.narrow(0, h, 1)?.squeeze(0)?;
                let k_h = k_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;
                let v_h = v_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;

                let out_h = self.compute_flash_attention_block(&q_h, &k_h, &v_h)?;
                head_outputs.push(out_h);
            }

            let batch_out = Tensor::stack(&head_outputs, 0)?;
            outputs.push(batch_out);
        }

        Tensor::stack(&outputs, 0)
    }

    fn compute_flash_attention_block(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let seq_len_k = k.dims()[0];
        let block_size = self.block_size.min(seq_len_k);

        let num_blocks = seq_len_k.div_ceil(block_size);
        let seq_len_q = q.dims()[0];

        // H-12 #2: removed `let scale_tensor = Tensor::new(self.scale, q.device())?;`.
        // Replaced per-block `broadcast_mul(&scale_tensor)` with `affine(self.scale, 0.0)`
        // to eliminate the 0-D scalar tensor allocation that was repeated per-block.
        let mut final_output = Tensor::zeros(
            (seq_len_q, self.head_dim),
            candle_core::DType::F32,
            q.device(),
        )?;

        let mut running_m = Tensor::zeros((seq_len_q, 1), candle_core::DType::F32, q.device())?;
        let mut running_l: Option<Tensor> = None;

        for block_idx in 0..num_blocks {
            let start_k = block_idx * block_size;
            let end_k = (start_k + block_size).min(seq_len_k);
            let actual_block_size = end_k - start_k;

            let k_block = k.narrow(0, start_k, actual_block_size)?;
            let v_block = v.narrow(0, start_k, actual_block_size)?;

            let qk_block = q.matmul(&k_block.t()?)?;
            let qk_scaled = qk_block.affine(f64::from(self.scale), 0.0)?;

            let block_m = qk_scaled.max_keepdim(1)?;
            let block_p = qk_scaled.broadcast_sub(&block_m)?.exp()?;
            let block_l = block_p.sum_keepdim(1)?;

            let m_diff = block_m.broadcast_sub(&running_m)?;
            let correction = m_diff.exp()?;

            let scaled_output = if let Some(ref running_l_val) = running_l {
                let scaled =
                    final_output.broadcast_mul(&running_l_val.broadcast_mul(&correction)?)?;
                scaled.broadcast_div(&block_l)?
            } else {
                final_output
            };

            let block_out = block_p.matmul(&v_block)?;
            final_output = scaled_output.broadcast_add(&block_out)?;

            running_m = block_m;
            running_l = Some(block_l);
        }

        if let Some(l) = running_l {
            final_output = final_output.broadcast_div(&l)?;
        }

        Ok(final_output)
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_with_causal_mask(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (_, _, _seq_len_q, _) = q.dims4()?;
        let (_, _, seq_len_k, _) = k.dims4()?;

        if seq_len_k <= 128 {
            return self.forward_standard(q, k, v);
        }

        self.forward_flash_v2_with_causal(q, k, v)
    }

    fn forward_flash_v2_with_causal(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (batch_size, num_heads_q, _seq_len_q, _head_dim) = q.dims4()?;
        let (_, num_heads_k, _seq_len_k, _) = k.dims4()?;

        let mut outputs = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let q_b = q.narrow(0, b, 1)?.squeeze(0)?;
            let k_b = k.narrow(0, b, 1)?.squeeze(0)?;
            let v_b = v.narrow(0, b, 1)?.squeeze(0)?;

            let mut head_outputs = Vec::with_capacity(num_heads_q);

            for h in 0..num_heads_q {
                let q_h = q_b.narrow(0, h, 1)?.squeeze(0)?;
                let k_h = k_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;
                let v_h = v_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;

                let out_h = self.compute_flash_attention_causal(&q_h, &k_h, &v_h)?;
                head_outputs.push(out_h);
            }

            let batch_out = Tensor::stack(&head_outputs, 0)?;
            outputs.push(batch_out);
        }

        Tensor::stack(&outputs, 0)
    }

    fn compute_flash_attention_causal(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let seq_len_k = k.dims()[0];
        let block_size = self.block_size.min(seq_len_k);
        let seq_len_q = q.dims()[0];

        let num_blocks = seq_len_k.div_ceil(block_size);

        // H-12 #2: removed `let scale_tensor = Tensor::new(self.scale, q.device())?;`.
        // Replaced per-block `broadcast_mul(&scale_tensor)` with `affine(self.scale, 0.0)`.
        let mut final_output = Tensor::zeros(
            (seq_len_q, self.head_dim),
            candle_core::DType::F32,
            q.device(),
        )?;
        let mut running_m = Tensor::zeros((seq_len_q, 1), candle_core::DType::F32, q.device())?;
        let mut running_l: Option<Tensor> = None;

        for block_idx in 0..num_blocks {
            let start_k = block_idx * block_size;
            let end_k = (start_k + block_size).min(seq_len_k);
            let actual_block_size = end_k - start_k;

            let k_block = k.narrow(0, start_k, actual_block_size)?;
            let v_block = v.narrow(0, start_k, actual_block_size)?;

            let qk_block = q.matmul(&k_block.t()?)?;
            let qk_scaled = qk_block.affine(f64::from(self.scale), 0.0)?;

            let causal_mask =
                Self::create_causal_mask(&[seq_len_q, actual_block_size], start_k, q.device())?;
            let qk_masked = qk_scaled.broadcast_add(&causal_mask)?;

            let block_m = qk_masked.max_keepdim(1)?;
            let block_p = qk_masked.broadcast_sub(&block_m)?.exp()?;
            let block_l = block_p.sum_keepdim(1)?;

            let m_diff = block_m.broadcast_sub(&running_m)?;
            let correction = m_diff.exp()?;

            let scaled_output = if let Some(ref running_l_val) = running_l {
                let scaled =
                    final_output.broadcast_mul(&running_l_val.broadcast_mul(&correction)?)?;
                scaled.broadcast_div(&block_l)?
            } else {
                final_output
            };

            let block_out = block_p.matmul(&v_block)?;
            final_output = scaled_output.broadcast_add(&block_out)?;

            running_m = block_m;
            running_l = Some(block_l);
        }

        if let Some(l) = running_l {
            final_output = final_output.broadcast_div(&l)?;
        }

        Ok(final_output)
    }

    fn create_causal_mask(
        dims: &[usize],
        start_k: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let seq_len_q = dims[0];
        let block_size = dims[1];

        let mut mask_data = Vec::with_capacity(seq_len_q * block_size);

        for q_idx in 0..seq_len_q {
            for k_idx in 0..block_size {
                let global_k_idx = start_k + k_idx;
                if q_idx > global_k_idx {
                    mask_data.push(f32::NEG_INFINITY);
                } else {
                    mask_data.push(0.0);
                }
            }
        }

        Tensor::from_slice(&mask_data, (seq_len_q, block_size), device)
    }
}

impl FlashAttention for FlashAttentionV2 {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        Self::forward(self, q, k, v)
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
        let sdpa = super::scaled_dot_product::ScaledDotProductAttention::new(self.head_dim)
            .with_tile_size(tile_size);
        sdpa.compute_tiled(q, k, v, tile_size)
    }
}
