//! `GqaAttention::forward` + the production dispatcher surface
//! (`paged_attention_fn`, `tiled_attention_fn`, `flash_attention_fn`,
//! `run_attention_fn`).
//!
//! `forward` is a low-level primitive — it does **not** apply causal
//! masking. Production code paths route through
//! `RopeGqaAttention::forward_prefill` / `forward_decode` (in
//! `rope_gqa.rs`), which call one of the `_fn` dispatchers below with
//! `causal = true`.
//!
//! Performance notes (H-11): see the `// H-11 #N` comments inline for
//! the three optimization attempts that were made and reverted.

use super::GqaAttention;
use crate::components::attention::{GqaFlashAttention, paged_attention, tiled_attention};
use candle_core::{Module, Result, Tensor};
use tracing::trace;

impl GqaAttention {
    /// Pre-scale Q by `attn_factor` (YaRN §3.3 attention-temperature scaling).
    ///
    /// Mathematically equivalent to applying `attn_factor` to the post-`Q@K^T`
    /// logits: `(Q * attn_factor) @ K^T = attn_factor * (Q @ K^T)`, and the
    /// lower-level attention functions (`paged_attention`, `tiled_attention`,
    /// `GqaFlashAttention::forward`) then apply their internal `1/sqrt(d)`
    /// scale. softmax is invariant to positive scalar multiplication, so the
    /// final attention distribution equals `softmax(Q @ K^T * attn_factor /
    /// sqrt(d))` — identical to the standard forward path's
    /// `qk.affine(attn_factor / sqrt(d), 0.0)`.
    ///
    /// **No-op** when `attn_factor` is `None` or `Some(1.0)` (within
    /// `f32::EPSILON`) — Q is returned unchanged, no allocation, no kernel
    /// launch. This is the common case for non-YaRN models.
    ///
    /// Critical caveat: must scale by `attn_factor` only, never by
    /// `attn_factor * base_scale` — the `1/sqrt(d)` factor is the
    /// responsibility of the downstream attention function.
    fn apply_attn_factor(&self, q: Tensor) -> Result<Tensor> {
        match self.attn_factor {
            Some(f) if (f - 1.0).abs() > f32::EPSILON => q.affine(f64::from(f), 0.0),
            _ => Ok(q),
        }
    }

    /// Run the layer forward pass over the input.
    /// # Caution: No causal masking
    ///
    /// This is a low-level primitive. It does **NOT** apply causal masking.
    /// Use [`crate::components::attention::RopeGqaAttention::forward_prefill`] or
    /// [`crate::components::attention::RopeGqaAttention::forward_decode`] (which
    /// dispatch to `paged_attention_fn`, `tiled_attention_fn`, or
    /// `flash_attention_fn`, all of which apply causal masking) for
    /// production inference.
    ///
    /// The fused path (`config.use_fused = true`) hardcodes `causal = false`
    /// because this method is not used in production — see the `// invariant:`
    /// comment at the branch site.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    #[allow(clippy::many_single_char_names)]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        trace!(
            batch_size,
            seq_len,
            head_dim = self.head_dim,
            "GqaAttention forward started"
        );

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        let q = self.apply_q_norm(q, batch_size, seq_len)?;
        let k = self.apply_k_norm(k, batch_size, seq_len)?;

        // H-11 #3 (attempted, reverted): removing `.contiguous()?` here broke
        // `Tensor::matmul(&q, &k_t)` downstream — candle's CPU matmul kernel
        // rejects non-contiguous LHS (`MatMulUnexpectedStriding` error). Keep
        // the contiguous call; the cost is one copy of Q at (B, H, S, D).
        let q = q.transpose(1, 2)?.contiguous()?;

        if self.config.use_fused {
            // Honour `attn_factor` (YaRN §3.3): pre-scale Q by `attn_factor`
            // before delegating to `GqaFlashAttention::forward`, which applies
            // its own `1/sqrt(d)` internally. Mathematically equivalent to the
            // standard path's `qk.affine(attn_factor / sqrt(d), 0.0)` —
            // softmax is invariant to positive scalar multiplication. No-op
            // when `attn_factor` is `None` or `Some(1.0)`.
            let q = self.apply_attn_factor(q)?;
            // H-11 #3 (attempted, reverted): removing `.contiguous()?` here broke
            // `Tensor::matmul(&attn_weights, &v_expanded)` inside `flash.forward`.
            // Even though flash internally calls `.transpose(2, 3)?.contiguous()?`
            // for the k path, the v path (`Tensor::matmul(attn, v_expanded)`) has
            // no such copy — v_expanded is strided from k_heads.transpose(1, 2).
            // Same `MatMulUnexpectedStriding` failure mode as q/v on the standard path.
            let k_heads = k.transpose(1, 2)?.contiguous()?;
            let v_heads = v.transpose(1, 2)?.contiguous()?;
            // invariant: causal is hardcoded to false here because
            // GqaAttention::forward is a low-level primitive; production code
            // paths route through RopeGqaAttention::forward_prefill/forward_decode
            // which use flash_attention_fn(..., causal=true) instead.
            let flash =
                GqaFlashAttention::new(self.num_heads, self.num_kv_heads, self.head_dim, false);
            let attn_output = flash.forward(&q, &k_heads, &v_heads)?;
            let attn_output = attn_output.transpose(1, 2)?;
            let attn_output =
                attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
            let o = self.o_proj.forward(&attn_output)?;
            trace!(output_shape = ?o.dims(), "GqaAttention fused forward completed");
            return Ok(o);
        }

        // H-11 #1 (attempted, reverted): tried to skip `expand_kv` materialization
        // by reshaping q from `(B, H_q, S_q, D)` to `(B, H_kv, repeat, S_q, D)`
        // and using `Tensor::matmul` against `k_t = (B, H_kv, D, S_k)`.
        //
        // FAILED because candle's `Tensor::matmul` (cpu_backend/mod.rs:1329-1351)
        // requires the batch-dim product to match between LHS and RHS — the
        // `(B, H_kv, repeat)` on q_r vs `(B, H_kv, 1)` on k_t (after unsqueeze)
        // gives products `B*H_kv*repeat` vs `B*H_kv*1` (e.g. 16 vs 8 for the
        // standard 8/4 test case), and matmul rejects with
        // "shape mismatch in matmul". `Tensor::broadcast_matmul` would handle
        // it but forces `.contiguous()?` on the broadcast side, which would
        // materialize `(B, H_kv, repeat, D, S_k)` — i.e. `repeat` × the original
        // k_t size, defeating the optimization entirely.
        //
        // The expected savings (`~12×` less K/V memory traffic on qwen3-7B
        // with repeat=7) do not justify the refactor risk without a custom
        // fused matmul kernel that supports implicit GQA broadcasting.
        // Deferred to follow-up work (H-12+ or a custom kernel PR).
        let k = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;

        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let qk = Tensor::matmul(&q, &k_t)?;

        // H-11 #2: replaced `qk.mul(broadcast(scalar_tensor))` with `qk.affine(scale, 0.0)`.
        // The scalar tensor was re-allocated and broadcast to O(B*H*S*S) every forward;
        // `affine` fuses the scaling into the existing kernel without materializing a broadcast tensor.
        //
        // When `attn_factor` is set, the score scale is multiplied
        // by it (YaRN §3.3 attention-temperature scaling). `attn_factor = 1.0`
        // (or None) is a no-op; `attn_factor < 1.0` sharpens the softmax,
        // `attn_factor > 1.0` flattens it.
        let base_scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_scale = self.attn_factor.unwrap_or(1.0) * base_scale;
        let qk = qk.affine(f64::from(attn_scale), 0.0)?;
        // H-11 #3: `candle_nn::ops::softmax` already returns a contiguous tensor
        // (verified in candle-nn 0.10.2 src/ops.rs:22-29 — final op is
        // `broadcast_div`, which produces a fresh contiguous tensor). The
        // explicit `.contiguous()?` is a redundant is_contiguous check + clone.
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        // H-11 #3 (attempted, reverted): removing `.contiguous()?` on v broke
        // `Tensor::matmul(&attn_weights, &v)` — candle's CPU matmul kernel
        // rejects non-contiguous batch dimensions in the RHS
        // (`MatMulUnexpectedStriding`). Same constraint as q.contiguous() above.
        let attn_output = Tensor::matmul(&attn_weights, &v.contiguous()?)?;

        let attn_output = attn_output.transpose(1, 2)?;

        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        let o = self.o_proj.forward(&attn_output)?;

        trace!(output_shape = ?o.dims(), "GqaAttention forward completed");

        Ok(o)
    }

    /// Paged attention. Honours `attn_factor` (YaRN §3.3
    /// attention-temperature scaling) — Q is pre-scaled by `attn_factor`
    /// before being passed to `paged_attention`, which applies its own
    /// `1/sqrt(d)` internally. Mathematically equivalent to the standard
    /// path's `qk.affine(attn_factor / sqrt(d), 0.0)`.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn paged_attention_fn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q = self.apply_attn_factor(q.clone())?;
        let attn_output = paged_attention(&q, k, v, self.num_heads, self.head_dim)?;
        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    /// Tiled attention. Honours `attn_factor` (YaRN §3.3) — Q is
    /// pre-scaled by `attn_factor` before `tiled_attention` applies its
    /// internal `1/sqrt(d)`. See [`Self::paged_attention_fn`] for the
    /// equivalence argument.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn tiled_attention_fn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q = self.apply_attn_factor(q.clone())?;
        let tile_size = self.config.tile_size.unwrap_or(16);
        let attn_output = tiled_attention(&q, k, v, self.num_heads, tile_size)?;
        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    /// Flash attention. Honours `attn_factor` (YaRN §3.3) — Q is
    /// pre-scaled by `attn_factor` before `GqaFlashAttention::forward`
    /// applies its internal `1/sqrt(d)`. See [`Self::paged_attention_fn`]
    /// for the equivalence argument.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn flash_attention_fn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q = self.apply_attn_factor(q.clone())?;
        let flash = GqaFlashAttention::new(self.num_heads, self.num_kv_heads, self.head_dim, true);
        let attn_output = flash.forward(&q, k, v)?;
        let batch_size = q.dims()[0];
        let seq_len = q.dims()[2];
        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }

    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// Prefill/decode attention: flash when `use_fused`, else tiled or paged matmul.
    pub fn run_attention_fn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        if self.config.use_fused {
            return self.flash_attention_fn(q, k, v);
        }
        let tile_size = self.config.tile_size.unwrap_or(16);
        if q.dims()[2] > tile_size {
            self.tiled_attention_fn(q, k, v)
        } else {
            self.paged_attention_fn(q, k, v)
        }
    }
}
