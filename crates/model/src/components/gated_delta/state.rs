//! Gated `DeltaNet` recurrent state: `GatedDeltaConfig` (init parameters) and `GatedDeltaState` (the per-sequence hidden-state tensor).
//!
//! Used by the Qwen3.5 hybrid architecture; the recurrence is
//! implemented in `rule.rs` and this file owns only the storage shapes
//! + initialisation helpers.

// crates/model/src/components/gated_delta/state.rs
//
// State-space state types for the Gated DeltaNet (GDN) layer:
// `GatedDeltaConfig` and `GatedDeltaState`.

use candle_core::{DType, Result as CandleResult, Tensor};

/// Configuration for `GatedDelta`. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaConfig {
    /// Number of K-projection heads (delta-rule keys).
    pub num_k_heads: usize,
    /// Number of V-projection heads (delta-rule values).
    pub num_v_heads: usize,
    /// Per-head K dimension.
    pub key_head_dim: usize,
    /// Per-head V dimension.
    pub value_head_dim: usize,
    /// Causal conv kernel size (typically 4 for Qwen3.5).
    pub conv_kernel_size: usize,
}

impl GatedDeltaConfig {
    /// Total key-projection dimension: `num_k_heads * key_head_dim`.
    #[must_use]
    pub const fn key_dim(&self) -> usize {
        self.num_k_heads * self.key_head_dim
    }

    /// Total value-projection dimension: `num_v_heads * value_head_dim`.
    #[must_use]
    pub const fn value_dim(&self) -> usize {
        self.num_v_heads * self.value_head_dim
    }

    /// Combined QKV projection dimension: `2 * key_dim + value_dim`.
    #[must_use]
    pub const fn qkv_proj_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// Width of the causal conv state buffer: `conv_kernel_size - 1`
    /// (saturating at 0 for a 1-tap conv).
    #[must_use]
    pub const fn conv_state_width(&self) -> usize {
        self.conv_kernel_size.saturating_sub(1)
    }
}

/// Fixed-size recurrent + causal conv cache for GDN decode.
#[derive(Clone, Debug)]
pub struct GatedDeltaState {
    /// Recurrent (key × value) state tensor, shape `[B, num_v_heads, key_head_dim, value_head_dim]`.
    pub recurrent: Tensor,
    /// Causal conv buffer, shape `[B, qkv_proj_dim, conv_kernel_size - 1]`.
    pub conv: Tensor,
}

impl GatedDeltaState {
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new(
        batch: usize,
        config: &GatedDeltaConfig,
        device: &candle_core::Device,
    ) -> CandleResult<Self> {
        Ok(Self {
            recurrent: Tensor::zeros(
                (
                    batch,
                    config.num_v_heads,
                    config.key_head_dim,
                    config.value_head_dim,
                ),
                DType::F32,
                device,
            )?,
            conv: Tensor::zeros(
                (batch, config.qkv_proj_dim(), config.conv_state_width()),
                DType::F32,
                device,
            )?,
        })
    }
}
