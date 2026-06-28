// crates/model/src/components/gated_delta/state.rs
//
// State-space state types for the Gated DeltaNet (GDN) layer:
// `GatedDeltaConfig` and `GatedDeltaState`.

use candle_core::{DType, Result as CandleResult, Tensor};

/// `GatedDeltaConfig`: gated delta configuration.
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaConfig {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
}

impl GatedDeltaConfig {
    #[must_use]
    pub const fn key_dim(&self) -> usize {
        self.num_k_heads * self.key_head_dim
    }

    #[must_use]
    pub const fn value_dim(&self) -> usize {
        self.num_v_heads * self.value_head_dim
    }

    #[must_use]
    pub const fn qkv_proj_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    #[must_use]
    pub const fn conv_state_width(&self) -> usize {
        self.conv_kernel_size.saturating_sub(1)
    }
}

/// Fixed-size recurrent + causal conv cache for GDN decode.
#[derive(Clone, Debug)]
pub struct GatedDeltaState {
    pub recurrent: Tensor,
    pub conv: Tensor,
}

impl GatedDeltaState {
    /// Runs the operation.
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
