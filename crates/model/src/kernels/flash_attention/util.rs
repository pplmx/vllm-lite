// crates/model/src/kernels/flash_attention/util.rs
//
// Internal helpers used by the flash attention kernels:
// `AttentionStats` and `softmax_last_dim`.

use candle_core::{Result, Tensor};

/// Telemetry snapshot for Attention: counters, gauges, and percentile latencies. Cloned and serialized on every metrics export.
#[derive(Debug, Clone, Default)]
pub struct AttentionStats {
    pub forward_count: u64,
    pub tiled_forward_count: u64,
    pub total_tokens: u64,
}

impl AttentionStats {
    pub const fn record_forward(&mut self, num_tokens: usize) {
        self.forward_count += 1;
        self.total_tokens += num_tokens as u64;
    }

    pub const fn record_tiled(&mut self, num_tokens: usize) {
        self.tiled_forward_count += 1;
        self.total_tokens += num_tokens as u64;
    }
}

/// Run the operation (see signature for params and return type).
/// # Errors
///
/// Returns `Err` if the operation fails.
pub fn softmax_last_dim(t: &Tensor) -> Result<Tensor> {
    let shape = t.dims();
    let max_vals = t.max_keepdim(shape.len() - 1)?;
    let t_shifted = t.broadcast_sub(&max_vals)?;
    let exp = t_shifted.exp()?;
    let sum = exp.sum_keepdim(shape.len() - 1)?;
    exp.broadcast_div(&sum)
}
