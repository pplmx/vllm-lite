//! Positional embedding namespace: standard `RoPE` (`rope`) and multi-modal `RoPE` (`mrope`).
//!
//! Re-exports [`RoPE`], [`MRoPE`], [`apply_rope`], and
//! [`precompute_rope_cache`] at the namespace root for ergonomic access
//! from model forward paths.
pub mod mrope;
pub mod rope;

pub use mrope::MRoPE;
pub use rope::{RoPE, apply_rope, precompute_rope_cache};
