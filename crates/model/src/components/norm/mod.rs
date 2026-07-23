#![allow(clippy::module_name_repetitions)]
//! Normalization utilities.
//!
//! Provides unified `RMSNorm` and `LayerNorm` implementations.

/// LayerNorm (weight + bias) implementation.
pub mod layer_norm;
/// RMSNorm implementation.
pub mod rms_norm;

pub use layer_norm::{LnLayerNorm, layer_norm};
pub use rms_norm::{RmsNorm, rms_norm};
