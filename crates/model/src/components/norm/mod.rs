//! Normalization utilities.
//!
//! Provides unified RMSNorm and LayerNorm implementations.

/// layer_norm: layer norm module.
pub mod layer_norm;
/// rms_norm: rms norm module.
pub mod rms_norm;

pub use layer_norm::{LnLayerNorm, layer_norm};
pub use rms_norm::{RmsNorm, rms_norm};
