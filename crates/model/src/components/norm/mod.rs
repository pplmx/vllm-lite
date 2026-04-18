//! Normalization utilities.
//!
//! Provides unified RMSNorm and LayerNorm implementations.

pub mod layer_norm;
pub mod rms_norm;

pub use layer_norm::{LnLayerNorm, layer_norm};
pub use rms_norm::{RmsNorm, rms_norm};
