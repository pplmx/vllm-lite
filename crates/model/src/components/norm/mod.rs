//! Normalization utilities.
//!
//! Provides unified RMSNorm and LayerNorm implementations.

pub mod layer_norm;
pub mod rms_norm;

pub use layer_norm::{layer_norm, LnLayerNorm};
pub use rms_norm::{rms_norm, RmsNorm};
