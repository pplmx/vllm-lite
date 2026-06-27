//! Gemma3 model architecture implementation.
//!
//! Gemma3 uses:
//! - Sliding Window Attention (SWA)
//! - Grouped-Query Attention (GQA)
//! - GeGLU activation
//! - RMSNorm with embedding table sharing

pub mod arch;
pub mod register;

pub use arch::Gemma3Architecture;
