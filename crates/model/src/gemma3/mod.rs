//! Gemma3 model architecture implementation.
//!
//! Gemma3 uses:
//! - Sliding Window Attention (SWA)
//! - Grouped-Query Attention (GQA)
//! - GeGLU activation
//! - RMSNorm with embedding table sharing

/// arch: arch module.
pub mod arch;
/// register: register module.
pub mod register;

pub use arch::Gemma3Architecture;
