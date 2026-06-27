//! Phi-4 model architecture implementation.
//!
//! Phi-4 uses:
//! - Grouped-Query Attention (GQA)
//! - SwiGLU MLP with partial rotation
//! - RMSNorm
//! - No positional embedding (rotary only)

/// arch: arch module.
pub mod arch;
/// register: register module.
pub mod register;

pub use arch::Phi4Architecture;
