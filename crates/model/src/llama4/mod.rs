//! Llama 4 model architecture implementation.
//!
//! Llama 4 uses:
//! - Mixture of Experts (MoE) with sparse activation
//! - Grouped-Query Attention (GQA)
//! - SwiGLU MLP
//! - RMSNorm

/// arch: arch module.
pub mod arch;
/// register: register module.
pub mod register;

pub use arch::Llama4Architecture;
