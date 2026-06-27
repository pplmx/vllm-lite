//! Llama 4 model architecture implementation.
//!
//! Llama 4 uses:
//! - Mixture of Experts (MoE) with sparse activation
//! - Grouped-Query Attention (GQA)
//! - SwiGLU MLP
//! - RMSNorm

pub mod arch;
pub mod register;

pub use arch::Llama4Architecture;
