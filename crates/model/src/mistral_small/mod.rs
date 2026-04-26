//! Mistral Small model architecture implementation.
//!
//! Mistral Small uses:
//! - Mixture of Experts (MoE) with expert routing
//! - Grouped-Query Attention (GQA)
//! - Sliding Window Attention (SWA)
//! - RMSNorm

pub mod arch;
pub mod register;

pub use arch::MistralSmallArchitecture;
