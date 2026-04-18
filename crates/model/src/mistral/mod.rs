//! Mistral model architecture implementation.
//!
//! This module provides the MistralModel and MistralBlock structs
//! that implement the ModelBackend trait with sliding window attention.
//!
//! # Differences from Llama
//! - Uses sliding window attention (4096 tokens by default)
//! - Uses Grouped-Query Attention with fewer KV heads
//! - Supports MistralSparseMoe block in Mixtral

pub mod arch;
pub mod block;
pub mod model;
pub mod register;

pub use block::MistralBlock;
pub use model::MistralModel;
