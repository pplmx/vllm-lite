//! Llama model architecture implementation.
//!
//! This module provides the LlamaModel and LlamaBlock structs
//! that implement the ModelBackend trait for Llama-style transformers.
//!
//! # Architecture
//! - Uses RMSNorm (instead of LayerNorm)
//! - Uses SwiGLU MLP
//! - Supports Grouped-Query Attention (GQA)
//!
//! # Example
//! ```rust
//! use crate::config::ModelConfig;
//! use candle_core::Device;
//!
//! let config = ModelConfig::llama_7b();
//! let model = LlamaModel::new(config, Device::Cpu, 1024).unwrap();
//! ```

pub mod block;
pub mod model;

pub use block::LlamaBlock;
pub use model::LlamaModel;
