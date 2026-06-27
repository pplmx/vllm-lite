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
//! ```rust,ignore
//! use vllm_model::config::ModelConfig;
//! use candle_core::Device;
//!
//! let config = ModelConfig::llama_7b();
//! let model = vllm_model::llama::LlamaModel::new(config, Device::Cpu, 1024).unwrap();
//! ```

pub mod arch;
pub mod block;
pub mod model;
pub mod register;

pub use arch::LlamaArchitecture;
pub use block::LlamaBlock;
pub use model::LlamaModel;
