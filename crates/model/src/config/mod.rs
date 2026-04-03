//! Model configuration types.
//!
//! Provides unified configuration for multiple model architectures
//! including Llama, Mistral, and Qwen3.
//!
//! # Usage
//! ```rust,ignore
//! use vllm_model::config::ModelConfig;
//!
//! // Use predefined configs
//! let llama = ModelConfig::llama_7b();
//! let mistral = ModelConfig::mistral_7b();
//! ```

pub mod architecture;
pub mod model_config;

pub use architecture::{Architecture, AttentionType, LayerType, MlpType, NormType, RoPEConfig};
pub use model_config::ModelConfig;
