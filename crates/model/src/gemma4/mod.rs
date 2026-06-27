//! Gemma 4 model architecture implementation.
//!
//! This module provides the Gemma4Model and Gemma4Block structs
//! that implement the ModelBackend trait with hybrid attention.

/// arch: arch module.
pub mod arch;
/// attention: attention module.
pub mod attention;
/// block: block module.
pub mod block;
/// mlp: mlp module.
pub mod mlp;
/// model: model module.
pub mod model;
/// register: register module.
pub mod register;
/// rope: rope module.
pub mod rope;

pub use block::Gemma4Block;
pub use model::Gemma4Model;
