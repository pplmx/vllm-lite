//! Gemma 4 model architecture implementation.
//!
//! This module provides the Gemma4Model and Gemma4Block structs
//! that implement the ModelBackend trait with hybrid attention.

pub mod arch;
pub mod attention;
pub mod block;
pub mod mlp;
pub mod model;
pub mod register;
pub mod rope;

pub use block::Gemma4Block;
pub use model::Gemma4Model;
