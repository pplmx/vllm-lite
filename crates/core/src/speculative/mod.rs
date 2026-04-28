//! Speculative decoding implementations
//!
//! This module provides infrastructure for speculative decoding,
//! including draft generation, verification, and acceptance strategies.

pub mod adaptive;
pub mod config;
pub mod model;
pub mod self_spec;
pub mod strategy;
pub mod verifier;

pub use adaptive::{AdaptiveSpeculativeDecoder, DraftAccuracyTracker};
pub use config::{SpeculationConfig, SpeculationConfigBuilder};
pub use model::SpeculativeModel;
pub use self_spec::SelfSpeculativeModel;
pub use strategy::RejectionStrategy;
pub use verifier::{DraftVerifier, VerificationResult, VerifierError};
