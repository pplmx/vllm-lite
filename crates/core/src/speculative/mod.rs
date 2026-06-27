//! Speculative decoding implementations
//!
//! This module provides infrastructure for speculative decoding,
//! including draft generation, verification, and acceptance strategies.

pub mod adaptive;
pub mod config;
pub mod draft_registry;
pub mod memory_budget;
pub mod model;
pub mod self_spec;
pub mod strategy;
pub mod verifier;

pub use adaptive::{AdaptiveSpeculativeDecoder, DraftAccuracyTracker};
pub use config::{SpeculationConfig, SpeculationConfigBuilder};
pub use draft_registry::{
    DraftId, DraftModelRegistry, DraftRegistryError, DraftSpec, DraftState, LoadedDraft,
};
pub use memory_budget::{
    DEFAULT_BLOCK_BYTES, MemoryBudget, MemoryBudgetExceeded, MemoryBudgetSnapshot,
};
pub use model::SpeculativeModel;
pub use self_spec::SelfSpeculativeModel;
pub use strategy::RejectionStrategy;
pub use verifier::{DraftVerifier, VerificationResult, VerifierError};
