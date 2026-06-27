//! Speculative decoding implementations
//!
//! This module provides infrastructure for speculative decoding,
//! including draft generation, verification, and acceptance strategies.

/// adaptive: adaptive module.
pub mod adaptive;
/// config: config module.
pub mod config;
/// draft_registry: draft registry module.
pub mod draft_registry;
/// draft_resolver: draft resolver module.
pub mod draft_resolver;
/// memory_budget: memory budget module.
pub mod memory_budget;
/// model: model module.
pub mod model;
/// self_spec: self spec module.
pub mod self_spec;
/// strategy: strategy module.
pub mod strategy;
/// verifier: verifier module.
pub mod verifier;

pub use adaptive::{AdaptiveSpeculativeDecoder, DraftAccuracyTracker};
pub use config::{SpeculationConfig, SpeculationConfigBuilder};
pub use draft_registry::{
    DraftId, DraftModelRegistry, DraftRegistryError, DraftSpec, DraftState, LoadedDraft,
};
pub use draft_resolver::{DraftLoader, DraftResolver, NoopLoader, ResolvedDraft};
pub use memory_budget::{
    DEFAULT_BLOCK_BYTES, MemoryBudget, MemoryBudgetExceeded, MemoryBudgetSnapshot,
};
pub use model::SpeculativeModel;
pub use self_spec::SelfSpeculativeModel;
pub use strategy::RejectionStrategy;
pub use verifier::{DraftVerifier, VerificationResult, VerifierError};
