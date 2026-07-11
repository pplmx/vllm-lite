#![allow(clippy::module_name_repetitions)]
//! Speculative decoding implementations
//!
//! This module provides infrastructure for speculative decoding,
//! including draft generation, verification, and acceptance strategies.

/// Adaptive draft-length tuning from observed acceptance rates.
pub mod adaptive;
/// [`SpeculationConfig`] and environment-variable overrides.
pub mod config;
/// Resolves draft checkpoints from registry entries into loaded backends.
pub mod draft_resolver;
/// KV memory budgeting for co-located draft + target models.
pub mod memory_budget;
/// [`SpeculativeModel`] wrapper coordinating draft-verify cycles.
pub mod model;
/// Draft model registry (load, unload, and lookup by [`DraftId`]).
pub mod registry;
/// Self-speculation path where the target model also drafts tokens.
pub mod self_spec;
/// Token- and block-level rejection sampling strategies.
pub mod strategy;
/// Draft generation and target-model verification trait surface.
pub mod verifier;

pub use adaptive::{AdaptiveSpeculativeDecoder, DraftAccuracyTracker};
pub use config::{SpeculationConfig, SpeculationConfigBuilder};
pub use draft_resolver::{DraftLoader, DraftResolver, NoopLoader, ResolvedDraft};
pub use memory_budget::{
    DEFAULT_BLOCK_BYTES, MemoryBudget, MemoryBudgetExceeded, MemoryBudgetSnapshot,
};
pub use model::SpeculativeModel;
pub use registry::{
    DraftId, DraftModelRegistry, DraftRegistryError, DraftSpec, DraftState, LoadedDraft,
};
pub use self_spec::SelfSpeculativeModel;
pub use strategy::RejectionStrategy;
pub use verifier::{DraftVerifier, StubDraftVerifier, VerificationResult, VerifierError};
