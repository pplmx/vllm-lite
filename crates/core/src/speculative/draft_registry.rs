//! Deprecated: draft registry has been reorganized into the `registry/`
//! sub-tree. This module re-exports the public surface for backward
//! compatibility and will be removed in the next minor release.
//!
//! New code should import from `vllm_core::speculative::registry` directly:
//! - `crate::speculative::registry::DraftId`
//! - `crate::speculative::registry::DraftModelRegistry`
//! - `crate::speculative::registry::DraftRegistryError`
//! - `crate::speculative::registry::DraftSpec`
//! - `crate::speculative::registry::DraftState`
//! - `crate::speculative::registry::LoadedDraft`
//!
//! See Phase 31 plan `31-01` (v21.1) for the split rationale.

#[deprecated(
    since = "0.21.0",
    note = "use `crate::speculative::registry` instead; this module is preserved as a re-export shim for one minor release"
)]
pub use crate::speculative::registry::{
    DraftId, DraftModelRegistry, DraftRegistryError, DraftSpec, DraftState, LoadedDraft,
};
