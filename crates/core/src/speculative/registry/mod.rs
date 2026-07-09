#![allow(clippy::module_name_repetitions)]
//! Draft registry — heterogeneous draft model management.
//!
//! v18.0 Multi-Model Speculative Decoding phase 1.
//!
//! Owns zero-or-more external draft models alongside the target.
//! Each draft is an independent [`vllm_traits::ModelBackend`] with a private
//! block allocator — no KV-cache state is shared with the target or other
//! drafts.
//!
//! Loading is lazy by convention: [`register`](DraftModelRegistry::register)
//! only stores the spec as [`DraftState::Unloaded`]. The first call that
//! observes the `Unloaded` state and needs the live backend calls
//! [`attach_loaded`](DraftModelRegistry::attach_loaded) to promote it to
//! [`DraftState::Loaded`]. The caller (Engine, server, or a wrapper) is
//! responsible for actually invoking the model loader and producing a
//! `Box<dyn ModelBackend>`.
//!
//! This module does NOT depend on `vllm-model` so it stays usable from
//! contexts where the model loader is unavailable (e.g. embedded builds).
//!
//! # Module Layout
//!
//! The registry is split into focused files:
//!
//! - `types` — data types: [`DraftId`], [`DraftSpec`], [`LoadedDraft`],
//!   [`DraftState`]
//! - `errors` — [`DraftRegistryError`]
//! - `loader` — `register`, `attach_loaded`, `attach_loaded_budgeted`
//! - `lifecycle` — unload, refcount, lookup, memory reporting

mod errors;
mod lifecycle;
mod loader;
mod types;

pub use errors::DraftRegistryError;
pub use types::{DraftId, DraftSpec, DraftState, LoadedDraft};

use crate::speculative::memory_budget::MemoryBudget;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Runtime registry for external draft models.
///
/// `register` adds metadata only; `attach_loaded` promotes an `Unloaded` entry
/// to `Loaded`; `unload` reverses the transition. Multiple operations are
/// serialized via a single `RwLock` for simplicity — the registry is not on
/// the hot path (lookups during step scheduling are read-locked and cheap).
///
/// v18.2 adds a shared [`MemoryBudget`] (optional, defaults to unlimited).
/// When set, `attach_loaded_budgeted` reserves the draft's estimated footprint
/// in the budget, and `unload` releases it. `decrement_ref` auto-unloads
#[derive(Debug)]
/// loaded drafts when their refcount hits zero.
pub struct DraftModelRegistry {
    pub(super) drafts: RwLock<HashMap<DraftId, DraftState>>,
    pub(super) budget: Arc<MemoryBudget>,
}

impl Default for DraftModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DraftModelRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            drafts: RwLock::new(HashMap::new()),
            budget: Arc::new(MemoryBudget::unlimited()),
        }
    }

    /// Construct a registry with a custom memory budget. The budget is shared
    /// via `Arc` so the Engine can hold a reference too.
    pub fn with_budget(budget: Arc<MemoryBudget>) -> Self {
        Self {
            drafts: RwLock::new(HashMap::new()),
            budget,
        }
    }
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this file
// under the 800-line soft cap. They cover identity / state transitions
// (register, attach_loaded, unload, refcount lifecycle), the lookup
// surface (ids, len, is_empty, contains), and the memory-budget
// integration (reserve on attach, release on unload, over-budget
// refusal).
#[cfg(test)]
mod tests;
