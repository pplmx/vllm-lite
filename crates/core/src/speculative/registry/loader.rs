//! Loading logic for the draft registry.
//!
//! Concerns covered:
//! - [`register`](DraftModelRegistry::register): add metadata for a new draft
//! - [`attach_loaded`](DraftModelRegistry::attach_loaded): promote Unloaded → Loaded
//! - [`attach_loaded_budgeted`](DraftModelRegistry::attach_loaded_budgeted): same with
//!   shared-memory-budget reservation
//!
//! See [`super::lifecycle`] for unload/refcount/lookup operations.

use super::errors::DraftRegistryError;
use super::types::{DraftId, DraftSpec, DraftState, LoadedDraft};
use crate::scheduler::memory::BlockAllocator;
use crate::speculative::DraftModelRegistry;
use std::sync::{Arc, Mutex};
use vllm_traits::ModelBackend;

impl DraftModelRegistry {
    /// Register a draft spec. The spec is stored as `Unloaded`; no I/O happens.
    ///
    /// Returns [`DraftRegistryError::AlreadyLoaded`] if an entry with the same
    ///
    /// # Errors
    ///
    /// Returns `Err` if registration fails (e.g. duplicate name or invalid input).
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// id already exists in either state.
    pub fn register(&self, spec: DraftSpec) -> Result<(), DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let mut guard = self
            .drafts
            .write()
            .expect("DraftModelRegistry mutex poisoned");
        if guard.contains_key(&spec.id) {
            return Err(DraftRegistryError::AlreadyLoaded(spec.id));
        }
        guard.insert(spec.id.clone(), DraftState::Unloaded(spec));
        drop(guard);
        Ok(())
    }

    /// Promote an `Unloaded` entry to `Loaded` using a caller-supplied backend.
    ///
    /// This is the lazy-load seam: the caller (Engine, server, etc.) is
    /// responsible for invoking the actual model loader before calling this
    /// method. The registry only owns the state transition.
    ///
    /// Errors:
    /// - `UnknownDraftId` if no entry with `id` exists
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// - `AlreadyLoaded` if the entry is already in `Loaded` state
    pub fn attach_loaded(
        &self,
        id: &DraftId,
        backend: Box<dyn ModelBackend>,
    ) -> Result<(), DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let mut guard = self
            .drafts
            .write()
            .expect("DraftModelRegistry mutex poisoned");
        let result = {
            let entry = guard
                .get_mut(id)
                .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
            let spec = match entry {
                DraftState::Unloaded(s) => s.clone(),
                DraftState::Loaded(_) => {
                    return Err(DraftRegistryError::AlreadyLoaded(id.clone()));
                }
            };
            let kv_blocks = spec.kv_blocks;
            let loaded = LoadedDraft {
                spec,
                backend: Arc::new(Mutex::new(backend)),
                block_allocator: BlockAllocator::new(kv_blocks),
            };
            *entry = DraftState::Loaded(loaded);
            Ok(())
        };
        drop(guard);
        result
    }

    /// Promote an `Unloaded` entry to `Loaded` AND reserve the draft's
    /// estimated footprint in the shared memory budget.
    ///
    /// On budget exhaustion, returns [`DraftRegistryError::MemoryBudgetExceeded`]
    /// without changing state.
    ///
    /// Errors:
    /// - `UnknownDraftId` if no entry with `id` exists
    /// - `AlreadyLoaded` if the entry is already `Loaded`
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// - `MemoryBudgetExceeded` if the budget can't accommodate this draft
    pub fn attach_loaded_budgeted(
        &self,
        id: &DraftId,
        backend: Box<dyn ModelBackend>,
    ) -> Result<(), DraftRegistryError> {
        // Stage 1: read-lock to inspect and clone the spec; release the read
        // lock before doing the budget reservation so we don't hold both
        // locks if budget fails.
        let (kv_blocks, estimated) = {
            // invariant: lock is only held for synchronous field access; no panic possible while holding.
            let guard = self
                .drafts
                .read()
                .expect("DraftModelRegistry mutex poisoned");
            let result = match guard.get(id) {
                None => return Err(DraftRegistryError::UnknownDraftId(id.clone())),
                Some(DraftState::Loaded(_)) => {
                    return Err(DraftRegistryError::AlreadyLoaded(id.clone()));
                }
                Some(DraftState::Unloaded(s)) => (s.kv_blocks, s.estimated_total_bytes()),
            };
            drop(guard);
            result
        };

        // Stage 2: budget reservation (may fail).
        self.budget
            .try_reserve_draft(estimated, Some(id.clone()))
            .map_err(DraftRegistryError::MemoryBudgetExceeded)?;

        // Stage 3: state transition under write lock.
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let mut guard = self
            .drafts
            .write()
            .expect("DraftModelRegistry mutex poisoned");
        let result = {
            let entry = guard
                .get_mut(id)
                .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
            // Re-check state — it may have changed between read and write.
            let spec = match entry {
                DraftState::Loaded(_) => {
                    // Roll back the budget reservation we just made.
                    self.budget.release_draft(estimated);
                    return Err(DraftRegistryError::AlreadyLoaded(id.clone()));
                }
                DraftState::Unloaded(s) => s.clone(),
            };
            let loaded = LoadedDraft {
                spec,
                backend: Arc::new(Mutex::new(backend)),
                block_allocator: BlockAllocator::new(kv_blocks),
            };
            *entry = DraftState::Loaded(loaded);
            Ok(())
        };
        drop(guard);
        result
    }
}
