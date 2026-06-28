//! Lifecycle and lookup operations for the draft registry.
//!
//! Concerns covered:
//! - Unload (`unload`, `force_unload`)
//! - Reference counting (`increment_ref`, `decrement_ref`, `ref_count`)
//! - State inspection (`lookup`, `is_loaded`, `contains`, `ids`, `len`,
//!   `is_empty`, `get_loaded_backend`)
//! - Memory reporting (`draft_allocated_bytes`, `draft_reserved_bytes`,
//!   `memory_budget`)
//!
//! See [`super::loader`] for register/attach operations.

use super::errors::DraftRegistryError;
use super::types::{DraftId, DraftState};
use crate::speculative::DraftModelRegistry;
use std::sync::{Arc, Mutex};
use vllm_traits::ModelBackend;

impl DraftModelRegistry {
    /// Transition a `Loaded` entry back to `Unloaded`, dropping the backend and
    /// reclaiming the block allocator's state (allocator is dropped here).
    /// Releases the budget reservation (if any).
    ///
    /// No-op (returns `Ok`) if the entry is already `Unloaded`.
    ///
    /// Errors:
    /// - `UnknownDraftId` if no entry with `id` exists
    /// - `InUse(refcount)` if the draft is `Loaded` and has refcount > 0.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    ///   Use `force_unload` to bypass.
    pub fn unload(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let mut guard = self
            .drafts
            .write()
            .expect("DraftModelRegistry mutex poisoned");
        let entry = guard
            .get_mut(id)
            .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
        match entry {
            DraftState::Unloaded(_) => Ok(()),
            DraftState::Loaded(loaded) => {
                if loaded.spec.ref_count > 0 {
                    return Err(DraftRegistryError::InUse(loaded.spec.ref_count));
                }
                let spec = loaded.spec.clone();
                // Release the budget reservation (if budgeted).
                self.budget.release_draft(spec.estimated_total_bytes());
                *entry = DraftState::Unloaded(spec);
                Ok(())
            }
        }
    }

    /// Force-unload a draft, ignoring its refcount. Used by admin tooling
    /// and tests. Logs the bypass at WARN level via `tracing`.
    ///
    /// Errors:
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// - `UnknownDraftId` if no entry with `id` exists
    pub fn force_unload(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let mut guard = self
            .drafts
            .write()
            .expect("DraftModelRegistry mutex poisoned");
        let entry = guard
            .get_mut(id)
            .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
        match entry {
            DraftState::Unloaded(_) => Ok(()),
            DraftState::Loaded(loaded) => {
                let forced_refcount = loaded.spec.ref_count;
                let spec = loaded.spec.clone();
                if forced_refcount > 0 {
                    tracing::warn!(
                        draft_id = %id,
                        refcount = forced_refcount,
                        "force_unload bypassing non-zero refcount"
                    );
                }
                self.budget.release_draft(spec.estimated_total_bytes());
                *entry = DraftState::Unloaded(spec);
                Ok(())
            }
        }
    }

    /// Increment the reference count for a registered draft.
    ///
    /// Updated in v18.3 — increment is driven by per-request routing logic;
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// see ADR-007 for the routing design.
    pub fn increment_ref(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let mut guard = self
            .drafts
            .write()
            .expect("DraftModelRegistry mutex poisoned");
        let entry = guard
            .get_mut(id)
            .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
        match entry {
            DraftState::Unloaded(spec) => spec.ref_count += 1,
            DraftState::Loaded(loaded) => loaded.spec.ref_count += 1,
        }
        Ok(())
    }

    /// Decrement the reference count. Floors at 0 (no underflow).
    /// If the new count is zero AND the draft is `Loaded`, auto-unloads it:
    /// releases the budget reservation and drops the backend.
    ///
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// Returns `true` if auto-unload was triggered by this call.
    pub fn decrement_ref(&self, id: &DraftId) -> Result<bool, DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let mut guard = self
            .drafts
            .write()
            .expect("DraftModelRegistry mutex poisoned");
        let entry = guard
            .get_mut(id)
            .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
        match entry {
            DraftState::Unloaded(spec) => {
                spec.ref_count = spec.ref_count.saturating_sub(1);
                Ok(false)
            }
            DraftState::Loaded(loaded) => {
                loaded.spec.ref_count = loaded.spec.ref_count.saturating_sub(1);
                if loaded.spec.ref_count == 0 {
                    let spec = loaded.spec.clone();
                    self.budget.release_draft(spec.estimated_total_bytes());
                    *entry = DraftState::Unloaded(spec);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// Snapshot the reference count for a registered draft.
    pub fn ref_count(&self, id: &DraftId) -> Result<usize, DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        let entry = guard
            .get(id)
            .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
        Ok(entry.spec().ref_count)
    }

    /// Get a clone of the `Arc<Mutex<Box<dyn ModelBackend>>>` for a loaded draft.
    /// Returns None if the draft is unloaded or unknown. Used by
    /// `DraftResolver` to hand the backend to the engine.
    pub fn get_loaded_backend(&self, id: &DraftId) -> Option<Arc<Mutex<Box<dyn ModelBackend>>>> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        match guard.get(id) {
            Some(DraftState::Loaded(loaded)) => Some(loaded.backend.clone()),
            _ => None,
        }
    }

    /// Read-only lookup of the current state. Does NOT trigger loading.
    ///
    /// Returns `None` if no entry with `id` is registered.
    pub fn lookup(&self, id: &DraftId) -> Option<DraftState> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        guard.get(id).map(|s| match s {
            DraftState::Unloaded(s) => DraftState::Unloaded(s.clone()),
            DraftState::Loaded(l) => {
                // We cannot clone `LoadedDraft` because `Box<dyn ModelBackend>`
                // is not Clone. Return the Loaded variant but callers should
                // use `get` for in-place access.
                DraftState::Unloaded(l.spec.clone())
            }
        })
    }

    /// Check whether a draft is registered (either state).
    pub fn contains(&self, id: &DraftId) -> bool {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        guard.contains_key(id)
    }

    /// Check whether a draft is currently loaded.
    pub fn is_loaded(&self, id: &DraftId) -> bool {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        matches!(guard.get(id), Some(DraftState::Loaded(_)))
    }

    /// Number of bytes currently allocated to KV cache blocks for this draft.
    /// Returns 0 if the draft is `Unloaded`. Used by the Engine for runtime
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// KV-cache growth tracking (MEM-02).
    pub fn draft_allocated_bytes(&self, id: &DraftId) -> Result<u64, DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        let entry = guard
            .get(id)
            .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
        Ok(match entry {
            DraftState::Unloaded(_) => 0,
            DraftState::Loaded(loaded) => loaded.block_allocator.allocated_bytes() as u64,
        })
    }

    /// Estimated total VRAM footprint reserved for this draft in the budget.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// Zero for `Unloaded` drafts.
    pub fn draft_reserved_bytes(&self, id: &DraftId) -> Result<u64, DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        let entry = guard
            .get(id)
            .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
        Ok(match entry {
            DraftState::Unloaded(_) => 0,
            DraftState::Loaded(loaded) => loaded.spec.estimated_total_bytes(),
        })
    }

    /// List all registered draft ids (sorted).
    pub fn ids(&self) -> Vec<DraftId> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        let mut ids: Vec<DraftId> = guard.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Count of registered drafts (both states).
    pub fn len(&self) -> usize {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        guard.len()
    }

    /// Whether the registry has no registered drafts.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Access the shared memory budget.
    pub const fn memory_budget(&self) -> &Arc<crate::speculative::memory_budget::MemoryBudget> {
        &self.budget
    }
}
