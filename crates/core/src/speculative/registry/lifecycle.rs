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
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    ///   Use `force_unload` to bypass.
    pub fn unload(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let mut guard = self.drafts.write()?;
        let result = {
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
        };
        drop(guard);
        result
    }

    /// Force-unload a draft, ignoring its refcount. Used by admin tooling
    /// and tests. Logs the bypass at WARN level via `tracing`.
    ///
    /// Errors:
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// - `UnknownDraftId` if no entry with `id` exists
    pub fn force_unload(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let mut guard = self.drafts.write()?;
        let result = {
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
        };
        drop(guard);
        result
    }

    /// Increment the reference count for a registered draft.
    ///
    /// Updated in v18.3 — increment is driven by per-request routing logic;
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// see ADR-007 for the routing design.
    pub fn increment_ref(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let mut guard = self.drafts.write()?;
        let result = {
            let entry = guard
                .get_mut(id)
                .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
            match entry {
                DraftState::Unloaded(spec) => spec.ref_count += 1,
                DraftState::Loaded(loaded) => loaded.spec.ref_count += 1,
            }
            Ok(())
        };
        drop(guard);
        result
    }

    /// Decrement the reference count. Floors at 0 (no underflow).
    /// If the new count is zero AND the draft is `Loaded`, auto-unloads it:
    /// releases the budget reservation and drops the backend.
    ///
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// Returns `true` if auto-unload was triggered by this call.
    pub fn decrement_ref(&self, id: &DraftId) -> Result<bool, DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let mut guard = self.drafts.write()?;
        let result = {
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
        };
        drop(guard);
        result
    }

    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// Snapshot the reference count for a registered draft.
    pub fn ref_count(&self, id: &DraftId) -> Result<usize, DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self.drafts.read()?;
        let ref_count = guard
            .get(id)
            .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?
            .spec()
            .ref_count;
        drop(guard);
        Ok(ref_count)
    }

    /// Get a clone of the `Arc<Mutex<Box<dyn ModelBackend>>>` for a loaded draft.
    /// Returns None if the draft is unloaded or unknown. Used by
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// `DraftResolver` to hand the backend to the engine.
    pub fn get_loaded_backend(&self, id: &DraftId) -> Option<Arc<Mutex<Box<dyn ModelBackend>>>> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        // Degrade to `None` on poison rather than panic; invariant makes this unreachable.
        let guard = self.drafts.read().ok()?;
        let result = match guard.get(id) {
            Some(DraftState::Loaded(loaded)) => Some(loaded.backend.clone()),
            _ => None,
        };
        drop(guard);
        result
    }

    /// Read-only lookup of the current state. Does NOT trigger loading.
    ///
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// Returns `None` if no entry with `id` is registered.
    pub fn lookup(&self, id: &DraftId) -> Option<DraftState> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        // Degrade to `None` on poison rather than panic; invariant makes this unreachable.
        let guard = self.drafts.read().ok()?;
        let result = guard.get(id).map(|s| match s {
            DraftState::Unloaded(s) => DraftState::Unloaded(s.clone()),
            DraftState::Loaded(l) => {
                // We cannot clone `LoadedDraft` because `Box<dyn ModelBackend>`
                // is not Clone. Return the Loaded variant but callers should
                // use `get` for in-place access.
                DraftState::Unloaded(l.spec.clone())
            }
        });
        drop(guard);
        result
    }

    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// Check whether a draft is registered (either state).
    pub fn contains(&self, id: &DraftId) -> bool {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        self.drafts
            .read()
            // invariant: lock is only held for sync field access; poisoning only happens on panic during a critical section.
            .expect("DraftModelRegistry mutex poisoned")
            .contains_key(id)
    }

    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// Check whether a draft is currently loaded.
    pub fn is_loaded(&self, id: &DraftId) -> bool {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        matches!(
            self.drafts
                .read()
                // invariant: lock is only held for sync field access; poisoning only happens on panic during a critical section.
                .expect("DraftModelRegistry mutex poisoned")
                .get(id),
            Some(DraftState::Loaded(_))
        )
    }

    /// Number of bytes currently allocated to KV cache blocks for this draft.
    /// Returns 0 if the draft is `Unloaded`. Used by the Engine for runtime
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// KV-cache growth tracking (MEM-02).
    pub fn draft_allocated_bytes(&self, id: &DraftId) -> Result<u64, DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self.drafts.read()?;
        let bytes = match guard.get(id) {
            None => return Err(DraftRegistryError::UnknownDraftId(id.clone())),
            Some(DraftState::Unloaded(_)) => 0,
            Some(DraftState::Loaded(loaded)) => loaded.block_allocator.allocated_bytes() as u64,
        };
        drop(guard);
        Ok(bytes)
    }

    /// Estimated total VRAM footprint reserved for this draft in the budget.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// Zero for `Unloaded` drafts.
    pub fn draft_reserved_bytes(&self, id: &DraftId) -> Result<u64, DraftRegistryError> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        let guard = self.drafts.read()?;
        let bytes = match guard.get(id) {
            None => return Err(DraftRegistryError::UnknownDraftId(id.clone())),
            Some(DraftState::Unloaded(_)) => 0,
            Some(DraftState::Loaded(loaded)) => loaded.spec.estimated_total_bytes(),
        };
        drop(guard);
        Ok(bytes)
    }

    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// List all registered draft ids (sorted).
    pub fn ids(&self) -> Vec<DraftId> {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        // Degrade to empty vec on poison rather than panic; invariant makes this unreachable.
        let Ok(guard) = self.drafts.read() else {
            return Vec::new();
        };
        let mut ids: Vec<DraftId> = guard.keys().cloned().collect();
        ids.sort();
        drop(guard);
        ids
    }

    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// Count of registered drafts (both states).
    pub fn len(&self) -> usize {
        // invariant: lock is only held for synchronous field access; no panic possible while holding.
        self.drafts
            .read()
            // invariant: lock is only held for sync field access; poisoning only happens on panic during a critical section.
            .expect("DraftModelRegistry mutex poisoned")
            .len()
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
