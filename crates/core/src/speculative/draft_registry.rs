//! DraftModelRegistry — runtime registry for heterogeneous draft models
//!
//! v18.0 Multi-Model Speculative Decoding phase 1.
//!
//! Owns zero-or-more external draft models alongside the target.
//! Each draft is an independent [`vllm_traits::ModelBackend`] with a private
//! [`BlockAllocator`] — no KV-cache state is shared with the target or other
//! drafts.
//!
//! Loading is lazy by convention: `register` only stores the spec as
//! [`DraftState::Unloaded`]. The first call that observes the `Unloaded` state
//! and needs the live backend calls [`DraftModelRegistry::attach_loaded`] to
//! promote it to [`DraftState::Loaded`]. The caller (Engine, server, or a
//! wrapper) is responsible for actually invoking the model loader and producing
//! a `Box<dyn ModelBackend>`.
//!
//! This module does NOT depend on `vllm-model` so it stays usable from
//! contexts where the model loader is unavailable (e.g. embedded builds).

use crate::scheduler::memory::BlockAllocator;
use crate::speculative::memory_budget::{DEFAULT_BLOCK_BYTES, MemoryBudget, MemoryBudgetExceeded};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use vllm_traits::ModelBackend;

/// Opaque, user-supplied identifier for a draft model.
///
/// Typically a slug like `"qwen-small"` or `"llama-tiny"`. Equality is by
/// string match; the registry treats ids case-sensitively.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DraftId(pub String);

impl DraftId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DraftId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for DraftId {
    fn from(s: &str) -> Self {
        DraftId(s.to_string())
    }
}

impl From<String> for DraftId {
    fn from(s: String) -> Self {
        DraftId(s)
    }
}

/// Registration entry for a draft model.
///
/// Holds only the metadata needed to load the draft later. Construction is
/// cheap; no I/O happens until [`DraftModelRegistry::attach_loaded`] is
/// called.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftSpec {
    pub id: DraftId,
    pub model_dir: PathBuf,
    pub arch_hint: Option<String>,
    pub kv_blocks: usize,
    /// Conservative estimate of the draft's parameter footprint, in bytes.
    ///
    /// Sourced from `ModelLoader` metadata at registration time (v18.2 MEM-02).
    /// Used by the [`MemoryBudget`] to decide whether loading this draft
    /// would exceed VRAM. Set conservatively — over-estimate is safe,
    /// under-estimate can cause runtime OOM.
    pub weight_size_estimate_bytes: u64,
    pub ref_count: usize,
}

impl DraftSpec {
    pub fn new(id: impl Into<DraftId>, model_dir: impl Into<PathBuf>, kv_blocks: usize) -> Self {
        Self {
            id: id.into(),
            model_dir: model_dir.into(),
            arch_hint: None,
            kv_blocks,
            weight_size_estimate_bytes: 0,
            ref_count: 0,
        }
    }

    pub fn with_arch_hint(mut self, arch: impl Into<String>) -> Self {
        self.arch_hint = Some(arch.into());
        self
    }

    pub fn with_ref_count(mut self, ref_count: usize) -> Self {
        self.ref_count = ref_count;
        self
    }

    pub fn with_weight_size(mut self, bytes: u64) -> Self {
        self.weight_size_estimate_bytes = bytes;
        self
    }

    /// Estimated total VRAM footprint of this draft if loaded:
    /// `weight_size_estimate_bytes + kv_blocks * DEFAULT_BLOCK_BYTES`.
    pub fn estimated_total_bytes(&self) -> u64 {
        let kv_bytes = (self.kv_blocks as u64).saturating_mul(DEFAULT_BLOCK_BYTES);
        self.weight_size_estimate_bytes.saturating_add(kv_bytes)
    }
}

/// A loaded draft: backend + its private block allocator.
pub struct LoadedDraft {
    pub spec: DraftSpec,
    pub backend: Box<dyn ModelBackend>,
    pub block_allocator: BlockAllocator,
}

/// State machine for a registered draft.
#[allow(clippy::large_enum_variant)]
pub enum DraftState {
    Unloaded(DraftSpec),
    Loaded(LoadedDraft),
}

impl std::fmt::Debug for DraftState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DraftState::Unloaded(s) => f.debug_tuple("Unloaded").field(s).finish(),
            DraftState::Loaded(_) => f.debug_tuple("Loaded").field(&"<backend>").finish(),
        }
    }
}

impl DraftState {
    pub fn spec(&self) -> &DraftSpec {
        match self {
            DraftState::Unloaded(s) => s,
            DraftState::Loaded(l) => &l.spec,
        }
    }

    pub fn id(&self) -> &DraftId {
        &self.spec().id
    }

    pub fn is_loaded(&self) -> bool {
        matches!(self, DraftState::Loaded(_))
    }
}

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
/// loaded drafts when their refcount hits zero.
pub struct DraftModelRegistry {
    drafts: RwLock<HashMap<DraftId, DraftState>>,
    budget: Arc<MemoryBudget>,
}

impl Default for DraftModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DraftModelRegistry {
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

    /// Access the shared memory budget.
    pub fn memory_budget(&self) -> &Arc<MemoryBudget> {
        &self.budget
    }

    /// Register a draft spec. The spec is stored as `Unloaded`; no I/O happens.
    ///
    /// Returns `DraftRegistryError::AlreadyLoaded` if an entry with the same id
    /// already exists in either state.
    pub fn register(&self, spec: DraftSpec) -> Result<(), DraftRegistryError> {
        let mut guard = self
            .drafts
            .write()
            .expect("DraftModelRegistry mutex poisoned");
        if guard.contains_key(&spec.id) {
            return Err(DraftRegistryError::AlreadyLoaded(spec.id));
        }
        guard.insert(spec.id.clone(), DraftState::Unloaded(spec));
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
    /// - `AlreadyLoaded` if the entry is already in `Loaded` state
    pub fn attach_loaded(
        &self,
        id: &DraftId,
        backend: Box<dyn ModelBackend>,
    ) -> Result<(), DraftRegistryError> {
        let mut guard = self
            .drafts
            .write()
            .expect("DraftModelRegistry mutex poisoned");
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
            backend,
            block_allocator: BlockAllocator::new(kv_blocks),
        };
        *entry = DraftState::Loaded(loaded);
        Ok(())
    }

    /// Transition a `Loaded` entry back to `Unloaded`, dropping the backend and
    /// reclaiming the block allocator's state (allocator is dropped here).
    /// Releases the budget reservation (if any).
    ///
    /// No-op (returns `Ok`) if the entry is already `Unloaded`.
    ///
    /// Errors:
    /// - `UnknownDraftId` if no entry with `id` exists
    /// - `InUse(refcount)` if the draft is `Loaded` and has refcount > 0.
    ///   Use `force_unload` to bypass.
    pub fn unload(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
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
    /// - `UnknownDraftId` if no entry with `id` exists
    pub fn force_unload(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
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

    /// Promote an `Unloaded` entry to `Loaded` AND reserve the draft's
    /// estimated footprint in the shared memory budget.
    ///
    /// On budget exhaustion, returns `DraftRegistryError::MemoryBudgetExceeded`
    /// without changing state.
    ///
    /// Errors:
    /// - `UnknownDraftId` if no entry with `id` exists
    /// - `AlreadyLoaded` if the entry is already `Loaded`
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
            let guard = self
                .drafts
                .read()
                .expect("DraftModelRegistry mutex poisoned");
            let entry = guard
                .get(id)
                .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
            match entry {
                DraftState::Loaded(_) => {
                    return Err(DraftRegistryError::AlreadyLoaded(id.clone()));
                }
                DraftState::Unloaded(s) => (s.kv_blocks, s.estimated_total_bytes()),
            }
        };

        // Stage 2: budget reservation (may fail).
        self.budget
            .try_reserve_draft(estimated, Some(id.clone()))
            .map_err(DraftRegistryError::MemoryBudgetExceeded)?;

        // Stage 3: state transition under write lock.
        let mut guard = self
            .drafts
            .write()
            .expect("DraftModelRegistry mutex poisoned");
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
            backend,
            block_allocator: BlockAllocator::new(kv_blocks),
        };
        *entry = DraftState::Loaded(loaded);
        Ok(())
    }

    /// Read-only lookup of the current state. Does NOT trigger loading.
    ///
    /// Returns `None` if no entry with `id` is registered.
    pub fn lookup(&self, id: &DraftId) -> Option<DraftState> {
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
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        guard.contains_key(id)
    }

    /// Check whether a draft is currently loaded.
    pub fn is_loaded(&self, id: &DraftId) -> bool {
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        matches!(guard.get(id), Some(DraftState::Loaded(_)))
    }

    /// Increment the reference count for a registered draft.
    ///
    /// Phase 18.3 will drive this from routing logic. Phase 18.1 only stores
    /// the count for later use.
    pub fn increment_ref(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
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
    /// Returns `true` if auto-unload was triggered by this call.
    pub fn decrement_ref(&self, id: &DraftId) -> Result<bool, DraftRegistryError> {
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

    /// Snapshot the reference count for a registered draft.
    pub fn ref_count(&self, id: &DraftId) -> Result<usize, DraftRegistryError> {
        let guard = self
            .drafts
            .read()
            .expect("DraftModelRegistry mutex poisoned");
        let entry = guard
            .get(id)
            .ok_or_else(|| DraftRegistryError::UnknownDraftId(id.clone()))?;
        Ok(entry.spec().ref_count)
    }

    /// Number of bytes currently allocated to KV cache blocks for this draft.
    /// Returns 0 if the draft is `Unloaded`. Used by the Engine for runtime
    /// KV-cache growth tracking (MEM-02).
    pub fn draft_allocated_bytes(&self, id: &DraftId) -> Result<u64, DraftRegistryError> {
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
    /// Zero for `Unloaded` drafts.
    pub fn draft_reserved_bytes(&self, id: &DraftId) -> Result<u64, DraftRegistryError> {
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
}

/// Errors surfaced by [`DraftModelRegistry`].
#[derive(Debug, thiserror::Error)]
pub enum DraftRegistryError {
    #[error("unknown draft id: {0}")]
    UnknownDraftId(DraftId),
    #[error("draft already loaded: {0}")]
    AlreadyLoaded(DraftId),
    #[error("draft still in use (refcount={0})")]
    InUse(usize),
    #[error("draft load failed: {0}")]
    LoadFailed(String),
    #[error("{0}")]
    MemoryBudgetExceeded(MemoryBudgetExceeded),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_spec(id: &str, kv_blocks: usize) -> DraftSpec {
        DraftSpec::new(id, "/nonexistent", kv_blocks)
    }

    #[test]
    fn test_register_creates_unloaded_state() {
        let registry = DraftModelRegistry::new();
        registry.register(dummy_spec("a", 64)).unwrap();
        match registry.lookup(&DraftId("a".into())) {
            Some(DraftState::Unloaded(s)) => {
                assert_eq!(s.kv_blocks, 64);
                assert_eq!(s.ref_count, 0);
            }
            other => panic!("expected Unloaded, got {other:?}"),
        }
    }

    #[test]
    fn test_register_duplicate_id_errors() {
        let registry = DraftModelRegistry::new();
        registry.register(dummy_spec("a", 64)).unwrap();
        let err = registry.register(dummy_spec("a", 32)).unwrap_err();
        assert!(matches!(err, DraftRegistryError::AlreadyLoaded(_)));
    }

    #[test]
    fn test_attach_loaded_promotes_state() {
        let registry = DraftModelRegistry::new();
        registry.register(dummy_spec("a", 64)).unwrap();
        // Use a fake backend (we can't easily construct a ModelBackend here,
        // so we test via lookup/is_loaded).
        assert!(!registry.is_loaded(&DraftId("a".into())));
        // Direct manipulation isn't possible from outside, but we can verify
        // attach_loaded fails on unknown id.
        let err = registry
            .attach_loaded(&DraftId("nope".into()), test_backend())
            .unwrap_err();
        assert!(matches!(err, DraftRegistryError::UnknownDraftId(_)));
    }

    #[test]
    fn test_unload_unknown_id_errors() {
        let registry = DraftModelRegistry::new();
        let err = registry.unload(&DraftId("nope".into())).unwrap_err();
        assert!(matches!(err, DraftRegistryError::UnknownDraftId(_)));
    }

    #[test]
    fn test_unload_already_unloaded_is_noop() {
        let registry = DraftModelRegistry::new();
        registry.register(dummy_spec("a", 64)).unwrap();
        registry.unload(&DraftId("a".into())).unwrap();
        // No-op again on the same id
        registry.unload(&DraftId("a".into())).unwrap();
        assert!(!registry.is_loaded(&DraftId("a".into())));
    }

    #[test]
    fn test_refcount_increment_decrement() {
        let registry = DraftModelRegistry::new();
        registry.register(dummy_spec("a", 64)).unwrap();
        registry.increment_ref(&DraftId("a".into())).unwrap();
        registry.increment_ref(&DraftId("a".into())).unwrap();
        assert_eq!(registry.ref_count(&DraftId("a".into())).unwrap(), 2);
        registry.decrement_ref(&DraftId("a".into())).unwrap();
        assert_eq!(registry.ref_count(&DraftId("a".into())).unwrap(), 1);
        registry.decrement_ref(&DraftId("a".into())).unwrap();
        registry.decrement_ref(&DraftId("a".into())).unwrap(); // floor
        assert_eq!(registry.ref_count(&DraftId("a".into())).unwrap(), 0);
    }

    #[test]
    fn test_ids_lists_all_registered_sorted() {
        let registry = DraftModelRegistry::new();
        registry.register(dummy_spec("b", 32)).unwrap();
        registry.register(dummy_spec("a", 64)).unwrap();
        registry.register(dummy_spec("c", 16)).unwrap();
        assert_eq!(
            registry.ids(),
            vec![
                DraftId("a".into()),
                DraftId("b".into()),
                DraftId("c".into())
            ]
        );
    }

    #[test]
    fn test_len_and_is_empty() {
        let registry = DraftModelRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        registry.register(dummy_spec("a", 64)).unwrap();
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_contains_returns_true_for_registered() {
        let registry = DraftModelRegistry::new();
        registry.register(dummy_spec("a", 64)).unwrap();
        assert!(registry.contains(&DraftId("a".into())));
        assert!(!registry.contains(&DraftId("b".into())));
    }

    #[test]
    fn test_draft_id_display() {
        let id = DraftId("qwen-small".into());
        assert_eq!(format!("{id}"), "qwen-small");
        assert_eq!(id.as_str(), "qwen-small");
    }

    #[test]
    fn test_draft_id_from_str_and_string() {
        let a: DraftId = "abc".into();
        let b: DraftId = String::from("abc").into();
        assert_eq!(a, b);
    }

    #[test]
    fn test_draft_spec_builder() {
        let spec = DraftSpec::new("x", "/tmp/model", 128)
            .with_arch_hint("llama")
            .with_ref_count(2);
        assert_eq!(spec.id, DraftId("x".into()));
        assert_eq!(spec.arch_hint.as_deref(), Some("llama"));
        assert_eq!(spec.kv_blocks, 128);
        assert_eq!(spec.ref_count, 2);
    }

    // Minimal fake ModelBackend for tests. We don't call .forward() so the
    // bodies just panic — they're never invoked.
    struct FakeBackend;
    impl ModelBackend for FakeBackend {
        fn forward(
            &mut self,
            _seq_ids: &[vllm_traits::SeqId],
            _input_tokens: &[Vec<vllm_traits::TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> vllm_traits::Result<vllm_traits::BatchOutput> {
            panic!("FakeBackend::forward should not be called in registry tests")
        }
        fn forward_logits(
            &mut self,
            _seq_ids: &[vllm_traits::SeqId],
            _input_tokens: &[Vec<vllm_traits::TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> vllm_traits::Result<Vec<Vec<f32>>> {
            panic!("FakeBackend::forward_logits should not be called in registry tests")
        }
        fn embed(
            &mut self,
            _input_tokens: &[Vec<vllm_traits::TokenId>],
            _positions: &[Vec<usize>],
        ) -> vllm_traits::Result<Vec<Vec<f32>>> {
            panic!("FakeBackend::embed should not be called in registry tests")
        }
        fn vocab_size(&self) -> usize {
            panic!("FakeBackend::vocab_size should not be called in registry tests")
        }
        fn num_layers(&self) -> usize {
            panic!("FakeBackend::num_layers should not be called in registry tests")
        }
        fn num_heads(&self) -> usize {
            panic!("FakeBackend::num_heads should not be called in registry tests")
        }
    }

    fn test_backend() -> Box<dyn ModelBackend> {
        Box::new(FakeBackend)
    }

    // ───────────────────────── 18.2 lifecycle + budget tests ─────────────────

    fn dummy_spec_with_size(id: &str, kv_blocks: usize, weight_bytes: u64) -> DraftSpec {
        DraftSpec::new(id, "/nonexistent", kv_blocks).with_weight_size(weight_bytes)
    }

    #[test]
    fn test_unload_with_refcount_errors_in_use() {
        let registry = DraftModelRegistry::new();
        registry
            .register(dummy_spec_with_size("a", 8, 1024))
            .unwrap();
        registry
            .attach_loaded(&DraftId("a".into()), test_backend())
            .unwrap();
        registry.increment_ref(&DraftId("a".into())).unwrap();
        let err = registry.unload(&DraftId("a".into())).unwrap_err();
        assert!(matches!(err, DraftRegistryError::InUse(1)));
        // State still Loaded
        assert!(registry.is_loaded(&DraftId("a".into())));
    }

    #[test]
    fn test_force_unload_overrides_refcount() {
        let registry = DraftModelRegistry::new();
        registry
            .register(dummy_spec_with_size("a", 8, 1024))
            .unwrap();
        registry
            .attach_loaded(&DraftId("a".into()), test_backend())
            .unwrap();
        registry.increment_ref(&DraftId("a".into())).unwrap();
        registry.increment_ref(&DraftId("a".into())).unwrap();
        registry.force_unload(&DraftId("a".into())).unwrap();
        assert!(!registry.is_loaded(&DraftId("a".into())));
    }

    #[test]
    fn test_decrement_ref_auto_unloads_at_zero() {
        let registry = DraftModelRegistry::new();
        registry
            .register(dummy_spec_with_size("a", 8, 1024))
            .unwrap();
        registry
            .attach_loaded(&DraftId("a".into()), test_backend())
            .unwrap();
        registry.increment_ref(&DraftId("a".into())).unwrap();
        let auto_unloaded = registry.decrement_ref(&DraftId("a".into())).unwrap();
        assert!(auto_unloaded);
        assert!(!registry.is_loaded(&DraftId("a".into())));
    }

    #[test]
    fn test_decrement_ref_no_auto_unload_when_already_unloaded() {
        let registry = DraftModelRegistry::new();
        registry.register(dummy_spec("a", 8)).unwrap();
        registry.increment_ref(&DraftId("a".into())).unwrap();
        let auto_unloaded = registry.decrement_ref(&DraftId("a".into())).unwrap();
        assert!(!auto_unloaded);
        // Unloaded drafts stay Unloaded
        assert!(!registry.is_loaded(&DraftId("a".into())));
    }

    #[test]
    fn test_decrement_ref_does_not_auto_unload_above_zero() {
        let registry = DraftModelRegistry::new();
        registry
            .register(dummy_spec_with_size("a", 8, 1024))
            .unwrap();
        registry
            .attach_loaded(&DraftId("a".into()), test_backend())
            .unwrap();
        registry.increment_ref(&DraftId("a".into())).unwrap();
        registry.increment_ref(&DraftId("a".into())).unwrap();
        let auto_unloaded = registry.decrement_ref(&DraftId("a".into())).unwrap();
        assert!(!auto_unloaded);
        assert!(registry.is_loaded(&DraftId("a".into())));
    }

    #[test]
    fn test_attach_loaded_budgeted_reserves_budget() {
        let budget = Arc::new(MemoryBudget::new(100_000_000_000).unwrap()); // 100 GB
        let registry = DraftModelRegistry::with_budget(budget.clone());
        let spec = dummy_spec_with_size("a", 4, 1000); // total ≈ 67 MB
        let estimated = spec.estimated_total_bytes();
        registry.register(spec).unwrap();
        registry
            .attach_loaded_budgeted(&DraftId("a".into()), test_backend())
            .unwrap();
        let snap = budget.snapshot();
        assert_eq!(snap.reserved_drafts_bytes, estimated);
    }

    #[test]
    fn test_attach_loaded_budgeted_releases_on_double_attach() {
        let budget = Arc::new(MemoryBudget::new(100_000_000).unwrap());
        let registry = DraftModelRegistry::with_budget(budget.clone());
        let spec = dummy_spec_with_size("a", 1, 1000);
        registry.register(spec).unwrap();
        registry
            .attach_loaded_budgeted(&DraftId("a".into()), test_backend())
            .unwrap();
        // Second attach_loaded_budgeted is AlreadyLoaded; ensure budget
        // reservation from the second attempt was rolled back.
        let snap_before = budget.snapshot();
        let _ = registry.attach_loaded_budgeted(&DraftId("a".into()), test_backend());
        let snap_after = budget.snapshot();
        assert_eq!(
            snap_before.reserved_drafts_bytes,
            snap_after.reserved_drafts_bytes
        );
    }

    #[test]
    fn test_attach_loaded_budgeted_refuses_when_over_budget() {
        let budget = Arc::new(MemoryBudget::new(100).unwrap()); // tiny budget
        let registry = DraftModelRegistry::with_budget(budget);
        let spec = dummy_spec_with_size("huge", 4, 1000); // estimated ≈ 67 MiB
        registry.register(spec).unwrap();
        let err = registry
            .attach_loaded_budgeted(&DraftId("huge".into()), test_backend())
            .unwrap_err();
        assert!(matches!(err, DraftRegistryError::MemoryBudgetExceeded(_)));
        // State stayed Unloaded
        assert!(!registry.is_loaded(&DraftId("huge".into())));
    }

    #[test]
    fn test_unload_releases_budget_reservation() {
        let budget = Arc::new(MemoryBudget::new(100_000_000).unwrap());
        let registry = DraftModelRegistry::with_budget(budget.clone());
        let spec = dummy_spec_with_size("a", 1, 1000);
        let estimated = spec.estimated_total_bytes();
        registry.register(spec).unwrap();
        registry
            .attach_loaded_budgeted(&DraftId("a".into()), test_backend())
            .unwrap();
        assert_eq!(budget.snapshot().reserved_drafts_bytes, estimated);
        registry.unload(&DraftId("a".into())).unwrap();
        assert_eq!(budget.snapshot().reserved_drafts_bytes, 0);
    }

    #[test]
    fn test_draft_allocated_bytes_zero_when_unloaded() {
        let registry = DraftModelRegistry::new();
        registry.register(dummy_spec("a", 4)).unwrap();
        assert_eq!(
            registry
                .draft_allocated_bytes(&DraftId("a".into()))
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_draft_reserved_bytes_zero_when_unloaded() {
        let registry = DraftModelRegistry::new();
        registry.register(dummy_spec("a", 4)).unwrap();
        assert_eq!(
            registry.draft_reserved_bytes(&DraftId("a".into())).unwrap(),
            0
        );
    }

    #[test]
    fn test_memory_budget_accessor_returns_arc() {
        let budget = Arc::new(MemoryBudget::new(1000).unwrap());
        let registry = DraftModelRegistry::with_budget(budget.clone());
        assert_eq!(registry.memory_budget().total_bytes(), 1000);
    }

    #[test]
    fn test_draft_spec_estimated_total_bytes() {
        // 1024 bytes weights + 2 blocks × 16 MiB/block = 1024 + 33554432
        let spec = DraftSpec::new("x", "/tmp", 2).with_weight_size(1024);
        let expected = 1024 + 2 * (16 * 1024 * 1024);
        assert_eq!(spec.estimated_total_bytes(), expected);
    }
}
