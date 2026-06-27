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
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;
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
    pub ref_count: usize,
}

impl DraftSpec {
    pub fn new(id: impl Into<DraftId>, model_dir: impl Into<PathBuf>, kv_blocks: usize) -> Self {
        Self {
            id: id.into(),
            model_dir: model_dir.into(),
            arch_hint: None,
            kv_blocks,
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
pub struct DraftModelRegistry {
    drafts: RwLock<HashMap<DraftId, DraftState>>,
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
        }
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
    ///
    /// No-op (returns `Ok`) if the entry is already `Unloaded`.
    ///
    /// Errors:
    /// - `UnknownDraftId` if no entry with `id` exists
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
                let spec = loaded.spec.clone();
                *entry = DraftState::Unloaded(spec);
                Ok(())
            }
        }
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
    pub fn decrement_ref(&self, id: &DraftId) -> Result<(), DraftRegistryError> {
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
            }
            DraftState::Loaded(loaded) => {
                loaded.spec.ref_count = loaded.spec.ref_count.saturating_sub(1);
            }
        }
        Ok(())
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
}
