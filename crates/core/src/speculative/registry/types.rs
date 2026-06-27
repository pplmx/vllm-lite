//! Draft registry data types.
//!
//! Defines the core value types used by [`DraftModelRegistry`](super::DraftModelRegistry):
//! the [`DraftId`] opaque identifier, [`DraftSpec`] registration metadata,
//! [`LoadedDraft`] loaded backend + allocator pair, and the [`DraftState`]
//! state machine.

use crate::scheduler::memory::BlockAllocator;
use crate::speculative::memory_budget::DEFAULT_BLOCK_BYTES;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use vllm_traits::ModelBackend;

/// Opaque, user-supplied identifier for a draft model.
///
/// Typically a slug like `"qwen-small"` or `"llama-tiny"`. Equality is by
/// string match; the registry treats ids case-sensitively.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DraftId(pub String);

impl DraftId {
    /// as_str: as str.
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
/// cheap; no I/O happens until [`DraftModelRegistry::attach_loaded`](super::DraftModelRegistry::attach_loaded)
/// is called.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftSpec {
    pub id: DraftId,
    pub model_dir: PathBuf,
    pub arch_hint: Option<String>,
    pub kv_blocks: usize,
    /// Conservative estimate of the draft's parameter footprint, in bytes.
    ///
    /// Sourced from `ModelLoader` metadata at registration time (v18.2 MEM-02).
    /// Used by the [`MemoryBudget`](crate::speculative::memory_budget::MemoryBudget)
    /// to decide whether loading this draft would exceed VRAM. Set conservatively —
    /// over-estimate is safe, under-estimate can cause runtime OOM.
    pub weight_size_estimate_bytes: u64,
    pub ref_count: usize,
}

impl DraftSpec {
    /// new: new.
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

    /// with_arch_hint: with arch hint.
    pub fn with_arch_hint(mut self, arch: impl Into<String>) -> Self {
        self.arch_hint = Some(arch.into());
        self
    }

    /// with_ref_count: with ref count.
    pub fn with_ref_count(mut self, ref_count: usize) -> Self {
        self.ref_count = ref_count;
        self
    }

    /// with_weight_size: with weight size.
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
    pub backend: Arc<Mutex<Box<dyn ModelBackend>>>,
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
    /// spec: spec.
    pub fn spec(&self) -> &DraftSpec {
        match self {
            DraftState::Unloaded(s) => s,
            DraftState::Loaded(l) => &l.spec,
        }
    }

    /// id: id.
    pub fn id(&self) -> &DraftId {
        &self.spec().id
    }

    /// is_loaded: is loaded.
    pub fn is_loaded(&self) -> bool {
        matches!(self, DraftState::Loaded(_))
    }
}
