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
    #[must_use]
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
        Self(s.to_string())
    }
}

impl From<String> for DraftId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Registration entry for a draft model.
///
/// Holds only the metadata needed to load the draft later. Construction is
/// cheap; no I/O happens until [`DraftModelRegistry::attach_loaded`](super::DraftModelRegistry::attach_loaded)
/// is called.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftSpec {
    /// User-supplied identifier (`"qwen-small"`, `"llama-tiny"`, ...).
    pub id: DraftId,
    /// Path to the checkpoint directory on disk.
    pub model_dir: PathBuf,
    /// Optional architecture hint to disambiguate the loader.
    pub arch_hint: Option<String>,
    /// Number of KV-cache blocks to allocate when this draft is loaded.
    pub kv_blocks: usize,
    /// Conservative estimate of the draft's parameter footprint, in bytes.
    ///
    /// Sourced from `ModelLoader` metadata at registration time (v18.2 MEM-02).
    /// Used by the [`MemoryBudget`](crate::speculative::memory_budget::MemoryBudget)
    /// to decide whether loading this draft would exceed VRAM. Set conservatively —
    /// over-estimate is safe, under-estimate can cause runtime OOM.
    pub weight_size_estimate_bytes: u64,
    /// Initial reference count (requests expected to use this draft concurrently).
    pub ref_count: usize,
}

impl DraftSpec {
    /// Construct a minimal `DraftSpec` with `weight_size_estimate_bytes = 0`
    /// and `ref_count = 0`. Use the `with_*` builder methods to fill in the
    /// remaining fields.
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

    /// Set the architecture hint (e.g. `"qwen3"`, `"llama"`). Used by the
    /// loader when the model directory's config is ambiguous.
    #[must_use]
    pub fn with_arch_hint(mut self, arch: impl Into<String>) -> Self {
        self.arch_hint = Some(arch.into());
        self
    }

    /// Set the initial reference count (number of requests expected to use
    /// this draft concurrently). The registry tracks the live ref-count
    /// separately; this is the seed value at registration time.
    #[must_use]
    pub const fn with_ref_count(mut self, ref_count: usize) -> Self {
        self.ref_count = ref_count;
        self
    }

    /// Set the conservative weight-size estimate in bytes. Over-estimate to
    /// stay safe; under-estimate risks OOM at load time.
    #[must_use]
    pub const fn with_weight_size(mut self, bytes: u64) -> Self {
        self.weight_size_estimate_bytes = bytes;
        self
    }

    /// Estimated total VRAM footprint of this draft if loaded:
    /// `weight_size_estimate_bytes + kv_blocks * DEFAULT_BLOCK_BYTES`.
    #[must_use]
    pub const fn estimated_total_bytes(&self) -> u64 {
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

impl std::fmt::Debug for LoadedDraft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedDraft")
            .field("spec", &self.spec)
            .field("backend", &"<dyn ModelBackend>")
            .field("block_allocator", &self.block_allocator)
            .finish()
    }
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
            Self::Unloaded(s) => f.debug_tuple("Unloaded").field(s).finish(),
            Self::Loaded(_) => f.debug_tuple("Loaded").field(&"<backend>").finish(),
        }
    }
}

impl DraftState {
    #[must_use]
    pub const fn spec(&self) -> &DraftSpec {
        match self {
            Self::Unloaded(s) => s,
            Self::Loaded(l) => &l.spec,
        }
    }

    #[must_use]
    pub const fn id(&self) -> &DraftId {
        &self.spec().id
    }

    #[must_use]
    pub const fn is_loaded(&self) -> bool {
        matches!(self, Self::Loaded(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `DraftState::is_loaded` must distinguish the two variants, and both
    /// `Debug` impls must name them.
    ///
    /// `cargo mutants` replaced `is_loaded` with the constants `true` and
    /// `false`, and swapped the `Debug for DraftState` and `Debug for
    /// LoadedDraft` bodies for `Ok(())`. All four survived: the registry tests
    /// match on the variants directly, so the accessor was never read, and
    /// nothing ever formatted these types. `is_loaded` is what the loader and
    /// the lifecycle metrics branch on.
    #[test]
    fn draft_state_is_loaded_and_debug_describe_the_variant() {
        let spec = DraftSpec::new("draft-7b", "models/draft", 512);
        let unloaded = DraftState::Unloaded(spec.clone());
        assert!(!unloaded.is_loaded(), "Unloaded must report not-loaded");
        assert_eq!(unloaded.id(), &spec.id);
        let rendered = format!("{unloaded:?}");
        assert!(
            rendered.contains("Unloaded"),
            "Debug must name the variant; got {rendered}"
        );

        // A loaded state needs a backend; the registry owns that construction,
        // so build it through the public path used by `attach_loaded`.
        let backend: Arc<Mutex<Box<dyn ModelBackend>>> = Arc::new(Mutex::new(Box::new(Noop)));
        let loaded = DraftState::Loaded(LoadedDraft {
            spec: spec.clone(),
            backend,
            block_allocator: BlockAllocator::new(8),
        });
        assert!(loaded.is_loaded(), "Loaded must report loaded");
        assert_eq!(loaded.id(), &spec.id, "both variants expose the same id");

        let rendered = format!("{loaded:?}");
        assert!(
            rendered.contains("Loaded"),
            "Debug must name the loaded variant; got {rendered}"
        );
        assert!(
            !rendered.contains("Unloaded"),
            "Debug must not confuse the variants; got {rendered}"
        );

        let rendered = format!(
            "{:?}",
            LoadedDraft {
                spec,
                backend: Arc::new(Mutex::new(Box::new(Noop))),
                block_allocator: BlockAllocator::new(8),
            }
        );
        assert!(
            rendered.contains("LoadedDraft") && rendered.contains("<dyn ModelBackend>"),
            "LoadedDraft's Debug must name the struct and hide the backend; got {rendered}"
        );
    }

    #[derive(Debug)]
    struct Noop;

    impl ModelBackend for Noop {
        fn forward(
            &mut self,
            _seq_ids: &[vllm_traits::SeqId],
            _input_tokens: &[Vec<vllm_traits::TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> vllm_traits::Result<vllm_traits::BatchOutput> {
            panic!("Noop::forward not used in types tests")
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
            panic!("Noop::forward_logits not used in types tests")
        }
        fn embed(
            &mut self,
            _input_tokens: &[Vec<vllm_traits::TokenId>],
            _positions: &[Vec<usize>],
        ) -> vllm_traits::Result<Vec<Vec<f32>>> {
            panic!("Noop::embed not used in types tests")
        }
        fn vocab_size(&self) -> usize {
            32000
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn num_heads(&self) -> usize {
            1
        }
    }
}
