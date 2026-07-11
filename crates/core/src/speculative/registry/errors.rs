//! Errors surfaced by [`DraftModelRegistry`](super::DraftModelRegistry).

use super::types::DraftId;
use crate::speculative::memory_budget::MemoryBudgetExceeded;

/// Errors surfaced by the draft registry.
///
/// All variants are typed; callers can `match` on the failure category
/// without parsing message strings. The previously-deprecated
/// `LoadFailed` / `LoadFailedWithSource` variants and the
/// `load_failed()` convenience constructor were removed — they had no
/// production callers and the typed variants (`IoLoad`, `Model`) cover
/// the same use cases.
#[derive(Debug, thiserror::Error)]
pub enum DraftRegistryError {
    #[error("unknown draft id: {0}")]
    UnknownDraftId(DraftId),
    #[error("draft already loaded: {0}")]
    AlreadyLoaded(DraftId),
    #[error("draft still in use (refcount={0})")]
    InUse(usize),

    /// Typed: an I/O error occurred while reading the draft checkpoint.
    #[error("I/O error loading draft {draft_id} from {path}: {source}")]
    IoLoad {
        draft_id: DraftId,
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// Typed: model construction / weight validation failed. Carries the
    /// structured [`vllm_traits::ModelError`] for callers that want to
    /// distinguish OOM from shape mismatch from corrupt weights.
    #[error("model construction failed for draft {0}: {1}")]
    Model(DraftId, #[source] vllm_traits::ModelError),

    #[error("{0}")]
    MemoryBudgetExceeded(MemoryBudgetExceeded),

    /// A `Mutex`/`RwLock` guard was poisoned by a panic while held.
    /// Returning this as a typed error preserves the failure mode
    /// instead of unwinding the caller.
    #[error("draft registry lock poisoned")]
    LockPoisoned,
}

impl DraftRegistryError {
    /// Construct [`Self::IoLoad`] for a draft whose checkpoint read failed.
    pub fn io_load(draft_id: DraftId, path: impl Into<String>, source: std::io::Error) -> Self {
        Self::IoLoad {
            draft_id,
            path: path.into(),
            source,
        }
    }
}

impl From<std::io::Error> for DraftRegistryError {
    fn from(source: std::io::Error) -> Self {
        // No draft_id or path available at this `?` conversion point; the
        // call site should construct the variant explicitly via `io_load`
        // when those are known.
        Self::IoLoad {
            draft_id: DraftId::from("<unknown>"),
            path: String::new(),
            source,
        }
    }
}

impl From<vllm_traits::ModelError> for DraftRegistryError {
    fn from(source: vllm_traits::ModelError) -> Self {
        Self::Model(DraftId::from("<unknown>"), source)
    }
}

/// Convert any `std::sync::PoisonError<T>` into [`DraftRegistryError::LockPoisoned`].
///
/// This lets callers write `let guard = self.foo.lock()?;` inside any function
/// returning `Result<_, DraftRegistryError>` instead of `.expect("poisoned")`,
/// which would panic the runtime on a poisoned lock.
impl<T> From<std::sync::PoisonError<T>> for DraftRegistryError {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Self::LockPoisoned
    }
}
