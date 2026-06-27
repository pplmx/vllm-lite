//! Errors surfaced by [`DraftModelRegistry`](super::DraftModelRegistry).

use super::types::DraftId;
use crate::speculative::memory_budget::MemoryBudgetExceeded;

/// Errors surfaced by the draft registry.
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
