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
    /// Free-form load failure (legacy — message-only).
    /// Prefer [`Self::LoadFailedWithSource`] for new code to preserve the
    /// underlying error chain.
    #[error("draft load failed: {0}")]
    LoadFailed(String),
    /// Typed load failure with preserved source chain.
    /// Use this when wrapping an underlying error (e.g., `candle_core::Error`).
    #[error("draft load failed: {message}")]
    LoadFailedWithSource {
        message: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("{0}")]
    MemoryBudgetExceeded(MemoryBudgetExceeded),
    /// A `Mutex`/`RwLock` guard was poisoned by a panic while held.
    /// Returning this as a typed error preserves the failure mode
    /// instead of unwinding the caller.
    #[error("draft registry lock poisoned")]
    LockPoisoned,
}

impl DraftRegistryError {
    /// Convenience constructor for [`Self::LoadFailedWithSource`].
    pub fn load_failed(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::LoadFailedWithSource {
            message: message.into(),
            source: Box::new(source),
        }
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
