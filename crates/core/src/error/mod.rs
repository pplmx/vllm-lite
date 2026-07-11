#![allow(clippy::module_name_repetitions)]
//! Engine-level error types and inter-crate `From` conversions.
//!
//! Every fallible operation in `vllm-core` returns [`Result<T, EngineError>`]
//! (alias [`Result`]). Variants cover the four major failure modes:
//! request validation, model forward errors, scheduler timeouts, and
//! resource exhaustion (e.g. KV-cache blocks, draft memory budget).

pub mod recovery;

/// Top-level engine error type. Every public API in `vllm-core` returns `Result<T, EngineError>`. Variants cover model load, scheduler, sampling, KV-cache, and tokenizer failures; see the enum definition for the full list.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("sequence {id} not found")]
    SeqNotFound { id: u64 },

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Free-form model error (legacy, kept for backward compat).
    /// New code should construct via the typed variant [`Self::Model`].
    #[error("model forward failed: {0}")]
    ModelError(String),

    /// Typed model error with preserved source chain.
    /// Use this when wrapping a `vllm_traits::ModelError` to preserve the
    /// `Error::source()` chain for log correlation.
    #[error("model error: {0}")]
    Model(#[source] vllm_traits::ModelError),

    #[error("sampling failed: {0}")]
    SamplingError(String),

    #[error("internal lock poisoned")]
    LockPoisoned,

    #[error("operation '{op}' timed out after {ms} ms")]
    Timeout { op: String, ms: u64 },

    #[error("request {request_id} was cancelled")]
    Cancelled { request_id: u64 },

    #[error("resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    #[error("backend unavailable: {backend}")]
    BackendUnavailable { backend: String },
}

impl From<vllm_traits::ModelError> for EngineError {
    fn from(err: vllm_traits::ModelError) -> Self {
        // Use the typed variant to preserve the source chain.
        Self::Model(err)
    }
}

impl From<crate::speculative::DraftRegistryError> for EngineError {
    fn from(err: crate::speculative::DraftRegistryError) -> Self {
        Self::Model(vllm_traits::ModelError::new(format!(
            "draft registry: {err}"
        )))
    }
}

impl From<crate::speculative::memory_budget::MemoryBudgetExceeded> for EngineError {
    fn from(err: crate::speculative::memory_budget::MemoryBudgetExceeded) -> Self {
        Self::ResourceExhausted {
            resource: format!("memory_budget: {err}"),
        }
    }
}

impl From<crate::circuit_breaker::breaker::CircuitBreakerError> for EngineError {
    fn from(err: crate::circuit_breaker::breaker::CircuitBreakerError) -> Self {
        Self::BackendUnavailable {
            backend: format!("circuit_breaker: {err}"),
        }
    }
}

impl From<crate::metrics::exporter::MetricsError> for EngineError {
    fn from(err: crate::metrics::exporter::MetricsError) -> Self {
        Self::Model(vllm_traits::ModelError::new(format!("metrics: {err}")))
    }
}

/// Convenience alias used by every public API in `vllm-core`.
pub type Result<T> = std::result::Result<T, EngineError>;

/// Convert any `std::sync::PoisonError<T>` into [`EngineError::LockPoisoned`].
///
/// This lets lock-guarded callers write `let guard = self.field.lock()?`
/// instead of `.expect("mutex poisoned")`, propagating the failure as a typed
/// `EngineError` rather than a panic. The wrapped `T` is discarded because
/// the engine cannot meaningfully continue with a poisoned lock — the
/// invariant of the protected data is broken until the lock is re-created.
impl<T> From<std::sync::PoisonError<T>> for EngineError {
    fn from(_err: std::sync::PoisonError<T>) -> Self {
        Self::LockPoisoned
    }
}

impl EngineError {
    /// Attach a `request_id` to this error for log correlation.
    ///
    /// No-op for variants that don't carry per-request context.
    /// Returns the modified error so callers can chain: `err.with_request_id(id)?`.
    #[must_use]
    pub const fn with_request_id(self, request_id: u64) -> Self {
        // Currently, no variant carries optional request_id — this is a no-op
        // hook for future per-variant additions. Kept as a stable API so
        // adding structured context later is non-breaking.
        let _ = request_id;
        self
    }

    /// Attach a `seq_id` to this error for log correlation.
    /// Same semantics as [`Self::with_request_id`].
    #[must_use]
    pub const fn with_seq_id(self, seq_id: u64) -> Self {
        let _ = seq_id;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seq_not_found_error_message() {
        let err = EngineError::SeqNotFound { id: 42 };
        assert_eq!(err.to_string(), "sequence 42 not found");
    }

    #[test]
    fn test_model_error_message() {
        let err = EngineError::ModelError("out of memory".to_string());
        assert_eq!(err.to_string(), "model forward failed: out of memory");
    }

    #[test]
    fn test_model_typed_preserves_source() {
        let inner = vllm_traits::ModelError::new("candle backend crashed");
        let err = EngineError::from(inner);
        match &err {
            EngineError::Model(_) => {}
            other => panic!("expected Model variant, got {other:?}"),
        }
        // Source chain must be preserved.
        let source = std::error::Error::source(&err);
        assert!(source.is_some(), "source chain must be preserved");
    }

    #[test]
    fn test_sampling_error_message() {
        let err = EngineError::SamplingError("invalid temperature".to_string());
        assert_eq!(err.to_string(), "sampling failed: invalid temperature");
    }

    #[test]
    fn test_timeout_error_message() {
        let err = EngineError::Timeout {
            op: "forward".to_string(),
            ms: 1500,
        };
        assert_eq!(
            err.to_string(),
            "operation 'forward' timed out after 1500 ms"
        );
    }

    #[test]
    fn test_cancelled_error_message() {
        let err = EngineError::Cancelled { request_id: 42 };
        assert_eq!(err.to_string(), "request 42 was cancelled");
    }

    #[test]
    fn test_resource_exhausted_error_message() {
        let err = EngineError::ResourceExhausted {
            resource: "kv_blocks".to_string(),
        };
        assert_eq!(err.to_string(), "resource exhausted: kv_blocks");
    }

    #[test]
    fn test_backend_unavailable_error_message() {
        let err = EngineError::BackendUnavailable {
            backend: "cuda".to_string(),
        };
        assert_eq!(err.to_string(), "backend unavailable: cuda");
    }

    #[test]
    fn test_with_request_id_returns_self() {
        let err = EngineError::InvalidRequest("test".into()).with_request_id(42);
        // No variant carries request_id yet — this is a stable no-op hook.
        // The test ensures the API exists and doesn't break.
        let _ = err;
    }

    #[test]
    fn test_with_seq_id_returns_self() {
        let err = EngineError::InvalidRequest("test".into()).with_seq_id(7);
        let _ = err;
    }
}
