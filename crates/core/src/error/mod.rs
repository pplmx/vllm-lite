//! mod: module.

/// recovery: recovery module.
pub mod recovery;

/// EngineError: engine error.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("sequence {id} not found")]
    SeqNotFound { id: u64 },

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("model forward failed: {0}")]
    ModelError(String),

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
        EngineError::ModelError(err.to_string())
    }
}

impl From<crate::speculative::DraftRegistryError> for EngineError {
    fn from(err: crate::speculative::DraftRegistryError) -> Self {
        EngineError::ModelError(format!("draft registry: {}", err))
    }
}

impl From<crate::speculative::memory_budget::MemoryBudgetExceeded> for EngineError {
    fn from(err: crate::speculative::memory_budget::MemoryBudgetExceeded) -> Self {
        EngineError::ResourceExhausted {
            resource: format!("memory_budget: {}", err),
        }
    }
}

impl From<crate::circuit_breaker::breaker::CircuitBreakerError> for EngineError {
    fn from(err: crate::circuit_breaker::breaker::CircuitBreakerError) -> Self {
        EngineError::BackendUnavailable {
            backend: format!("circuit_breaker: {}", err),
        }
    }
}

impl From<crate::metrics::exporter::MetricsError> for EngineError {
    fn from(err: crate::metrics::exporter::MetricsError) -> Self {
        EngineError::ModelError(format!("metrics: {}", err))
    }
}

/// Result: result.
pub type Result<T> = std::result::Result<T, EngineError>;

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
}
