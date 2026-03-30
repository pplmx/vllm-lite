#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("sequence {id} not found")]
    SeqNotFound { id: u64 },

    #[error("model forward failed: {0}")]
    ModelError(String),

    #[error("sampling failed: {0}")]
    SamplingError(String),
}

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
}
