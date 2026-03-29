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
