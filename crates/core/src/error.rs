#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("generic error: {0}")]
    Generic(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;
