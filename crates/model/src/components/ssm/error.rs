//! SSM error type.

use thiserror::Error;

/// Error type for SSM. Returned from every fallible public API; covers I/O, validation, and resource-limit failures. Use [`Result<T>`] alias in the same module.
#[derive(Debug, Error)]
pub enum SSMError {
    #[error("{0}")]
    Msg(String),
}

impl From<std::convert::Infallible> for SSMError {
    fn from(_: std::convert::Infallible) -> Self {
        Self::Msg("Infallible error".to_string())
    }
}
