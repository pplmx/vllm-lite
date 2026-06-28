//! SSM error type.

use thiserror::Error;

/// `SSMError`: ssm error.
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
