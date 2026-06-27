//! Typed errors for model configuration parsing.
//!
//! Replaces the previous `Box<dyn std::error::Error>` return types in
//! [`crate::config::ModelConfig::from_config_json`] and
//! [`crate::qwen3::config::Qwen3Config::from_file`] (Phase 32 / API-03).

use thiserror::Error;

/// Errors that can occur while parsing model configuration from JSON.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The architecture detected from config JSON is not recognized.
    #[error("unknown architecture: {0}")]
    UnknownArchitecture(String),

    /// A required field is missing from the config JSON.
    #[error("missing required field: {0}")]
    MissingField(&'static str),

    /// A field has an invalid value (wrong type, out of range, etc.).
    #[error("invalid field {field}: {message}")]
    InvalidField { field: String, message: String },

    /// Underlying JSON parse/format error.
    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// Underlying I/O error while reading a config file.
    #[error("io error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Result alias for configuration operations.
pub type ConfigResult<T> = Result<T, ConfigError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknown_architecture_display() {
        let err = ConfigError::UnknownArchitecture("foo".into());
        assert_eq!(err.to_string(), "unknown architecture: foo");
    }

    #[test]
    fn test_missing_field_display() {
        let err = ConfigError::MissingField("hidden_size");
        assert_eq!(err.to_string(), "missing required field: hidden_size");
    }

    #[test]
    fn test_invalid_field_display() {
        let err = ConfigError::InvalidField {
            field: "num_heads".into(),
            message: "must be > 0".into(),
        };
        assert_eq!(
            err.to_string(),
            "invalid field num_heads: must be > 0"
        );
    }

    #[test]
    fn test_io_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let err = ConfigError::Io {
            path: "/tmp/missing.json".into(),
            source: io_err,
        };
        assert!(err.to_string().contains("io error reading /tmp/missing.json"));
    }

    #[test]
    fn test_source_chain_preserved() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err = ConfigError::Io {
            path: "/secret".into(),
            source: io_err,
        };
        // Source chain must be preserved for `Error::source()` lookups.
        let source = std::error::Error::source(&err);
        assert!(source.is_some(), "source chain must be preserved");
    }
}
