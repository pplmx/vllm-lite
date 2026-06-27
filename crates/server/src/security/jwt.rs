//! jwt: jwt.

use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// JwtError: jwt error.
#[derive(Debug, Error)]
pub enum JwtError {
    #[error("Invalid token format: {0}")]
    InvalidFormat(String),
    #[error("Token expired")]
    TokenExpired,
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("Invalid issuer: {0}")]
    InvalidIssuer(String),
    #[error("Missing required claim: {0}")]
    MissingClaim(String),
}

/// Claims: claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub iss: String,
    pub aud: String,
    pub exp: u64,
    pub iat: u64,
    #[serde(default)]
    pub roles: Vec<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// JwtConfig: jwt configuration.
#[derive(Debug, Clone)]
pub struct JwtConfig {
    pub secret: Option<String>,
    pub public_key_pem: Option<String>,
    pub issuer: String,
    pub audience: String,
    pub validate_exp: bool,
}

impl JwtConfig {
/// with_secret: with secret.
    pub fn with_secret(secret: impl Into<String>) -> Self {
        Self {
            secret: Some(secret.into()),
            public_key_pem: None,
            issuer: "vllm".to_string(),
            audience: "vllm-api".to_string(),
            validate_exp: true,
        }
    }

/// with_public_key: with public key.
    pub fn with_public_key(public_key_pem: impl Into<String>) -> Self {
        Self {
            secret: None,
            public_key_pem: Some(public_key_pem.into()),
            issuer: "vllm".to_string(),
            audience: "vllm-api".to_string(),
            validate_exp: true,
        }
    }

/// with_issuer: with issuer.
    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = issuer.into();
        self
    }

/// with_audience: with audience.
    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.audience = audience.into();
        self
    }
}

/// JwtValidator: jwt validator.
pub struct JwtValidator {
    config: JwtConfig,
}

impl JwtValidator {
/// new: new.
    pub fn new(config: JwtConfig) -> Self {
        Self { config }
    }

/// validate: validate.
    pub fn validate(&self, token: &str) -> Result<Claims, JwtError> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(JwtError::InvalidFormat(
                "Token must have 3 parts".to_string(),
            ));
        }

        let payload_part = parts[1];

        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(payload_part)
            .map_err(|_| JwtError::InvalidFormat("Invalid base64".to_string()))?;

        let claims: Claims =
            serde_json::from_slice(&payload).map_err(|e| JwtError::InvalidFormat(e.to_string()))?;

        if self.config.validate_exp {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            if claims.exp < now {
                return Err(JwtError::TokenExpired);
            }
        }

        if claims.iss != self.config.issuer {
            return Err(JwtError::InvalidIssuer(claims.iss));
        }

        if claims.aud != self.config.audience {
            return Err(JwtError::InvalidIssuer(claims.aud));
        }

        Ok(claims)
    }

/// extract_token: extract token.
    pub fn extract_token(auth_header: &str) -> Option<&str> {
        auth_header.strip_prefix("Bearer ")
    }
}

/// JwtAuthMiddleware: jwt auth middleware.
pub struct JwtAuthMiddleware {
    validator: Arc<RwLock<JwtValidator>>,
}

impl JwtAuthMiddleware {
/// new: new.
    pub fn new(config: JwtConfig) -> Self {
        Self {
            validator: Arc::new(RwLock::new(JwtValidator::new(config))),
        }
    }

/// validate_request: validate request.
    pub async fn validate_request(&self, auth_header: &str) -> Result<Claims, JwtError> {
        let token = JwtValidator::extract_token(auth_header)
            .ok_or_else(|| JwtError::InvalidFormat("Missing Bearer token".to_string()))?;

        let validator = self.validator.read().await;
        validator.validate(token)
    }

/// update_config: update config.
    pub async fn update_config(&self, config: JwtConfig) {
        let mut validator = self.validator.write().await;
        *validator = JwtValidator::new(config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_token() {
        assert_eq!(JwtValidator::extract_token("Bearer abc123"), Some("abc123"));
        assert_eq!(JwtValidator::extract_token("Basic abc"), None);
        assert_eq!(JwtValidator::extract_token("bearer abc"), None);
    }

    #[test]
    fn test_jwt_config_with_secret() {
        let config = JwtConfig::with_secret("my-secret");
        assert_eq!(config.secret, Some("my-secret".to_string()));
        assert!(config.public_key_pem.is_none());
    }

    #[test]
    fn test_jwt_config_with_public_key() {
        let config = JwtConfig::with_public_key("-----BEGIN PUBLIC KEY-----");
        assert!(config.secret.is_none());
        assert!(config.public_key_pem.is_some());
    }
}
