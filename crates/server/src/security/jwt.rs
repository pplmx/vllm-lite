use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

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

#[derive(Debug, Clone)]
pub struct JwtConfig {
    pub secret: Option<String>,
    pub public_key_pem: Option<String>,
    pub issuer: String,
    pub audience: String,
    pub validate_exp: bool,
}

impl JwtConfig {
    pub fn with_secret(secret: impl Into<String>) -> Self {
        Self {
            secret: Some(secret.into()),
            public_key_pem: None,
            issuer: "vllm".to_string(),
            audience: "vllm-api".to_string(),
            validate_exp: true,
        }
    }

    pub fn with_public_key(public_key_pem: impl Into<String>) -> Self {
        Self {
            secret: None,
            public_key_pem: Some(public_key_pem.into()),
            issuer: "vllm".to_string(),
            audience: "vllm-api".to_string(),
            validate_exp: true,
        }
    }

    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = issuer.into();
        self
    }

    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.audience = audience.into();
        self
    }
}

pub struct JwtValidator {
    config: JwtConfig,
    secret_key: Option<Vec<u8>>,
}

impl JwtValidator {
    pub fn new(config: JwtConfig) -> Self {
        let secret_key = config.secret.as_ref().map(|s| s.as_bytes().to_vec());
        Self {
            config,
            secret_key,
        }
    }

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

        let claims: Claims = serde_json::from_slice(&payload)
            .map_err(|e| JwtError::InvalidFormat(e.to_string()))?;

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

    pub fn extract_token(auth_header: &str) -> Option<&str> {
        if auth_header.starts_with("Bearer ") {
            Some(&auth_header[7..])
        } else {
            None
        }
    }
}

pub struct JwtAuthMiddleware {
    validator: Arc<RwLock<JwtValidator>>,
}

impl JwtAuthMiddleware {
    pub fn new(config: JwtConfig) -> Self {
        Self {
            validator: Arc::new(RwLock::new(JwtValidator::new(config))),
        }
    }

    pub async fn validate_request(&self, auth_header: &str) -> Result<Claims, JwtError> {
        let token = JwtAuthMiddleware::extract_token(auth_header)
            .ok_or_else(|| JwtError::InvalidFormat("Missing Bearer token".to_string()))?;

        let validator = self.validator.read().await;
        validator.validate(token)
    }

    pub async fn update_config(&self, config: JwtConfig) {
        let mut validator = self.validator.write().await;
        *validator = JwtValidator::new(config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_claims(exp: u64) -> String {
        let claims = Claims {
            sub: "user123".to_string(),
            iss: "vllm".to_string(),
            aud: "vllm-api".to_string(),
            exp,
            iat: 0,
            roles: vec!["user".to_string()],
            scope: None,
            extra: HashMap::new(),
        };
        let json = serde_json::to_string(&claims).unwrap();
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json.as_bytes());
        format!("header.{}.signature", encoded)
    }

    #[test]
    fn test_extract_token() {
        assert_eq!(
            JwtAuthMiddleware::extract_token("Bearer abc123"),
            Some("abc123")
        );
        assert_eq!(JwtAuthMiddleware::extract_token("Basic abc"), None);
        assert_eq!(JwtAuthMiddleware::extract_token("bearer abc"), None);
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
