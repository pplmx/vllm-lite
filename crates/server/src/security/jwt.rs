#![allow(clippy::module_name_repetitions)]
//!
//! `JwtValidator::validate` parses and verifies a JWT's signature. The
//! verification algorithm is selected from an allowlist based on the
//! configured `JwtConfig`:
//!
//! - `secret` (HMAC path): HS256 only.
//! - `public_key_pem` (asymmetric path): RS256/RS384/RS512 (RSA) and
//!   ES256/ES384 (ECDSA). The actual algorithm is taken from the JWT
//!   header's `alg` field and must be in the allowlist.
//!
//! Tokens with `alg: none` or any algorithm outside the allowlist are
//! rejected with `JwtError::InvalidSignature`. Issuer/audience/exp
//! validation is delegated to `jsonwebtoken::Validation`.

use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

// Unit tests are extracted to `tests.rs` to keep this file under the
// 800-line soft cap. See `tests.rs` for the test surface.
#[cfg(test)]
mod tests;

/// Errors raised during JWT validation. Every variant maps to a
/// distinct failure mode so callers (auth middleware, batch job
/// validation) can branch on the specific cause rather than parsing
/// a single error message string.
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
    #[error("Signature verification failed: {0}")]
    SignatureVerification(String),
    #[error("Unsupported algorithm: {0}")]
    UnsupportedAlgorithm(String),
    #[error("Key configuration error: {0}")]
    KeyConfig(String),
}

/// JWT claim set. Standard registered claims (`sub`, `iss`, `aud`,
/// `exp`, `iat`) plus an open `extra: HashMap<String, Value>` for
/// application-specific fields (e.g. tenant id, scope list).
///
/// Serialized as the JWT body and verified against the configured
/// algorithm and audience before granting access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject — typically the user / service-account id.
    pub sub: String,
    /// Issuer — must match `JwtConfig::issuer`.
    pub iss: String,
    /// Audience — must match `JwtConfig::audience`.
    pub aud: String,
    /// Expiry as a UNIX timestamp (seconds).
    pub exp: u64,
    /// Issued-at as a UNIX timestamp (seconds).
    pub iat: u64,
    /// Optional role list; consumed by authorization checks downstream.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Optional OAuth-style space-separated scope string.
    #[serde(default)]
    pub scope: Option<String>,
    /// Catch-all bucket for application-specific claims (tenant id, etc.).
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// JWT verification configuration. Either `secret` (for HMAC HS256)
/// or `public_key_pem` (for RS*/ES*) must be set — the validator
/// refuses to start without one. Issuer and audience defaults are
/// `"vllm"` / `"vllm-api"` and can be overridden via the builder
/// methods.
#[derive(Debug, Clone)]
pub struct JwtConfig {
    /// HMAC shared secret; required for HS256.
    pub secret: Option<String>,
    /// PEM-encoded public key for RS256/RS384/RS512/ES256/ES384.
    pub public_key_pem: Option<String>,
    /// Expected issuer claim (`iss`); tokens with a different `iss` are rejected.
    pub issuer: String,
    /// Expected audience claim (`aud`); tokens with a different `aud` are rejected.
    pub audience: String,
    /// Whether to enforce the `exp` claim and reject expired tokens.
    pub validate_exp: bool,
}

impl JwtConfig {
    /// Build a [`JwtConfig`] configured for HS256 (HMAC) verification
    /// using `secret` as the shared key. `public_key_pem` is cleared
    /// and issuer/audience default to `"vllm"` / `"vllm-api"`.
    ///
    /// NOTE: This constructor is part of the public API — it is used by
    /// integration tests in `crates/server/tests/` which require external
    /// (`pub`) visibility. Keep as `pub`.
    #[must_use]
    pub fn with_secret(secret: impl Into<String>) -> Self {
        Self {
            secret: Some(secret.into()),
            public_key_pem: None,
            issuer: "vllm".to_string(),
            audience: "vllm-api".to_string(),
            validate_exp: true,
        }
    }

    /// Build a [`JwtConfig`] configured for RS*/ES* (asymmetric)
    /// verification using `public_key_pem`. `secret` is cleared and
    /// issuer/audience default to `"vllm"` / `"vllm-api"`.
    ///
    /// NOTE: Part of the public API — symmetric counterpart to
    /// [`JwtConfig::with_secret`]. External embedders configure
    /// [`JwtConfig`] via these constructors; keep as `pub`.
    #[must_use]
    pub fn with_public_key(public_key_pem: impl Into<String>) -> Self {
        Self {
            secret: None,
            public_key_pem: Some(public_key_pem.into()),
            issuer: "vllm".to_string(),
            audience: "vllm-api".to_string(),
            validate_exp: true,
        }
    }

    /// Override the expected `iss` claim. Tokens carrying a different
    /// issuer are rejected.
    ///
    /// NOTE: Part of the public API builder chain — used by integration
    /// tests in `crates/server/tests/`. Keep as `pub`.
    #[must_use]
    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = issuer.into();
        self
    }

    /// Override the expected `aud` claim. Tokens targeting a different
    /// audience are rejected.
    ///
    /// NOTE: Part of the public API builder chain — used by integration
    /// tests in `crates/server/tests/`. Keep as `pub`.
    #[must_use]
    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.audience = audience.into();
        self
    }
}

/// Algorithms allowed for HMAC (symmetric) verification.
const HMAC_ALGORITHMS: &[Algorithm] = &[Algorithm::HS256];

/// Algorithms allowed for asymmetric verification.
const ASYMMETRIC_ALGORITHMS: &[Algorithm] = &[
    Algorithm::RS256,
    Algorithm::RS384,
    Algorithm::RS512,
    Algorithm::ES256,
    Algorithm::ES384,
];

/// JWT validator. Wraps a [`JwtConfig`] and exposes a single
/// `validate(token) -> Result<Claims, JwtError>` method that performs
/// algorithm allowlist check + signature verification + standard
/// claim validation.
#[derive(Debug)]
pub struct JwtValidator {
    config: JwtConfig,
}

impl JwtValidator {
    #[must_use]
    pub const fn new(config: JwtConfig) -> Self {
        Self { config }
    }

    ///
    /// Verifies the JWT's signature, then validates standard claims
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// (iss, aud, exp). Returns the deserialized `Claims` on success.
    pub fn validate(&self, token: &str) -> Result<Claims, JwtError> {
        // Structural sanity check (3 parts) before delegating to
        // jsonwebtoken (which will reject malformed tokens anyway, but
        // the explicit guard surfaces a clearer error class).
        if token.split('.').count() != 3 {
            return Err(JwtError::InvalidFormat(
                "Token must have 3 parts".to_string(),
            ));
        }

        // Header sanity — peek at the alg before constructing the
        // DecodingKey so we can route to the right allowlist.
        let header = decode_header(token)
            .map_err(|e| JwtError::InvalidFormat(format!("Invalid header: {e}")))?;

        // Reject `alg: none` and any algorithm outside our allowlist
        // BEFORE attempting verification. jsonwebtoken's `Validation`
        // also enforces this, but the explicit guard makes the error
        // class more discoverable in logs.
        let allowed = if self.config.secret.is_some() {
            HMAC_ALGORITHMS
        } else {
            ASYMMETRIC_ALGORITHMS
        };
        if !allowed.contains(&header.alg) {
            return Err(JwtError::UnsupportedAlgorithm(format!("{:?}", header.alg)));
        }

        // Construct the decoding key for this token's alg family.
        let key = self.decoding_key(header.alg)?;

        // Build validation with the single allowed algorithm so
        // jsonwebtoken cannot be tricked into a different algorithm
        // (e.g. via alg-confusion: HMAC key + RS256 token).
        let mut validation = Validation::new(header.alg);
        validation.set_issuer(&[self.config.issuer.as_str()]);
        validation.set_audience(&[self.config.audience.as_str()]);
        validation.validate_exp = self.config.validate_exp;

        let token_data = decode::<Claims>(token, &key, &validation).map_err(|e| {
            use jsonwebtoken::errors::ErrorKind;
            match e.kind() {
                ErrorKind::ExpiredSignature => JwtError::TokenExpired,
                ErrorKind::InvalidIssuer => JwtError::InvalidIssuer(self.config.issuer.clone()),
                ErrorKind::InvalidAudience => JwtError::InvalidIssuer(self.config.audience.clone()),
                ErrorKind::InvalidSignature | ErrorKind::InvalidAlgorithm => {
                    JwtError::InvalidSignature
                }
                _ => JwtError::SignatureVerification(e.to_string()),
            }
        })?;

        Ok(token_data.claims)
    }

    fn decoding_key(&self, alg: Algorithm) -> Result<DecodingKey, JwtError> {
        if HMAC_ALGORITHMS.contains(&alg) {
            let secret = self
                .config
                .secret
                .as_ref()
                .ok_or_else(|| JwtError::KeyConfig("HMAC alg requires `secret`".into()))?;
            Ok(DecodingKey::from_secret(secret.as_bytes()))
        } else if matches!(alg, Algorithm::RS256 | Algorithm::RS384 | Algorithm::RS512) {
            let pem =
                self.config.public_key_pem.as_ref().ok_or_else(|| {
                    JwtError::KeyConfig("RSA alg requires `public_key_pem`".into())
                })?;
            DecodingKey::from_rsa_pem(pem.as_bytes())
                .map_err(|e| JwtError::KeyConfig(format!("Invalid RSA PEM: {e}")))
        } else if matches!(alg, Algorithm::ES256 | Algorithm::ES384) {
            let pem =
                self.config.public_key_pem.as_ref().ok_or_else(|| {
                    JwtError::KeyConfig("ECDSA alg requires `public_key_pem`".into())
                })?;
            DecodingKey::from_ec_pem(pem.as_bytes())
                .map_err(|e| JwtError::KeyConfig(format!("Invalid ECDSA PEM: {e}")))
        } else {
            Err(JwtError::UnsupportedAlgorithm(format!("{alg:?}")))
        }
    }

    #[must_use]
    pub(crate) fn extract_token(auth_header: &str) -> Option<&str> {
        auth_header.strip_prefix("Bearer ")
    }
}
#[derive(Debug)]

/// `JwtAuthMiddleware`. See the type definition for fields and behavior.
pub struct JwtAuthMiddleware {
    validator: Arc<RwLock<JwtValidator>>,
}

impl JwtAuthMiddleware {
    #[must_use]
    pub fn new(config: JwtConfig) -> Self {
        Self {
            validator: Arc::new(RwLock::new(JwtValidator::new(config))),
        }
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub async fn validate_request(&self, auth_header: &str) -> Result<Claims, JwtError> {
        let token = JwtValidator::extract_token(auth_header)
            .ok_or_else(|| JwtError::InvalidFormat("Missing Bearer token".to_string()))?;

        let validator = self.validator.read().await;
        validator.validate(token)
    }

    pub async fn update_config(&self, config: JwtConfig) {
        let mut validator = self.validator.write().await;
        *validator = JwtValidator::new(config);
    }
}
