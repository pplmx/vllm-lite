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

#[cfg(test)]
use base64::Engine;

/// `JwtError`: jwt error.
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

/// `JwtConfig`: jwt configuration.
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

#[derive(Debug)]
/// `JwtValidator`: jwt validator.
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
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
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
        let key = self.decoding_key(&header.alg)?;

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

    fn decoding_key(&self, alg: &Algorithm) -> Result<DecodingKey, JwtError> {
        if HMAC_ALGORITHMS.contains(alg) {
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
    pub fn extract_token(auth_header: &str) -> Option<&str> {
        auth_header.strip_prefix("Bearer ")
    }
}
#[derive(Debug)]

/// `JwtAuthMiddleware`: jwt auth middleware.
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

    /// Runs the operation.
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

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use serde_json::json;

    const TEST_ISS: &str = "vllm-test";
    const TEST_AUD: &str = "vllm-test-api";

    fn claims(now: u64, exp_offset_secs: i64) -> Claims {
        Claims {
            sub: "user-1".into(),
            iss: TEST_ISS.into(),
            aud: TEST_AUD.into(),
            exp: (now as i64 + exp_offset_secs).max(0) as u64,
            iat: now,
            roles: vec![],
            scope: None,
            extra: HashMap::new(),
        }
    }

    fn config_secret(secret: &str) -> JwtConfig {
        JwtConfig::with_secret(secret)
            .with_issuer(TEST_ISS)
            .with_audience(TEST_AUD)
    }

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

    #[test]
    fn test_validate_accepts_valid_hmac_token() {
        let secret = "test-secret-32-bytes-long-aaaa";
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let c = claims(now, 3600);
        let token = encode(
            &Header::new(Algorithm::HS256),
            &c,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();

        let validator = JwtValidator::new(config_secret(secret));
        let parsed = validator.validate(&token).expect("valid token must parse");
        assert_eq!(parsed.sub, "user-1");
        assert_eq!(parsed.iss, TEST_ISS);
    }

    #[test]
    fn test_validate_rejects_token_signed_with_wrong_secret() {
        let now = 1_700_000_000;
        let c = claims(now, 3600);
        let token = encode(
            &Header::new(Algorithm::HS256),
            &c,
            &EncodingKey::from_secret(b"signer-A-secret"),
        )
        .unwrap();

        let validator = JwtValidator::new(config_secret("verifier-B-secret"));
        let err = validator
            .validate(&token)
            .expect_err("wrong-secret token must fail");
        assert!(matches!(err, JwtError::InvalidSignature));
    }

    #[test]
    fn test_validate_rejects_tampered_token() {
        let secret = "test-secret-32-bytes-long-aaaa";
        let now = 1_700_000_000;
        let c = claims(now, 3600);
        let token = encode(
            &Header::new(Algorithm::HS256),
            &c,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();

        // Flip the last char of the payload section.
        let mut parts: Vec<&str> = token.split('.').collect();
        let payload = parts[1];
        let mut tampered_payload = payload.to_string();
        let last = tampered_payload.pop().unwrap();
        let flipped = if last == 'A' { 'B' } else { 'A' };
        tampered_payload.push(flipped);
        parts[1] = &tampered_payload;
        let tampered = parts.join(".");

        let validator = JwtValidator::new(config_secret(secret));
        let err = validator
            .validate(&tampered)
            .expect_err("tampered token must fail");
        assert!(matches!(err, JwtError::InvalidSignature));
    }

    #[test]
    fn test_validate_rejects_expired_token() {
        let secret = "test-secret-32-bytes-long-aaaa";
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let c = claims(now, -3600); // expired 1h ago
        let token = encode(
            &Header::new(Algorithm::HS256),
            &c,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();

        let validator = JwtValidator::new(config_secret(secret));
        let err = validator
            .validate(&token)
            .expect_err("expired token must fail");
        assert!(matches!(err, JwtError::TokenExpired));
    }

    #[test]
    fn test_validate_rejects_invalid_issuer() {
        let secret = "test-secret-32-bytes-long-aaaa";
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut c = claims(now, 3600);
        c.iss = "evil-corp".into();
        let token = encode(
            &Header::new(Algorithm::HS256),
            &c,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();

        let validator = JwtValidator::new(config_secret(secret));
        let err = validator.validate(&token).expect_err("wrong iss must fail");
        assert!(
            matches!(
                err,
                JwtError::InvalidIssuer(_) | JwtError::SignatureVerification(_)
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn test_validate_rejects_invalid_audience() {
        let secret = "test-secret-32-bytes-long-aaaa";
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut c = claims(now, 3600);
        c.aud = "evil-aud".into();
        let token = encode(
            &Header::new(Algorithm::HS256),
            &c,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();

        let validator = JwtValidator::new(config_secret(secret));
        let err = validator.validate(&token).expect_err("wrong aud must fail");
        assert!(
            matches!(
                err,
                JwtError::InvalidIssuer(_) | JwtError::SignatureVerification(_)
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn test_validate_rejects_none_algorithm() {
        // Manually craft an `alg: none` token (jsonwebtoken refuses to
        // produce one). Header: {"alg":"none","typ":"JWT"} ->
        // base64url(JSON). Payload: claims. Signature: empty.
        let header_json = json!({"alg": "none", "typ": "JWT"}).to_string();
        let header_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(header_json);
        let payload_json = json!({
            "sub": "user-1",
            "iss": TEST_ISS,
            "aud": TEST_AUD,
            "exp": 9_999_999_999u64,
            "iat": 1_700_000_000u64,
        })
        .to_string();
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload_json);
        let token = format!("{header_b64}.{payload_b64}.");

        let validator = JwtValidator::new(config_secret("any-secret"));
        let err = validator.validate(&token).expect_err("alg=none must fail");
        // alg=none can surface as UnsupportedAlgorithm (caught by our
        // explicit guard), as InvalidFormat (jsonwebtoken refuses to
        // parse `none` as a known algorithm at the header-decode step),
        // OR as InvalidSignature / SignatureVerification (caught by
        // jsonwebtoken downstream). All four reject the token — what
        // matters is that the validator does not accept the unsigned
        // token.
        assert!(
            matches!(
                err,
                JwtError::UnsupportedAlgorithm(_)
                    | JwtError::InvalidFormat(_)
                    | JwtError::InvalidSignature
                    | JwtError::SignatureVerification(_)
            ),
            "alg=none should be rejected, got: {err:?}"
        );
    }

    #[test]
    fn test_validate_rsa_without_pem_returns_key_config() {
        // With a public_key_pem config and an RSA-typed token, the
        // validator needs a valid PEM. Constructing one inline via
        // jsonwebtoken v9 is not exposed in the public API (no
        // generate_rsa). We verify the error-path instead: a config
        // missing `secret` (HMAC) but missing `public_key_pem` for an
        // RSA token will return KeyConfig.
        let now: u64 = 1_700_000_000;
        let c = claims(now, 3600);

        // Construct an RS256-typed token header and payload manually so
        // the validator's algorithm routing path runs (without needing
        // a generated RSA keypair, which jsonwebtoken v9 does not
        // expose publicly).
        let header_json = serde_json::json!({"alg": "RS256", "typ": "JWT"}).to_string();
        let header_b64 =
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(header_json.as_bytes());
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&c).unwrap());
        let token = format!("{header_b64}.{payload_b64}.AAAA");

        let validator = JwtValidator::new(
            JwtConfig::with_public_key("not-a-real-pem")
                .with_issuer(TEST_ISS)
                .with_audience(TEST_AUD),
        );
        let err = validator
            .validate(&token)
            .expect_err("invalid RSA setup must fail");
        assert!(matches!(err, JwtError::KeyConfig(_)));
    }

    #[test]
    fn test_validate_ecdsa_without_pem_returns_key_config() {
        let now: u64 = 1_700_000_000;
        let c = claims(now, 3600);

        // Construct an ES256-typed token header and payload manually so
        // the validator's algorithm routing path runs (without needing
        // a generated ECDSA keypair, which jsonwebtoken v9 does not
        // expose publicly).
        let header_json = serde_json::json!({"alg": "ES256", "typ": "JWT"}).to_string();
        let header_b64 =
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(header_json.as_bytes());
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&c).unwrap());
        let token = format!("{header_b64}.{payload_b64}.AAAA");

        let validator = JwtValidator::new(
            JwtConfig::with_public_key("not-a-real-pem")
                .with_issuer(TEST_ISS)
                .with_audience(TEST_AUD),
        );
        let err = validator
            .validate(&token)
            .expect_err("invalid ECDSA setup must fail");
        assert!(matches!(err, JwtError::KeyConfig(_)));
    }
}
