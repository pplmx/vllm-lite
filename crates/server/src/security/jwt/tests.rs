//! Unit tests for the JWT validator (`JwtValidator::validate`,
//! `JwtValidator::extract_token`, `JwtConfig` builders) and the
//! `JwtAuthMiddleware` wrapper.
//!
//! Extracted from `jwt.rs` to keep the implementation file under the
//! project's 800-line soft cap. Exercises the production forward paths
//! across:
//!
//! - HMAC happy-path (valid HS256 token accepted)
//! - HMAC error paths (wrong secret, tampered token, expired,
//!   wrong issuer, wrong audience, `alg: none`)
//! - Asymmetric error paths (RSA / ECDSA without a real PEM →
//!   `KeyConfig` error)
//! - Builder / extractor helpers (`with_secret`, `with_public_key`,
//!   `extract_token`)

use super::*;
use base64::Engine;
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde_json::json;

const TEST_ISS: &str = "vllm-test";
const TEST_AUD: &str = "vllm-test-api";

fn claims(now: u64, exp_offset_secs: i64) -> Claims {
    Claims {
        sub: "user-1".into(),
        iss: TEST_ISS.into(),
        aud: TEST_AUD.into(),
        exp: u64::try_from(i64::try_from(now).expect("bounded test timestamp") + exp_offset_secs)
            .unwrap_or(0),
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
    // invariant: pre-conditions make this infallible at this call site.
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
    let payload_b64 =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&c).unwrap());
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
    let payload_b64 =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&c).unwrap());
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
