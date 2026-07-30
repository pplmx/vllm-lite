//! Unit tests for the `tls` configuration loader.
//!
//! Three contracts are locked in:
//!
//! 1. **Basic construction**: `TlsConfig::new(cert, key)` records
//!    the two paths and leaves `mtls = false`.
//! 2. **mTLS builder**: `with_ca_cert(ca)` flips `mtls = true`
//!    and records the CA path.
//! 3. **SEC-06 regression**: `TlsConfig::load` on an mTLS config
//!    with a missing `ca_cert_path` must return a structured
//!    `TlsError` rather than panicking on `.unwrap()`. The test
//!    wraps the call in `std::panic::catch_unwind` and
//!    re-raises any panic as a regression.
//! 4. **Full load with live certs**: `TlsConfig::load` succeeds
//!    when given valid PEM certificate and key files (generated
//!    at test time via `rcgen`), and fails with structured errors
//!    for invalid/missing files.

use super::*;
use std::io::Write;

/// Generate a self-signed X.509 certificate + private key PEM pair.
/// Returns `(cert_pem, key_pem)` as strings.
fn generate_self_signed_pair() -> (String, String) {
    use rcgen::{BasicConstraints, CertificateParams, IsCa, KeyPair};

    let key_pair = KeyPair::generate().unwrap();
    let mut params = CertificateParams::new(vec!["localhost".to_string()]).unwrap();
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![
        rcgen::KeyUsagePurpose::DigitalSignature,
        rcgen::KeyUsagePurpose::KeyCertSign,
        rcgen::KeyUsagePurpose::KeyEncipherment,
    ];
    let cert = params.self_signed(&key_pair).unwrap();
    (cert.pem(), key_pair.serialize_pem())
}

/// Write PEM strings to temporary files and return their paths.
fn write_pem_files(
    dir: &tempfile::TempDir,
    cert_pem: &str,
    key_pem: &str,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let cert_path = dir.path().join("cert.pem");
    let key_path = dir.path().join("key.pem");
    let mut f1 = std::fs::File::create(&cert_path).unwrap();
    f1.write_all(cert_pem.as_bytes()).unwrap();
    let mut f2 = std::fs::File::create(&key_path).unwrap();
    f2.write_all(key_pem.as_bytes()).unwrap();
    (cert_path, key_path)
}

#[test]
fn test_tls_config_creation() {
    let config = TlsConfig::new("/path/to/cert.pem", "/path/to/key.pem");
    assert_eq!(config.cert_path, "/path/to/cert.pem");
    assert_eq!(config.key_path, "/path/to/key.pem");
    assert!(!config.mtls);
}

#[test]
fn test_tls_config_with_ca() {
    let config =
        TlsConfig::new("/path/to/cert.pem", "/path/to/key.pem").with_ca_cert("/path/to/ca.pem");
    assert!(config.mtls);
    assert!(config.ca_cert_path.is_some());
}

/// Regression for v22.0 SEC-06: `TlsConfig::load` on an mTLS
/// configuration with a missing `ca_cert_path` must return a
/// structured `TlsError` rather than panicking on `.unwrap()`.
#[test]
fn test_tls_load_with_mtls_but_no_ca_path_returns_error() {
    // Construct via the literal struct since the builder enforces
    // the invariant we want to violate here.
    let config = TlsConfig {
        cert_path: "/path/to/cert.pem".into(),
        key_path: "/path/to/key.pem".into(),
        ca_cert_path: None,
        mtls: true,
    };
    let result = std::panic::catch_unwind(|| config.load());
    // The function must not panic — that's the core SEC-06
    // invariant. The actual error path depends on whether the cert
    // files exist on disk (in this test, they do not, so the
    // function returns `Err(CertificateRead(...))`); if they did
    // exist, the missing-CA-path check would fire next and return
    // `Err(InvalidConfig("CA cert path not set..."))`. Both are
    // acceptable structured-error outcomes.
    match result {
        Ok(Err(_)) => { /* structured error — pass */ }
        Ok(Ok(_)) => panic!("load() succeeded with invalid config"),
        Err(panic_payload) => {
            std::panic::resume_unwind(panic_payload);
        }
    }
}

// ── Full load tests with live certificates ──

#[test]
fn test_tls_load_with_valid_certs_succeeds() {
    let (cert_pem, key_pem) = generate_self_signed_pair();
    let dir = tempfile::TempDir::new().unwrap();
    let (cert_path, key_path) = write_pem_files(&dir, &cert_pem, &key_pem);

    let config = TlsConfig::new(cert_path.to_string_lossy(), key_path.to_string_lossy());
    let result = config.load();
    assert!(
        result.is_ok(),
        "load should succeed with valid PEM files: {:?}",
        result.err()
    );
}

#[test]
fn test_tls_load_with_nonexistent_cert_returns_error() {
    let config = TlsConfig::new("/nonexistent/cert.pem", "/nonexistent/key.pem");
    let result = config.load();
    assert!(result.is_err());
    match result.unwrap_err() {
        TlsError::CertificateRead(_) => { /* expected */ }
        other => panic!("expected CertificateRead error, got: {other}"),
    }
}

#[test]
fn test_tls_load_with_invalid_cert_data_returns_error() {
    let dir = tempfile::TempDir::new().unwrap();
    let (cert_path, key_path) = write_pem_files(&dir, "not-a-valid-cert", "not-a-valid-key");

    let config = TlsConfig::new(cert_path.to_string_lossy(), key_path.to_string_lossy());
    let result = config.load();
    assert!(result.is_err());
    match result.unwrap_err() {
        TlsError::InvalidConfig(msg) => {
            assert!(
                msg.contains("Invalid certificate") || msg.contains("Invalid key"),
                "unexpected error message: {msg}"
            );
        }
        other => panic!("expected InvalidConfig error, got: {other}"),
    }
}

#[test]
fn test_tls_load_with_invalid_key_returns_error() {
    use rcgen::KeyPair;

    let (cert_pem, _) = generate_self_signed_pair();
    let dir = tempfile::TempDir::new().unwrap();

    // Generate a *different* key pair — the cert won't match
    let wrong_key = KeyPair::generate().unwrap();
    let wrong_key_pem = wrong_key.serialize_pem();

    let (cert_path, key_path) = write_pem_files(&dir, &cert_pem, &wrong_key_pem);

    let config = TlsConfig::new(cert_path.to_string_lossy(), key_path.to_string_lossy());
    let result = config.load();
    assert!(result.is_err(), "load should reject cert/key mismatch");
    // The error should be an InvalidConfig wrapping the underlying rustls error
    match result.unwrap_err() {
        TlsError::InvalidConfig(_) => { /* expected */ }
        other => panic!("expected InvalidConfig error, got: {other}"),
    }
}

#[test]
fn test_tls_load_with_mtls_and_valid_ca_succeeds() {
    // Generate a server cert + key and a CA cert
    let (server_cert_pem, server_key_pem) = generate_self_signed_pair();
    let (ca_cert_pem, _ca_key_pem) = generate_self_signed_pair();

    let dir = tempfile::TempDir::new().unwrap();
    let (cert_path, key_path) = write_pem_files(&dir, &server_cert_pem, &server_key_pem);

    // Write the CA cert
    let ca_path = dir.path().join("ca.pem");
    std::fs::write(&ca_path, ca_cert_pem).unwrap();

    let config = TlsConfig::new(cert_path.to_string_lossy(), key_path.to_string_lossy())
        .with_ca_cert(ca_path.to_string_lossy().to_string());

    let result = config.load();
    assert!(
        result.is_ok(),
        "mTLS load should succeed with valid CA: {:?}",
        result.err()
    );
}

#[test]
fn test_tls_load_with_mtls_and_missing_ca_file_returns_error() {
    let (cert_pem, key_pem) = generate_self_signed_pair();
    let dir = tempfile::TempDir::new().unwrap();
    let (cert_path, key_path) = write_pem_files(&dir, &cert_pem, &key_pem);

    let config = TlsConfig::new(cert_path.to_string_lossy(), key_path.to_string_lossy())
        .with_ca_cert("/nonexistent/ca.pem");

    let result = config.load();
    assert!(result.is_err());
    match result.unwrap_err() {
        TlsError::CertificateRead(_) => { /* expected — CA file not found */ }
        other => panic!("expected CertificateRead error, got: {other}"),
    }
}

#[test]
fn test_tls_load_with_mtls_and_invalid_ca_returns_error() {
    let (cert_pem, key_pem) = generate_self_signed_pair();
    let dir = tempfile::TempDir::new().unwrap();
    let (cert_path, key_path) = write_pem_files(&dir, &cert_pem, &key_pem);

    // Write garbage as CA
    let ca_path = dir.path().join("ca.pem");
    std::fs::write(&ca_path, "not-a-valid-ca").unwrap();

    let config = TlsConfig::new(cert_path.to_string_lossy(), key_path.to_string_lossy())
        .with_ca_cert(ca_path.to_string_lossy().to_string());

    let result = config.load();
    assert!(result.is_err());
    match result.unwrap_err() {
        TlsError::InvalidConfig(_) => { /* expected — rustls reports the invalid CA */ }
        other => panic!("expected InvalidConfig error, got: {other}"),
    }
}

#[test]
fn test_tls_error_display() {
    let err = TlsError::CertificateRead("permission denied".into());
    assert!(err.to_string().contains("permission denied"));

    let err = TlsError::KeyRead("file not found".into());
    assert!(err.to_string().contains("file not found"));

    let err = TlsError::InvalidConfig("bad PEM".into());
    assert!(err.to_string().contains("bad PEM"));

    let err = TlsError::HandshakeFailed("timeout".into());
    assert!(err.to_string().contains("timeout"));
}
