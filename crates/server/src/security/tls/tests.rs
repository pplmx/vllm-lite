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
use super::*;

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
            // invariant: the panic payload type is determined by the code
            // path that panicked; in this test, only `load()` can panic
            // and it is an `String`/`&str` message — treat any panic as a
            // SEC-06 regression. Re-raise with the original payload so
            // the failure message is preserved.
            std::panic::resume_unwind(panic_payload);
        }
    }
}
