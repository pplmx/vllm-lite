#![allow(clippy::module_name_repetitions)]
//! TLS configuration for the axum server: rustls-based server identity, optional client mTLS, and ALPN protocol negotiation.
//!
//! Activated when `AppConfig.tls.enabled = true`. Certificates and keys
//! are loaded from PEM files at startup; invalid certs cause a startup
//! abort rather than a runtime failure.
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::fs;
use std::sync::Arc;
use thiserror::Error;
use tokio_rustls::rustls::RootCertStore;
use tokio_rustls::rustls::server::WebPkiClientVerifier;
use tokio_rustls::rustls::{self, ServerConfig};

/// Errors raised during TLS configuration load or handshake. Each
/// variant carries the underlying error context as a `String` so
/// startup logs surface the exact PEM file or handshake failure
/// without exposing raw filesystem paths to the client.
#[derive(Debug, Error)]
pub enum TlsError {
    #[error("Failed to read certificate: {0}")]
    CertificateRead(String),
    #[error("Failed to read private key: {0}")]
    KeyRead(String),
    #[error("Invalid TLS configuration: {0}")]
    InvalidConfig(String),
    #[error("TLS handshake failed: {0}")]
    HandshakeFailed(String),
}

/// TLS configuration: PEM-encoded server certificate + private key,
/// plus optional CA bundle for client-certificate verification. Plain
/// TLS is selected when `mtls = false`; calling `TlsConfig::with_ca_cert`
/// flips the flag and installs the CA path for mTLS.
#[derive(Debug)]
pub struct TlsConfig {
    pub cert_path: String,
    pub key_path: String,
    pub ca_cert_path: Option<String>,
    pub mtls: bool,
}

impl TlsConfig {
    /// Build a plain (server-only) TLS configuration. Call
    /// `TlsConfig::with_ca_cert` to upgrade to mTLS.
    pub fn new(cert_path: impl Into<String>, key_path: impl Into<String>) -> Self {
        Self {
            cert_path: cert_path.into(),
            key_path: key_path.into(),
            ca_cert_path: None,
            mtls: false,
        }
    }

    /// Load the PEM files referenced by this config and assemble a
    /// rustls [`ServerConfig`] ready to wrap the axum listener.
    ///
    /// # Errors
    ///
    /// Returns [`TlsError::CertificateRead`] / [`TlsError::KeyRead`]
    /// if a referenced file cannot be read, [`TlsError::InvalidConfig`]
    /// if the PEM contents are malformed or the mTLS verifier builder
    /// rejects the CA bundle.
    pub fn load(&self) -> Result<ServerConfig, TlsError> {
        let cert_bytes =
            fs::read(&self.cert_path).map_err(|e| TlsError::CertificateRead(e.to_string()))?;
        let cert_chain: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(&cert_bytes)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| TlsError::InvalidConfig(format!("Invalid certificate: {e:?}")))?;

        let key_bytes = fs::read(&self.key_path).map_err(|e| TlsError::KeyRead(e.to_string()))?;
        let key = PrivateKeyDer::from_pem_slice(&key_bytes)
            .map_err(|e| TlsError::InvalidConfig(format!("Invalid key: {e:?}")))?;

        let config = if self.mtls {
            let ca_cert_path = self.ca_cert_path.as_ref().ok_or_else(|| {
                TlsError::InvalidConfig(
                    "CA cert path not set despite mtls=true (constructor invariant violated)"
                        .to_string(),
                )
            })?;
            let ca_bytes =
                fs::read(ca_cert_path).map_err(|e| TlsError::CertificateRead(e.to_string()))?;
            let ca_chain: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(&ca_bytes)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| TlsError::InvalidConfig(format!("Invalid CA: {e:?}")))?;

            let mut root_store = RootCertStore::empty();
            for cert in ca_chain {
                root_store
                    .add(cert)
                    .map_err(|e| TlsError::InvalidConfig(format!("Invalid CA cert: {e:?}")))?;
            }
            let verifier = WebPkiClientVerifier::builder(Arc::new(root_store))
                .build()
                .map_err(|e| TlsError::InvalidConfig(e.to_string()))?;

            ServerConfig::builder()
                .with_client_cert_verifier(verifier)
                .with_single_cert(cert_chain, key)
                .map_err(|e: rustls::Error| TlsError::InvalidConfig(e.to_string()))?
        } else {
            ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(cert_chain, key)
                .map_err(|e| TlsError::InvalidConfig(e.to_string()))?
        };

        Ok(config)
    }
}

#[cfg(test)]
impl TlsConfig {
    /// Enable mutual TLS by recording `ca_cert_path` (PEM bundle of
    /// trusted client CAs) and flipping `mtls = true`.
    #[must_use]
    pub(crate) fn with_ca_cert(mut self, ca_cert_path: impl Into<String>) -> Self {
        self.ca_cert_path = Some(ca_cert_path.into());
        self.mtls = true;
        self
    }
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this TLS
// config module under the 800-line soft cap. They cover the basic
// `TlsConfig::new` / `with_ca_cert` builder chain and the v22.0
// SEC-06 regression: `load()` on an mTLS config with a missing
// `ca_cert_path` must surface a structured `TlsError` (not panic).
#[cfg(test)]
mod tests;
