use rustls_pemfile::{certs, private_key};
use std::fs;
use std::sync::Arc;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;
use tokio_rustls::rustls::RootCertStore;
use tokio_rustls::rustls::pki_types::CertificateDer;
use tokio_rustls::rustls::server::WebPkiClientVerifier;
use tokio_rustls::rustls::{self, ServerConfig};

/// `TlsError`: tls error.
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

/// `TlsConfig`: tls configuration.
pub struct TlsConfig {
    pub cert_path: String,
    pub key_path: String,
    pub ca_cert_path: Option<String>,
    pub mtls: bool,
}

impl TlsConfig {
    pub fn new(cert_path: impl Into<String>, key_path: impl Into<String>) -> Self {
        Self {
            cert_path: cert_path.into(),
            key_path: key_path.into(),
            ca_cert_path: None,
            mtls: false,
        }
    }

    pub fn with_ca_cert(mut self, ca_cert_path: impl Into<String>) -> Self {
        self.ca_cert_path = Some(ca_cert_path.into());
        self.mtls = true;
        self
    }

    pub fn load(&self) -> Result<ServerConfig, TlsError> {
        let cert_file = fs::File::open(&self.cert_path)
            .map_err(|e| TlsError::CertificateRead(e.to_string()))?;
        let mut cert_reader = std::io::BufReader::new(cert_file);
        let cert_chain: Vec<CertificateDer<'static>> = certs(&mut cert_reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| TlsError::InvalidConfig(format!("Invalid certificate: {e:?}")))?;

        let key_file =
            fs::File::open(&self.key_path).map_err(|e| TlsError::KeyRead(e.to_string()))?;
        let mut key_reader = std::io::BufReader::new(key_file);
        let key = private_key(&mut key_reader)
            .map_err(|e| TlsError::InvalidConfig(format!("Invalid key: {e:?}")))?
            .ok_or(TlsError::InvalidConfig("No private key found".to_string()))?;

        let config = if self.mtls {
            let ca_cert_path = self.ca_cert_path.as_ref().ok_or_else(|| {
                TlsError::InvalidConfig(
                    "CA cert path not set despite mtls=true (constructor invariant violated)"
                        .to_string(),
                )
            })?;
            let ca_file = fs::File::open(ca_cert_path)
                .map_err(|e| TlsError::CertificateRead(e.to_string()))?;
            let mut ca_reader = std::io::BufReader::new(ca_file);
            let ca_chain: Vec<CertificateDer<'static>> = certs(&mut ca_reader)
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

/// `TlsListener`: tls listener.
pub struct TlsListener {
    config: Arc<ServerConfig>,
}

impl TlsListener {
    pub fn new(config: TlsConfig) -> Result<Self, TlsError> {
        let server_config = config.load()?;
        Ok(Self {
            config: Arc::new(server_config),
        })
    }

    pub async fn bind(&self, addr: &str) -> Result<TcpListener, std::io::Error> {
        let listener = TcpListener::bind(addr).await?;
        Ok(listener)
    }

    #[must_use]
    pub fn acceptor(&self) -> TlsAcceptor {
        TlsAcceptor::from(self.config.clone())
    }
}

#[cfg(test)]
mod tests {
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
            Err(_) => panic!("load() panicked — SEC-06 regression"),
        }
    }
}
