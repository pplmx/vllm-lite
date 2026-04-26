use std::fs;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio_rustls::rustls::{
    internal::pemfile::{certs, pkcs8_private_keys},
    server::AllowAnyAuthenticatedClient,
    CertRequest, Dumenor, NoClientAuth, PrivateKey, ServerConfig, ServerSession,
};
use tokio_rustls::TlsAcceptor;

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
        let cert_chain = certs(&mut cert_reader)
            .map_err(|e| TlsError::InvalidConfig(format!("Invalid certificate: {:?}", e)))?;

        let key_file = fs::File::open(&self.key_path)
            .map_err(|e| TlsError::KeyRead(e.to_string()))?;
        let mut key_reader = std::io::BufReader::new(key_file);
        let mut keys: Vec<PrivateKey> = pkcs8_private_keys(&mut key_reader)
            .map_err(|e| TlsError::InvalidConfig(format!("Invalid key: {:?}", e)))?;

        if keys.is_empty() {
            return Err(TlsError::InvalidConfig("No private key found".to_string()));
        }

        let mut config = if self.mtls {
            let ca_cert_file = fs::File::open(self.ca_cert_path.as_ref().unwrap())
                .map_err(|e| TlsError::CertificateRead(e.to_string()))?;
            let mut ca_reader = std::io::BufReader::new(ca_cert_file);
            let ca_chain = certs(&mut ca_reader)
                .map_err(|e| TlsError::InvalidConfig(format!("Invalid CA certificate: {:?}", e)))?;

            let verifier = AllowAnyAuthenticatedClient::new(ca_chain.into());
            let mut cfg = ServerConfig::new(verifier);
            cfg.set_single_cert(cert_chain, keys.remove(0))
                .map_err(|e| TlsError::InvalidConfig(e.to_string()))?;
            cfg
        } else {
            let verifier = NoClientAuth::new();
            let mut cfg = ServerConfig::new(verifier);
            cfg.set_single_cert(cert_chain, keys.remove(0))
                .map_err(|e| TlsError::InvalidConfig(e.to_string()))?;
            cfg
        };

        config.set_protocols(&["h2".into(), "http/1.1".into()]);

        Ok(config)
    }
}

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
        let config = TlsConfig::new("/path/to/cert.pem", "/path/to/key.pem")
            .with_ca_cert("/path/to/ca.pem");
        assert!(config.mtls);
        assert!(config.ca_cert_path.is_some());
    }
}
