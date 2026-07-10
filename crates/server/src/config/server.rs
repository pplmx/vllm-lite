//! `ServerConfig` — HTTP server bind address, TCP port, log level,
//! optional structured-log directory.

use serde::{Deserialize, Serialize};

/// HTTP server section: bind address, TCP port, log level, optional
/// structured-log directory. Constructed either from YAML/JSON via
/// [`super::AppConfig::load`] or programmatically via `ServerConfig::default()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::derivable_impls)]
pub struct ServerConfig {
    /// Bind address (e.g. `"0.0.0.0"` for all interfaces).
    #[serde(default = "default_host")]
    pub host: String,
    /// TCP port to listen on; validated to be non-zero.
    #[serde(default = "default_port")]
    pub port: u16,
    /// One of `trace`, `debug`, `info`, `warn`, `error`.
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// If set, JSON-formatted log lines are also written to this directory.
    #[serde(default)]
    pub log_dir: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            log_level: default_log_level(),
            log_dir: None,
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

const fn default_port() -> u16 {
    8000
}

fn default_log_level() -> String {
    "info".to_string()
}
