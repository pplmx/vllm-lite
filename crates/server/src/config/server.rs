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

/// SEC-01 (technical due diligence): classify a bind address as loopback
/// or non-loopback so the server can warn (or refuse) to start when it
/// would expose the inference API to the network without authentication.
///
/// The classification is intentionally conservative:
/// - `127.0.0.0/8`, `::1`, `localhost`, and empty values are loopback.
/// - Everything else (including `0.0.0.0`, `::`, `192.168.x.x`,
///   `10.x.x.x`, DNS names) is non-loopback.
///
/// We deliberately do not try to resolve DNS names here; a hostname that
/// points at a loopback address is still treated as non-loopback because
/// the operator should make the bind explicit. The check exists to
/// catch the most common foot-gun (`--host 0.0.0.0` without auth), not
/// to be a complete trust-boundary analyzer.
#[must_use]
pub fn is_loopback_address(host: &str) -> bool {
    let h = host.trim();
    if h.is_empty() || h.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if h == "::1" {
        return true;
    }
    // IPv4 loopback range: 127.0.0.0/8.
    if let Some(rest) = h.strip_prefix("127.") {
        return rest
            .split('.')
            .all(|octet| !octet.is_empty() && octet.parse::<u8>().is_ok());
    }
    false
}

#[cfg(test)]
mod loopback_tests {
    use super::is_loopback_address;

    #[test]
    fn loopback_addresses_classified_as_loopback() {
        for host in [
            "127.0.0.1",
            "127.0.0.42",
            "127.255.255.254",
            "::1",
            "localhost",
            "LOCALHOST",
            "",
        ] {
            assert!(
                is_loopback_address(host),
                "{host:?} should be classified as loopback"
            );
        }
    }

    #[test]
    fn non_loopback_addresses_classified_as_exposed() {
        for host in [
            "0.0.0.0",
            "::",
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "example.com",
            "vllm.internal",
        ] {
            assert!(
                !is_loopback_address(host),
                "{host:?} must be classified as non-loopback so the SEC-01 warning fires"
            );
        }
    }

    #[test]
    fn malformed_loopback_prefix_does_not_slide_through() {
        // 127.999.x.x is not a real IPv4 address. We must not classify it
        // as loopback just because of the prefix.
        assert!(!is_loopback_address("127.999.0.1"));
        assert!(!is_loopback_address("127..1.1"));
    }
}

