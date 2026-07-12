//! CORS configuration: explicit allowlist of origins, methods, headers.
//!
//! Production-readiness recommendation §9: the CORS layer MUST be
//! configured via an explicit allowlist — never default to `*`. The
//! matching layer ([`crate::security::cors::CorsConfig`]) is closed
//! by default; operators opt in to browser-direct access by setting
//! `cors.allow_origins` in the YAML / JSON config file.
//!
//! Example:
//!
//! ```yaml
//! cors:
//!   allow_origins:
//!     - https://app.example.com
//!   allow_methods:
//!     - GET
//!     - POST
//!   allow_headers:
//!     - content-type
//!     - authorization
//!   allow_credentials: true
//! ```

use serde::{Deserialize, Serialize};

/// CORS configuration. All fields default to a closed state — no
/// `Access-Control-Allow-Origin` is emitted until the operator
/// explicitly lists at least one origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::derivable_impls)]
pub struct CorsConfigFile {
    /// Origins allowed by `Access-Control-Allow-Origin`. Empty =
    /// closed (no browser-direct access).
    #[serde(default)]
    pub allow_origins: Vec<String>,
    /// HTTP methods allowed by `Access-Control-Allow-Methods`.
    /// Empty = defaults to a permissive set (handled by the layer
    /// when empty).
    #[serde(default)]
    pub allow_methods: Vec<String>,
    /// Headers allowed by `Access-Control-Allow-Headers`. Empty =
    /// defaults to a permissive set.
    #[serde(default)]
    pub allow_headers: Vec<String>,
    /// Whether to set `Access-Control-Allow-Credentials: true`.
    /// MUST be `false` when `allow_origins` contains `*` (browsers
    /// reject that combination). The wire format accepts the value
    /// as-is; the runtime layer (`security::cors`) does not currently
    /// parse-and-reject, so operators are responsible for not setting
    /// this combination.
    #[serde(default)]
    pub allow_credentials: bool,
}

impl Default for CorsConfigFile {
    fn default() -> Self {
        Self {
            allow_origins: Vec::new(),
            allow_methods: Vec::new(),
            allow_headers: Vec::new(),
            allow_credentials: false,
        }
    }
}

impl CorsConfigFile {
    /// Convert to the runtime [`crate::security::cors::CorsConfig`]
    /// consumed by [`crate::security::cors::with_cors`]. Identity
    /// mapping today (the file config is intentionally shaped like
    /// the runtime config) but the indirection lets us add fields
    /// to the runtime config without breaking the on-disk format.
    #[must_use]
    pub fn into_runtime(self) -> crate::security::cors::CorsConfig {
        crate::security::cors::CorsConfig {
            allow_origins: self.allow_origins,
            allow_methods: self.allow_methods,
            allow_headers: self.allow_headers,
            allow_credentials: self.allow_credentials,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_closed() {
        let cfg = CorsConfigFile::default();
        assert!(cfg.allow_origins.is_empty());
        assert!(cfg.allow_methods.is_empty());
        assert!(cfg.allow_headers.is_empty());
        assert!(!cfg.allow_credentials);
    }

    #[test]
    fn into_runtime_passes_through() {
        let cfg = CorsConfigFile {
            allow_origins: vec!["https://example.com".to_string()],
            allow_methods: vec!["GET".to_string()],
            allow_headers: vec!["content-type".to_string()],
            allow_credentials: true,
        };
        let rt = cfg.into_runtime();
        assert_eq!(rt.allow_origins, vec!["https://example.com"]);
        assert_eq!(rt.allow_methods, vec!["GET"]);
        assert_eq!(rt.allow_headers, vec!["content-type"]);
        assert!(rt.allow_credentials);
    }
}
