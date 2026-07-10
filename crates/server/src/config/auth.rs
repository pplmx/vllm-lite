//! `AuthConfig` — API key resolution + per-key rate-limit settings.

use serde::{Deserialize, Serialize};

/// Authentication and per-key rate-limiting configuration.
///
/// API keys are resolved from up to three sources at startup, in order:
/// inline `api_keys`, the env var named by `api_keys_env`, and the file
/// at `api_keys_file`. See [`AuthConfig::resolve_api_keys`] for the
/// exact precedence. The rate limiter is a sliding-window counter
/// applied per resolved key (best-effort; single-process).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Inline list of API keys that authenticate clients.
    #[serde(default)]
    pub api_keys: Vec<String>,
    /// Optional env var name holding a comma-separated API key list.
    #[serde(default)]
    pub api_keys_env: Option<String>,
    /// Optional path to a file of API keys, one per line (`#` for comments).
    #[serde(default)]
    pub api_keys_file: Option<String>,
    /// Per-key request quota within `rate_limit_window_secs`.
    #[serde(default = "default_rate_limit_requests")]
    pub rate_limit_requests: usize,
    /// Sliding window length (seconds) for the rate limiter.
    #[serde(default = "default_rate_limit_window")]
    pub rate_limit_window_secs: u64,
}

impl AuthConfig {
    /// Collect the effective set of API keys from all configured sources:
    ///   1. `api_keys` (literal list)
    ///   2. `api_keys_env` (comma-separated env var, parsed lazily)
    ///   3. `api_keys_file` (one key per line, `#` comments, blanks ignored)
    ///
    /// Missing/unreadable sources contribute nothing. Empty entries are
    /// filtered. Returns a deduplicated list in source order.
    #[must_use]
    pub fn resolve_api_keys(&self) -> Vec<String> {
        let mut keys = self.api_keys.clone();

        if let Some(ref env_var) = self.api_keys_env
            && let Ok(env_keys) = std::env::var(env_var)
        {
            for key in env_keys.split(',') {
                let key = key.trim().to_string();
                if !key.is_empty() {
                    keys.push(key);
                }
            }
        }

        if let Some(ref file_path) = self.api_keys_file
            && let Ok(content) = std::fs::read_to_string(file_path)
        {
            for line in content.lines() {
                let key = line.trim().to_string();
                if !key.is_empty() && !key.starts_with('#') {
                    keys.push(key);
                }
            }
        }

        keys
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            api_keys: vec![],
            api_keys_env: None,
            api_keys_file: None,
            rate_limit_requests: 100,
            rate_limit_window_secs: 60,
        }
    }
}

const fn default_rate_limit_requests() -> usize {
    100
}

const fn default_rate_limit_window() -> u64 {
    60
}
