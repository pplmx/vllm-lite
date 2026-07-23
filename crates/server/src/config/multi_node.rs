//! Multi-node KV block transfer config (Phase 41 OPS-32a second-half).
//!
//! When `multi_node.enabled = true`, the server constructs a
//! `PagedKvCacheWrapper` and starts a gRPC server that answers
//! `TransferKVBlock` calls with real K/V bytes from the local cache.
//!
//! Single-node builds (no `multi-node` Cargo feature) compile this
//! module out so the call sites in `main.rs` short-circuit at compile
//! time — the config still exists (it's part of the on-disk schema)
//! but is parsed as plain YAML and never acted on.
//!
//! Example:
//!
//! ```yaml
//! server:
//!   multi_node:
//!     enabled: true
//!     node_id: my-node-01      # optional; auto-generated UUID v4 if absent
//!     bind_addr: 0.0.0.0:50051 # default; matches OPERATIONS.md quickstart
//! ```

use serde::{Deserialize, Serialize};

/// Multi-node KV block replication config (Phase 41 OPS-32a second-half).
///
/// When [`Self::enabled`] is `true`, the server constructs a
/// `PagedKvCacheWrapper` and starts a gRPC server that answers
/// `TransferKVBlock` calls with real K/V bytes from the local cache.
/// Default is `false` (single-node); operators opt in by setting
/// `server.multi_node.enabled: true` in the YAML / JSON config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::derivable_impls)]
pub struct MultiNodeConfig {
    /// When `true`, the server enables multi-node KV block transfer.
    /// Default: `false` (single-node).
    #[serde(default)]
    pub enabled: bool,

    /// Optional explicit node id. When `None`, the bootstrap generates
    /// a fresh `uuid::Uuid::new_v4().to_string()` and logs it.
    #[serde(default)]
    pub node_id: Option<String>,

    /// gRPC bind address. Default: `0.0.0.0:50051` (matches the
    /// OPERATIONS.md quickstart).
    #[serde(default = "default_multi_node_bind_addr")]
    pub bind_addr: String,
}

impl Default for MultiNodeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            node_id: None,
            bind_addr: default_multi_node_bind_addr(),
        }
    }
}

fn default_multi_node_bind_addr() -> String {
    "0.0.0.0:50051".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_disabled_with_sensible_bind_addr() {
        let cfg = MultiNodeConfig::default();
        assert!(!cfg.enabled);
        assert!(cfg.node_id.is_none());
        assert_eq!(cfg.bind_addr, "0.0.0.0:50051");
    }

    #[test]
    fn round_trip_yaml_parses_back() {
        let yaml = r"
            enabled: true
            node_id: my-node-01
            bind_addr: 127.0.0.1:60000
        ";
        let cfg: MultiNodeConfig = serde_saphyr::from_str(yaml).expect("parse");
        assert!(cfg.enabled);
        assert_eq!(cfg.node_id.as_deref(), Some("my-node-01"));
        assert_eq!(cfg.bind_addr, "127.0.0.1:60000");
    }

    #[test]
    fn empty_yaml_uses_defaults() {
        let cfg: MultiNodeConfig = serde_saphyr::from_str("").expect("parse empty");
        assert!(!cfg.enabled);
        assert!(cfg.node_id.is_none());
        assert_eq!(cfg.bind_addr, "0.0.0.0:50051");
    }
}
