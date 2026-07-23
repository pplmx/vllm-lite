//! `ObservabilityConfig` — OTLP exporter section of the server config.
//!
//! Only compiled when the `opentelemetry` Cargo feature is enabled on
//! `vllm-server` (which propagates `vllm-core/opentelemetry`).

use serde::{Deserialize, Serialize};

/// Observability configuration. Currently only OTLP is configurable;
/// Prometheus scraping stays on by default at `/metrics`.
///
/// Loaded from the `observability` top-level section of the server YAML/JSON
/// config or overridden by the `--otlp-endpoint` CLI flag.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ObservabilityConfig {
    /// OTLP exporter settings (endpoint, sampling, export interval).
    /// Defaults to `OtlpConfig::default()` (disabled).
    #[serde(default)]
    pub otlp: vllm_core::metrics::exporter::OtlpConfig,
}
