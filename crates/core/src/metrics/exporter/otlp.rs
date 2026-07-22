//! OpenTelemetry (OTLP) push-based exporter — gated by the `opentelemetry`
//! feature on `vllm-core`. Streams engine metrics + tracing spans to any
//! OTel-compatible collector (Jaeger / Tempo / Datadog / Honeycomb / etc.).

#![cfg(feature = "opentelemetry")]

use serde::{Deserialize, Serialize};

/// Wire protocol for the OTLP exporter. Only `Grpc` is supported in v43;
/// the enum is reserved so adding `Http` later is a non-breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OtlpProtocol {
    #[default]
    Grpc,
}

/// Configuration for the OTLP exporter. Loaded from the
/// `app_config.observability.otlp` YAML section or the `--otlp-endpoint`
/// CLI override.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpConfig {
    /// Master switch — when `false`, the bootstrap skips OTLP entirely.
    pub enabled: bool,
    /// OTLP collector endpoint (gRPC). Default: `"http://localhost:4317"`.
    pub endpoint: String,
    /// OTel `service.name` resource attribute.
    pub service_name: String,
    /// OTel `service.version` resource attribute (synced from the release manifest).
    pub service_version: String,
    /// Metrics export interval in seconds. Default: `30`.
    pub metrics_export_interval_secs: u64,
    /// Trace sampling ratio in `[0.0, 1.0]`. Default: `1.0` (always sample).
    pub trace_sampling_ratio: f64,
    /// OTLP transport protocol. Default: `Grpc`.
    pub protocol: OtlpProtocol,
}

impl Default for OtlpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://localhost:4317".to_string(),
            service_name: "vllm-lite".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            metrics_export_interval_secs: 30,
            trace_sampling_ratio: 1.0,
            protocol: OtlpProtocol::Grpc,
        }
    }
}

impl OtlpConfig {
    /// Validate field ranges. Returns `Err(OtlpError::Config)` for any
    /// out-of-range value. Safe to call repeatedly.
    pub fn validate(&self) -> Result<(), OtlpError> {
        if !self.trace_sampling_ratio.is_finite() || !(0.0..=1.0).contains(&self.trace_sampling_ratio) {
            return Err(OtlpError::Config(format!(
                "trace_sampling_ratio = {} is out of range [0.0, 1.0]",
                self.trace_sampling_ratio
            )));
        }
        if self.metrics_export_interval_secs == 0 {
            return Err(OtlpError::Config(
                "metrics_export_interval_secs must be > 0".to_string(),
            ));
        }
        if self.endpoint.trim().is_empty() {
            return Err(OtlpError::Config("endpoint must be non-empty".to_string()));
        }
        Ok(())
    }
}

/// Typed errors for the OTLP exporter / tracing-init layer.
#[derive(Debug, thiserror::Error)]
pub enum OtlpError {
    #[error("otlp config invalid: {0}")]
    Config(String),
    #[error("otlp export failed: {0}")]
    Export(String),
    #[error("otlp collector unreachable: {0}")]
    CollectorUnreachable(String),
    #[error("otlp builder failed: {0}")]
    Builder(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otlp_config_default_matches_spec() {
        let cfg = OtlpConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.endpoint, "http://localhost:4317");
        assert_eq!(cfg.service_name, "vllm-lite");
        assert_eq!(cfg.metrics_export_interval_secs, 30);
        assert!((cfg.trace_sampling_ratio - 1.0).abs() < f64::EPSILON);
        assert!(matches!(cfg.protocol, OtlpProtocol::Grpc));
    }

    #[test]
    fn otlp_config_rejects_negative_sampling_ratio() {
        let cfg = OtlpConfig { trace_sampling_ratio: -0.1, ..OtlpConfig::default() };
        assert!(matches!(cfg.validate(), Err(OtlpError::Config(_))));
    }

    #[test]
    fn otlp_config_rejects_above_one_sampling_ratio() {
        let cfg = OtlpConfig { trace_sampling_ratio: 1.5, ..OtlpConfig::default() };
        assert!(matches!(cfg.validate(), Err(OtlpError::Config(_))));
    }

    #[test]
    fn otlp_config_rejects_zero_metrics_interval() {
        let cfg = OtlpConfig { metrics_export_interval_secs: 0, ..OtlpConfig::default() };
        assert!(matches!(cfg.validate(), Err(OtlpError::Config(_))));
    }

    #[test]
    fn otlp_config_validates_default_as_ok() {
        assert!(OtlpConfig::default().validate().is_ok());
    }
}

