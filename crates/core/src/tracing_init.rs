//! Tracing-subscriber bootstrap with optional OpenTelemetry bridge.
//! Gated by the `opentelemetry` feature on `vllm-core`.

use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider};
use opentelemetry_semantic_conventions::attribute as semattr;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::metrics::exporter::otlp::{OtlpConfig, OtlpError};

/// Initialise `tracing-subscriber` with the requested `EnvFilter`.
///
/// When `config.enabled = true`, adds a `tracing-opentelemetry` layer
/// that bridges every `tracing::info_span!` to an `OTel` span. The returned
/// `OtlpGuard` flushes pending spans on drop (graceful shutdown).
///
/// # Errors
///
/// Returns [`OtlpError::Config`] if the configuration is invalid, or
/// [`OtlpError::Builder`] if the subscriber or span exporter cannot be
/// initialised (e.g., the tracing subscriber has already been initialised).
pub fn init_tracing_with_otlp(
    env_filter: EnvFilter,
    config: OtlpConfig,
) -> Result<OtlpGuard, OtlpError> {
    config.validate()?;

    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true);

    if !config.enabled {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .try_init()
            .map_err(|e| OtlpError::Builder(format!("subscriber init: {e}")))?;
        return Ok(OtlpGuard::disabled());
    }

    let span_exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&config.endpoint)
        .build()
        .map_err(|e| OtlpError::Builder(format!("span exporter: {e}")))?;

    let resource = Resource::builder()
        .with_attributes([
            KeyValue::new(semattr::SERVICE_NAME, config.service_name.clone()),
            KeyValue::new(semattr::SERVICE_VERSION, config.service_version.clone()),
        ])
        .build();

    let sampler = Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
        config.trace_sampling_ratio,
    )));

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(span_exporter)
        .with_resource(resource)
        .with_sampler(sampler)
        .build();

    let tracer = provider.tracer("vllm-lite");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(otel_layer)
        .try_init()
        .map_err(|e| OtlpError::Builder(format!("subscriber init: {e}")))?;

    Ok(OtlpGuard::enabled(provider))
}

/// Drop-flush guard for the OTLP tracing bridge. Dropping the guard
/// calls the tracer provider's `shutdown()`, flushing pending spans.
pub struct OtlpGuard {
    provider: Option<SdkTracerProvider>,
}

impl OtlpGuard {
    const fn disabled() -> Self {
        Self { provider: None }
    }
    const fn enabled(provider: SdkTracerProvider) -> Self {
        Self {
            provider: Some(provider),
        }
    }

    /// Returns `true` if OTLP tracing was initialised (drop will flush).
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.provider.is_some()
    }
}

impl Drop for OtlpGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.provider.take() {
            // `shutdown()` returns `Vec<SpanData>` on success; we discard.
            let _ = provider.shutdown();
        }
    }
}

impl std::fmt::Debug for OtlpGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtlpGuard")
            .field("is_enabled", &self.is_enabled())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otlp_guard_disabled_reports_not_enabled() {
        let guard = OtlpGuard::disabled();
        assert!(!guard.is_enabled());
        // Drop is a no-op when disabled.
        drop(guard);
    }

    #[test]
    fn otlp_config_validate_is_idempotent() {
        // Validates the same OtlpConfig repeatedly yields the same result.
        let cfg = OtlpConfig::default();
        assert!(cfg.validate().is_ok());
        assert!(cfg.validate().is_ok());
        assert!(!cfg.enabled);
    }

    #[test]
    fn otlp_config_enabled_with_zero_sampling_ratio_is_allowed() {
        // Sampling 0.0 is in-range (always off). Used for "disable tracing
        // but keep metrics" deployments.
        let cfg = OtlpConfig {
            trace_sampling_ratio: 0.0,
            enabled: true,
            ..OtlpConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }
}
