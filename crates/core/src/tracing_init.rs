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
    config: &OtlpConfig,
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

    /// Wrap a caller-built provider (test-only).
    ///
    /// The production path builds its own `SdkTracerProvider` around an OTLP
    /// exporter, which cannot be observed from a test. This hook lets a test
    /// supply a provider whose span processor counts `shutdown()` calls, which
    /// is the whole point of the guard.
    #[cfg(test)]
    pub(crate) const fn with_provider_for_test(provider: SdkTracerProvider) -> Self {
        Self::enabled(provider)
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
    use opentelemetry_sdk::error::OTelSdkResult;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn otlp_guard_disabled_reports_not_enabled() {
        let guard = OtlpGuard::disabled();
        assert!(!guard.is_enabled());
        // Drop is a no-op when disabled.
        drop(guard);
    }

    /// The guard's `Debug` must name the struct and show the enabled flag, and
    /// dropping an ENABLED guard must shut the provider down.
    ///
    /// `cargo mutants` replaced the `Debug` body with `Ok(())` and the whole
    /// `Drop` impl with `()`. Dropping without `shutdown()` loses every span
    /// still buffered in the batch processor — the graceful-shutdown contract
    /// the guard exists for. The provider is built around a counting span
    /// processor so the shutdown is an observable fact, not a doc comment.
    #[test]
    fn otlp_guard_debug_and_drop_shuts_the_provider_down() {
        let counter = Arc::new(AtomicUsize::new(0));
        let provider = SdkTracerProvider::builder()
            .with_span_processor(CountingProcessor {
                shutdowns: Arc::clone(&counter),
            })
            .build();
        let guard = OtlpGuard::with_provider_for_test(provider);

        let rendered = format!("{guard:?}");
        assert!(
            rendered.contains("OtlpGuard"),
            "Debug must name the type; got {rendered}"
        );
        assert!(
            rendered.contains("is_enabled: true"),
            "Debug must report the enabled flag; got {rendered}"
        );
        assert!(guard.is_enabled());
        assert_eq!(counter.load(Ordering::Relaxed), 0, "nothing shut down yet");

        drop(guard);
        assert_eq!(
            counter.load(Ordering::Relaxed),
            1,
            "dropping an enabled guard must shut the provider down exactly once"
        );
    }

    /// A DISABLED config must install the plain subscriber and return a guard
    /// that is not enabled.
    ///
    /// `cargo mutants` deleted the `!` in `if !config.enabled`, which sends a
    /// disabled config down the OTLP path: it builds a real exporter and hands
    /// back a guard whose drop then shuts down a provider the operator never
    /// asked for. This is the only place in the process that installs a global
    /// subscriber, so the test runs exactly once (it is the sole `try_init`
    /// caller in this crate's test binary).
    #[test]
    fn disabled_config_installs_no_otel_layer() {
        let cfg = OtlpConfig {
            enabled: false,
            ..OtlpConfig::default()
        };
        let guard = init_tracing_with_otlp(EnvFilter::new("error"), &cfg)
            .expect("a disabled config never fails to install");
        assert!(
            !guard.is_enabled(),
            "a disabled config must not return an enabled guard"
        );
        assert!(
            format!("{guard:?}").contains("is_enabled: false"),
            "Debug must report the disabled state"
        );
        drop(guard);
    }

    #[derive(Debug)]
    struct CountingProcessor {
        shutdowns: Arc<AtomicUsize>,
    }

    impl opentelemetry_sdk::trace::SpanProcessor for CountingProcessor {
        fn on_start(
            &self,
            _span: &mut opentelemetry_sdk::trace::Span,
            _cx: &opentelemetry::Context,
        ) {
        }
        fn on_end(&self, _span: opentelemetry_sdk::trace::SpanData) {}
        fn force_flush(&self) -> OTelSdkResult {
            Ok(())
        }
        fn shutdown_with_timeout(&self, _timeout: std::time::Duration) -> OTelSdkResult {
            self.shutdowns.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    /// The provider's `shutdown()` reaches the span processor only through the
    /// `OtlpGuard::drop` path, so a `Drop` impl replaced by `()` must show up
    /// as a zero count. This documents the mechanism the previous test relies
    /// on, so a future provider change cannot make that test vacuous.
    #[test]
    fn provider_shutdown_is_observable_through_the_span_processor() {
        let counter = Arc::new(AtomicUsize::new(0));
        let provider = SdkTracerProvider::builder()
            .with_span_processor(CountingProcessor {
                shutdowns: Arc::clone(&counter),
            })
            .build();
        provider.shutdown().expect("shutdown succeeds");
        assert_eq!(
            counter.load(Ordering::Relaxed),
            1,
            "SdkTracerProvider::shutdown must call the span processor exactly once"
        );
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
