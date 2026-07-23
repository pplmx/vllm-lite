//! OTLP exporter bootstrap: wires the engine's `EnhancedMetricsCollector`
//! to a background `OtlpExporter` task. Returns an `OtlpHandle` that
//! flushes pending spans + metrics on drop (graceful shutdown).
// This module is gated by `#[cfg(feature = "opentelemetry")] pub mod observability;`
// in `bootstrap/mod.rs`, so the `#![cfg(...)]` attribute here would be
// a duplicated attribute. The module simply doesn't compile without the feature.

use std::sync::Arc;

use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::metrics::exporter::{OtlpConfig, OtlpError, OtlpExporter};

/// Handle returned by [`spawn_otlp_exporter`]. Holds the running background
/// task's `JoinHandle`. Drop = abort the task + flush pending metrics via
/// `OtlpExporter::shutdown`.
pub struct OtlpHandle {
    /// The `OtlpExporter` that owns the `SdkMeterProvider`. We only call
    /// `shutdown()` on it (sync flush) — `run()` has already consumed its
    /// `Arc` clone internally, so the provider is still alive via the
    /// background task's reference until we abort + flush.
    exporter: Option<OtlpExporter>,
    /// Background task running `OtlpExporter::run`. Aborted on drop.
    task: Option<tokio::task::JoinHandle<Result<(), OtlpError>>>,
}

impl Drop for OtlpHandle {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
            // We don't await in Drop (fn signature doesn't allow it). The
            // abort stops the polling loop; the exporter flush happens via
            // the tracer-provider `OtlpGuard` returned by
            // `init_tracing_with_otlp` in the server bootstrap.
        }
        // Belt-and-suspenders: if the exporter is still held (e.g. `run()`
        // was never called), flush it directly.
        if let Some(exporter) = self.exporter.take() {
            let _ = exporter.shutdown();
        }
    }
}

impl std::fmt::Debug for OtlpHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtlpHandle")
            .field("has_task", &self.task.is_some())
            .field("has_exporter", &self.exporter.is_some())
            .finish()
    }
}

impl OtlpHandle {
    /// Returns `true` if the background task is still running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.task.as_ref().is_some_and(|t| !t.is_finished())
    }
}

/// Spawn the OTLP metrics background task.
///
/// When `config.enabled` is `false`, returns `Ok(None)` so callers can use
/// `if let Some(handle) = ...` without branching on the flag themselves.
///
/// The `collector` is the engine's `EnhancedMetricsCollector` (shared via
/// `Arc` so the engine and the exporter see the same counters). The
/// exporter's background task polls every
/// `config.metrics_export_interval_secs` and ships via OTLP `grpc-tonic`.
///
/// # Errors
///
/// Returns [`OtlpError`] if the config is invalid or the exporter cannot be
/// constructed (e.g. the OTLP endpoint cannot be resolved at build time).
pub fn spawn_otlp_exporter(
    collector: Arc<EnhancedMetricsCollector>,
    config: OtlpConfig,
) -> Result<Option<OtlpHandle>, OtlpError> {
    if !config.enabled {
        return Ok(None);
    }

    let exporter = OtlpExporter::new(collector, config)?;

    // `OtlpExporter::run` borrows `self` by reference (it holds `&self`),
    // so we can clone the exporter — one for the task to drive, one for
    // the handle to flush on shutdown.
    let run_clone = exporter.clone();
    let task = tokio::spawn(async move { run_clone.run().await });

    Ok(Some(OtlpHandle {
        exporter: Some(exporter),
        task: Some(task),
    }))
}
