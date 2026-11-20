// crates/server/src/health_handlers.rs
//
// HTTP handlers for `/health/live`, `/health/ready`, and `/metrics`.
// These are wired in `main.rs` for both the canonical K8s probe paths
// (`/health/live`, `/health/ready`) and the legacy aliases (`/health`,
// `/ready`).
//
// Lives in `lib.rs` so integration tests in `crates/server/tests/`
// can mount the routes against a real `axum::Router` with a
// controlled `ApiState`. Production wiring in `main.rs` references
// the same handlers via `crate::health_handlers::*`.

use axum::{extract::State, http::StatusCode, response::Response};
use serde_json::json;
use vllm_core::metrics::PrometheusExporter;

use crate::ApiState;

/// Mailbox fill ratio above which `/health/ready` reports
/// `NotReady`. Below this ratio the engine is still able to absorb
/// bursts; above it, the orchestrator should treat the pod as
/// saturated and drain traffic. 90 % matches the convention used
/// by Kubernetes' HPA `target.averageUtilization` defaults.
const READY_MAILBOX_FILL_RATIO: f64 = 0.90;

/// Health check endpoint - liveness probe
///
/// A poisoned health lock reports `Ok`: poison means some other thread
/// panicked, which is not evidence that this process stopped serving. Failing
/// the liveness probe would make the orchestrator restart a pod that is still
/// answering traffic. The writers (`main.rs` shutdown hook, `/shutdown`) also
/// degrade with a warning on poison, so the readers are not stricter.
///
/// Must be `async` for axum 0.8 `Handler` trait — the function
/// body contains no `.await` points because all work is synchronous.
#[allow(clippy::unused_async)]
pub async fn health_handler(State(state): State<ApiState>) -> Response {
    let status = match state.health.read() {
        Ok(checker) => checker.check_liveness(),
        Err(poisoned) => {
            tracing::warn!("health lock poisoned; reporting liveness as Ok");
            poisoned.into_inner().check_liveness()
        }
    };
    let http_status = StatusCode::from_u16(status.http_status()).unwrap_or(StatusCode::OK);

    let body = json!({ "status": status.as_str() });
    Response::builder()
        .status(http_status)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&body).unwrap_or_default().into())
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}

/// Readiness check endpoint
///
/// Production-readiness recommendation 7: a readiness probe must
/// reflect the engine's actual ability to accept work, not a
/// static boolean. We OR the static `HealthChecker` flag with
/// the live mailbox-fill ratio: when the engine's bounded mpsc
/// queue (REL-01) crosses `READY_MAILBOX_FILL_RATIO` (90 %),
/// readiness flips to `NotReady` so Kubernetes stops routing new
/// traffic to the pod and the existing connections can drain.
///
/// tokio's `Sender` doesn't expose `len()`; the queue size is
/// derived as `max_capacity() - capacity()` (max capacity is the
/// buffer size at channel construction, capacity is the current
/// available slots).
///
/// The body is always JSON so operators can scrape readiness
/// details (`status`, `mailbox_len`, `mailbox_capacity`,
/// `mailbox_fill_ratio`) without parsing the HTTP status code.
///
/// A poisoned health lock reports `NotReady`: poison means the state it held
/// can no longer be trusted, so the pod should be drained rather than
/// declared healthy (contrast [`health_handler`], which stays `Ok` because
/// liveness asks a weaker question).
///
/// Must be `async` for axum 0.8 `Handler` trait — the function
/// body contains no `.await` points (all work is synchronous).
#[allow(clippy::unused_async)]
pub async fn ready_handler(State(state): State<ApiState>) -> Response {
    let static_status = match state.health.read() {
        Ok(checker) => checker.check_readiness(),
        Err(poisoned) => {
            tracing::warn!("health lock poisoned; reporting readiness as NotReady");
            poisoned.into_inner().check_readiness()
        }
    };

    let max_capacity = state.engine_tx.max_capacity();
    let available_capacity = state.engine_tx.capacity();
    // saturating_sub: `max_capacity >= available_capacity` always holds
    // (capacity cannot exceed max), so this never underflows. The
    // `saturating_sub` is defensive against a future tokio API change.
    let len = max_capacity.saturating_sub(available_capacity);
    #[allow(clippy::cast_precision_loss)]
    let fill_ratio = if max_capacity == 0 {
        0.0
    } else {
        (len as f64 / max_capacity as f64).min(1.0)
    };
    let saturated = fill_ratio >= READY_MAILBOX_FILL_RATIO;

    let status = match static_status {
        crate::health::HealthStatus::Ok if saturated => crate::health::HealthStatus::NotReady,
        other => other,
    };

    let http_status =
        StatusCode::from_u16(status.http_status()).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);

    let body = json!({
        "status": status.as_str(),
        "mailbox_len": len,
        "mailbox_capacity": max_capacity,
        "mailbox_fill_ratio": fill_ratio,
        "mailbox_threshold": READY_MAILBOX_FILL_RATIO,
        "saturated": saturated,
    });
    Response::builder()
        .status(http_status)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&body).unwrap_or_default().into())
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}

/// Prometheus metrics endpoint
pub async fn metrics_handler(State(state): State<ApiState>) -> Response {
    let exporter = PrometheusExporter::new(state.metrics.clone(), 9090);
    let output = exporter.export_to_string().await;

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/plain; charset=utf-8")
        .body(output.into())
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures;
    use vllm_model::config::Architecture;

    /// Poison the health lock by panicking while holding its write guard.
    fn poison_health(state: &ApiState) {
        let health = state.health.clone();
        let _ = std::thread::spawn(move || {
            let _guard = health.write().unwrap();
            panic!("intentional panic while holding the health write lock");
        })
        .join();
    }

    /// A poisoned health lock must not take the pod down.
    ///
    /// The liveness and readiness probes are what the orchestrator polls;
    /// panicking in them makes the pod fail its probe and get restarted for
    /// a fault that is entirely unrelated to serving traffic. The writers
    /// (`main.rs` shutdown hook, `/shutdown`) already degrade with a warning
    /// on poison, so the readers must not be stricter than the writers.
    #[tokio::test]
    async fn probes_survive_a_poisoned_health_lock() {
        let state = test_fixtures::api_state(Architecture::Llama);
        poison_health(&state);
        assert!(
            state.health.read().is_err(),
            "fixture must actually poison the lock"
        );

        let live = health_handler(State(state.clone())).await;
        assert_eq!(live.status(), StatusCode::OK);

        let ready = ready_handler(State(state)).await;
        assert!(
            ready.status() == StatusCode::OK || ready.status() == StatusCode::SERVICE_UNAVAILABLE,
            "readiness must answer with a status, not panic: got {}",
            ready.status()
        );
    }
}
