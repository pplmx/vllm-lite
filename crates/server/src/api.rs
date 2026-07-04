//! Top-level axum router and the `/health` / `/v1/models` / `/metrics` handlers that aren't part of the OpenAI surface.
//!
//! `build_router` is the entry point called from `main.rs`; it wires
//! every middleware (auth, RBAC, backpressure, correlation, audit) and
//! the OpenAI sub-router under `/v1`.
use axum::{Json, extract::State};
use serde::Serialize;
use tokio::sync::mpsc;
use vllm_core::metrics::MetricsSnapshot;
use vllm_core::types::EngineMessage;

use crate::ApiState;

/// Sender side of the engine mailbox. Each handler holds a clone of this
/// `UnboundedSender`; sending an [`EngineMessage`] enqueues work for the
/// engine's run loop.
///
/// Backpressure is intentionally not applied here — the engine drains every
/// message each loop iteration.
pub type EngineHandle = mpsc::UnboundedSender<EngineMessage>;

/// Response payload for Health. Returned from handlers, serialized to JSON for the HTTP boundary.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
}

/// Response payload for HealthDetail. Returned from handlers, serialized to JSON for the HTTP boundary.
#[derive(Debug, Serialize)]
pub struct HealthDetailResponse {
    pub status: String,
    pub gpu_available: bool,
    pub gpu_utilization: Option<f32>,
    pub kv_cache_usage_percent: Option<f32>,
}

/// `/health/details` handler. Returns a richer status payload including
/// live metrics from the engine (prefill throughput, KV-cache utilization).
///
/// Used by ops dashboards that need more than the liveness/readiness probe.
// invariant: throughput/percent values are bounded metrics; f64 -> f32
// precision loss / truncation is acceptable for the public health snapshot.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub async fn health_details(State(state): State<ApiState>) -> Json<HealthDetailResponse> {
    let (response_tx, mut response_rx) = mpsc::unbounded_channel();
    let _ = state
        .engine_tx
        .send(EngineMessage::GetMetrics { response_tx });

    let metrics = response_rx.recv().await.unwrap_or_default();

    let gpu_utilization = metrics.prefill_throughput as f32;
    let kv_cache_usage_percent = metrics.kv_cache_usage_percent as f32;
    Json(HealthDetailResponse {
        status: "ok".to_string(),
        gpu_available: true,
        gpu_utilization: Some(gpu_utilization),
        kv_cache_usage_percent: Some(kv_cache_usage_percent),
    })
}

/// `/shutdown` handler. Sends [`EngineMessage::Shutdown`] to the engine and
/// returns immediately; the HTTP server keeps serving until the process
/// exits.
///
/// Useful for graceful drain during orchestrator rolling updates.
#[allow(clippy::unused_async)]
pub async fn shutdown(State(state): State<ApiState>) -> &'static str {
    let _ = state.engine_tx.send(EngineMessage::Shutdown);
    "Shutting down"
}

/// `/metrics` Prometheus exposition handler. Returns a text/plain payload in
/// the Prometheus 0.0.4 format containing the engine's current
/// [`MetricsSnapshot`].
///
/// # Panics
///
/// Panics only if the engine channel is closed before this handler runs
/// (programmer error — the API state holds a sender cloned from the same
/// channel the engine reads from).
pub async fn get_prometheus(State(state): State<ApiState>) -> String {
    let (response_tx, mut response_rx) = mpsc::unbounded_channel();
    state
        .engine_tx
        // invariant: engine is shutdown only after all senders are dropped; the sender
        // outlives the receiver for the lifetime of this request handler.
        .send(EngineMessage::GetMetrics { response_tx })
        // invariant: pre-conditions make this infallible at this call site.
        .expect("Engine channel should be available");
    let m = response_rx.recv().await.unwrap_or(MetricsSnapshot {
        tokens_total: 0,
        requests_total: 0,
        avg_latency_ms: 0.0,
        p50_latency_ms: 0.0,
        p90_latency_ms: 0.0,
        p99_latency_ms: 0.0,
        avg_batch_size: 0.0,
        current_batch_size: 0,
        requests_in_flight: 0,
        kv_cache_usage_percent: 0.0,
        prefix_cache_hit_rate: 0.0,
        prefill_throughput: 0.0,
        decode_throughput: 0.0,
        avg_scheduler_wait_time_ms: 0.0,
    });
    format!(
        "vllm_tokens_total {}\n\
         vllm_requests_total {}\n\
         vllm_avg_latency_ms {:.2}\n\
         vllm_p50_latency_ms {:.2}\n\
         vllm_p90_latency_ms {:.2}\n\
         vllm_p99_latency_ms {:.2}\n\
         vllm_avg_batch_size {:.2}\n\
         vllm_current_batch_size {}\n\
         vllm_requests_in_flight {}\n\
         vllm_kv_cache_usage_percent {:.2}\n\
         vllm_prefix_cache_hit_rate {:.2}\n\
         vllm_prefill_throughput {:.2}\n\
         vllm_decode_throughput {:.2}\n\
         vllm_avg_scheduler_wait_time_ms {:.2}\n",
        m.tokens_total,
        m.requests_total,
        m.avg_latency_ms,
        m.p50_latency_ms,
        m.p90_latency_ms,
        m.p99_latency_ms,
        m.avg_batch_size,
        m.current_batch_size,
        m.requests_in_flight,
        m.kv_cache_usage_percent,
        m.prefix_cache_hit_rate,
        m.prefill_throughput,
        m.decode_throughput,
        m.avg_scheduler_wait_time_ms
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode},
        response::Response,
        routing::get,
    };
    use tower::ServiceExt;

    async fn health() -> Json<HealthResponse> {
        Json(HealthResponse {
            status: "ok".to_string(),
        })
    }

    async fn send_request(app: Router, request: Request<Body>) -> Response {
        // invariant: pre-conditions make this infallible at this call site.
        app.oneshot(request).await.expect("Failed to send request")
    }

    #[tokio::test]
    async fn test_health_ok() {
        let app = Router::new().route("/health", get(health));

        let response = send_request(
            app,
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_json_response() {
        let app = Router::new().route("/health", get(health));

        let response = send_request(
            app,
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("\"status\""));
        assert!(body_str.contains("ok"));
    }
}
