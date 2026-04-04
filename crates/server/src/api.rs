#![allow(dead_code)]

use axum::{Json, extract::State};
use serde::Serialize;
use tokio::sync::mpsc;
use vllm_core::metrics::MetricsSnapshot;
use vllm_core::types::EngineMessage;

use crate::ApiState;

pub type EngineHandle = mpsc::UnboundedSender<EngineMessage>;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
}

#[derive(Serialize)]
pub struct HealthDetailResponse {
    pub status: String,
    pub gpu_available: bool,
    pub gpu_utilization: Option<f32>,
    pub kv_cache_usage_percent: Option<f32>,
}

pub(crate) async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

pub(crate) async fn health_details(State(state): State<ApiState>) -> Json<HealthDetailResponse> {
    let (response_tx, mut response_rx) = mpsc::unbounded_channel();
    let _ = state.engine_tx.send(EngineMessage::GetMetrics { response_tx });

    let metrics = response_rx.recv().await.unwrap_or_default();

    Json(HealthDetailResponse {
        status: "ok".to_string(),
        gpu_available: true,
        gpu_utilization: Some(metrics.prefill_throughput as f32),
        kv_cache_usage_percent: Some(metrics.kv_cache_usage_percent as f32),
    })
}

pub async fn shutdown(State(engine_tx): State<EngineHandle>) -> &'static str {
    let _ = engine_tx.send(EngineMessage::Shutdown);
    "Shutting down"
}

pub async fn get_prometheus(State(state): State<ApiState>) -> String {
    let (response_tx, mut response_rx) = mpsc::unbounded_channel();
    state
        .engine_tx
        .send(EngineMessage::GetMetrics { response_tx })
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

    async fn send_request(app: Router, request: Request<Body>) -> Response {
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
