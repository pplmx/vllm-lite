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

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
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
