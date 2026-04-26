//! Debug Utilities
//!
//! Provides debug endpoints for inspecting internal state:
//! - Request tracing via tracing spans
//! - KV cache dump
//! - Metrics snapshot

use crate::ApiState;
use axum::{Json, extract::State};
use serde::Serialize;
use std::collections::HashMap;
use vllm_core::types::EngineMessage;

#[derive(Serialize)]
pub struct MetricsSnapshotResponse {
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
    pub queue_depth: u64,
    pub active_sequences: u64,
    pub cuda_graph_hit_rate: f64,
}

pub async fn metrics_snapshot(State(state): State<ApiState>) -> Json<MetricsSnapshotResponse> {
    let metrics = state.metrics;
    let counters: HashMap<String, u64> = [
        (
            "cuda_graph_hits_total".to_string(),
            metrics.get_counter("cuda_graph_hits_total"),
        ),
        (
            "cuda_graph_misses_total".to_string(),
            metrics.get_counter("cuda_graph_misses_total"),
        ),
        (
            "packing_sequences_total".to_string(),
            metrics.get_counter("packing_sequences_total"),
        ),
        (
            "speculative_adjustments_total".to_string(),
            metrics.get_counter("speculative_adjustments_total"),
        ),
        (
            "requests_total".to_string(),
            metrics.get_counter("requests_total"),
        ),
        (
            "errors_total".to_string(),
            metrics.get_counter("errors_total"),
        ),
    ]
    .into_iter()
    .collect();

    let gauges: HashMap<String, f64> = [
        (
            "packing_efficiency".to_string(),
            metrics.get_gauge("packing_efficiency") as f64 / 1000.0,
        ),
        (
            "speculative_acceptance_rate".to_string(),
            metrics.get_gauge("speculative_acceptance_rate") as f64 / 1000.0,
        ),
    ]
    .into_iter()
    .collect();

    let queue_depth = metrics.get_gauge("request_queue_depth");
    let active_sequences = metrics.get_gauge("active_sequences");

    let hit_total = metrics.get_counter("cuda_graph_hits_total");
    let miss_total = metrics.get_counter("cuda_graph_misses_total");
    let cuda_graph_hit_rate = if hit_total + miss_total > 0 {
        hit_total as f64 / (hit_total + miss_total) as f64
    } else {
        0.0
    };

    Json(MetricsSnapshotResponse {
        counters,
        gauges,
        queue_depth,
        active_sequences,
        cuda_graph_hit_rate,
    })
}

#[derive(Serialize)]
pub struct KvCacheDumpResponse {
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub available_blocks: usize,
    pub usage_percent: f64,
    pub prefix_cache_nodes: usize,
    pub prefix_cache_hit_rate: f64,
}

pub async fn kv_cache_dump(State(state): State<ApiState>) -> Json<KvCacheDumpResponse> {
    let (response_tx, mut response_rx) = tokio::sync::mpsc::unbounded_channel();

    let _ = state
        .engine_tx
        .send(EngineMessage::GetMetrics { response_tx });

    let metrics = response_rx.recv().await.unwrap_or_default();

    let available_blocks = metrics.current_batch_size as usize;
    let kv_cache_usage_percent = metrics.kv_cache_usage_percent;

    Json(KvCacheDumpResponse {
        total_blocks: 1024,
        used_blocks: (1024.0 * kv_cache_usage_percent / 100.0) as usize,
        available_blocks,
        usage_percent: kv_cache_usage_percent,
        prefix_cache_nodes: 0,
        prefix_cache_hit_rate: metrics.prefix_cache_hit_rate,
    })
}

#[derive(Serialize)]
pub struct TraceStatusResponse {
    pub tracing_enabled: bool,
    pub log_level: String,
    pub spans_active: usize,
}

pub async fn trace_status(State(_state): State<ApiState>) -> Json<TraceStatusResponse> {
    Json(TraceStatusResponse {
        tracing_enabled: true,
        log_level: std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        spans_active: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_snapshot_response_serialization() {
        let mut counters = HashMap::new();
        counters.insert("requests_total".to_string(), 100);
        let mut gauges = HashMap::new();
        gauges.insert("active_sequences".to_string(), 5.0);

        let response = MetricsSnapshotResponse {
            counters,
            gauges,
            queue_depth: 10,
            active_sequences: 5,
            cuda_graph_hit_rate: 0.95,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("requests_total"));
        assert!(json.contains("active_sequences"));
        assert!(json.contains("cuda_graph_hit_rate"));
    }

    #[test]
    fn test_kv_cache_dump_response_serialization() {
        let response = KvCacheDumpResponse {
            total_blocks: 1024,
            used_blocks: 512,
            available_blocks: 512,
            usage_percent: 50.0,
            prefix_cache_nodes: 100,
            prefix_cache_hit_rate: 0.75,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("total_blocks"));
        assert!(json.contains("usage_percent"));
        assert!(json.contains("prefix_cache_hit_rate"));
    }
}
