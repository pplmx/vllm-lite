//! Debug Utilities
//!
//! Provides debug endpoints for inspecting internal state:
//! - Request tracing via tracing spans
//! - KV cache dump
//! - Metrics snapshot
//!
//! SEC-01 (technical due diligence): every handler in this module
//! is gated by `require_admin` — when no API keys are configured
//! the endpoint refuses with `503 admin_disabled` (so it can't be
//! silently reachable on a non-loopback bind), and when keys are
//! configured the caller must present a valid `Bearer` token. This
//! is a deliberately crude check; the long-term fix is to bind
//! the RBAC role to a verified JWT claim rather than the
//! user-supplied `X-User-Role` header.

use crate::ApiState;
use crate::security::audit::AuditEvent;
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode, header::AUTHORIZATION},
    response::{IntoResponse, Response},
};
use serde::Serialize;
use std::collections::HashMap;
use vllm_core::types::EngineMessage;

/// SEC-01: enforce admin auth at the debug-handler boundary. Returns
/// `Ok(())` if the request is authorized to reach the handler body,
/// otherwise an `(StatusCode, JSON body)` tuple suitable for
/// `IntoResponse`.
///
/// Policy (intentionally simple — see module docs):
///
/// 1. **No API keys configured** → all callers get `503 admin_disabled`.
///    This is the fail-closed half of the SEC-01 fix: if the operator
///    has not set up auth, debug endpoints refuse rather than silently
///    exposing internal state. Reachable in non-loopback environments
///    via `--insecure-allow-public-no-auth` (CLI) or
///    `VLLM_INSECURE_ALLOW_PUBLIC_NO_AUTH=true` (env), but those
///    affect *only* the startup warning — admin gating here stays
///    strict because the cost of a debug leak is much higher than
///    the cost of an operator having to set up an API key for
///    legitimate debugging.
/// 2. **API keys configured, no/malformed `Authorization` header**
///    → `401 unauthorized`.
/// 3. **API keys configured, valid `Bearer` key** → `200 ok`.
///
/// We deliberately do NOT trust the `X-User-Role` header here even
/// though the existing RBAC middleware does. The debug surface
/// should not assume the request passed through RBAC, and the RBAC
/// role extraction is a known vulnerability (SEC-01 again). Treating
/// any valid key as admin is consistent with the current model
/// where every API key is equivalent; once keys are bound to roles
/// via JWT, this check should narrow to require the `admin` claim.
#[allow(clippy::result_large_err)] // Response is the natural Err shape for axum handlers
fn require_admin(state: &ApiState, headers: &HeaderMap) -> Result<(), Response> {
    let Some(auth) = state.auth.as_ref() else {
        let body = Json(serde_json::json!({
            "error": "admin_disabled",
            "message": "debug endpoints require API key auth to be configured; \
                        set --api-key or VLLM_API_KEY to enable admin access",
        }));
        return Err((StatusCode::SERVICE_UNAVAILABLE, body).into_response());
    };

    let api_key = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));
    match api_key {
        Some(key) if auth.api_keys().iter().any(|k| k == key) => Ok(()),
        _ => {
            let body = Json(serde_json::json!({
                "error": "unauthorized",
                "message": "valid Bearer token required for admin endpoint",
            }));
            Err((StatusCode::UNAUTHORIZED, body).into_response())
        }
    }
}

/// Response payload for `MetricsSnapshot`. Returned from handlers, serialized to JSON for the HTTP boundary.
#[derive(Debug, Serialize)]
pub struct MetricsSnapshotResponse {
    /// Monotonic counters (events since process start).
    pub counters: HashMap<String, u64>,
    /// Current gauge values (point-in-time samples).
    pub gauges: HashMap<String, f64>,
    /// Current scheduler request queue depth.
    pub queue_depth: u64,
    /// Number of sequences currently being processed.
    pub active_sequences: u64,
    /// CUDA-Graph replay hit rate (0.0–1.0).
    pub cuda_graph_hit_rate: f64,
}

/// Admin-only endpoint: return a JSON snapshot of key metrics counters and gauges.
#[allow(clippy::unused_async)]
pub async fn metrics_snapshot(
    State(state): State<ApiState>,
    headers: axum::http::HeaderMap,
) -> Response {
    if let Err(response) = require_admin(&state, &headers) {
        return response;
    }
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
        // invariant: gauge values are bounded counts/ratios; u64 -> f64 precision
        // loss is acceptable for snapshot display.
        (
            "packing_efficiency".to_string(),
            #[allow(clippy::cast_precision_loss)]
            {
                metrics.get_gauge("packing_efficiency") as f64 / 1000.0
            },
        ),
        (
            "speculative_acceptance_rate".to_string(),
            #[allow(clippy::cast_precision_loss)]
            {
                metrics.get_gauge("speculative_acceptance_rate") as f64 / 1000.0
            },
        ),
    ]
    .into_iter()
    .collect();

    let queue_depth = metrics.get_gauge("request_queue_depth");
    let active_sequences = metrics.get_gauge("active_sequences");

    let hit_total = metrics.get_counter("cuda_graph_hits_total");
    let miss_total = metrics.get_counter("cuda_graph_misses_total");
    // invariant: hit/miss totals are bounded counts; u64 -> f64 precision loss is
    // acceptable for a ratio display.
    let cuda_graph_hit_rate = if hit_total + miss_total > 0 {
        #[allow(clippy::cast_precision_loss)]
        {
            hit_total as f64 / (hit_total + miss_total) as f64
        }
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
    .into_response()
}

/// Response payload for `KvCacheDump`. Returned from handlers, serialized to JSON for the HTTP boundary.
#[derive(Debug, Serialize)]
pub struct KvCacheDumpResponse {
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub available_blocks: usize,
    pub usage_percent: f64,
    pub prefix_cache_nodes: usize,
    pub prefix_cache_hit_rate: f64,
}

/// Admin-only endpoint: dump KV-cache block allocation and metadata as JSON.
pub async fn kv_cache_dump(
    State(state): State<ApiState>,
    headers: axum::http::HeaderMap,
) -> Response {
    if let Err(response) = require_admin(&state, &headers) {
        return response;
    }
    let (response_tx, mut response_rx) = tokio::sync::mpsc::unbounded_channel();

    // REL-01: `try_send` so this debug endpoint never blocks. We
    // ignore the failure (Full or Closed) and fall back to the
    // default `MetricsSnapshot` below — debug endpoints should
    // never fail the caller because of an overloaded engine.
    let _ = state
        .engine_tx
        .try_send(EngineMessage::GetMetrics { response_tx });

    let metrics = response_rx.recv().await.unwrap_or_default();

    let available_blocks = metrics.current_batch_size;
    let kv_cache_usage_percent = metrics.kv_cache_usage_percent;

    // invariant: kv_cache_usage_percent is a 0..=100 ratio; 1024 * pct / 100 is
    // bounded by 1024, well within usize range.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let used_blocks = (1024.0 * kv_cache_usage_percent / 100.0) as usize;

    Json(KvCacheDumpResponse {
        total_blocks: 1024,
        used_blocks,
        available_blocks,
        usage_percent: kv_cache_usage_percent,
        prefix_cache_nodes: 0,
        prefix_cache_hit_rate: metrics.prefix_cache_hit_rate,
    })
    .into_response()
}

/// Response payload for `TraceStatus`. Returned from handlers, serialized to JSON for the HTTP boundary.
#[derive(Debug, Serialize)]
pub struct TraceStatusResponse {
    pub tracing_enabled: bool,
    pub log_level: String,
    pub spans_active: usize,
}

/// Response payload for `AuditDump`. Returned from handlers, serialized to JSON for the HTTP boundary.
#[derive(Debug, Serialize)]
pub struct AuditDumpResponse {
    /// Total number of audit events currently in the in-memory
    /// ring buffer (may exceed `events.len()` if events were
    /// evicted between this snapshot and the next caller).
    pub count: usize,
    /// The `events.len()` actually returned (≤ `count`). When the
    /// buffer has been pruned, `count > events.len()` and operators
    /// know the older rows are gone — they must reconstruct them
    /// from the structured `tracing` log stream.
    pub returned: usize,
    /// Hard cap we applied before serializing — protects the JSON
    /// response from runaway growth on a long-lived process.
    pub cap: usize,
    /// Most recent audit events (newest first). Empty for a server
    /// that hasn't processed any requests yet.
    pub events: Vec<AuditEvent>,
}

const AUDIT_DUMP_DEFAULT_CAP: usize = 1000;

/// Admin-only endpoint: return the most recent audit events (newest-first).
pub async fn audit_dump(State(state): State<ApiState>, headers: axum::http::HeaderMap) -> Response {
    if let Err(response) = require_admin(&state, &headers) {
        return response;
    }
    let cap = AUDIT_DUMP_DEFAULT_CAP;
    // Snapshot the events and reverse so callers see newest-first.
    // Audit rows are append-only; reversing is cheap for the
    // bounded buffer (10 000 entries by default) and matches the
    // mental model "what just happened?".
    let mut all_events = state.audit.get_events().await;
    let count = all_events.len();
    // Take only the trailing `cap` events to keep the response
    // bounded. We do this BEFORE the reverse so we get the most
    // recent N (the tail of the buffer).
    let start = count.saturating_sub(cap);
    all_events.drain(..start);
    let returned = all_events.len();
    all_events.reverse();

    Json(AuditDumpResponse {
        count,
        returned,
        cap,
        events: all_events,
    })
    .into_response()
}

/// Admin-only endpoint: report the current tracing/observability configuration.
#[allow(clippy::unused_async)]
pub async fn trace_status(
    State(state): State<ApiState>,
    headers: axum::http::HeaderMap,
) -> Response {
    if let Err(response) = require_admin(&state, &headers) {
        return response;
    }
    Json(TraceStatusResponse {
        tracing_enabled: true,
        log_level: std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        spans_active: 0,
    })
    .into_response()
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
