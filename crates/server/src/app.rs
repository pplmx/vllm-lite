//! Production router assembly.
//!
//! Extracted from `main.rs` so integration tests can exercise the
//! *real* production middleware stack (including its ordering) rather
//! than a simplified replica.
//!
//! # Middleware order (outermost → innermost)
//!
//! ```text
//! cors_layer                    ← browser-direct callers: headers on every response
//! correlation_id_middleware     ← sets X-Request-ID on every request AND response
//! audit_middleware              ← records 401/413/success rows (sees rejections)
//! body_size_limit               ← 413 before auth reads a multi-MiB body
//! auth_middleware               ← 401/429 + rate-limit headers (optional)
//! handlers                      ← business logic
//! ```
//!
//! Ordering invariants (see `security::audit_middleware` module docs):
//! - **correlation is outermost of the security stack** so even
//!   rejected requests (401/413/429) carry `X-Request-ID` in the
//!   response.
//! - **body-size limit sits above auth**: `auth_middleware` buffers
//!   the whole body to estimate token cost; if it ran outside the
//!   limit, an oversized body would be read (and silently truncated
//!   by the middleware's 1 MiB `to_bytes` cap) *before* the limit
//!   could reject it with 413. Putting the limit above auth keeps
//!   the documented 413 contract intact even when API keys are
//!   enabled.
//! - **audit sits above both** so 413s and 401s are recorded.
//!
//! `axum::Router::layer` wraps the router with the *most recently
//! applied* layer outermost, so the layers below are applied in the
//! order listed (auth first → cors last).
#![allow(clippy::module_name_repetitions)]

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};

use crate::ApiState;
use crate::api;
use crate::auth::AuthMiddleware;
use crate::config::AuthConfig;
use crate::debug;
use crate::health_handlers;
use crate::openai::batch::handler::{
    cancel_batch, create_batch, get_batch, get_batch_results, list_batches,
};
use crate::openai::chat::chat_completions;
use crate::openai::completions::completions as openai_completions;
use crate::openai::embeddings::embeddings;
use crate::openai::models::models_handler;
use crate::security::audit::AuditLogger;
use crate::security::audit_middleware::audit_middleware;
use crate::security::correlation::correlation_id_middleware;
use crate::security::cors::{CorsConfig, with_cors};
use crate::security::size_limit::with_default_body_limit;
use std::collections::HashMap;

/// Build the auth middleware from the *effective* API keys — all three
/// configured sources (inline `api_keys`, the `api_keys_env` var, and the
/// `api_keys_file` file), per [`AuthConfig::resolve_api_keys`]. Returns
/// `None` only when no key is configured anywhere.
///
/// (RIL ISS-080: the production binary previously gated the middleware on
/// the inline `api_keys` list alone, so an operator using
/// `--api-key-file` / `VLLM_API_KEYS_FILE` / `api_keys_env` got a
/// non-empty SEC-01 posture — no startup warning — while the middleware
/// stayed `None` and the inference API ran **completely unauthenticated**.
/// Enforcement must use exactly what the auth posture computes.)
pub fn build_auth_middleware(auth: &AuthConfig) -> Option<Arc<AuthMiddleware>> {
    let keys = auth.resolve_api_keys();
    if keys.is_empty() {
        return None;
    }
    // Convert RateLimitOverride into (max_requests, window_secs) pairs.
    let overrides: HashMap<String, (usize, u64)> = auth
        .rate_limit_overrides
        .iter()
        .map(|(k, v)| (k.clone(), (v.max_requests, v.rate_limit_window_secs)))
        .collect();
    Some(Arc::new(AuthMiddleware::new_with_overrides(
        keys,
        auth.rate_limit_requests,
        auth.rate_limit_window_secs,
        overrides,
    )))
}

/// Build the production HTTP router.
///
/// `state` is the shared [`ApiState`] the handlers receive;
/// `auth_middleware` is `Some(...)` only when API keys are
/// configured; `audit_logger` records one row per request;
/// `cors` is the resolved runtime CORS config.
///
/// The returned router is `Router` (state erased) and can be served
/// directly or mounted under additional layers by the caller.
pub fn build_app(
    state: ApiState,
    auth_middleware: Option<Arc<AuthMiddleware>>,
    audit_logger: Arc<AuditLogger>,
    cors: &CorsConfig,
) -> Router {
    let mut app = Router::new()
        // OpenAI API
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(openai_completions))
        .route("/v1/embeddings", post(embeddings))
        // Batch API
        .route("/v1/batches", post(create_batch))
        .route("/v1/batches", get(list_batches))
        .route("/v1/batches/{id}", get(get_batch))
        .route("/v1/batches/{id}/cancel", post(cancel_batch))
        .route("/v1/batches/{id}/results", get(get_batch_results))
        // Health, readiness, and metrics endpoints (K8s-compatible paths)
        .route("/health/live", get(health_handlers::health_handler))
        .route("/health/ready", get(health_handlers::ready_handler))
        .route("/health", get(health_handlers::health_handler))
        .route("/ready", get(health_handlers::ready_handler))
        .route("/metrics", get(health_handlers::metrics_handler))
        .route("/health/details", get(api::health_details))
        // Debug endpoints
        .route("/debug/metrics", get(debug::metrics_snapshot))
        .route("/debug/kv-cache", get(debug::kv_cache_dump))
        .route("/debug/trace", get(debug::trace_status))
        .route("/debug/audit", get(debug::audit_dump))
        // Shutdown
        .route("/shutdown", get(api::shutdown))
        .with_state(state);

    // auth (innermost of the security stack): reads the body to
    // estimate token cost, then either rejects (401/429) or runs the
    // inner stack. Only mounted when API keys are configured.
    if let Some(auth) = auth_middleware {
        app = app.layer(axum::middleware::from_fn_with_state(
            auth,
            crate::auth::auth_middleware,
        ));
    }

    // body-size limit: reject oversized bodies with 413 *before*
    // auth buffers them (auth's `to_bytes` caps at 1 MiB).
    app = with_default_body_limit(app);

    // audit: sees 413s (from the limit) and 401/429s (from auth)
    // plus every successful request.
    app = app.layer(axum::middleware::from_fn_with_state(
        audit_logger,
        audit_middleware,
    ));

    // correlation: outermost of the security stack so even rejected
    // requests carry X-Request-ID in the response and audit rows.
    app = app.layer(axum::middleware::from_fn(correlation_id_middleware));

    // CORS: outermost overall so even 413/401 responses carry the
    // Access-Control-Allow-Origin header for browser-direct callers.
    with_cors(app, cors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AuthConfig;

    #[test]
    fn build_auth_middleware_none_when_no_keys_anywhere() {
        let auth = AuthConfig::default();
        assert!(build_auth_middleware(&auth).is_none());
    }

    #[test]
    fn build_auth_middleware_uses_inline_keys() {
        let auth = AuthConfig {
            api_keys: vec!["sk-inline".to_string()],
            ..AuthConfig::default()
        };
        let mw = build_auth_middleware(&auth).expect("inline key must yield a middleware");
        assert_eq!(mw.api_keys(), &["sk-inline".to_string()]);
    }

    #[test]
    fn build_auth_middleware_enforces_file_keys() {
        // RIL ISS-080 regression: a config carrying ONLY file keys must
        // still produce a middleware. Pre-fix the binary gated on the
        // inline `api_keys` list alone, so `--api-key-file` /
        // `VLLM_API_KEYS_FILE` deployments saw no SEC-01 warning (the
        // resolved posture looked configured) yet the middleware stayed
        // `None` -- the inference API ran completely unauthenticated.
        // Enforcement must use the same all-source key set the posture
        // computes.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("keys.txt");
        std::fs::write(&file, "# comment\nsk-file-key-1\nsk-file-key-2\n").unwrap();
        let auth = AuthConfig {
            api_keys: vec![],
            api_keys_env: None,
            api_keys_file: Some(file.to_string_lossy().into_owned()),
            ..AuthConfig::default()
        };
        let mw = build_auth_middleware(&auth).expect("file keys must yield a middleware");
        assert_eq!(
            mw.api_keys(),
            &["sk-file-key-1".to_string(), "sk-file-key-2".to_string()]
        );
    }
}
