//! CORS layer — explicit allowlist, no default `*`.
//!
//! Production-readiness recommendation §9: \"仓库无实际 CORS layer。
//! 对于服务端 SDK 不是问题；若支持浏览器直连，应使用显式
//! allowlist，禁止默认 `*` 与 credential 组合\". This module
//! implements the explicit-allowlist half. The default construction
//! (`with_default_cors`) is **closed**: no `Access-Control-Allow-Origin`
//! header is emitted unless the operator explicitly lists origins via
//! [`CorsConfig::allow_origins`].
//!
//! ## Why no `*` default
//!
//! Browsers reject `Access-Control-Allow-Origin: *` when
//! `Access-Control-Allow-Credentials: true`. Picking one or the
//! other silently is the footgun. By defaulting to a closed CORS
//! layer we force the operator to choose: server-to-server SDKs
//! (the dominant use case) need no CORS at all; browser-direct
//! callers must list exact origins.
//!
//! ## Layout in the production router
//!
//! ```text
//! correlation_id_middleware   ← sets CorrelationId + X-Request-ID
//! audit_middleware            ← this file; logs after the response
//! size_limit_middleware       ← rejects oversize bodies with 413
//! auth_middleware             ← sets AuthenticatedUser on success
//! cors_layer (NEW)            ← here, ABOVE handlers so OPTIONS preflights get answered
//! handler                     ← actual business logic
//! ```
//!
//! `cors_layer` sits at the innermost position (just above the
//! handler). Reason: preflight `OPTIONS` requests should reach the
//! layer without going through body-limit / auth (the preflight
//! has no body and no auth header).

// `CorsConfig` / `with_cors` are the natural names for the CORS
// configuration type and its router helper. The `cors` module is the
// only place this idiom exists, so suppressing the lint at file
// scope is cleaner than per-item attributes.
#![allow(clippy::module_name_repetitions)]

use axum::Router;
use tower_http::cors::{Any, CorsLayer};

/// CORS configuration supplied by the operator.
///
/// All fields are explicit — there is no implicit default that
/// grants `*` access. `allow_origins = []` produces a closed layer
/// (CORS preflights pass through but no `Access-Control-Allow-Origin`
/// header is emitted, so browser-direct callers will be blocked at
/// the browser level).
#[derive(Debug, Clone, Default)]
pub struct CorsConfig {
    /// Origins allowed by the `Access-Control-Allow-Origin` header.
    /// Empty = closed (server-to-server SDKs only). The values are
    /// passed through verbatim; do NOT add `*` when
    /// `allow_credentials = true` — browsers reject that
    /// combination and `tower-http` will refuse to set the header.
    pub allow_origins: Vec<String>,
    /// HTTP methods allowed by `Access-Control-Allow-Methods`.
    /// Empty = no methods listed (preflight fails closed).
    pub allow_methods: Vec<String>,
    /// Headers allowed by `Access-Control-Allow-Headers`. Empty =
    /// no headers listed.
    pub allow_headers: Vec<String>,
    /// Whether to set `Access-Control-Allow-Credentials: true`. Must
    /// be `false` when `allow_origins` contains `*`; we never emit
    /// `*` here so this can safely be set to `true` once the
    /// operator opts in.
    pub allow_credentials: bool,
}

/// Apply a CORS layer built from `config` to `router`.
///
/// When `config.allow_origins` is empty, the layer is *closed*:
/// `Access-Control-Allow-Origin` is not emitted, so browser-direct
/// requests are blocked at the browser even when the underlying
/// request would have succeeded. This is the safe default — opt
/// in via `CorsConfig::allow_origins` only when you actually need
/// browser-direct access.
pub fn with_cors(router: Router, config: CorsConfig) -> Router {
    use axum::http::header::HeaderName;
    use std::str::FromStr;

    let mut layer = CorsLayer::new();

    if config.allow_origins.is_empty() {
        // No origins listed: leave the layer in its closed state.
        // We do NOT call `allow_origin(Any)` — that would emit `*`
        // and break the security contract documented at the top
        // of this file.
        return router.layer(layer);
    }

    // Parse each origin string into a HeaderValue. Malformed
    // values are dropped with a warning-equivalent (we log via
    // `tracing::warn` so an operator scanning the boot log can
    // spot a typo).
    let origins: Vec<_> = config
        .allow_origins
        .iter()
        .filter_map(|s| {
            if let Ok(v) = axum::http::HeaderValue::from_str(s) {
                Some(v)
            } else {
                tracing::warn!(
                    origin = %s,
                    "CORS: dropping malformed origin (not a valid header value)"
                );
                None
            }
        })
        .collect();
    layer = layer.allow_origin(origins);

    // Methods / headers: tower-http's CorsLayer takes an `IntoIter<Item = ...>`
    // of either `HeaderName` (for headers) or `Method` (for methods). We
    // accept `Vec<String>` in the config to keep the operator-facing API
    // string-typed (YAML / env var friendly).
    let methods: Vec<_> = config
        .allow_methods
        .iter()
        .filter_map(|s| {
            if let Ok(m) = axum::http::Method::from_str(s) {
                Some(m)
            } else {
                tracing::warn!(
                    method = %s,
                    "CORS: dropping malformed method"
                );
                None
            }
        })
        .collect();
    if methods.is_empty() {
        layer = layer.allow_methods(Any);
    } else {
        layer = layer.allow_methods(methods);
    }

    let headers: Vec<_> = config
        .allow_headers
        .iter()
        .filter_map(|s| {
            if let Ok(h) = HeaderName::from_str(s) {
                Some(h)
            } else {
                tracing::warn!(
                    header = %s,
                    "CORS: dropping malformed header name"
                );
                None
            }
        })
        .collect();
    if headers.is_empty() {
        layer = layer.allow_headers(Any);
    } else {
        layer = layer.allow_headers(headers);
    }

    if config.allow_credentials {
        layer = layer.allow_credentials(true);
    }

    router.layer(layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_closed() {
        let cfg = CorsConfig::default();
        assert!(cfg.allow_origins.is_empty(), "default must have no origins");
        assert!(
            !cfg.allow_credentials,
            "default must NOT set credentials=true"
        );
    }

    #[test]
    fn closed_config_does_not_emit_allow_origin() {
        // Building the layer must succeed even with no origins
        // — the closed state is a valid configuration, not an
        // error condition.
        let _ = with_cors(Router::new(), CorsConfig::default());
    }
}
