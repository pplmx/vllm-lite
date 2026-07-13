//!
//! Wires a real Axum app with `JwtAuthMiddleware` (validating tokens),
//! `RbacMiddleware` (enforcing role-based access), and an
//! `AuditLogger` (recording events). Each test fires a request through
//! the stack and asserts the expected audit events are emitted.
//!
//! SEC-01 (v31.0 P4): the role is now installed into the
//! [`AuthenticatedRole`] request extension by the JWT middleware,
//! not read from the `X-User-Role` request header. Tests that
//! previously forged `X-User-Role: admin` to reach protected paths
//! now install `AuthenticatedRole` directly via the extension —
//! matching the production contract enforced by `rbac_middleware`.

#![cfg(test)]

use axum::Router;
use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::middleware::{Next, from_fn_with_state};
use axum::response::Response;
use axum::routing::get;
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde_json::json;
use std::sync::Arc;
use tower::ServiceExt;
use vllm_server::security::{
    Role,
    audit::AuditLogger,
    jwt::JwtAuthMiddleware,
    rbac::{AuthenticatedRole, rbac_middleware},
};

/// Test JWT secret used by all cases in this file.
const TEST_SECRET: &str = "test-secret-32-bytes-long-aaaa";
const TEST_ISS: &str = "vllm-test";
const TEST_AUD: &str = "vllm-test-api";

fn mint_token(role: &str) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let payload = json!({
        "sub": "user-1",
        "iss": TEST_ISS,
        "aud": TEST_AUD,
        "exp": now + 3600,
        "iat": now,
        "roles": [role],
    });
    encode(
        &Header::new(Algorithm::HS256),
        &payload,
        &EncodingKey::from_secret(TEST_SECRET.as_bytes()),
    )
    .unwrap()
}

/// Resolve a role string to the workspace [`Role`] enum. Mirrors the
/// production JWT-middleware behaviour: the role comes from the JWT
/// claim, not the request header.
fn role_from_str(role: &str) -> Role {
    match role {
        "admin" => Role::Admin,
        "operator" => Role::Operator,
        "user" => Role::User,
        _ => Role::Anonymous,
    }
}

#[derive(Clone)]
struct AuditState {
    jwt: Arc<JwtAuthMiddleware>,
    audit: Arc<AuditLogger>,
}

/// Audit middleware used by these integration tests. Mirrors what
/// the production stack would do: on a successful JWT, install the
/// `AuthenticatedRole` extension from the validated claims so the
/// downstream `rbac_middleware` can authorise (or deny) the request
/// **without** ever consulting a client-supplied header.
async fn audit_middleware(
    axum::extract::State(state): axum::extract::State<AuditState>,
    mut req: axum::extract::Request,
    next: Next,
) -> Response {
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .map(std::string::ToString::to_string);
    if let Some(h) = auth_header {
        match state.jwt.validate_request(&h).await {
            Ok(claims) => {
                // Promote the highest-privileged role claim to the
                // request extension. The production JWT middleware
                // will eventually do this directly; for now we do it
                // here so the RBAC middleware can authorise the
                // request from a verified source.
                let role = claims
                    .roles
                    .iter()
                    .map(|r| role_from_str(r))
                    .max_by_key(|r| match r {
                        Role::Admin => 3,
                        Role::Operator => 2,
                        Role::User => 1,
                        Role::Anonymous => 0,
                    })
                    .unwrap_or(Role::Anonymous);
                req.extensions_mut().insert(AuthenticatedRole(role));
                state.audit.log_auth_success(&claims.sub, "test-req").await;
            }
            Err(e) => {
                state
                    .audit
                    .log_auth_failure(&e.to_string(), "test-req")
                    .await;
            }
        }
    } else {
        state
            .audit
            .log_auth_failure("missing Authorization", "test-req")
            .await;
    }
    next.run(req).await
}

/// Build an Axum router with JWT auth + RBAC + audit wired.
fn build_app(audit: Arc<AuditLogger>, jwt: Arc<JwtAuthMiddleware>) -> Router {
    async fn models() -> &'static str {
        "models"
    }
    async fn admin_users() -> &'static str {
        "admin"
    }
    async fn health() -> &'static str {
        "ok"
    }

    let state = AuditState { jwt, audit };

    let protected = Router::new()
        .route("/v1/models", get(models))
        .route("/admin/users", get(admin_users))
        // RBAC runs first (innermost), so it can 403 before any other
        // processing. Audit runs second (outermost) so it observes the
        // request regardless of the RBAC outcome.
        .layer(axum::middleware::from_fn(rbac_middleware))
        .layer(from_fn_with_state(state, audit_middleware));

    protected.route("/health", get(health))
}

#[tokio::test]
async fn test_audit_emits_success_on_valid_jwt() {
    let audit = Arc::new(AuditLogger::new(100));
    let jwt = Arc::new(JwtAuthMiddleware::new(
        vllm_server::security::jwt::JwtConfig::with_secret(TEST_SECRET)
            .with_issuer(TEST_ISS)
            .with_audience(TEST_AUD),
    ));
    let app = build_app(audit.clone(), jwt);

    // JWT carries `admin` role; the audit middleware promotes it to
    // an `AuthenticatedRole` extension. No forged `X-User-Role`
    // header — post SEC-01 the middleware ignores those entirely.
    let token = mint_token("admin");
    let req = HttpRequest::builder()
        .method("GET")
        .uri("/v1/models")
        .header("authorization", format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let events = audit.get_events().await;
    assert!(
        events
            .iter()
            .any(|e| e.action == "authenticate" && e.result == "success"),
        "expected auth success event, got: {events:?}"
    );
}

#[tokio::test]
async fn test_audit_emits_failure_on_invalid_jwt() {
    let audit = Arc::new(AuditLogger::new(100));
    let jwt = Arc::new(JwtAuthMiddleware::new(
        vllm_server::security::jwt::JwtConfig::with_secret(TEST_SECRET)
            .with_issuer(TEST_ISS)
            .with_audience(TEST_AUD),
    ));
    let app = build_app(audit.clone(), jwt);

    let req = HttpRequest::builder()
        .method("GET")
        .uri("/v1/models")
        .header("authorization", "Bearer not.a.real.jwt")
        .body(Body::empty())
        .unwrap();
    let _resp = app.oneshot(req).await.unwrap();

    let events = audit.get_events().await;
    assert!(
        events
            .iter()
            .any(|e| { e.action == "authenticate" && e.result.starts_with("failure") }),
        "expected auth failure event, got: {events:?}"
    );
}

#[tokio::test]
async fn test_audit_no_event_for_health_endpoint() {
    let audit = Arc::new(AuditLogger::new(100));
    let jwt = Arc::new(JwtAuthMiddleware::new(
        vllm_server::security::jwt::JwtConfig::with_secret(TEST_SECRET)
            .with_issuer(TEST_ISS)
            .with_audience(TEST_AUD),
    ));
    let app = build_app(audit.clone(), jwt);

    let req = HttpRequest::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let events = audit.get_events().await;
    assert!(
        events.is_empty(),
        "health endpoint should not emit audit events, got: {events:?}"
    );
}

#[tokio::test]
async fn test_audit_emits_success_even_when_rbac_denies() {
    let audit = Arc::new(AuditLogger::new(100));
    let jwt = Arc::new(JwtAuthMiddleware::new(
        vllm_server::security::jwt::JwtConfig::with_secret(TEST_SECRET)
            .with_issuer(TEST_ISS)
            .with_audience(TEST_AUD),
    ));
    let app = build_app(audit.clone(), jwt);

    // JWT carries `user` role — too weak for `/admin/users` even
    // when the JWT validates. RBAC must still deny, audit must
    // still record success.
    let token = mint_token("user");
    let req = HttpRequest::builder()
        .method("GET")
        .uri("/admin/users")
        .header("authorization", format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    let events = audit.get_events().await;
    assert!(
        events
            .iter()
            .any(|e| e.action == "authenticate" && e.result == "success"),
        "expected auth success event even on RBAC denial, got: {events:?}"
    );
}

/// SEC-01 regression: a forged `X-User-Role: admin` header MUST NOT
/// grant admin access, even when paired with a valid `user` JWT.
/// Pre-fix, this combination succeeded because RBAC trusted the
/// header; post-fix the header is ignored and only the
/// JWT-derived role counts.
#[tokio::test]
async fn test_audit_forged_role_header_is_ignored_by_rbac() {
    let audit = Arc::new(AuditLogger::new(100));
    let jwt = Arc::new(JwtAuthMiddleware::new(
        vllm_server::security::jwt::JwtConfig::with_secret(TEST_SECRET)
            .with_issuer(TEST_ISS)
            .with_audience(TEST_AUD),
    ));
    let app = build_app(audit.clone(), jwt);

    let token = mint_token("user");
    let req = HttpRequest::builder()
        .method("GET")
        .uri("/admin/users")
        .header("authorization", format!("Bearer {token}"))
        .header("X-User-Role", "admin")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::FORBIDDEN,
        "forged X-User-Role header must not grant admin (SEC-01)"
    );
}
