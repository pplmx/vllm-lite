//! Role-based access control middleware: `Role` enum, `Permission` enum, and the axum extractor that gates endpoints behind the configured role set.
//!
//! Default policy: `admin` > `operator` > `user` > `anonymous`. Roles and
//! endpoint→role mappings are configured in [`AppConfig`](crate::config::AppConfig).
//!
//! ## SEC-01 (residual) — untrusted-header forgery closed
//!
//! The original `rbac_middleware` extracted the role from the
//! client-supplied `X-User-Role` request header, which let any caller
//! claim `admin` and reach `/metrics`, `/admin/*`, etc. without a
//! valid API key. As of the v31.0 P4 follow-up batch the role must
//! come from the `AuthenticatedRole` request extension, which can
//! only be inserted by server-side middleware (JWT claim extraction,
//! or a future role-aware auth path). Headers are no longer consulted
//! at any decision point. See `docs/technical-due-diligence/production-readiness.md`
//! §2 (SEC-01) and the follow-up note in `debug.rs` for the residual
//! vulnerability that this commit closes.
#![allow(clippy::module_name_repetitions)]
use axum::{
    Json,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;
use std::sync::Arc;

/// Authenticated identity tier. Strict ordering:
/// `Admin > Operator > User > Anonymous`. Each variant maps to a
/// fixed capability set in `RbacMiddleware::check_permission` —
/// `Admin` is the only role allowed to call `manage_*` actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Admin,
    Operator,
    User,
    Anonymous,
}

impl Role {
    /// Parse a role name case-insensitively. Unknown values fall back
    /// to [`Role::Anonymous`] (fail-closed) rather than `Admin`, so a
    /// typo can never grant elevated permissions.
    ///
    /// Only intended for trusted server-side inputs (e.g. JWT claim
    /// strings parsed by `security::jwt`). The HTTP middleware MUST NOT
    /// call this on values derived from the client.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "admin" => Self::Admin,
            "operator" => Self::Operator,
            "user" => Self::User,
            _ => Self::Anonymous,
        }
    }

    /// `true` iff this role can read model metadata (`GET /v1/models`).
    #[must_use]
    pub const fn can_read_models(&self) -> bool {
        !matches!(self, Self::Anonymous)
    }

    /// `true` iff this role can load or unload model weights (admin-only).
    #[must_use]
    pub const fn can_write_models(&self) -> bool {
        matches!(self, Self::Admin)
    }

    /// `true` iff this role can invoke `/admin/*` user-management endpoints.
    #[must_use]
    pub const fn can_manage_users(&self) -> bool {
        matches!(self, Self::Admin)
    }

    /// `true` iff this role can read the Prometheus `/metrics` endpoint.
    #[must_use]
    pub const fn can_view_metrics(&self) -> bool {
        matches!(self, Self::Admin | Self::Operator)
    }

    /// `true` iff this role can reach the admin UI / dashboard.
    #[must_use]
    pub const fn can_access_admin(&self) -> bool {
        matches!(self, Self::Admin)
    }
}

/// Server-injected role claim for the current request.
///
/// Inserted into request extensions by `auth_middleware` (after JWT
/// claim parsing) or by future role-aware middleware. The
/// `rbac_middleware` then reads this value to decide whether the
/// request has the required permission.
///
/// **This type must never be constructible from a request header.**
/// The whole point of the v31.0 P4 SEC-01 fix is that the only way
/// `rbac_middleware` learns a role is through this extension, and the
/// only writers of this extension are server-side middleware that have
/// already authenticated the caller.
#[derive(Debug, Clone, Copy)]
pub struct AuthenticatedRole(pub Role);

/// RBAC middleware. Holds the default role for requests with no role
/// extension installed (always `Anonymous` in production — the only
/// way to bypass `Anonymous` is for upstream middleware to insert
/// [`AuthenticatedRole`]) and the (role → permitted actions) table
/// consulted by `Self::check_permission`.
#[derive(Debug)]
pub struct RbacMiddleware {
    /// Role assigned to requests missing an [`AuthenticatedRole`] extension.
    /// Always `Anonymous` in production; tests may override for fixture setup.
    default_role: Role,
    /// Static (role → permitted actions) policy table.
    role_permissions: Arc<Vec<(Role, Vec<&'static str>)>>,
}

impl RbacMiddleware {
    /// Construct an [`RbacMiddleware`] that grants `default_role` to
    /// requests missing the [`AuthenticatedRole`] extension. The
    /// static (role → action) policy table is baked in: `Admin → ["*"]`,
    /// `Operator → ["read", "execute"]`, `User → ["read", "execute"]`,
    /// `Anonymous → []`.
    ///
    /// **SEC-01**: `default_role` is the *fallback* when no extension
    /// is set, not a privilege granted by an absent auth header. To
    /// preserve the v31.0 P4 fix, production callers must pass
    /// `Role::Anonymous` here.
    #[must_use]
    pub fn new(default_role: Role) -> Self {
        let role_permissions = vec![
            (Role::Admin, vec!["*"]),
            (Role::Operator, vec!["read", "execute"]),
            (Role::User, vec!["read", "execute"]),
            (Role::Anonymous, vec![]),
        ];

        Self {
            default_role,
            role_permissions: Arc::new(role_permissions),
        }
    }

    #[must_use]
    pub(crate) fn check_permission(&self, role: Role, action: &str) -> bool {
        for (r, actions) in self.role_permissions.iter() {
            if *r == role {
                return actions.iter().any(|a| *a == "*" || *a == action);
            }
        }
        false
    }

    /// Resolve the effective [`Role`] for a request.
    ///
    /// Looks at the [`AuthenticatedRole`] request extension installed
    /// by upstream middleware. Falls back to `self.default_role` when
    /// no extension is present, so a request that bypassed auth is
    /// `Anonymous` rather than implicitly `User`.
    ///
    /// **Never reads from request headers** — that was the SEC-01
    /// residual vulnerability.
    #[must_use]
    pub(crate) fn resolve_role(&self, extension: Option<&AuthenticatedRole>) -> Role {
        extension.map_or(self.default_role, |r| r.0)
    }

    ///
    /// Static path → action mapping. Used by `rbac_middleware` to
    /// decide whether the requesting role has the required permission.
    /// Returns `None` for paths that have no RBAC requirement (i.e.
    /// public endpoints like `/health`).
    #[must_use]
    pub(crate) fn required_action_for_path(path: &str) -> Option<&'static str> {
        // Strip trailing slash for matching.
        let p = path.trim_end_matches('/');
        match p {
            "/health" | "/ready" => None,
            "/v1/models" => Some("read"),
            "/v1/chat/completions" | "/v1/completions" | "/v1/embeddings" => Some("execute"),
            "/metrics" => Some("view_metrics"),
            p if p.starts_with("/admin") => Some("manage_users"),
            // Default: unknown paths require `read` (least-privilege
            // default; admin wildcard still grants access).
            _ => Some("read"),
        }
    }
}

///
/// Enforces role-based access control.
///
/// Resolves the role from the [`AuthenticatedRole`] request extension
/// installed by upstream auth middleware (JWT claim path in production,
/// fixtures in tests). Rejects the request with HTTP 403 if the role
/// lacks the required permission for the path.
///
/// **Never reads from request headers** — the SEC-01 residual
/// vulnerability (X-User-Role forgery) is closed by relying solely on
/// the server-installed extension.
pub async fn rbac_middleware(request: Request, next: Next) -> Response {
    let rbac = RbacMiddleware::new(Role::Anonymous);
    let path = request.uri().path().to_string();
    let Some(required) = RbacMiddleware::required_action_for_path(&path) else {
        return next.run(request).await;
    };

    let extension = request.extensions().get::<AuthenticatedRole>().copied();
    let role = rbac.resolve_role(extension.as_ref());
    if rbac.check_permission(role, required) {
        next.run(request).await
    } else {
        let body = Json(json!({
            "error": "forbidden",
            "message": format!("Role {:?} lacks required permission '{}' for {}", role, required, path),
            "required_action": required,
        }));
        (StatusCode::FORBIDDEN, body).into_response()
    }
}

// Unit + integration tests live in `tests.rs` (sibling) to keep this
// middleware module under the 800-line soft cap. They cover the
// pure-unit role/permission matrix (4 tests including the new SEC-01
// forgery-resistance assertions) and an axum middleware integration
// that exercises the full request → middleware → response path through
// `tower::ServiceExt::oneshot` (5 tests, all updated to install
// `AuthenticatedRole` via request extensions rather than forging
// `X-User-Role` headers).
#[cfg(test)]
mod tests;
