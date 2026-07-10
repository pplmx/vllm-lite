//! Role-based access control middleware: `Role` enum, `Permission` enum, and the axum extractor that gates endpoints behind the configured role set.
//!
//! Default policy: `admin` > `operator` > `user` > `anonymous`. Roles and
//! endpoint→role mappings are configured in [`AppConfig`](crate::config::AppConfig).
#![allow(clippy::module_name_repetitions)]
use axum::{
    Json,
    extract::Request,
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;
use std::sync::Arc;

/// Authenticated identity tier. Strict ordering:
/// `Admin > Operator > User > Anonymous`. Each variant maps to a
/// fixed capability set in [`RbacMiddleware::check_permission`] —
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

#[derive(Debug)]
/// RBAC middleware. Holds the default role for unauthenticated requests and the
/// (role → permitted actions) table consulted by [`Self::check_permission`].
pub struct RbacMiddleware {
    /// Role assigned to requests without an `X-User-Role` header.
    default_role: Role,
    /// Static (role → permitted actions) policy table.
    role_permissions: Arc<Vec<(Role, Vec<&'static str>)>>,
}

impl RbacMiddleware {
    /// Construct an [`RbacMiddleware`] that grants `default_role` to
    /// requests missing the `X-User-Role` header. The static
    /// (role → action) policy table is baked in: `Admin → ["*"]`,
    /// `Operator → ["read", "execute"]`, `User → ["read", "execute"]`,
    /// `Anonymous → []`.
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
    pub fn check_permission(&self, role: Role, action: &str) -> bool {
        for (r, actions) in self.role_permissions.iter() {
            if *r == role {
                return actions.iter().any(|a| *a == "*" || *a == action);
            }
        }
        false
    }

    pub fn extract_role_from_headers(&self, headers: &HeaderMap) -> Role {
        headers
            .get("X-User-Role")
            .and_then(|v| v.to_str().ok())
            .map_or(self.default_role, Role::from_str)
    }

    ///
    /// Static path → action mapping. Used by `rbac_middleware` to
    /// decide whether the requesting role has the required permission.
    /// Returns `None` for paths that have no RBAC requirement (i.e.
    /// public endpoints like `/health`).
    #[must_use]
    pub fn required_action_for_path(path: &str) -> Option<&'static str> {
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
/// Extracts the role from either the JWT-claims-style `X-User-Role`
/// header (set upstream by the auth middleware) or the configured
/// default role, then denies the request with HTTP 403 + a structured
/// JSON error if the role lacks the required permission for the
/// requested path.
pub async fn rbac_middleware(request: Request, next: Next) -> Response {
    let rbac = RbacMiddleware::new(Role::Anonymous);
    let path = request.uri().path().to_string();
    let Some(required) = RbacMiddleware::required_action_for_path(&path) else {
        return next.run(request).await;
    };

    let role = rbac.extract_role_from_headers(request.headers());
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
// pure-unit role/permission matrix (3 tests) and an axum middleware
// integration that exercises the full request → middleware →
// response path through `tower::ServiceExt::oneshot` (5 tests).
#[cfg(test)]
mod tests;
