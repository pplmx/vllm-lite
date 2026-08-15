//! Public API surface of the security subsystem: re-exports the JWT, RBAC, audit, TLS, CORS, and size-limit modules under one namespace.
//!
//! ## What is actually mounted (RIL ISS-071 / DEC-049)
//!
//! The request-path middlewares installed by `app::build_app` are: CORS,
//! correlation-id, audit logging, the body-size limit, and — when API keys
//! are configured — the API-key auth + rate-limit middleware (`auth.rs`).
//! That auth layer is the **only** access control on the live router.
//!
//! The RBAC and JWT modules in this directory are **declared but not
//! enforced**: `RbacMiddleware` has no production mount site, `auth_middleware`
//! only ever inserts `AuthenticatedUser` (never `AuthenticatedRole`, the value
//! RBAC gates on), and `JwtValidator` has no constructor call anywhere in
//! `main.rs`/`app.rs`. An operator reading the old module doc would believe
//! RBAC tiers (admin/operator/user/anonymous) protect `/v1/models` and
//! `/metrics` — they do not; only API-key verification does. Treat these
//! modules as the designed-but-deferred RBAC/JWT milestone, not as active
//! controls. See `docs/technical-due-diligence/production-readiness.md` §2
//! and RIL DEC-049 for the deferral rationale.
/// Structured audit logging: in-memory ring buffer + `/debug/audit` export.
pub mod audit;
/// Audit-logging request-layer middleware.
pub mod audit_middleware;
/// Request correlation ID assignment and propagation.
pub mod correlation;
/// CORS middleware with explicit allowlist (no wildcard).
pub mod cors;
/// JWT validation and auth middleware.
pub mod jwt;
/// Role-based access control middleware.
pub mod rbac;
/// Request body size-limit enforcement.
pub mod size_limit;
pub mod timing;
/// TLS configuration and certificate loading.
pub mod tls;

pub use audit::AuditLogger;
pub use audit_middleware::audit_middleware;
pub use correlation::CorrelationIdMiddleware;
pub use cors::{CorsConfig, with_cors};
pub use jwt::JwtValidator;
pub use rbac::{RbacMiddleware, Role};
pub use size_limit::{DEFAULT_BODY_LIMIT_BYTES, with_body_size_limit, with_default_body_limit};
pub use tls::TlsConfig;
