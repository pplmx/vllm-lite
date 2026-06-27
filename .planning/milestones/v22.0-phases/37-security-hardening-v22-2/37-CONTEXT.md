# Phase 37: Security Hardening (v22.2) - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Smart discuss auto-accepted (infrastructure phase)

<domain>

## Phase Boundary

Make security middleware actually enforce its policies — currently JWT parses
but doesn't verify, RBAC is a no-op pass-through, TLS has unwrap() panics in
the mTLS path, Grafana credentials are hardcoded, and no integration tests
verify the auth/middleware pipeline end-to-end.

Six concrete deliverables:

1. **SEC-01**: JWT signature verification — HMAC-SHA256 for `secret`-based
   tokens; RSA + ECDSA for `public_key_pem`-based tokens. Reject
   invalid/expired/tampered tokens with structured errors. Integration tests
   covering each rejection case.
2. **SEC-02**: Wire `RbacMiddleware` into the request pipeline (currently a
   no-op pass-through at `rbac.rs:96-98`). Enforce role-based permission
   checks; deny with HTTP 403 + structured error. Integration tests for
   permission-required paths.
3. **SEC-03**: `tower_http::limit::RequestBodyLimitLayer` with configurable
   size limit; requests over the limit return HTTP 413 Payload Too Large.
4. **SEC-04**: Audit log integration test asserting `AuditLogger` emits
   entries for each auth scenario (success, JWT failure, RBAC denial).
5. **SEC-05**: `docker-compose.yml` — `GRAFANA_ADMIN_USER` and
   `GRAFANA_ADMIN_PASSWORD` read from `.env` via Compose variable
   substitution. No hardcoded credentials in the compose file.
6. **SEC-06**: Replace `.unwrap()` on `ca_cert_path` in `security/tls.rs:70`
   with a structured `TlsError::KeyRead`/`CertificateRead` variant.

FINAL-01: All 1155+ tests remain green post-fix.

</domain>

<decisions>

## Implementation Decisions

### the agent's Discretion

All implementation choices are at the agent's discretion — pure
security-hardening phase. Use the codebase scout below to drive decisions.
Specific guidance:

- **JWT verification**: Use the `jsonwebtoken` crate (already in tree for
  `cargo` indirect deps; pull in as direct dep). Use `Validation` builder
  for issuer/audience/exp checks; `EncodingKey`/`DecodingKey` for HMAC vs
  RSA/ECDSA. Algorithm must be picked from a small allowlist (avoid
  `alg: none` and key-confusion attacks).
- **RBAC wiring**: Add a `Permission` enum + path → action mapping table
  inside `RbacMiddleware`. The middleware should be optional in the route
  builder so tests can opt-out. Extract role from JWT claims (preferred)
  or `X-User-Role` header (back-compat).
- **Request body limit**: Default limit (e.g. 1 MiB) configurable via
  `ServerConfig`. Use `tower_http::limit::RequestBodyLimitLayer` applied
  via `tower::ServiceBuilder` to all routes.
- **Audit log integration**: Test must build a real Axum app with
  middleware wired, fire requests through `tower::ServiceExt::oneshot`,
  assert events appear in the audit log.
- **Grafana credentials**: Use `${GRAFANA_ADMIN_USER:-admin}` / similar
  syntax in docker-compose.yml. Add a `.env.example` for documentation.
- **TLS unwrap**: Replace `self.ca_cert_path.as_ref().unwrap()` with
  `.ok_or(TlsError::InvalidConfig(...))?` for symmetry with the existing
  `KeyRead` / `CertificateRead` error handling.

### Constraints (apply to all sub-plans)

- No new feature surface beyond SEC-01..06.
- All public API changes use `#[deprecated(since, note)]` markers + migration
  path (DEP-01/02 from v20.6).
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  remains clean after each sub-plan.
- No new dependencies beyond `jsonwebtoken` (already in dep tree).

</decisions>

<code_context>

## Existing Code Insights

### Reusable Assets

- **`JwtValidator`** (`crates/server/src/security/jwt.rs:88`) — currently
  parses tokens but does not verify signatures. Has `JwtConfig` with
  `secret` (HMAC path) and `public_key_pem` (RSA/ECDSA path) fields.
- **`JwtAuthMiddleware`** (`crates/server/src/security/jwt.rs:144`) —
  async wrapper around `JwtValidator`. Currently no callers wired in.
- **`RbacMiddleware`** (`crates/server/src/security/rbac.rs:54`) — defines
  `Role` enum + `check_permission(role, action)`. The `rbac_middleware`
  free function is a no-op pass-through.
- **`AuditLogger`** (`crates/server/src/security/audit.rs:21`) — has
  `log_auth_success`, `log_auth_failure`, `log_api_request`, `get_events`.
- **`TlsConfig::load`** (`crates/server/src/security/tls.rs:54`) — has
  `unwrap()` on `ca_cert_path.as_ref()` at line 70 (mtls path).
- **`ServerConfig`** — entry point for runtime config; check existing
  `body_size_limit` field or add one if missing.

### Established Patterns

- All security middleware uses `axum::middleware::from_fn` / `from_fn_with_state`.
- Errors propagate via `thiserror::Error` enums; HTTP responses via
  `axum::Json` + `StatusCode`.
- Integration tests use `tower::ServiceExt::oneshot` + `axum::body::Body`.
- Audit events are clones into a `Vec`; `len() > max_events` evicts the
  oldest.
- Tracing spans use `tracing::info_span!` / `info!` / `warn!` / `error!`.

### Integration Points

- **Auth pipeline**: `crates/server/src/lib.rs` (or wherever the Axum
  router is built) — middleware stack needs:
  `JwtAuthMiddleware → RbacMiddleware → handler`.
- **Body limit**: applied via `tower::ServiceBuilder::layer(
  RequestBodyLimitLayer::new(limit))` on the router.
- **TLS error path**: `tls.rs:70` — single `.unwrap()` to replace.
- **docker-compose.yml**: top-level — `GRAFANA_ADMIN_USER` /
  `GRAFANA_ADMIN_PASSWORD` env blocks for the `grafana` service.

</code_context>

<specifics>

## Specific Ideas

- **JWT verification algorithm allowlist**: accept only HS256 (HMAC),
  RS256/RS384/RS512 (RSA), ES256/ES384 (ECDSA). Reject `none` and other
  algorithms. The `jsonwebtoken` crate enforces this when
  `validation.algorithms` is set explicitly.
- **RBAC path→action mapping**: define a small enum or table mapping
  `POST /v1/chat/completions → "execute"`, `GET /v1/models → "read"`,
  `GET /admin/users → "manage_users"`, etc. RBAC middleware extracts the
  path, looks up the action, then calls `check_permission`.
- **Audit log test**: spawn a test app with `AuditLogger::new(100)`,
  fire a request through `oneshot`, read `get_events()`, assert the
  expected event is present.
- **Body limit default**: 1 MiB (1_048_576 bytes). Document in `ServerConfig`.
- **docker-compose env**: `${GRAFANA_ADMIN_USER:-admin}` so defaults
  remain backward-compatible.

</specifics>

<deferred>

## Deferred Ideas

- **JWT refresh-token flow** — out of scope; SEC-01 covers verification
  only.
- **mTLS client-cert enforcement at HTTP layer** — TLS supports it via
  `TlsConfig::mtls = true`; the Rustls handshake validates the cert.
  Higher-level request-binding to cert subject is future work.
- **Per-route RBAC actions** beyond the canonical mapping — current
  scope covers the documented endpoints; new endpoints add their own
  actions in subsequent phases.

</deferred>
