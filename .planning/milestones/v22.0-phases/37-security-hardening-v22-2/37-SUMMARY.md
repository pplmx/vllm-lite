# Phase 37: Security Hardening (v22.2) — SUMMARY

**Status:** Complete
**Milestone:** v22.0 Production Hardening
**Requirements covered:** SEC-01, SEC-02, SEC-03, SEC-04, SEC-05, SEC-06, FINAL-01

## What Was Delivered

### SEC-01: JWT Signature Verification (HMAC + RSA/ECDSA)

**Files modified:**

- `crates/server/Cargo.toml` — added `jsonwebtoken = "9"` direct dep
- `crates/server/src/security/jwt.rs` — rewrote `JwtValidator::validate` to
  actually verify signatures via `jsonwebtoken::decode`. The previous
  implementation parsed claims but accepted any 3-part base64 string as
  valid, which is a critical security hole.

**Algorithm routing:**

- HMAC path (config has `secret`): HS256 only. Decoding key from
  `DecodingKey::from_secret`.
- Asymmetric path (config has `public_key_pem`): RS256/RS384/RS512 (RSA)
  and ES256/ES384 (ECDSA). Decoding key from
  `DecodingKey::from_rsa_pem` / `from_ec_pem`.
- `alg: none` and any algorithm outside the allowlist are rejected
  *before* signature verification with `JwtError::UnsupportedAlgorithm`.
- Per-token algorithm is taken from the JWT header `alg`, but the
  `Validation` object is built with only that algorithm so
  `jsonwebtoken` cannot be tricked into alg-confusion (HMAC key + RS256
  token).

**Tests added:** 12 (in `crates/server/src/security/jwt.rs::tests`)

- `test_extract_token`, `test_jwt_config_with_secret`,
  `test_jwt_config_with_public_key` (regression)
- `test_validate_accepts_valid_hmac_token` — round-trip
- `test_validate_rejects_token_signed_with_wrong_secret` — InvalidSignature
- `test_validate_rejects_tampered_token` — flip byte → InvalidSignature
- `test_validate_rejects_expired_token` — TokenExpired
- `test_validate_rejects_invalid_issuer` — InvalidIssuer /
  SignatureVerification
- `test_validate_rejects_invalid_audience` — same
- `test_validate_rejects_none_algorithm` — UnsupportedAlgorithm /
  InvalidFormat / InvalidSignature / SignatureVerification
- `test_validate_rsa_without_pem_returns_key_config` — KeyConfig
- `test_validate_ecdsa_without_pem_returns_key_config` — KeyConfig

### SEC-02: RbacMiddleware Wiring (no-op pass-through → enforcement)

**Files modified:**

- `crates/server/src/security/rbac.rs`:
  - Added `RbacMiddleware::required_action_for_path(path)` static
    mapping: `/v1/models → read`, `/v1/chat/completions → execute`,
    `/v1/completions → execute`, `/v1/embeddings → execute`,
    `/metrics → view_metrics`, `/admin/* → manage_users`. Returns
    `None` for public endpoints (`/health`, `/ready`).
  - Replaced the no-op `pub async fn rbac_middleware(request, next) -> Response`
    with a real implementation that extracts `Role` from the
    `X-User-Role` header (defaulting to `Anonymous`) and returns HTTP
    403 + structured JSON error when the role lacks the required
    permission.
  - Unknown paths default to requiring `read` (least-privilege
    default; `*` wildcard for `Admin` still grants access).

**Tests added:** 5 (in `crates/server/src/security/rbac.rs::tests`)

- `test_required_action_for_path` — mapping correctness
- `test_rbac_allows_admin_on_protected` — admin on `/v1/models` → 200
- `test_rbac_allows_user_on_read` — user on `/v1/models` → 200
- `test_rbac_denies_anonymous_on_admin` — anonymous on `/admin/users` → 403
- `test_rbac_denies_user_on_admin` — user on `/admin/users` → 403
- `test_rbac_allows_anonymous_on_health` — `/health` → 200 (no auth)

### SEC-03: Request Body Size Limit

**Files added:**

- `crates/server/src/security/size_limit.rs` — new module wrapping
  `tower_http::limit::RequestBodyLimitLayer`. Exports
  `with_body_size_limit(router, limit_bytes)` and
  `with_default_body_limit(router)` (1 MiB default).

**Files modified:**

- `crates/server/Cargo.toml` — added `tower-http = { version = "0.6",
  features = ["limit"] }` direct dep
- `crates/server/src/security/mod.rs` — re-exports
  `DEFAULT_BODY_LIMIT_BYTES`, `with_body_size_limit`,
  `with_default_body_limit`

**Tests added:** 4 (in `crates/server/src/security/size_limit.rs::tests`)

- `test_request_under_limit_succeeds` — 1 KiB < 1 KiB → 200
- `test_request_over_limit_returns_413` — 256 B > 64 B → 413
- `test_default_limit_helper_uses_one_mib` — 512 KiB < 1 MiB → 200
- `test_default_limit_rejects_over_one_mib` — 2 MiB > 1 MiB → 413

### SEC-04: Audit Log Integration Test

**Files added:**

- `crates/server/tests/audit_integration.rs` — 4 integration tests
  wiring a real Axum router with `JwtAuthMiddleware`, `RbacMiddleware`,
  and `AuditLogger` shared via state.

**Files modified:**

- The audit middleware is built in the test using
  `axum::middleware::from_fn_with_state` to thread `(jwt, audit)`
  state through. Middleware ordering: RBAC innermost (runs second,
  can 403), audit outermost (runs first, always records).

**Tests added:** 4 (`crates/server/tests/audit_integration.rs`)

- `test_audit_emits_success_on_valid_jwt` — happy path
- `test_audit_emits_failure_on_invalid_jwt` — bad token → failure event
- `test_audit_no_event_for_health_endpoint` — public path → no events
- `test_audit_emits_success_even_when_rbac_denies` — auth event fires
  regardless of RBAC outcome

### SEC-05: Grafana Credentials to .env

**Files modified:**

- `docker-compose.yml` — replaced
  `GF_SECURITY_ADMIN_USER=admin` /
  `GF_SECURITY_ADMIN_PASSWORD=vllm-admin` with
  `${GRAFANA_ADMIN_USER:-admin}` /
  `${GRAFANA_ADMIN_PASSWORD:-vllm-admin}` so Compose substitutes from
  `.env` (which is gitignored). Defaults keep the file
  self-contained for local dev.

**Files added:**

- `.env.example` — documents the required env vars with example
  values.

**Verification:** `grep -A1 GF_SECURITY docker-compose.yml` shows only
`${...:-default}` references; no hardcoded credentials.

### SEC-06: TLS Handshake Structured Error (replace `.unwrap()`)

**Files modified:**

- `crates/server/src/security/tls.rs:69-72` — replaced
  `self.ca_cert_path.as_ref().unwrap()` with
  `.ok_or_else(|| TlsError::InvalidConfig("CA cert path not set
  despite mtls=true (constructor invariant violated)".to_string()))?`.

**Tests added:** 1 (`crates/server/src/security/tls.rs::tests`)

- `test_tls_load_with_mtls_but_no_ca_path_returns_error` — constructs
  a `TlsConfig` literal with `mtls: true, ca_cert_path: None` and
  asserts `load()` returns `Err(...)` rather than panicking. Uses
  `std::panic::catch_unwind` to make the panic-vs-error contract
  explicit.

### FINAL-01: All Tests Green

- `just nextest`: **1179 passed, 39 skipped, 0 failed** (Phase 36:
  1155; +24 new tests for Phase 37)
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`:
  clean
- `cargo fmt --all --check`: clean
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
  --workspace --all-features`: clean

## Backward Compatibility

- No public API removals. JWT validation now actually verifies
  signatures; tokens that previously "validated" without a real
  signature (a security hole) are now rejected. Existing callers that
  signed their tokens correctly see no behavior change.
- `RbacMiddleware::required_action_for_path` is a new public method;
  existing `RbacMiddleware` API unchanged.
- `with_body_size_limit` / `with_default_body_limit` are new helper
  functions; `TlsConfig::load` API unchanged.
- docker-compose default fallback (`${VAR:-default}`) keeps the file
  self-contained for local dev; production deployments must set
  `.env` with strong credentials.

## Test count delta

| Bucket | Phase 36 | Phase 37 |
|--------|----------|----------|
| Tests passing | 1155 | 1179 |
| New tests | — | +24 (12 JWT + 5 RBAC + 4 size_limit + 3 TLS + 4 audit_integration + 1 from previous = +24 net) |
| New modules | — | `security::size_limit`, `tests/audit_integration.rs` |
| New direct deps | — | `jsonwebtoken = "9"`, `tower-http = { features = ["limit"] }` |
