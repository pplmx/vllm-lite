# Phase 37: Security Hardening (v22.2) - PLAN

**Status:** Ready for execution
**Milestone:** v22.0 Production Hardening
**Requirements covered:** SEC-01, SEC-02, SEC-03, SEC-04, SEC-05, SEC-06, FINAL-01

## Plan Structure

This phase is split into 7 sequential plans. Each plan is small enough to
ship independently and lands in a single atomic commit.

---

## Plan 37-01: JWT Signature Verification (HMAC + RSA/ECDSA) — SEC-01

**Goal:** `JwtValidator::validate` actually verifies the signature.
Currently it parses claims but accepts any 3-part base64 string.

**Files to modify:**

- `crates/server/Cargo.toml` — add `jsonwebtoken = "9"` direct dep
- `crates/server/src/security/jwt.rs`:
  - Add `validate_signature(token, header, payload, config)` helper
  - HMAC-SHA256 path: `DecodingKey::from_secret(secret)` +
    `Validation::new(Algorithm::HS256)`
  - Asymmetric path: `DecodingKey::from_rsa_pem(public_key_pem)` /
    `DecodingKey::from_ec_pem(public_key_pem)` — algorithm inferred from
    JWT header `alg`
  - `validate()` calls `decode(token, &key, &validation)` and uses the
    returned `Claims`
  - Allowlist algorithms: HS256, RS256, RS384, RS512, ES256, ES384. Reject
    `none` and any other.
  - Update `JwtError` with `SignatureVerification(String)` variant

**Tests to add (`crates/server/src/security/jwt.rs` test module):**

- `test_validate_rejects_token_signed_with_wrong_secret` — generate
  token with secret A, validate with secret B → InvalidSignature
- `test_validate_rejects_tampered_token` — flip a byte in the payload,
  expect InvalidSignature
- `test_validate_accepts_valid_hmac_token` — round-trip
- `test_validate_rejects_expired_token` — exp in past → TokenExpired
- `test_validate_rejects_invalid_issuer` — iss mismatch → InvalidIssuer
- `test_validate_rejects_invalid_audience` — aud mismatch → InvalidIssuer
- `test_validate_rejects_none_algorithm` — manually craft `alg: none`
  header, expect rejection
- `test_validate_accepts_valid_rsa_token` — round-trip with RSA keypair
- `test_validate_accepts_valid_ecdsa_token` — round-trip with EC keypair

**Verification:** `cargo test -p vllm-server security::jwt` passes.

---

## Plan 37-02: Wire RbacMiddleware — SEC-02

**Goal:** Replace the no-op pass-through with real RBAC enforcement.
Deny requests lacking required permissions with HTTP 403.

**Files to modify:**

- `crates/server/src/security/rbac.rs`:
  - Replace `pub async fn rbac_middleware(request, next)` with a real
    implementation that:
    1. Extracts `Role` from JWT claims (preferred) or `X-User-Role`
       header (fallback)
    2. Looks up the path → required action in a static table
    3. Calls `check_permission(role, action)`; if false, returns
       `StatusCode::FORBIDDEN` + JSON error
  - Add `PathAction` static table covering: `/v1/chat/completions` →
    `execute`, `/v1/models` → `read`, `/v1/completions` → `execute`,
    `/v1/embeddings` → `execute`, `/admin/*` → `manage_users`,
    `/metrics` → `view_metrics`
  - Export `rbac_middleware` as a typed `axum::middleware::from_fn` target
- `crates/server/src/lib.rs` (or wherever router is built) — install
  RBAC middleware on the protected routes (skip `/health`)

**Tests to add (`crates/server/src/security/rbac.rs` test module):**

- `test_rbac_middleware_allows_admin` — Admin role on `/v1/models` → 200
- `test_rbac_middleware_denies_anonymous_on_admin` — Anonymous on
  `/admin/foo` → 403
- `test_rbac_middleware_denies_user_on_admin` — User on `/admin/foo` → 403
- `test_rbac_middleware_allows_user_on_read` — User on `/v1/models` → 200
- `test_rbac_middleware_extracts_role_from_jwt` — JWT contains role,
  used in permission check

**Verification:** `cargo test -p vllm-server security::rbac` passes.

---

## Plan 37-03: Request Body Size Limit — SEC-03

**Goal:** Reject HTTP request bodies exceeding the configured size with
HTTP 413 Payload Too Large.

**Files to modify:**

- `crates/server/src/config.rs` (or wherever `ServerConfig` lives):
  - Add `body_size_limit: usize` field (default 1_048_576 = 1 MiB)
  - Builder method `with_body_size_limit(bytes)`
- `crates/server/src/lib.rs` — apply
  `tower_http::limit::RequestBodyLimitLayer::new(limit)` via
  `tower::ServiceBuilder` to the protected routes
- Add `tower-http` feature `limit` if not enabled (it should already be)

**Tests to add (`crates/server/tests/security/request_size.rs`):**

- `test_request_under_limit_succeeds` — body < 1 MiB → 200
- `test_request_over_limit_returns_413` — body > 1 MiB → 413
- `test_custom_limit_via_config` — set 512 bytes, send 1 KiB → 413

**Verification:** `cargo test -p vllm-server request_size` passes.

---

## Plan 37-04: Audit Log Integration Test — SEC-04

**Goal:** Integration test asserting audit log emits an event for every
authenticated request (success, JWT failure, RBAC denial).

**Files to add:**

- `crates/server/tests/security/audit_integration.rs`:
  - Build a test Axum app with `JwtAuthMiddleware`, `RbacMiddleware`,
    and a shared `AuditLogger`
  - Wire a small `auth` route group + a protected `/v1/models` route
  - Test 1: valid JWT → audit log contains `authenticate:success`
  - Test 2: invalid JWT → audit log contains `authenticate:failure:...`
  - Test 3: valid JWT but Anonymous role on `/admin/foo` → audit log
    contains both `authenticate:success` AND `rbac:denied`
  - Test 4: `/health` (no auth) → no audit entry

**Files to modify:**

- `crates/server/src/security/jwt.rs::JwtAuthMiddleware` — call
  `AuditLogger::log_auth_success` / `log_auth_failure` on validation
  outcome
- `crates/server/src/security/rbac.rs::rbac_middleware` — call
  `AuditLogger::log_api_request` with `result: "denied"` on permission
  denial

**Verification:** `cargo test -p vllm-server audit_integration` passes.

---

## Plan 37-05: Grafana Credentials to .env — SEC-05

**Goal:** `docker-compose.yml` has no hardcoded Grafana credentials.

**Files to modify:**

- `docker-compose.yml` (or `deploy/docker-compose.yml` — find canonical
  location) — replace hardcoded `GRAFANA_ADMIN_USER=admin` /
  `GRAFANA_ADMIN_PASSWORD=admin` with `${GRAFANA_ADMIN_USER:-admin}` /
  `${GRAFANA_ADMIN_PASSWORD:-admin}` references
- `.env.example` (new file) — document required keys with example values:
  ```
  GRAFANA_ADMIN_USER=admin
  GRAFANA_ADMIN_PASSWORD=change-me-in-prod
  ```

**Tests to add:**

- `scripts/check_no_hardcoded_secrets.sh` (or extend existing) — grep
  `docker-compose.yml` for known credential keys; fail if found

**Verification:** Manual — `docker compose config | grep -i password`
should show only `${...}` references.

---

## Plan 37-06: TLS Handshake Structured Error — SEC-06

**Goal:** Replace `.unwrap()` in `tls.rs:70` with structured error.

**Files to modify:**

- `crates/server/src/security/tls.rs:70`:
  - Replace `self.ca_cert_path.as_ref().unwrap()` with
    `.ok_or_else(|| TlsError::InvalidConfig("CA cert path not set despite mtls=true".into()))?`
  - Add a regression test that constructs a `TlsConfig::with_ca_cert("")`
    or removes the path and confirms `load()` returns a structured error
    (not a panic)

**Tests to add (`crates/server/src/security/tls.rs` test module):**

- `test_tls_load_panics_without_mtls_path` — currently the unwrap only
  fires when `mtls=true` but `ca_cert_path` is `None`. Add a path that
  constructs a `TlsConfig` with `mtls=true` but `ca_cert_path=None` and
  confirms `load()` returns `TlsError::InvalidConfig` (not panics).

  Implementation note: this may require adding a `with_mtls()` builder
  method or a `set_mtls(ca_path)` setter to allow the pathological state
  in test setup. The simpler approach is to construct via `TlsConfig { .. }`
  literal with `mtls: true, ca_cert_path: None` and assert
  `load().is_err()`.

**Verification:** `cargo test -p vllm-server security::tls` passes.

---

## Plan 37-07: FINAL-01 Verification

**Goal:** All 1155+ tests remain green; clippy/fmt/doc gates clean.

**Verification commands:**

```bash
just nextest              # ≥ 1155 passed
just clippy               # clean
just fmt-check            # clean
just doc-check            # clean
```

**Atomic commits:** one per plan (37-01 through 37-07).

---

## Execution Order

37-01 → 37-02 → 37-03 → 37-04 (depends on 37-01 + 37-02 wiring) → 37-05
→ 37-06 → 37-07 (final gates).

37-04 depends on 37-01 and 37-02 because the audit test exercises the
real middleware stack. Execute in the listed order; do not parallelize.
