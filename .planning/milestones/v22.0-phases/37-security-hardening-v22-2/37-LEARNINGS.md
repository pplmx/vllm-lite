---
phase: 37
phase_name: "Security Hardening"
project: "vllm-lite"
generated: "2026-06-27T15:17:22Z"
counts:
  decisions: 12
  lessons: 4
  patterns: 9
  surprises: 6
missing_artifacts:
  - "37-VERIFICATION.md"
  - "37-UAT.md"
---

# Phase 37 Learnings: Security Hardening

## Decisions

### Use `jsonwebtoken = "9"` crate for JWT signature verification

What was decided: Promote `jsonwebtoken` from indirect to direct dependency in
`crates/server/Cargo.toml` and use it as the single source of truth for JWT
decode/encode. `JwtValidator::validate` now delegates to
`jsonwebtoken::decode` with `DecodingKey`/`Validation` rather than hand-rolled
claim parsing.

**Rationale:** The crate already existed in the dependency tree (pulled in
indirectly), so adding it as a direct dep is zero-cost. It has audited
implementations of HMAC, RSA, and ECDSA verification and the `Validation`
builder handles `exp`/`iss`/`aud` checks out of the box.
**Source:** 37-PLAN.md, 37-CONTEXT.md

---

### Algorithm allowlist enforced before signature verification

What was decided: Accept only `HS256`, `RS256`, `RS384`, `RS512`, `ES256`,
`ES384`. Reject `alg: none` and any other algorithm with
`JwtError::UnsupportedAlgorithm` *before* invoking `decode`.

**Rationale:** Standard alg-confusion mitigation — verifying with a
hand-picked `Validation` object prevents an attacker from swapping the
algorithm in the header (e.g. presenting an RS256 token while the server is
configured with an HMAC secret, causing the server to use the public key as
the HMAC secret).
**Source:** 37-PLAN.md, 37-SUMMARY.md

---

### Static path → action mapping table for RBAC enforcement

What was decided: Add `RbacMiddleware::required_action_for_path(path)` as a
static lookup. Mappings: `/v1/models → read`, `/v1/chat/completions → execute`,
`/v1/completions → execute`, `/v1/embeddings → execute`,
`/metrics → view_metrics`, `/admin/* → manage_users`. Public endpoints
(`/health`, `/ready`) return `None`.

**Rationale:** Explicit lookup table is easy to extend, easy to test in
isolation, and avoids per-route handler annotations (which would require
touching every route).
**Source:** 37-PLAN.md, 37-SUMMARY.md

---

### Least-privilege default for unknown paths

What was decided: Unknown paths default to requiring `read` action. The `*`
wildcard for `Admin` still grants access even with the least-privilege
default.

**Rationale:** New endpoints added without an explicit RBAC mapping fail
closed (require authentication) instead of failing open (allow everyone).
This shifts the burden of remembering to register permissions onto the
endpoint author, but the cost of a forgotten mapping is "users get 403
until we add it," not "users get unauthenticated access."
**Source:** 37-SUMMARY.md

---

### Role extraction precedence: JWT claims → header fallback

What was decided: Extract `Role` from JWT claims first; fall back to the
`X-User-Role` header only when claims are absent. Anonymous is the default
when neither is present.

**Rationale:** JWT claims are authoritative (cryptographically verified) and
prevent header forgery. The header path exists only for back-compat with
existing callers that haven't migrated to claim-based roles yet.
**Source:** 37-PLAN.md, 37-CONTEXT.md

---

### Middleware ordering: audit outermost, RBAC innermost

What was decided: In the middleware stack, `AuditLogger` runs outermost (first
to see the request, last to write) and `RbacMiddleware` runs innermost
(second, can short-circuit with 403).

**Rationale:** Audit must record every request, including ones denied by
RBAC. Putting audit outermost guarantees the auth:success event fires even
when RBAC returns 403 — without this ordering, a denial would suppress the
audit trail of the auth attempt.
**Source:** 37-SUMMARY.md

---

### `tower_http::limit::RequestBodyLimitLayer` for body size cap

What was decided: Wrap the Axum router with
`RequestBodyLimitLayer::new(limit)` via `tower::ServiceBuilder`. Default 1
MiB (`1_048_576` bytes). Configurable through `ServerConfig::body_size_limit`
with builder method `with_body_size_limit(bytes)`.

**Rationale:** Body-limit middleware is a well-trodden tower pattern; using
the canonical crate keeps the implementation consistent with the rest of
the tower-http ecosystem. 1 MiB is generous for chat-completion JSON payloads
and tight enough to bound memory.
**Source:** 37-PLAN.md, 37-SUMMARY.md

---

### `${VAR:-default}` shell substitution in docker-compose

What was decided: Replace hardcoded `GF_SECURITY_ADMIN_USER=admin` /
`GF_SECURITY_ADMIN_PASSWORD=vllm-admin` with
`${GRAFANA_ADMIN_USER:-admin}` / `${GRAFANA_ADMIN_PASSWORD:-vllm-admin}`.
Add `.env.example` documenting the keys.

**Rationale:** `${VAR:-default}` keeps the compose file self-contained for
local development (defaults fire when `.env` is absent) while still
allowing production deployments to inject strong credentials via `.env`
(which is gitignored). No hardcoded credentials in the repo.
**Source:** 37-PLAN.md, 37-SUMMARY.md

---

### Replace `.unwrap()` with structured `TlsError::InvalidConfig`

What was decided: In `crates/server/src/security/tls.rs:69-72`, replace
`self.ca_cert_path.as_ref().unwrap()` with
`.ok_or_else(|| TlsError::InvalidConfig(...))?`.

**Rationale:** The constructor invariant `mtls=true ⇒ ca_cert_path=Some` is
not compiler-enforced; if a future refactor breaks the invariant, the
current code panics. A structured error surfaces the bug as an HTTP-level
failure with a diagnostic message instead of crashing the server process.
**Source:** 37-PLAN.md, 37-SUMMARY.md

---

### Regression test with `std::panic::catch_unwind`

What was decided: Add `test_tls_load_with_mtls_but_no_ca_path_returns_error`
which constructs a `TlsConfig` literal with `mtls: true, ca_cert_path: None`
and asserts `load()` returns `Err(...)`. Uses `std::panic::catch_unwind` to
make the panic-vs-error contract explicit.

**Rationale:** A bare `assert!(load().is_err())` cannot distinguish "load
returned an error" from "load panicked but the test runner swallowed it."
`catch_unwind` lets the test fail loudly if a future regression reintroduces
the panic.
**Source:** 37-PLAN.md, 37-SUMMARY.md

---

### Expose `required_action_for_path` as a public method

What was decided: Make `RbacMiddleware::required_action_for_path(path)` a
`pub fn` (not `pub(crate)`) so external callers can inspect the RBAC
mapping.

**Rationale:** Future auditing/reporting tools need to enumerate which
routes require which permissions without re-implementing the mapping.
**Source:** 37-SUMMARY.md

---

### Atomic commits, one per plan

What was decided: Each of the 7 plans (37-01 through 37-07) lands in a
single atomic commit. Plans execute sequentially with no parallelization.

**Rationale:** SEC-04 depends on SEC-01 and SEC-02 being wired in
production-style middleware; parallelizing risks a green test suite from a
half-wired stack. Atomic commits keep each plan bisectable.
**Source:** 37-PLAN.md

---

## Lessons

### Allowlist enforcement must precede signature verification

What was learned: When verifying JWTs, the algorithm allowlist must be
enforced *before* the signature check. If `jsonwebtoken::decode` is invoked
with a `Validation` that lists multiple algorithms (or `alg: none`), the
attacker can swap the algorithm in the header to downgrade verification.

**Context:** The `jsonwebtoken` crate, when given a `Validation` object
constructed with a single explicit algorithm, will reject any token whose
header disagrees. Per-token algorithm comes from the JWT header `alg`, but
the `Validation` set determines what is acceptable. Building `Validation`
with only the expected algorithm closes the alg-confusion class of attacks
without a separate pre-check.
**Source:** 37-PLAN.md, 37-SUMMARY.md

---

### Auth audit event fires even when RBAC denies

What was learned: The audit pipeline must record auth outcomes
independently of authorization outcomes. The integration test
`test_audit_emits_success_even_when_rbac_denies` asserts that
`authenticate:success` is logged for a valid JWT *and* `rbac:denied` is
logged for the subsequent permission failure.

**Context:** If audit were inner-most (after RBAC), a 403 from RBAC would
suppress the auth event and create an audit gap where successful
authentication attempts are not visible. The fix is middleware ordering,
not the audit logger itself.
**Source:** 37-SUMMARY.md

---

### Explicit public-endpoint allowlist is required for path-based RBAC

What was learned: A path-based RBAC middleware needs an explicit list of
public endpoints (those returning `None` from
`required_action_for_path`) rather than relying on absence of a mapping to
mean "allow."

**Context:** With least-privilege defaults (`unknown → read`), a missing
mapping defaults to requiring auth, which is safe but blocks legitimate
public endpoints like `/health`. The current implementation returns `None`
for `/health` and `/ready`. Any new public endpoint must be added to this
list explicitly.
**Source:** 37-SUMMARY.md

---

### Audit logger as middleware state, not global

What was learned: The `AuditLogger` is shared into the middleware stack
through Axum state (`axum::middleware::from_fn_with_state`) rather than
being a global singleton. This lets integration tests construct isolated
loggers per-test and assert against `get_events()`.

**Context:** Globals make per-test isolation awkward (must reset between
tests; tests can race on shared state). State injection via
`from_fn_with_state` is the idiomatic Axum pattern and integrates cleanly
with `tower::ServiceExt::oneshot` for testing.
**Source:** 37-SUMMARY.md

---

## Patterns

### Algorithm allowlist before signature verification

Description: When verifying any JWT (or any signed token format that carries
the algorithm in-band), check the algorithm against a small allowlist
*before* invoking the cryptographic verify function.

**When to use:** Any time you wire a JWT validator that accepts tokens with
embedded algorithm metadata. The pattern defends against
`alg: none` downgrade and HMAC-vs-RSA confusion attacks.
**Source:** 37-PLAN.md

---

### Static lookup table for path-based authorization

Description: Define a single static function that maps `&str` path to
`Option<Action>` (returning `None` for public endpoints, `Some(action)` for
protected ones). Callers iterate or look up the action before consulting
the role → permission table.

**When to use:** Authorization models where the route surface is stable
enough to enumerate (typical for REST APIs with a small number of
endpoints). Avoids per-route annotations and keeps the RBAC policy in one
readable location.
**Source:** 37-PLAN.md, 37-SUMMARY.md

---

### Layered middleware with explicit ordering invariant

Description: Compose middleware as nested layers with a documented
ordering: outermost layers (e.g., audit) always run; innermost layers
(e.g., authorization) may short-circuit. State passes inward; events
propagate outward.

**When to use:** Anywhere you need guaranteed observability of an event
class (auth attempts, request counts, errors) regardless of how the request
ultimately terminates. The pattern prevents authorization from suppressing
the audit trail.
**Source:** 37-SUMMARY.md

---

### `tower::ServiceExt::oneshot` for middleware integration tests

Description: Build the middleware stack as a `tower::Service`, then fire
requests through it with `ServiceExt::oneshot(request).await.unwrap()`
instead of binding a real socket.

**When to use:** Unit/integration testing of Axum middleware where you want
to assert response status, body, and side effects without network
plumbing. Pairs naturally with `axum::body::Body` and a small in-memory
`AuditLogger` shared via state.
**Source:** 37-PLAN.md

---

### `axum::middleware::from_fn_with_state` for stateful middleware

Description: When middleware needs shared state (logger, metrics, config),
wire it with `from_fn_with_state` and pass the state through the router
builder (`Router::new().route(...).layer(from_fn_with_state(state.clone(),
middleware_fn))`).

**When to use:** Any middleware that depends on a per-app resource
(`AuditLogger`, rate limiter, feature flag store). Cleaner than globals and
trivially testable.
**Source:** 37-SUMMARY.md

---

### `std::panic::catch_unwind` for panic-vs-error regression tests

Description: When migrating from `.unwrap()` / `.expect()` to structured
errors, write a regression test that explicitly catches panics and asserts
the function returned `Err`.

**When to use:** Any `.unwrap()` removal where the panic represents a real
bug being fixed (invariant violation, malformed config). A bare
`assert!(fn().is_err())` cannot tell the difference between "returned an
error" and "panicked and was swallowed."
**Source:** 37-PLAN.md, 37-SUMMARY.md

---

### `${VAR:-default}` shell substitution for env-injected config

Description: In `docker-compose.yml` (and similar env-substitution
contexts), write `${VAR_NAME:-default_value}` so the file works with or
without a `.env` file.

**When to use:** Container manifests where the same file must work for
local dev (defaults) and production (overrides via `.env` or `--env-file`).
Avoids the two-file problem of "compose.yml" vs "compose.prod.yml."
**Source:** 37-SUMMARY.md

---

### `tower_http::limit::RequestBodyLimitLayer` via `ServiceBuilder`

Description: Apply a request-body size cap by stacking
`RequestBodyLimitLayer::new(limit_bytes)` onto a `ServiceBuilder` and
applying it to the router. The default is configurable through
`ServerConfig::body_size_limit`.

**When to use:** Any HTTP server that accepts user-supplied JSON or file
uploads. Pairs with a `413 Payload Too Large` response code that tower-http
emits automatically.
**Source:** 37-PLAN.md, 37-SUMMARY.md

---

### Builder method for security-related config fields

Description: Add a `with_<field>(value)` builder method to the
`ServerConfig` builder for any new security knob (body limit, JWT issuer
list, allowed CORS origins, etc.).

**When to use:** New tunable knobs that downstream code (especially tests)
will want to override. Keeps the constructor backward-compatible while
making configuration explicit.
**Source:** 37-PLAN.md

---

## Surprises

### Previous JWT "validation" parsed but never verified signatures

What was surprising: The pre-Phase-37 `JwtValidator::validate` accepted
*any* 3-part base64 string as a valid JWT. It decoded the header and
payload (so claims like `exp` and `iss` were checkable) but did not verify
the cryptographic signature at all. Tokens issued by anyone — not just the
configured issuer — would pass validation.

**Impact:** Critical security hole. Any production deployment relying on
JWT auth before Phase 37 was effectively unauthenticated. The fix in
SEC-01 is the single most impactful change in v22.0 from a security
standpoint. Backward-compat note: existing callers that signed tokens
correctly see no behavior change; only callers that previously issued
unsigned tokens break (and that's the intended fix).
**Source:** 37-CONTEXT.md, 37-SUMMARY.md

---

### `RbacMiddleware` was a no-op pass-through

What was surprising: `rbac_middleware` at `crates/server/src/security/rbac.rs:96-98`
was a free function that accepted every request and forwarded it
downstream. The `Role` enum and `check_permission(role, action)` helper
existed and were tested in isolation, but they were never invoked in the
HTTP request path.

**Impact:** A second critical security gap. Even authenticated users with
the `Anonymous` role could hit `/admin/*` endpoints. SEC-02 wires the
existing primitives into the request pipeline, turning the
already-implemented policy into an enforced one with no new policy logic.
**Source:** 37-CONTEXT.md, 37-SUMMARY.md

---

### `tls.rs:70` `.unwrap()` could panic the server at runtime

What was surprising: The mTLS path in `TlsConfig::load` had
`self.ca_cert_path.as_ref().unwrap()`. The constructor does not enforce
`mtls=true ⇒ ca_cert_path=Some`, so any code that constructed a
`TlsConfig` programmatically with mTLS enabled but no CA path would crash
the server process at first request.

**Impact:** Denial-of-service vector for any operator who configured mTLS
incorrectly (typo in config, missing environment variable, partial
migration). SEC-06 turns the panic into a structured error and the
regression test makes the contract explicit.
**Source:** 37-CONTEXT.md, 37-SUMMARY.md

---

### `docker-compose.yml` shipped with hardcoded Grafana password

What was surprising: The committed `docker-compose.yml` contained
`GF_SECURITY_ADMIN_USER=admin` / `GF_SECURITY_ADMIN_PASSWORD=vllm-admin`
in plaintext. Anyone with read access to the repository had production
credentials for any deployment that ran the file unmodified.

**Impact:** Credential exposure. Even if local-only deployments are not
production, the leak trains bad habits and trips credential-scanning tools
on first commit. SEC-05 replaces with `${VAR:-default}` so production
deployments must override via `.env`.
**Source:** 37-CONTEXT.md, 37-SUMMARY.md

---

### Test count delta was tighter than initial estimate

What was surprising: The 6 security deliverables added **+24 net tests**
(not the larger number implied by the per-section count of 30+ tests).
Some tests in the same module were consolidated, and a few expected tests
collapsed into single test functions covering multiple cases (e.g., the
RSA-without-PEM and ECDSA-without-PEM cases share one assertion path).

**Impact:** Test count grew from 1155 → 1179. This is consistent with the
project's preference for "tight" tests that assert one invariant per test
function, with related cases grouped where they share setup.
**Source:** 37-SUMMARY.md

---

### Per-plan atomic commits were mandatory despite internal dependencies

What was surprising: SEC-04 (audit integration) depends on SEC-01 (JWT
verification) and SEC-02 (RBAC wiring) being live. Despite the dependency,
each plan still lands as an atomic commit. The audit test in SEC-04 builds
its own middleware stack from scratch (`tower::ServiceExt::oneshot` against
a test Axum app) rather than depending on the production router, so the
SEC-04 commit is shippable even if it landed first.

**Impact:** Faster bisection when a regression is found. Each commit
corresponds to exactly one SEC requirement, so `git log --oneline
--grep=SEC-01` cleanly tracks the lifecycle of a single requirement.
**Source:** 37-PLAN.md, 37-SUMMARY.md
