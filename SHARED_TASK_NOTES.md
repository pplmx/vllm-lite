# Autonomous Loop — Per-Tenant Rate Limiting Enhancement

> **Project:** vLLM-lite (`vllm-lite` Rust workspace)
> **Theme:** v31.0 "Perfection & Elegance" — production-readiness follow-up
> **Skill:** `autonomous-loops` (dynamic two-prompt system)
> **Goal:** Enhance the existing per-API-key rate limiter to be production-grade:
> token-bucket algorithm, token-cost awareness, rate-limit headers, and reduced lock contention.

## Current State — What Exists

| File                               | Role                                                                   |
| ---------------------------------- | ---------------------------------------------------------------------- |
| `crates/server/src/auth.rs`        | `AuthMiddleware` + `RateLimiter` (sliding-window `Vec<Instant>`)       |
| `crates/server/src/config/auth.rs` | `AuthConfig` — global `rate_limit_requests` / `rate_limit_window_secs` |
| `crates/server/src/main.rs`        | Wires `AuthMiddleware` into axum router                                |

**Limitations:**

1. Sliding-window with `Vec<Instant>` per key — O(n) per request, memory grows within window
2. Single `RwLock` guards all keys — contention under high concurrency
3. No token-cost awareness — all requests cost 1 (but LLM requests vary by prompt/max_tokens)
4. No rate-limit response headers (`X-RateLimit-Remaining`, `Retry-After`)
5. Global rate limit settings — no per-key overrides
6. No `429` response body explaining the limit

## Implementation Plan (Incremental)

### Phase 1 — Token Bucket with Cost Awareness (TDD)

- [ ] Write tests first: `TokenBucketRateLimiter` with token-cost `check_and_consume`
- [ ] Refactor `RateLimiter` to use token bucket (`tokens: f64`, `last_refill: Instant`)
- [ ] Estimate request cost: `min(prompt_tokens + max_tokens, cap)` from the incoming request
- [ ] Add rate-limit headers to `429` response: `Retry-After`, `X-RateLimit-Remaining`
- [ ] Verify: `just ci` green (fmt, clippy, doc, test)

### Phase 2 — Per-Key Configuration

- [ ] Extend `AuthConfig` with optional `HashMap<String, RateLimitOverride>`
- [ ] `RateLimiter` looks up per-key limits at startup
- [ ] Tests for per-key override behavior
- [ ] Verify: `just ci` green

### Phase 3 — Lock Contention Reduction

- [ ] Replace single `RwLock` with sharded locks (`DashMap`-style) or per-key `Mutex`
- [ ] Benchmark: verify contention reduction (optional — if time permits)
- [ ] Verify: `just ci` green

## Progress Log

<!-- Loop iterations append here -->

## Iteration 1 — Token Bucket + Cost Awareness + Rate Limit Headers

**Status:** ✅ Complete, CI green

### Changes

- `crates/server/src/auth.rs`: Replaced sliding-window `RateLimiter` with token-bucket
    - `TokenBucket` struct: lazy refill, fractional tokens, `O(1)` per request
    - `RateLimitResult` struct: `allowed`, `remaining`, `retry_after`, `limit`
    - `RateLimiter::check_and_consume(key, cost)` — cost-aware consumption
    - `AuthError` enum (typed): `MissingHeader`, `InvalidKey`, `RateLimited`
    - `AuthMiddleware::verify_with_meta(headers, cost)` — returns full rate-limit metadata
    - `AuthMiddleware::verify(headers)` — backward-compatible wrapper (cost=1.0)
    - `auth_middleware` now adds `X-RateLimit-Remaining`, `X-RateLimit-Limit`, `Retry-After` headers
    - `estimate_request_cost(body)` — JSON body parser for prompt_tokens + max_tokens (Phase 2)
- `crates/server/tests/rate_limit_headers.rs`: 4 new integration tests

### Tests

- 25 unit tests + 4 integration tests, all pass
- 343 total lib tests pass (all pre-existing + new)
- Clippy clean on auth.rs; docs build clean

## Iteration 2 — Cost-Aware Middleware Wiring

**Status:** ✅ Complete, CI green

### Changes

- `crates/server/src/auth.rs`: `auth_middleware` now reads the request body,
  estimates token cost via `estimate_request_cost(body)`, and passes the
  cost to `verify_with_meta`. Body is reconstructed so downstream handlers
  still receive it.
- `crates/server/tests/rate_limit_headers.rs`: 2 new integration tests
    - `test_cost_aware_rate_limiting_with_large_body` — large JSON body
  (3 prompt words + max_tokens=2 = cost 5) exhausts a capacity-5 bucket
    - `test_small_body_costs_one_token` — empty body costs 1 token

### Tests

- 25 unit tests + 6 integration tests, all pass
- 343 total lib tests pass
- Clippy clean on auth.rs; docs build clean

## Iteration 3 — Per-Key Rate Limit Configuration

**Status:** ✅ Complete, CI green

### Changes

- `crates/server/src/auth.rs`: `RateLimiter` now supports per-key limits via
  `new_with_overrides()`. `AuthMiddleware::new_with_overrides()` accepts a
  `HashMap<String, (usize, u64)>` of per-key overrides.
- `crates/server/src/config/auth.rs`: Added `RateLimitOverride` struct and
  `rate_limit_overrides` field to `AuthConfig` (deserialized from YAML).
- `crates/server/src/main.rs`: Wires `app_config.auth.rate_limit_overrides`
  into `AuthMiddleware::new_with_overrides()`.

### Tests

- 29 unit tests + 6 integration tests, all pass
- 347 total lib tests pass
- Clippy clean; docs build clean

## Iteration 4 — Lock Contention Reduction (Sharded Locks)

**Status:** ✅ Complete, CI green

### Changes

- `crates/server/src/auth.rs`: `RateLimiter` already uses sharded locking
  — the bucket map is partitioned across `NUM_SHARDS = 16` independent
  `parking_lot::RwLock` instances so concurrent requests for *different*
  API keys don't contend on a single lock. The `shard_of()` function
  distributes keys via `DefaultHasher`.
- The shared config (`capacity`, `refill_rate`, `per_key_limits`) is
  read-only after construction and needs no lock.
- `check_and_consume` acquires only the shard write-lock for the
  critical section, extracting result values before releasing.

### Tests

- `test_token_bucket_separate_keys_independent` in `auth.rs` lib tests
- `test_separate_keys_have_independent_rate_limits` in `rate_limit_headers.rs`
- All 347 lib tests + 6 integration tests pass; clippy clean; docs build clean.
