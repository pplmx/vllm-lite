//! Authentication: API-key lookup table + JWT verification.
//!
//! Wraps the [`AuthMiddleware`] which selects between API-key and JWT
//! verification at startup based on `AppConfig.auth.method`. Mounted as
//! an axum middleware before every request reaches the router.
//!
//! Rate limiting uses a **token bucket** algorithm (see `TokenBucket`).
//! Each API key gets a bucket with `max_requests` capacity that refills
//! at `max_requests / window_secs` tokens per second. This is more
//! memory-efficient than a sliding window (`O(1)` per request vs
//! `O(n)`) and provides precise `Retry-After` values for clients.
#![allow(clippy::module_name_repetitions)]
use axum::{
    extract::Request,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::Response,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Header name for rate-limit metadata sent on every response.
const HEADER_RATE_LIMIT_REMAINING: HeaderName = HeaderName::from_static("x-ratelimit-remaining");
const HEADER_RATE_LIMIT_LIMIT: HeaderName = HeaderName::from_static("x-ratelimit-limit");
const HEADER_RETRY_AFTER: HeaderName = HeaderName::from_static("retry-after");

/// A token bucket for one key.
///
/// Holds `tokens` (current count, up to `capacity`) and `last_refill`
/// (when tokens were last replenished). Refill is lazy: we compute
/// elapsed tokens on each `consume` call rather than spawning a task.
#[derive(Debug)]
pub(crate) struct TokenBucket {
    /// Current token count (may be fractional due to partial refills).
    tokens: f64,
    /// Wall-clock time of the last refill.
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a full bucket at capacity.
    fn new(capacity: f64) -> Self {
        Self {
            tokens: capacity,
            last_refill: Instant::now(),
        }
    }

    /// Attempt to consume `cost` tokens after lazy refilling.
    ///
    /// Returns `Some(remaining)` if enough tokens were available (and
    /// deducts them), or `None` if the bucket is depleted. When
    /// `None`, the caller can use [`Self::wait_for`] to compute
    /// how long to wait for enough tokens.
    fn consume(&mut self, cost: f64, capacity: f64, refill_rate: f64) -> Option<f64> {
        // Lazy refill: add tokens based on elapsed wall-clock time.
        // When refill_rate is infinite (window_secs == 0) the bucket
        // refills instantly — tokens snap to capacity.
        if refill_rate.is_infinite() {
            self.tokens = capacity;
        } else {
            let elapsed = self.last_refill.elapsed().as_secs_f64();
            self.tokens = elapsed.mul_add(refill_rate, self.tokens).min(capacity);
        }
        self.last_refill = Instant::now();

        if self.tokens >= cost {
            self.tokens -= cost;
            Some(self.tokens)
        } else {
            None
        }
    }

    /// Compute how many seconds the caller must wait to accumulate
    /// at least `cost` tokens.
    #[must_use]
    fn wait_for(&self, cost: f64, refill_rate: f64) -> Duration {
        if refill_rate.is_infinite() || self.tokens >= cost {
            return Duration::ZERO;
        }
        let needed = cost - self.tokens;
        let secs = needed / refill_rate;
        // Clamp to avoid absurdly tiny durations that round to 0s.
        Duration::from_secs_f64(secs.max(0.0))
    }
}

/// Result of a rate-limit check.
#[derive(Debug)]
pub(crate) struct RateLimitResult {
    /// Whether the request was allowed.
    pub allowed: bool,
    /// Tokens remaining in the bucket after the check.
    pub remaining: f64,
    /// How long to wait before retrying (for `Retry-After` header).
    pub retry_after: Option<Duration>,
    /// The configured bucket capacity (limit).
    pub limit: f64,
}

/// Number of independent shards for the rate limiter.
///
/// Each shard has its own lock, so requests for different keys can
/// be rate-limited concurrently without contention.
const NUM_SHARDS: usize = 16;

/// `RateLimiter`. See the type definition for fields and behavior.
///
/// Uses a token bucket per key. The bucket capacity is `max_requests`
/// and refills at `max_requests / window_secs` tokens per second.
///
/// **Sharded locking:** the bucket map is partitioned across
/// [`NUM_SHARDS`] independent `parking_lot::RwLock` instances so that
/// concurrent requests for *different* API keys don't contend on a
/// single lock. The shared config (`capacity`, `refill_rate`,
/// `per_key_limits`) is read-only after construction and needs no
/// lock.
#[derive(Debug)]
pub(crate) struct RateLimiter {
    /// Global default capacity (tokens).
    capacity: f64,
    /// Global default tokens per second. Infinite when `window_secs == 0`.
    refill_rate: f64,
    /// Per-key overrides: `key → (capacity, refill_rate)`.
    /// When a key is absent here, the global defaults are used.
    per_key_limits: HashMap<String, (f64, f64)>,
    /// Sharded bucket maps — one lock per shard.
    shards: Vec<RwLock<HashMap<String, TokenBucket>>>,
}

/// Hash a key to its shard index.
fn shard_of(key: &str) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    (hasher.finish() as usize) % NUM_SHARDS
}

impl RateLimiter {
    fn new(max_requests: usize, window_secs: u64) -> Self {
        let capacity = f64::from(u32::try_from(max_requests).unwrap_or(u32::MAX));
        let refill_rate = if window_secs == 0 {
            f64::INFINITY
        } else {
            capacity / window_secs as f64
        };
        Self {
            capacity,
            refill_rate,
            per_key_limits: HashMap::new(),
            shards: (0..NUM_SHARDS)
                .map(|_| RwLock::new(HashMap::new()))
                .collect(),
        }
    }

    /// Create a rate limiter with per-key overrides.
    ///
    /// `overrides` maps an API key to its own `(max_requests, window_secs)`.
    /// Keys not in the map use the global defaults.
    fn new_with_overrides(
        max_requests: usize,
        window_secs: u64,
        overrides: HashMap<String, (usize, u64)>,
    ) -> Self {
        let capacity = f64::from(u32::try_from(max_requests).unwrap_or(u32::MAX));
        let global_refill_rate = if window_secs == 0 {
            f64::INFINITY
        } else {
            capacity / window_secs as f64
        };
        let per_key_limits = overrides
            .into_iter()
            .map(|(key, (max, win))| {
                let cap = f64::from(u32::try_from(max).unwrap_or(u32::MAX));
                let rate = if win == 0 {
                    f64::INFINITY
                } else {
                    cap / win as f64
                };
                (key, (cap, rate))
            })
            .collect();
        Self {
            capacity,
            refill_rate: global_refill_rate,
            per_key_limits,
            shards: (0..NUM_SHARDS)
                .map(|_| RwLock::new(HashMap::new()))
                .collect(),
        }
    }

    /// Check whether `key` can proceed, deducting `cost` tokens.
    ///
    /// Returns a [`RateLimitResult`] with remaining tokens and (if
    /// denied) the `retry_after` duration.
    ///
    /// Locks only the shard responsible for `key`, so concurrent checks
    /// for different keys proceed without contention.
    #[allow(clippy::significant_drop_tightening)]
    fn check_and_consume(&self, key: &str, cost: f64) -> RateLimitResult {
        let (bucket_capacity, bucket_refill_rate) = self
            .per_key_limits
            .get(key)
            .copied()
            .unwrap_or((self.capacity, self.refill_rate));

        let shard_idx = shard_of(key);
        // Acquire the shard write-lock only for the critical section — extract
        // the result values (all Copy: f64 + Option<Duration>) before
        // releasing so the lock isn't held during RateLimitResult
        // construction, reducing contention across API keys on the same shard.
        let (allowed, remaining, retry_after) = {
            let mut buckets = self.shards[shard_idx].write();
            let bucket = buckets
                .entry(key.to_string())
                .or_insert_with(|| TokenBucket::new(bucket_capacity));

            match bucket.consume(cost, bucket_capacity, bucket_refill_rate) {
                Some(remaining) => (true, remaining, None),
                None => (
                    false,
                    bucket.tokens,
                    Some(bucket.wait_for(cost, bucket_refill_rate)),
                ),
            }
        };

        RateLimitResult {
            allowed,
            remaining,
            retry_after,
            limit: bucket_capacity,
        }
    }

    /// Backward-compatible wrapper: check with a default cost of 1.0.
    ///
    /// Only used by tests; production code calls [`Self::check_and_consume`]
    /// directly (via [`AuthMiddleware::verify_with_meta`]).
    #[allow(dead_code)]
    fn check_rate_limit(&self, key: &str) -> bool {
        self.check_and_consume(key, 1.0).allowed
    }
}

/// Error from [`AuthMiddleware::verify_with_meta`].
// NOTE: `Eq` is intentionally omitted — the `RateLimited { limit: f64 }`
// variant means `AuthError` cannot satisfy `Eq`'s reflexivity guarantee
// (f64::NAN != f64::NAN). `PartialEq` is sufficient for all current use
// (pattern matching via `matches!`, not `==` equality).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum AuthError {
    /// No `Authorization: Bearer <key>` header present.
    MissingHeader,
    /// API key not in the configured list.
    InvalidKey,
    /// Rate limit exceeded.
    ///
    /// `retry_after_secs` is how long to wait; `limit` is the
    /// per-key bucket capacity (may differ from the global default
    /// when overrides are configured).
    RateLimited { retry_after_secs: u64, limit: f64 },
}

impl From<AuthError> for StatusCode {
    fn from(err: AuthError) -> Self {
        use AuthError::{InvalidKey, MissingHeader, RateLimited};
        match err {
            MissingHeader | InvalidKey => Self::UNAUTHORIZED,
            RateLimited { .. } => Self::TOO_MANY_REQUESTS,
        }
    }
}

#[derive(Debug)]
/// `AuthMiddleware`. See the type definition for fields and behavior.
pub struct AuthMiddleware {
    api_keys: Arc<Vec<String>>,
    rate_limiter: Arc<RateLimiter>,
}

impl AuthMiddleware {
    /// Create an auth + rate-limiting middleware for the given API keys.
    #[must_use]
    pub fn new(api_keys: Vec<String>, max_requests: usize, window_secs: u64) -> Self {
        Self {
            api_keys: Arc::new(api_keys),
            rate_limiter: Arc::new(RateLimiter::new(max_requests, window_secs)),
        }
    }

    /// Create an auth + rate-limiting middleware with per-key overrides.
    ///
    /// `overrides` maps an API key to its own `(max_requests, window_secs)`,
    /// allowing privileged keys to have higher or lower limits than the
    /// global default.
    #[must_use]
    pub fn new_with_overrides(
        api_keys: Vec<String>,
        max_requests: usize,
        window_secs: u64,
        overrides: HashMap<String, (usize, u64)>,
    ) -> Self {
        Self {
            api_keys: Arc::new(api_keys),
            rate_limiter: Arc::new(RateLimiter::new_with_overrides(
                max_requests,
                window_secs,
                overrides,
            )),
        }
    }

    /// SEC-01 (technical due diligence): read-only access to the
    /// configured API keys. Debug and shutdown endpoints use this to
    /// verify a `Bearer` token against the same list the global auth
    /// middleware consults, so the admin gate cannot drift from the
    /// regular auth gate.
    ///
    /// Returns the list in declaration order. Callers MUST treat this
    /// as read-only — keys are sensitive material and should not be
    /// cloned, logged, or persisted by the caller.
    #[must_use]
    pub fn api_keys(&self) -> &[String] {
        &self.api_keys
    }

    /// Verify a request's API key and consume one rate-limit token.
    ///
    /// Returns the authenticated key on success, or a [`StatusCode`]
    /// error (401 / 429).
    ///
    /// # Errors
    ///
    /// Returns `Err(StatusCode::UNAUTHORIZED)` if the key is missing or
    /// invalid, or `Err(StatusCode::TOO_MANY_REQUESTS)` if the rate
    /// limit has been exceeded.
    pub fn verify(&self, headers: &HeaderMap) -> Result<String, StatusCode> {
        self.verify_with_meta(headers, 1.0)
            .map(|decision| decision.0)
            .map_err(Into::into)
    }

    /// Verify a request's API key and consume `cost` rate-limit tokens.
    ///
    /// Returns the authenticated key and a [`RateLimitResult`] on
    /// success, or an [`AuthError`] on failure.
    ///
    /// # Errors
    ///
    /// Returns `Err(AuthError::MissingHeader)` if no `Authorization`
    /// header is present, `Err(AuthError::InvalidKey)` if the key is
    /// not configured, or `Err(AuthError::RateLimited)` if the bucket
    /// is depleted.
    pub(crate) fn verify_with_meta(
        &self,
        headers: &HeaderMap,
        cost: f64,
    ) -> Result<(String, RateLimitResult), AuthError> {
        let auth_header = headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .ok_or(AuthError::MissingHeader)?;

        let api_key = auth_header
            .strip_prefix("Bearer ")
            .ok_or(AuthError::MissingHeader)?;

        if !self.api_keys.is_empty() && !self.api_keys.contains(&api_key.to_string()) {
            return Err(AuthError::InvalidKey);
        }

        // Sharded rate limiter: locks only the shard for this key.
        let result = self.rate_limiter.check_and_consume(api_key, cost);

        if result.allowed {
            Ok((api_key.to_string(), result))
        } else {
            let retry_after_secs = result.retry_after.map_or(1, |d| d.as_secs().max(1));
            Err(AuthError::RateLimited {
                retry_after_secs,
                limit: result.limit,
            })
        }
    }
}

/// Opaque marker: an authenticated request, with a stable,
/// **non-secret** user identifier that the audit layer can log.
///
/// We store only a truncated prefix of the api key — the full key
/// must never appear in audit events because audit logs are
/// exported to `/debug/audit` (and, in production, to an external
/// SIEM). Logging the full key would leak credentials to anyone
/// who can read the audit trail.
///
/// Inserted into request extensions by [`auth_middleware`] on a
/// successful verify so downstream layers (audit, structured
/// logs) can read the user without re-parsing the
/// `Authorization` header.
#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedUser(pub String);

/// Compute a short, stable identifier for an authenticated user
/// from the api key. First 8 chars is enough to disambiguate keys
/// in normal use without exposing the credential itself.
#[must_use]
pub(crate) fn user_id_from_key(api_key: &str) -> String {
    let prefix: String = api_key.chars().take(8).collect();
    format!("key:{prefix}")
}

/// Run the operation (see signature for params and return type).
/// # Panics
///
/// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
pub async fn auth_middleware(
    auth: axum::extract::State<Arc<AuthMiddleware>>,
    request: Request,
    next: Next,
) -> Response {
    // Read the body to estimate token cost before rate limiting.
    // The body is reconstructed so downstream handlers still see it.
    let (parts, body) = request.into_parts();
    let body_bytes = axum::body::to_bytes(body, 1 << 20)
        .await
        .unwrap_or_default();

    let cost = if body_bytes.is_empty() {
        1.0
    } else {
        let body_str = String::from_utf8_lossy(&body_bytes);
        estimate_request_cost(&body_str)
    };

    let mut request = Request::from_parts(parts, axum::body::Body::from(body_bytes));

    match auth.verify_with_meta(request.headers(), cost) {
        Ok((api_key, rate_result)) => {
            // Stamp the user id on the request so the audit
            // middleware (and any future per-user handler logic)
            // can read it from extensions without re-parsing the
            // Authorization header.
            request
                .extensions_mut()
                .insert(AuthenticatedUser(user_id_from_key(&api_key)));

            // Add rate-limit headers to the downstream response.
            // `.max(0.0)` before the cast is defensive: `remaining`/`limit`
            // are always ≥ 0 in practice, but a negative or NaN f64 would
            // wrap to a huge u64 via `as`, producing a misleading header.
            let remaining = rate_result.remaining.round().max(0.0) as u64;
            let limit = rate_result.limit.round().max(0.0) as u64;
            let mut response = next.run(request).await;
            response
                .headers_mut()
                .insert(HEADER_RATE_LIMIT_REMAINING, HeaderValue::from(remaining));
            response
                .headers_mut()
                .insert(HEADER_RATE_LIMIT_LIMIT, HeaderValue::from(limit));
            response
        }
        Err(AuthError::RateLimited {
            retry_after_secs,
            limit,
        }) => {
            // Rate-limited: include Retry-After + rate-limit headers.
            // The limit reflects the per-key bucket capacity (may differ
            // from the global default when overrides are configured).
            Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header(HEADER_RETRY_AFTER, HeaderValue::from(retry_after_secs))
                .header(HEADER_RATE_LIMIT_REMAINING, HeaderValue::from(0))
                .header(
                    HEADER_RATE_LIMIT_LIMIT,
                    HeaderValue::from(limit.round().max(0.0) as u64),
                )
                // invariant: a `Response` with an empty body cannot fail to build.
                .body("".into())
                .unwrap()
        }
        Err(_) => {
            // Unauthorized / missing header.
            Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                // invariant: a `Response` with an empty body cannot fail to build.
                .body("".into())
                .unwrap()
        }
    }
}

/// Estimate the token cost of an LLM inference request from its raw
/// JSON body.
///
/// For completions, cost = `min(estimated_prompt_tokens + max_tokens, cap)`.
/// For chat, cost = `min(estimated_prompt_tokens + max_tokens, cap)`.
/// Prompt tokens are approximated by counting whitespace-separated words
/// in the prompt / message contents (±3x error vs. a real BPE tokenizer —
/// sufficient for rate-limiting purposes where exact counts are not
/// required).
///
/// Unknown payload shapes or parse failures default to a cost of `1.0`
/// so that unrecognised requests are not silently free.
#[must_use]
pub(crate) fn estimate_request_cost(body: &str) -> f64 {
    use serde_json::Value;

    let Ok(json) = serde_json::from_str::<Value>(body) else {
        return 1.0;
    };

    // Only proceed if the body looks like a completions or chat request.
    // Unknown shapes default to 1.0 cost.
    let prompt_tokens = if let Some(messages) = json.get("messages").and_then(Value::as_array) {
        // Chat: sum content lengths across all messages.
        messages
            .iter()
            .filter_map(|m| m.get("content").and_then(Value::as_str))
            .map(|s| s.split_whitespace().count() as f64)
            .sum()
    } else if let Some(prompt) = json.get("prompt") {
        // Completions: prompt may be a string, array of strings, or
        // array of pre-tokenized integer IDs.
        match prompt {
            Value::String(s) => s.split_whitespace().count() as f64,
            Value::Array(arr) => {
                // If every element is a string, sum word counts across
                // all strings (each is a separate prompt in batch mode).
                // Otherwise (integers, arrays, mixed), fall back to
                // `arr.len()` for pre-tokenized or opaque content.
                let all_strings = arr.iter().all(serde_json::Value::is_string);
                if all_strings {
                    arr.iter()
                        .filter_map(|e| e.as_str())
                        .map(|s| s.split_whitespace().count() as f64)
                        .sum()
                } else {
                    arr.len() as f64
                }
            }
            _ => return 1.0,
        }
    } else {
        return 1.0;
    };

    // `max_tokens` is not validated here (it's validated later by
    // `validate_completion_request_fields` / `validate_chat_request_fields`),
    // but negative or zero values would *reduce* the estimated cost and
    // let an attacker under-pay their rate-limit budget. Filter to
    // positive — consistent with how `n` / `best_of` are handled above.
    let max_tokens: f64 = json
        .get("max_tokens")
        .and_then(Value::as_i64)
        .filter(|&m| m > 0)
        .map_or(100.0, |m| m as f64);

    // For completions with `n > 1` or `best_of > 1`, each candidate generates
    // `max_tokens` output tokens — multiply the cost so rate limits can't be
    // bypassed by sending a single request with a large `n` or `best_of`
    // value. The validator enforces `best_of` and `n` are not both > 1, so
    // `max(n, best_of)` is the correct compute multiplier.
    let n: f64 = json
        .get("n")
        .and_then(Value::as_i64)
        .filter(|&n| n > 0)
        .map_or(1.0, |n| n as f64);
    let best_of: f64 = json
        .get("best_of")
        .and_then(Value::as_i64)
        .filter(|&b| b > 0)
        .map_or(1.0, |b| b as f64);
    let multiplier = n.max(best_of);

    // Cost = (prompt tokens + max_tokens) * multiplier, clamped to [1.0, 100_000].
    ((prompt_tokens + max_tokens) * multiplier).clamp(1.0, 100_000.0)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;
    use axum::http::header::AUTHORIZATION;
    use tokio::time::{Duration, sleep};

    // ------------------------------------------------------------------
    // TokenBucket unit tests
    // ------------------------------------------------------------------

    #[test]
    fn test_token_bucket_full_at_start() {
        let limiter = RateLimiter::new(3, 60);
        let result = limiter.check_and_consume("key1", 1.0);
        assert!(result.allowed);
        assert_eq!(result.remaining, 2.0);
    }

    #[test]
    fn test_token_bucket_blocks_over_capacity() {
        let limiter = RateLimiter::new(2, 60);
        assert!(limiter.check_rate_limit("key1"));
        assert!(limiter.check_rate_limit("key1"));
        let result = limiter.check_and_consume("key1", 1.0);
        assert!(!result.allowed);
        assert!(result.retry_after.is_some());
    }

    #[test]
    fn test_token_bucket_separate_keys_independent() {
        let limiter = RateLimiter::new(1, 60);
        assert!(limiter.check_rate_limit("key1"));
        // key1 exhausted, but key2 should be fine
        assert!(limiter.check_rate_limit("key2"));
        // key1 still blocked
        assert!(!limiter.check_rate_limit("key1"));
    }

    #[test]
    fn test_token_bucket_cost_deducts_proportionally() {
        let limiter = RateLimiter::new(10, 60);
        // Consume 3 tokens in one call
        let result = limiter.check_and_consume("key1", 3.0);
        assert!(result.allowed);
        assert!((result.remaining - 7.0).abs() < 1e-6);
        // 7 tokens should still be enough for a cost-5 request
        let result = limiter.check_and_consume("key1", 5.0);
        assert!(result.allowed);
        // Allow tiny floating-point drift from lazy refill between calls.
        assert!((result.remaining - 2.0).abs() < 1e-3);
        // 3 tokens should NOT be enough
        let result = limiter.check_and_consume("key1", 3.0);
        assert!(!result.allowed);
    }

    #[tokio::test]
    async fn test_token_bucket_zero_window_refills_immediately() {
        let limiter = RateLimiter::new(1, 0);
        assert!(limiter.check_rate_limit("key1"));
        // With window_secs=0, refill_rate is infinite → bucket refills
        sleep(Duration::from_millis(10)).await;
        assert!(limiter.check_rate_limit("key1"));
    }

    #[test]
    fn test_token_bucket_retry_after_is_deterministic() {
        let limiter = RateLimiter::new(4, 60);
        // Capacity = 4, refill_rate = 4/60 ≈ 0.0667 tokens/sec
        assert!(limiter.check_and_consume("key1", 4.0).allowed);
        // Bucket is now empty, cost=1
        let result = limiter.check_and_consume("key1", 1.0);
        assert!(!result.allowed);
        // retry_after = 1 / (4/60) = 15s → we check it's Some and > 0
        let retry = result.retry_after.expect("retry_after should be set");
        assert!(retry >= Duration::from_secs(14));
        assert!(retry <= Duration::from_secs(15));
    }

    #[test]
    fn test_token_bucket_result_includes_limit() {
        let limiter = RateLimiter::new(100, 60);
        let result = limiter.check_and_consume("key1", 5.0);
        assert_eq!(result.limit, 100.0);
    }

    #[test]
    fn test_per_key_limit_reflects_override_capacity() {
        let mut overrides = HashMap::new();
        overrides.insert("premium".to_string(), (5, 60));
        let limiter = RateLimiter::new_with_overrides(10, 60, overrides);

        // The global default capacity is 10, but "premium" has an override of 5.
        // The `limit` field in the result should reflect the per-key capacity.
        let result = limiter.check_and_consume("premium", 1.0);
        assert_eq!(result.limit, 5.0);
        assert!(result.allowed);

        // A key without an override should report the global default.
        let result = limiter.check_and_consume("standard", 1.0);
        assert_eq!(result.limit, 10.0);
    }

    // ------------------------------------------------------------------
    // RateLimitResult / AuthError tests
    // ------------------------------------------------------------------

    #[test]
    fn test_auth_error_converts_to_status_code() {
        assert_eq!(
            StatusCode::from(AuthError::MissingHeader),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            StatusCode::from(AuthError::InvalidKey),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            StatusCode::from(AuthError::RateLimited {
                retry_after_secs: 3,
                limit: 10.0,
            }),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    // ------------------------------------------------------------------
    // AuthMiddleware tests (backward-compatible verify + new verify_with_meta)
    // ------------------------------------------------------------------

    #[test]
    fn test_verify_with_meta_returns_rate_result() {
        let auth = AuthMiddleware::new(vec!["test_key".to_string()], 10, 60);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer test_key".parse().unwrap());
        let (key, result) = auth.verify_with_meta(&headers, 2.0).unwrap();
        assert_eq!(key, "test_key");
        assert!(result.allowed);
        // 10 capacity - 2 cost = 8 remaining
        assert_eq!(result.remaining, 8.0);
    }

    #[test]
    fn test_verify_with_meta_rate_limited_error() {
        let auth = AuthMiddleware::new(vec!["test_key".to_string()], 2, 60);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer test_key".parse().unwrap());

        auth.verify(&headers).unwrap();
        auth.verify(&headers).unwrap();
        let err = auth.verify_with_meta(&headers, 1.0).unwrap_err();
        assert!(matches!(
            err,
            AuthError::RateLimited { retry_after_secs, .. } if retry_after_secs >= 1
        ));
    }

    #[test]
    fn test_verify_with_meta_missing_header() {
        let auth = AuthMiddleware::new(vec!["test_key".to_string()], 10, 60);
        let headers = HeaderMap::new();
        let err = auth.verify_with_meta(&headers, 1.0).unwrap_err();
        assert_eq!(err, AuthError::MissingHeader);
    }

    #[test]
    fn test_verify_with_meta_invalid_key() {
        let auth = AuthMiddleware::new(vec!["test_key".to_string()], 10, 60);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer wrong_key".parse().unwrap());
        let err = auth.verify_with_meta(&headers, 1.0).unwrap_err();
        assert_eq!(err, AuthError::InvalidKey);
    }

    #[test]
    fn test_auth_middleware_valid_key() {
        let auth = AuthMiddleware::new(vec!["test_key".to_string()], 10, 60);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer test_key".parse().unwrap());
        let result = auth.verify(&headers);
        assert!(result.is_ok());
    }

    #[test]
    fn test_auth_middleware_invalid_key() {
        let auth = AuthMiddleware::new(vec!["test_key".to_string()], 10, 60);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer wrong_key".parse().unwrap());
        let result = auth.verify(&headers);
        assert_eq!(result.unwrap_err(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_auth_middleware_no_key() {
        let auth = AuthMiddleware::new(vec!["test_key".to_string()], 10, 60);
        let headers = HeaderMap::new();
        let result = auth.verify(&headers);
        assert_eq!(result.unwrap_err(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_auth_middleware_no_keys_allow_all() {
        let auth = AuthMiddleware::new(vec![], 10, 60);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer any_key".parse().unwrap());
        let result = auth.verify(&headers);
        assert!(result.is_ok());
    }

    #[test]
    fn test_auth_middleware_rate_limit_exceeded() {
        let auth = AuthMiddleware::new(vec!["test_key".to_string()], 2, 60);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer test_key".parse().unwrap());

        assert!(auth.verify(&headers).is_ok());
        assert!(auth.verify(&headers).is_ok());
        assert_eq!(
            auth.verify(&headers).unwrap_err(),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    #[test]
    fn test_auth_middleware_missing_bearer_prefix() {
        let auth = AuthMiddleware::new(vec!["test_key".to_string()], 10, 60);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "test_key".parse().unwrap());
        let result = auth.verify(&headers);
        assert_eq!(result.unwrap_err(), StatusCode::UNAUTHORIZED);
    }

    // ------------------------------------------------------------------
    // estimate_request_cost tests
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_cost_completions_string_prompt() {
        let body = r#"{"prompt": "hello world foo bar", "max_tokens": 50}"#;
        let cost = estimate_request_cost(body);
        // 4 words + 50 max_tokens = 54
        assert_eq!(cost, 54.0);
    }

    #[test]
    fn test_estimate_cost_completions_default_max_tokens() {
        let body = r#"{"prompt": "hello"}"#;
        let cost = estimate_request_cost(body);
        // 1 word + 100 (default) = 101
        assert_eq!(cost, 101.0);
    }

    #[test]
    fn test_estimate_cost_completions_token_array() {
        let body = r#"{"prompt": [1, 2, 3, 4, 5], "max_tokens": 10}"#;
        let cost = estimate_request_cost(body);
        // 5 tokens + 10 max_tokens = 15
        assert_eq!(cost, 15.0);
    }

    #[test]
    fn test_estimate_cost_chat_messages() {
        let body = r#"{"messages": [{"role": "user", "content": "hello world"}, {"role": "assistant", "content": "foo bar baz"}], "max_tokens": 20}"#;
        let cost = estimate_request_cost(body);
        // 2 + 3 + 20 = 25
        assert_eq!(cost, 25.0);
    }

    #[test]
    fn test_estimate_cost_invalid_json_defaults_to_one() {
        let cost = estimate_request_cost("not json");
        assert_eq!(cost, 1.0);
    }

    #[test]
    fn test_estimate_cost_clamped_to_max() {
        let prompt = "word ".repeat(60_000);
        let body = format!(r#"{{"prompt": "{prompt}", "max_tokens": 50000}}"#);
        let cost = estimate_request_cost(&body);
        assert_eq!(cost, 100_000.0);
    }

    #[test]
    fn test_estimate_cost_negative_max_tokens_defaults_to_100() {
        // A negative max_tokens would reduce the estimated cost and
        // let an attacker under-pay their rate-limit budget. It must
        // fall back to the default (100) — same treatment as n=0 /
        // best_of=0 above.
        let body = r#"{"prompt": "hello world", "max_tokens": -1000}"#;
        let cost = estimate_request_cost(body);
        // 2 words + 100 (default) = 102
        assert_eq!(cost, 102.0);
    }

    #[test]
    fn test_estimate_cost_zero_max_tokens_defaults_to_100() {
        // max_tokens = 0 is also treated as 100 (filtered as non-positive).
        let body = r#"{"prompt": "hello", "max_tokens": 0}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 101.0);
    }

    #[test]
    fn test_estimate_cost_unknown_shape_defaults_to_one() {
        let body = r#"{"foo": "bar"}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 1.0);
    }

    #[test]
    fn test_estimate_cost_n_multiplies_cost() {
        // n=3 should triple the cost: (4 words + 10 max_tokens) * 3 = 42
        let body = r#"{"prompt": "hello world foo bar", "max_tokens": 10, "n": 3}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 42.0);
    }

    #[test]
    fn test_estimate_cost_n_defaults_to_one() {
        // Without n, cost is just prompt tokens + max_tokens.
        let body = r#"{"prompt": "hello", "max_tokens": 100}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 101.0);
    }

    #[test]
    fn test_estimate_cost_n_zero_defaults_to_one() {
        // n=0 is treated as n=1 (filter excludes non-positive values).
        let body = r#"{"prompt": "hello", "max_tokens": 100, "n": 0}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 101.0);
    }

    #[test]
    fn test_estimate_cost_n_negative_defaults_to_one() {
        // n=-1 is treated as n=1 (filter excludes non-positive values).
        let body = r#"{"prompt": "hello", "max_tokens": 100, "n": -1}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 101.0);
    }

    #[test]
    fn test_estimate_cost_n_with_chat_messages() {
        // Chat requests also accept n (for multiple candidates).
        // (2 + 3 words + 20 max_tokens) * 2 = 50
        let body = r#"{"messages": [{"role": "user", "content": "hello world"}, {"role": "assistant", "content": "foo bar baz"}], "max_tokens": 20, "n": 2}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 50.0);
    }

    #[test]
    fn test_estimate_cost_n_clamped_to_max() {
        // Large n * (prompt + max_tokens) should still be clamped to 100_000.
        let prompt = "word ".repeat(60_000);
        let body = format!(r#"{{"prompt": "{prompt}", "max_tokens": 50000, "n": 10}}"#);
        let cost = estimate_request_cost(&body);
        assert_eq!(cost, 100_000.0);
    }

    #[test]
    fn test_estimate_cost_best_of_multiplies_cost() {
        // best_of=5 should multiply the cost: (4 words + 10) * 5 = 70
        let body = r#"{"prompt": "hello world foo bar", "max_tokens": 10, "best_of": 5}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 70.0);
    }

    #[test]
    fn test_estimate_cost_best_overrides_n() {
        // When both n and best_of are set, the larger one is the multiplier.
        // best_of=10 > n=2 → multiplier = 10 → (4 + 10) * 10 = 140
        let body = r#"{"prompt": "hello world foo bar", "max_tokens": 10, "n": 2, "best_of": 10}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 140.0);
    }

    #[test]
    fn test_estimate_cost_n_overrides_best() {
        // n=10 > best_of=2 → multiplier = 10 → (4 + 10) * 10 = 140
        let body = r#"{"prompt": "hello world foo bar", "max_tokens": 10, "n": 10, "best_of": 2}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 140.0);
    }

    #[test]
    fn test_estimate_cost_best_of_defaults_to_one() {
        // Without best_of, cost is 1x (same as n=1 default).
        let body = r#"{"prompt": "hello", "max_tokens": 100}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 101.0);
    }

    #[test]
    fn test_estimate_cost_best_of_zero_defaults_to_one() {
        // best_of=0 is treated as 1 (filter excludes non-positive values).
        let body = r#"{"prompt": "hello", "max_tokens": 100, "best_of": 0}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 101.0);
    }

    #[test]
    fn test_estimate_cost_completions_string_array_prompt() {
        // OpenAI completions accept `prompt` as an array of strings
        // (batch prompts). Each string is a separate prompt; the cost
        // must sum word counts across all strings, not just count
        // array elements. Failing to do so is a rate-limit bypass: a
        // single 10 000-word string in a 1-element array would be
        // charged as 1 token instead of 10 000.
        let body = r#"{"prompt": ["hello world", "foo bar baz"], "max_tokens": 10}"#;
        let cost = estimate_request_cost(body);
        // 2 + 3 + 10 = 15
        assert_eq!(cost, 15.0);
    }

    #[test]
    fn test_estimate_cost_stop_does_not_affect_cost() {
        // The `stop` parameter specifies sequences that terminate generation
        // early, but for rate-limiting we charge for the maximum possible
        // output (max_tokens). `stop` is intentionally NOT included in the
        // cost formula — documenting this invariant prevents a future PR
        // from accidentally reducing the cost estimate (which would be a
        // rate-limit bypass).
        let body_with_stop = r#"{"prompt": "hello world", "max_tokens": 50, "stop": ["\n\n"]}"#;
        let body_without_stop = r#"{"prompt": "hello world", "max_tokens": 50}"#;
        assert_eq!(
            estimate_request_cost(body_with_stop),
            estimate_request_cost(body_without_stop),
            "stop parameter must not change the rate-limit cost"
        );
    }

    #[test]
    fn test_estimate_cost_empty_string_prompt_charges_max_tokens() {
        // An empty prompt string has 0 words but the model still
        // generates up to max_tokens tokens, so the cost should be
        // 0 + max_tokens = 100.
        let body = r#"{"prompt": "", "max_tokens": 100}"#;
        let cost = estimate_request_cost(body);
        assert_eq!(cost, 100.0);
    }

    #[test]
    fn test_estimate_cost_empty_messages_array_charges_max_tokens() {
        // An empty messages array (chat with no messages) should still
        // charge max_tokens default (100) since the cost is clamped to
        // a minimum of 1.
        let body = r#"{"messages": [], "max_tokens": 100}"#;
        let cost = estimate_request_cost(body);
        assert!(cost >= 1.0, "cost should be at least 1 for empty messages");
    }

    // ------------------------------------------------------------------
    // Per-key override tests
    // ------------------------------------------------------------------

    #[test]
    fn test_per_key_override_allows_more_requests() {
        let mut overrides = HashMap::new();
        overrides.insert("premium".to_string(), (5, 60));

        let auth = AuthMiddleware::new_with_overrides(
            vec!["premium".to_string(), "standard".to_string()],
            2, // global default: 2
            60,
            overrides,
        );
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer premium".parse().unwrap());

        // Premium key should get 5 requests (override)
        for _ in 0..5 {
            assert!(auth.verify(&headers).is_ok());
        }
        // 6th request should be rate-limited
        assert_eq!(
            auth.verify(&headers).unwrap_err(),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    #[test]
    fn test_per_key_override_limit_in_error() {
        let mut overrides = HashMap::new();
        overrides.insert("premium".to_string(), (5, 60));

        let auth = AuthMiddleware::new_with_overrides(
            vec!["premium".to_string()],
            10, // global default: 10
            60,
            overrides,
        );
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer premium".parse().unwrap());

        // Exhaust the premium override (capacity 5).
        for _ in 0..5 {
            assert!(auth.verify(&headers).is_ok());
        }
        // 6th request is rate-limited; the error should carry the override limit.
        let err = auth.verify_with_meta(&headers, 1.0).unwrap_err();
        match err {
            AuthError::RateLimited { limit, .. } => {
                assert_eq!(limit, 5.0, "per-key limit should be reported in the error");
            }
            _ => panic!("expected RateLimited, got {err:?}"),
        }
    }

    #[test]
    fn test_api_keys_returns_configured_keys_in_order() {
        let auth = AuthMiddleware::new(vec!["alpha".to_string(), "beta".to_string()], 10, 60);
        let keys = auth.api_keys();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0], "alpha");
        assert_eq!(keys[1], "beta");
    }

    #[test]
    fn test_user_id_from_key_truncates_to_8_chars() {
        // Keys longer than 8 chars → truncated prefix.
        assert_eq!(user_id_from_key("sk-abcdefghijklmnop"), "key:sk-abcde");
        // Keys shorter than 8 chars → full key.
        assert_eq!(user_id_from_key("sk-short"), "key:sk-short");
        // Empty key → empty prefix.
        assert_eq!(user_id_from_key(""), "key:");
    }

    #[test]
    fn test_per_key_override_standard_key_uses_global() {
        let mut overrides = HashMap::new();
        overrides.insert("premium".to_string(), (5, 60));

        let auth = AuthMiddleware::new_with_overrides(
            vec!["premium".to_string(), "standard".to_string()],
            2, // global default: 2
            60,
            overrides,
        );
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer standard".parse().unwrap());

        // Standard key should only get 2 (global default)
        assert!(auth.verify(&headers).is_ok());
        assert!(auth.verify(&headers).is_ok());
        assert_eq!(
            auth.verify(&headers).unwrap_err(),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    #[test]
    fn test_per_key_override_lower_limit() {
        let mut overrides = HashMap::new();
        overrides.insert("restricted".to_string(), (1, 60));

        let auth = AuthMiddleware::new_with_overrides(
            vec!["restricted".to_string()],
            10, // global default: 10
            60,
            overrides,
        );
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer restricted".parse().unwrap());

        // Restricted key should only get 1 (override is lower)
        assert!(auth.verify(&headers).is_ok());
        assert_eq!(
            auth.verify(&headers).unwrap_err(),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    #[test]
    fn test_no_overrides_uses_global() {
        let auth =
            AuthMiddleware::new_with_overrides(vec!["key1".to_string()], 3, 60, HashMap::new());
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer key1".parse().unwrap());

        for _ in 0..3 {
            assert!(auth.verify(&headers).is_ok());
        }
        assert_eq!(
            auth.verify(&headers).unwrap_err(),
            StatusCode::TOO_MANY_REQUESTS
        );
    }
}
