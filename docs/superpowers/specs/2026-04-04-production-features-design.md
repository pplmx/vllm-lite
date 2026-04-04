# Production Features Design

**Date**: 2026-04-04
**Status**: Approved
**Goal:** Complete production-ready features: Auth, Multi-tenancy, Rate Limiting, Audit, Monitoring, Security

---

## 1. Authentication & Authorization

### 1.1 API Key Authentication (Secure)

```rust
// Server config
pub struct AuthConfig {
    pub api_keys: Vec<ApiKey>,          // Stored with salt + hash
    pub jwt_secret_path: Option<String>,  // Path to JWT secret file (not inline!)
    pub token_expiry_secs: u64,          // Token expiration (default: 3600)
    pub max_auth_failures: u32,          // Max failures before lockout (default: 5)
    pub auth_lockout_secs: u64,          // Lockout duration (default: 300)
}

pub struct ApiKey {
    pub id: String,                 // key_xxx
    pub key_salt: String,           // Random salt for this key
    pub key_hash: String,           // SHA256(salt + key), NOT just SHA256(key)
    pub tenant_id: String,
    pub name: String,
    pub rate_limit: RateLimitConfig,
    pub quota_monthly: Option<u64>,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub rotated_from: Option<String>,   // Previous key ID (for rotation)
    pub is_active: bool,
}

// Key generation (secure)
impl ApiKey {
    pub fn generate(tenant_id: &str, name: &str) -> (Self, String) {
        let salt = generate_random_string(32);
        let key = generate_random_string(64);  // Return to user once
        let key_hash = sha256(format!("{}{}", salt, key));
        
        let api_key = Self {
            id: format!("key_{}", generate_id()),
            key_salt: salt,
            key_hash,
            // ... other fields
        };
        
        (api_key, key)  // Return plain key to user ONLY once
    }
}
```

### 1.2 Token-based Auth (JWT)

```rust
pub struct JwtClaims {
    pub sub: String,          // tenant_id
    pub exp: i64,
    pub iat: i64,
    pub roles: Vec<String>,
    pub key_id: Option<String>,  // Which API key was used
    pub jti: String,          // Unique token ID for revocation
}

// JWT secret loaded from file (not environment!)
impl JwtConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        let secret = std::fs::read_to_string(path)?;
        // Validate minimum length
        if secret.len() < 32 {
            return Err(Error::msg("JWT secret too short"));
        }
        Ok(Self { secret })
    }
}
```

### 1.3 Brute Force Protection

```rust
pub struct AuthFailureTracker {
    failures: HashMap<String, (u32, i64)>,  // key -> (count, first_failure_time)
}

impl AuthFailureTracker {
    pub fn record_failure(&mut self, identifier: &str) -> bool {
        // After max_auth_failures, reject all requests for lockout duration
    }
    
    pub fn check_and_record(&mut self, identifier: &str) -> Result<()>;
    pub fn clear(&mut self, identifier: &str);
}
```

---

## 2. Multi-tenancy

### 2.1 Tenant Model with Management API

```rust
pub struct Tenant {
    pub id: String,
    pub name: String,
    pub quota: TenantQuota,
    pub config: TenantConfig,
    pub created_at: i64,
    pub is_active: bool,
}

// Admin API for tenant management
pub enum TenantApi {
    // GET /admin/tenants - List all tenants
    // POST /admin/tenants - Create tenant
    // GET /admin/tenants/{id} - Get tenant
    // PUT /admin/tenants/{id} - Update tenant
    // DELETE /admin/tenants/{id} - Delete tenant
    
    // API Key management
    // GET /admin/tenants/{id}/keys - List keys
    // POST /admin/tenants/{id}/keys - Create key
    // DELETE /admin/tenants/{id}/keys/{key_id} - Revoke key
    // POST /admin/tenants/{id}/keys/{key_id}/rotate - Rotate key
}

pub struct TenantQuota {
    pub requests_per_minute: u32,
    pub requests_per_day: u64,
    pub tokens_per_minute: u64,
    pub max_concurrent_requests: u32,
    pub storage_mb: Option<u64>,       // KV cache storage limit
}

pub struct TenantConfig {
    pub default_model: Option<String>,
    pub max_tokens: u32,
    pub allowed_models: Vec<String>,
    pub custom_system_prompt: Option<String>,
}
```

### 2.2 Tenant Isolation

```
Request Flow:
1. Extract API Key from header
2. Lookup tenant_id from key (cache hit)
3. Check X-Tenant-ID header (optional override)
4. Load tenant config from cache
5. Apply rate limits and quotas
6. Set tenant context in async local storage
7. Route to model
8. Track usage per tenant

Tenant Context (async local storage):
impl TenantContext {
    pub fn current() -> TenantContext;
    pub fn set(tenant_id: String, config: TenantConfig);
    pub fn quota() -> TenantQuota;
}
```

### 2.3 Data Isolation (Detailed)

```rust
pub enum CacheIsolation {
    // Option 1: Prefix each key with tenant_id (simple, OK for small scale)
    Prefix { tenant_prefix: String },
    
    // Option 2: Separate cache regions (better for large scale)
    SeparateRegions { per_tenant_blocks: usize },
    
    // Option 3: Shared cache with tenant-aware eviction (best for high utilization)
    SharedWithIsolation,
}

impl PagedKvCache {
    pub fn with_isolation(self, isolation: CacheIsolation) -> Self;
}
```

---

## 3. Rate Limiting

### 3.1 Rate Limit Config (In-Memory, No Redis)

```rust
pub struct RateLimitConfig {
    pub requests_per_minute: u32,
    pub requests_per_second: u32,
    pub tokens_per_minute: u64,
    pub burst_size: u32,  // Token bucket burst
}

impl RateLimit {
    // Token bucket algorithm (in-memory)
    // No Redis needed for single instance
    // For distributed部署, can add Redis later if needed
    pub fn check(&mut self, cost: u32) -> Result<()>;
}
```

### 3.2 Rate Limit Strategy

```
Per-Tenant Rate Limiting:
- Token bucket per tenant (in-memory)
- Sliding window for minute/day limits
- No external dependencies (simple, fast)
- For multi-instance: can extend with Redis later

Current vllm-lite already has in-memory rate limiter!
```

---

## 4. Audit Logging

### 4.1 Audit Events

```rust
pub enum AuditEvent {
    Request { tenant_id, model, prompt_tokens, completion_tokens },
    AuthSuccess { tenant_id, api_key_id },
    AuthFailure { reason, ip },
    RateLimitHit { tenant_id, limit_type },
    QuotaExceeded { tenant_id, quota_type },
    Error { tenant_id, error_type, message },
}

pub struct AuditLog {
    pub id: String,
    pub timestamp: i64,
    pub tenant_id: String,
    pub event_type: String,
    pub metadata: HashMap<String, String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}
```

### 4.2 Log Storage

- **In-memory**: Ring buffer for recent logs
- **File**: JSON lines for persistent storage
- **External**: Hook to send to external logging service

---

## 5. Monitoring & Alerting

### 5.1 Metrics (Already Implemented!)

vllm-lite already has comprehensive metrics in `crates/core/src/metrics.rs`:

```rust
// Already implemented:
pub struct MetricsSnapshot {
    pub tokens_total: u64,
    pub requests_total: u64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p90_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub avg_batch_size: f64,
    pub current_batch_size: usize,
    pub requests_in_flight: u64,
    pub kv_cache_usage_percent: f64,
    pub prefix_cache_hit_rate: f64,
    pub prefill_throughput: f64,
    pub decode_throughput: f64,
    pub avg_scheduler_wait_time_ms: f64,
}

// Existing /metrics endpoint at /metrics
// Just add per-tenant metrics to existing system
```

### 5.2 Extended Metrics (Add)

```rust
pub struct ExtendedMetrics {
    // Add to existing MetricsSnapshot:
    pub tenant_requests: HashMap<String, u64>,     // Per-tenant request count
    pub tenant_tokens: HashMap<String, u64>,        // Per-tenant token count
    pub tenant_errors: HashMap<String, u64>,        // Per-tenant error count
    pub rate_limit_hits: u64,
    pub auth_failures: u64,
}
```

### 5.3 Prometheus Export (Already Exists!)

```rust
// Already at /metrics endpoint in server
// Just verify works correctly
```

### 5.4 Alerting (Add)

```rust
pub struct AlertConfig {
    pub rate_limit_threshold: f64,    // 0.9 = 90% of limit
    pub error_rate_threshold: f64,      // 0.05 = 5% errors
    pub latency_threshold_ms: u64,       // 5000ms
    pub quota_usage_threshold: f64,      // 95%
}

pub enum Alert {
    HighErrorRate { tenant_id, rate: f64 },
    HighLatency { tenant_id, p99_ms: u64 },
    QuotaExceeded { tenant_id, quota_type: String },
    RateLimitNear { tenant_id, usage: f64 },
    SystemError { error_type: String },
}
```

### 5.2 Prometheus Export

```rust
// /metrics endpoint
pub fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).unwrap();
   (buffer, ContentType::TextPlain)
}
```

### 5.3 Alerting

```rust
pub struct AlertConfig {
    pub rate_limit_threshold: f64,    // 0.9 = 90% of limit
    pub error_rate_threshold: f64,      // 0.05 = 5% errors
    pub latency_threshold_ms: u64,       // 5000ms
    pub quota_usage_threshold: f64,      // 0.95 = 95%
}

pub enum Alert {
    HighErrorRate { tenant_id, rate: f64 },
    HighLatency { tenant_id, p99_ms: u64 },
    QuotaExceeded { tenant_id, quota_type: String },
    RateLimitNear { tenant_id, usage: f64 },
    SystemError { error_type: String },
}
```

---

## 6. Security

### 6.1 TLS/HTTPS

```rust
pub struct TlsConfig {
    pub cert_path: PathBuf,
    pub key_path: PathBuf,
    pub min_tls_version: String,  // "1.2" or "1.3"
}

impl Server {
    pub fn with_tls(self, config: TlsConfig) -> Self;
}
```

### 6.2 Request Validation

```rust
pub struct SecurityConfig {
    pub max_request_size_bytes: usize,    // 10MB default
    pub max_prompt_tokens: usize,          // 128K
    pub max_completion_tokens: usize,        // 8K
    pub allowed_content_types: Vec<String>,
    pub cors_origin: Option<String>,
}
```

### 6.3 Input Sanitization

- Prompt injection detection
- Token limit enforcement
- Special token validation

---

## 7. Architecture

```
server/
├── auth/                    # Already exists, enhance
│   ├── mod.rs              # AuthMiddleware (exists)
│   ├── api_key.rs          # NEW: API key management with salt
│   ├── jwt.rs              # NEW: JWT handling
│   └── middleware.rs       # Already exists
│
├── tenant/                 # NEW
│   ├── mod.rs
│   ├── model.rs
│   ├── quota.rs
│   └── isolation.rs
│
├── rate_limit/             # Already exists (in-memory), enhance
│   └── (extend existing)
│
├── audit/                 # NEW
│   ├── mod.rs
│   ├── logger.rs
│   └── storage.rs
│
├── metrics/               # Already exists, extend
│   └── (add per-tenant)
│
└── security/             # NEW
    ├── tls.rs
    └── validation.rs
```

---

## 8. Implementation Order

Based on existing code analysis:

| Phase | Component | Existing | New Work |
|-------|-----------|----------|----------|
| 1 | Authentication | Partial (basic) | Add salt+hash, JWT, key rotation |
| 1 | API Key Middleware | ✅ Exists | Add per-tenant |
| 2 | Multi-tenancy | None | Full implementation |
| 2 | Tenant Model | None | Full implementation |
| 2 | Tenant Isolation | None | Full implementation |
| 3 | Rate Limiting | ✅ Exists | Per-tenant enhancement |
| 3 | Redis | Not needed | In-memory is sufficient |
| 4 | Audit Logging | None | Full implementation |
| 5 | Metrics | ✅ Exists | Add per-tenant |
| 5 | Prometheus | ✅ Exists | Works |
| 5 | Alerting | None | Basic implementation |
| 6 | Security | None | TLS + validation |

**Revised Plan:**

1. **Phase 1: Auth Enhancement**
   - Add salt to API key storage
   - Add JWT support
   - Add key rotation

2. **Phase 2: Multi-tenancy**
   - Tenant model and CRUD
   - Tenant isolation

3. **Phase 3: Rate Limit Enhancement**
   - Per-tenant rate limits

4. **Phase 4: Audit Logging**
   - Event capture
   - Log storage

5. **Phase 5: Extended Metrics**
   - Per-tenant metrics

6. **Phase 6: Security**
   - TLS config

---

## 9. Acceptance Criteria

- [ ] API key authentication works
- [ ] JWT token authentication works
- [ ] Tenant isolation is enforced
- [ ] Quota limits are enforced
- [ ] Rate limiting works per tenant
- [ ] Audit logs are captured
- [ ] Prometheus metrics available at /metrics
- [ ] Alerts can be triggered
- [ ] TLS can be configured
- [ ] Request validation works
