# Production Features Design

**Date**: 2026-04-04
**Status**: Approved
**Goal:** Complete production-ready features: Auth, Multi-tenancy, Rate Limiting, Audit, Monitoring, Security

---

## 1. Authentication & Authorization

### 1.1 API Key Authentication

```rust
// Server config
pub struct AuthConfig {
    pub api_keys: Vec<ApiKey>,      // Stored hashed
    pub jwt_secret: Option<String>,  // Optional JWT
    pub token_expiry_secs: u64,      // Token expiration
}

pub struct ApiKey {
    pub id: String,                 // key_xxx
    pub key_hash: String,            // SHA256 hash
    pub tenant_id: String,
    pub name: String,
    pub rate_limit: RateLimitConfig,
    pub quota_monthly: Option<u64>,  // Monthly quota (requests)
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub is_active: bool,
}

// Middleware extracts tenant from:
// 1. X-API-Key header → lookup tenant_id
// 2. X-Tenant-ID header (optional) → override tenant
```

### 1.2 Token-based Auth (JWT)

```rust
pub struct JwtClaims {
    pub sub: String,          // tenant_id
    pub exp: i64,
    pub iat: i64,
    pub roles: Vec<String>,   // admin, user, etc.
}
```

---

## 2. Multi-tenancy

### 2.1 Tenant Model

```rust
pub struct Tenant {
    pub id: String,
    pub name: String,
    pub quota: TenantQuota,
    pub config: TenantConfig,
    pub created_at: i64,
}

pub struct TenantQuota {
    pub requests_per_minute: u32,
    pub requests_per_day: u64,
    pub tokens_per_minute: u64,
    pub max_concurrent_requests: u32,
}

pub struct TenantConfig {
    pub default_model: Option<String>,
    pub max_tokens: u32,
    pub allowed_models: Vec<String>,
}
```

### 2.2 Tenant Isolation

```
Request Flow:
1. Extract API Key from header
2. Lookup tenant_id from key
3. Check X-Tenant-ID header (optional override)
4. Load tenant config from cache
5. Apply rate limits and quotas
6. Route to model
7. Track usage per tenant
```

### 2.3 Data Isolation

- **KV Cache**: Tenant prefix or separate cache regions
- **Model Weights**: Shared (read-only), OK
- **Request State**: Per-tenant context in async local storage

---

## 3. Rate Limiting

### 3.1 Rate Limit Config

```rust
pub struct RateLimitConfig {
    pub requests_per_minute: u32,
    pub requests_per_second: u32,
    pub tokens_per_minute: u64,
    pub burst_size: u32,  // Token bucket burst
}

impl RateLimit {
    // Token bucket algorithm
    pub fn check(&mut self, cost: u32) -> Result<()>;
}
```

### 3.2 Rate Limit Strategy

```
Per-Tenant Rate Limiting:
- Token bucket per tenant
- Sliding window for minute/day limits
- Redis-based for distributed deployments
- In-memory fallback for single instance
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

### 5.1 Metrics

```rust
pub struct Metrics {
    // Request metrics
    pub requests_total: Counter,
    pub requests_by_tenant: Counter,
    pub request_duration_seconds: Histogram,
    
    // Model metrics
    pub tokens_generated_total: Counter,
    pub tokens_by_model: Counter,
    
    // System metrics
    pub kv_cache_usage: Gauge,
    pub memory_usage: Gauge,
    pub gpu_utilization: Gauge,
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
├── auth/
│   ├── mod.rs
│   ├── api_key.rs      # API key management
│   ├── jwt.rs          # JWT handling
│   └── middleware.rs   # Axum middleware
│
├── tenant/
│   ├── mod.rs
│   ├── model.rs        # Tenant model
│   ├── quota.rs        # Quota management
│   └── isolation.rs    # Tenant isolation
│
├── rate_limit/
│   ├── mod.rs
│   ├── token_bucket.rs # Rate limit algorithm
│   └── storage.rs     # Redis/in-memory
│
├── audit/
│   ├── mod.rs
│   ├── logger.rs      # Audit logging
│   └── storage.rs     # Log persistence
│
├── metrics/
│   ├── mod.rs
│   ├── prometheus.rs  # Prometheus export
│   └── alerting.rs    # Alert checking
│
└── security/
    ├── mod.rs
    ├── tls.rs         # TLS config
    └── validation.rs  # Input validation
```

---

## 8. Implementation Order

1. **Phase 1: Authentication**
   - API key storage and validation
   - JWT token generation/validation
   - Auth middleware

2. **Phase 2: Multi-tenancy**
   - Tenant model and config
   - Tenant isolation in request processing
   - Tenant-specific rate limits

3. **Phase 3: Rate Limiting**
   - Token bucket implementation
   - Per-tenant limits
   - Redis storage (optional)

4. **Phase 4: Audit Logging**
   - Event capture
   - Log storage
   - Query API

5. **Phase 5: Monitoring**
   - Prometheus metrics
   - Alert configuration
   - Dashboard endpoints

6. **Phase 6: Security**
   - TLS configuration
   - Request validation
   - CORS settings

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
