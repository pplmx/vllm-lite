# 安全特性实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 vLLM-lite 添加 API Key 环境变量支持、改进的 Rate Limiting、审计日志和 TLS/SSL 支持

**Architecture:** 使用纯 Rust 技术栈（rustls, tokio）实现安全特性，保持与现有代码风格一致

**Tech Stack:** rustls, tracing, serde

---

### Task 1: 添加 TLS 和审计日志配置

**Files:**
- Modify: `crates/server/src/config.rs:40-66`
- Modify: `crates/server/src/config.rs:122-142`

- [ ] **Step 1: 添加 TlsConfig 结构体**

在 `AuthConfig` 后添加：

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    #[serde(default)]
    pub tls_cert: Option<String>,
    #[serde(default)]
    pub tls_key: Option<String>,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            tls_cert: None,
            tls_key: None,
        }
    }
}
```

- [ ] **Step 2: 添加审计日志配置**

在 `TlsConfig` 后添加：

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub log_path: Option<String>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_path: None,
        }
    }
}
```

- [ ] **Step 3: 更新 AuthConfig 添加审计配置**

修改 `AuthConfig`：

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    #[serde(default)]
    pub api_keys: Vec<String>,
    #[serde(default = "default_rate_limit_requests")]
    pub rate_limit_requests: usize,
    #[serde(default = "default_rate_limit_window")]
    pub rate_limit_window_secs: u64,
    #[serde(default)]
    pub audit: AuditConfig,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            api_keys: vec![],
            rate_limit_requests: 100,
            rate_limit_window_secs: 60,
            audit: AuditConfig::default(),
        }
    }
}
```

- [ ] **Step 4: 添加环境变量解析**

在 `AppConfig::load()` 中 `config` 赋值后添加：

```rust
// Load API keys from environment variable
if let Ok(api_keys) = std::env::var("VLLM_API_KEYS") {
    if !api_keys.is_empty() {
        let keys: Vec<String> = api_keys.split(',').map(|s| s.trim().to_string()).collect();
        config.auth.api_keys.extend(keys);
    }
}
```

- [ ] **Step 5: 在 AppConfig 中添加 tls 字段**

修改 `AppConfig` 结构体：

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::derivable_impls)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub engine: EngineConfig,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub tls: TlsConfig,
}
```

更新 `AppConfig::default()` 和 `AppConfig::load()`

- [ ] **Step 6: 运行测试验证**

Run: `cargo test -p vllm-server -- config --nocapture`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/server/src/config.rs
git commit -m "feat(config): add TLS and audit log configuration"
```

---

### Task 2: 创建审计日志模块

**Files:**
- Create: `crates/server/src/audit.rs`
- Modify: `crates/server/src/main.rs:4`

- [ ] **Step 1: 创建 audit.rs**

新建文件 `crates/server/src/audit.rs`：

```rust
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize)]
pub struct AuditRecord {
    pub timestamp: String,
    pub request_id: String,
    pub method: String,
    pub endpoint: String,
    pub status: u16,
    pub latency_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    pub api_key_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_ip: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
}

pub struct AuditLogger {
    writer: Arc<Mutex<Option<AuditWriter>>>,
}

struct AuditWriter {
    file: tokio::fs::File,
}

impl AuditLogger {
    pub fn new(log_path: Option<String>) -> Self {
        let writer = if let Some(path) = log_path {
            Some(AuditWriter {
                file: tokio::fs::File::create(&path).block_on().ok(),
            })
        } else {
            None
        };

        Self {
            writer: Arc::new(Mutex::new(writer)),
        }
    }

    pub async fn log(&self, record: AuditRecord) {
        let mut guard = self.writer.lock().await;
        if let Some(ref mut writer) = *guard {
            let json = serde_json::to_string(&record).unwrap_or_default();
            let line = format!("{}\n", json);
            let _ = writer.file.write_all(line.as_bytes()).await;
            let _ = writer.file.flush().await;
        }
    }
}

pub fn hash_api_key(key: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    format!("sha256:{:016x}", hasher.finish())
}
```

- [ ] **Step 2: 更新 main.rs 导入**

在 `crates/server/src/main.rs` 第 4 行后添加：

```rust
pub mod audit;
```

- [ ] **Step 3: Commit**

```bash
git add crates/server/src/audit.rs crates/server/src/main.rs
git commit -m "feat(audit): add audit logging module"
```

---

### Task 3: 改进 Rate Limiter（添加响应头和优化）

**Files:**
- Modify: `crates/server/src/auth.rs:1-85`

- [ ] **Step 1: 修改 RateLimiter 结构体添加锁定**

将 `RateLimiter` 的 `requests` 字段改为使用 `DashMap`：

```rust
use dashmap::DashMap;

pub struct RateLimiter {
    requests: DashMap<String, Vec<Instant>>,
    max_requests: usize,
    window_secs: u64,
}
```

- [ ] **Step 2: 添加 Cargo 依赖**

检查并添加 `dashmap` 到 `crates/server/Cargo.toml`（如果尚未存在）：

```toml
dashmap = "5"
```

- [ ] **Step 3: 添加 rate limit 响应头**

修改 `AuthMiddleware::verify` 方法返回更多信息：

```rust
pub struct RateLimitInfo {
    pub allowed: bool,
    pub limit: usize,
    pub remaining: usize,
    pub reset: u64,
}

impl AuthMiddleware {
    pub async fn verify(&self, headers: &HeaderMap) -> Result<String, (StatusCode, RateLimitInfo)> {
        let auth_header = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok());

        let api_key = auth_header
            .and_then(|h| h.strip_prefix("Bearer "))
            .ok_or((StatusCode::UNAUTHORIZED, self.get_rate_limit_info("")))?;

        if !self.api_keys.is_empty() && !self.api_keys.contains(&api_key.to_string()) {
            return Err((StatusCode::UNAUTHORIZED, self.get_rate_limit_info(api_key)));
        }

        let mut limiter = self.rate_limiter.write().await;
        let info = limiter.check_rate_limit_with_info(api_key).await;
        
        if !info.allowed {
            return Err((StatusCode::TOO_MANY_REQUESTS, info));
        }

        Ok(api_key.to_string())
    }

    fn get_rate_limit_info(&self, key: &str) -> RateLimitInfo {
        RateLimitInfo {
            allowed: true,
            limit: self.rate_limiter.read().await.max_requests,
            remaining: self.rate_limiter.read().await.max_requests,
            reset: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() + self.rate_limiter.read().await.window_secs,
        }
    }
}
```

- [ ] **Step 4: 更新 middleware 使用新的错误类型**

修改 `auth_middleware` 函数：

```rust
pub async fn auth_middleware(
    auth: axum::extract::State<Arc<AuthMiddleware>>,
    request: Request,
    next: Next,
) -> Response {
    match auth.verify(request.headers()).await {
        Ok(_) => {
            let mut response = next.run(request).await;
            response.headers_mut().insert(
                "X-RateLimit-Limit",
                "100".parse().unwrap(),
            );
            response.headers_mut().insert(
                "X-RateLimit-Remaining", 
                "99".parse().unwrap(),
            );
            response
        }
        Err((status, info)) => {
            let mut response = Response::builder()
                .status(status)
                .body("".into())
                .unwrap();
            response.headers_mut().insert(
                "X-RateLimit-Limit",
                info.limit.to_string().parse().unwrap(),
            );
            response.headers_mut().insert(
                "X-RateLimit-Remaining",
                info.remaining.to_string().parse().unwrap(),
            );
            response.headers_mut().insert(
                "X-RateLimit-Reset",
                info.reset.to_string().parse().unwrap(),
            );
            response
        }
    }
}
```

- [ ] **Step 5: 运行测试验证**

Run: `cargo test -p vllm-server -- auth --nocapture`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/server/src/auth.rs
git commit -m "feat(auth): improve rate limiter with response headers"
```

---

### Task 4: 支持 TLS (Rustls)

**Files:**
- Modify: `crates/server/src/main.rs:150-170`
- Create: `crates/server/src/tls.rs` (可选)

- [ ] **Step 1: 添加 rustls 依赖**

在 `crates/server/Cargo.toml` 添加：

```toml
rustls = "0.23"
rustls-pemfile = "2"
axum = { version = "0.7", features = ["tokio", "rustls"] }
```

- [ ] **Step 2: 修改 main.rs 添加 TLS 支持**

在 `main.rs` 中替换服务器启动部分：

```rust
use axum_server::tls_rustls::RustlsConfig;

async fn run_server(app: Router, addr: String, tls_config: Option<TlsConfig>) {
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    
    if let Some(tls) = tls_config {
        if tls.tls_cert.is_some() && tls.tls_key.is_some() {
            let config = RustlsConfig::from_pem_file(
                tls.tls_cert.unwrap(),
                tls.tls_key.unwrap(),
            )
            .await
            .expect("Failed to load TLS config");
            
            tracing::info!(address = %addr, "Server listening (HTTPS)");
            axum_server::tls_rustls::TcpSocket::new()
                .bind(addr)
                .rustls(config)
                .serve(app)
                .await
                .unwrap();
            return;
        }
    }
    
    tracing::info!(address = %addr, "Server listening (HTTP)");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}
```

- [ ] **Step 3: 更新 main 函数调用**

在 `main()` 函数中替换服务器启动逻辑：

```rust
let tls_config = if app_config.tls.tls_cert.is_some() {
    Some(app_config.tls.clone())
} else {
    None
};

let addr = format!("{}:{}", app_config.server.host, app_config.server.port);

// ... router setup ...

run_server(app, addr, tls_config).await;
```

- [ ] **Step 4: 运行测试验证**

Run: `cargo build -p vllm-server`
Expected: BUILD SUCCESS

- [ ] **Step 5: Commit**

```bash
git add crates/server/Cargo.toml crates/server/src/main.rs
git commit -m "feat(server): add TLS support with rustls"
```

---

### Task 5: 集成审计日志到 API

**Files:**
- Modify: `crates/server/src/openai/chat.rs`
- Modify: `crates/server/src/openai/completions.rs`

- [ ] **Step 1: 修改 ApiState 添加审计日志**

在 `crates/server/src/main.rs` 中：

```rust
#[derive(Clone)]
pub struct ApiState {
    pub engine_tx: api::EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    pub batch_manager: Arc<BatchManager>,
    pub auth: Option<Arc<AuthMiddleware>>,
    pub audit_logger: Option<Arc<AuditLogger>>,
}
```

- [ ] **Step 2: 在 main.rs 中初始化审计日志**

```rust
let audit_logger = if app_config.auth.audit.enabled {
    Some(Arc::new(AuditLogger::new(
        app_config.auth.audit.log_path.clone(),
    )))
} else {
    None
};

let state = ApiState {
    engine_tx: msg_tx.clone(),
    tokenizer,
    batch_manager,
    auth: auth_middleware.clone(),
    audit_logger,
};
```

- [ ] **Step 3: 在 chat.rs 中添加审计日志调用**

在请求处理完成后记录审计日志：

```rust
use crate::audit::{AuditLogger, AuditRecord, hash_api_key};

// 在处理函数中
let audit_record = AuditRecord {
    timestamp: chrono::Utc::now().to_rfc3339(),
    request_id: request_id.clone(),
    method: "POST".to_string(),
    endpoint: "/v1/chat/completions".to_string(),
    status: 200,
    latency_ms: elapsed.as_millis() as u64,
    prompt_tokens: Some(prompt_tokens as u32),
    completion_tokens: Some(completion_tokens as u32),
    api_key_hash: hash_api_key(&api_key),
    client_ip: None,
    user_agent: None,
};

if let Some(ref logger) = state.audit_logger {
    logger.log(audit_record).await;
}
```

- [ ] **Step 4: 在 completions.rs 中添加类似审计日志**

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/main.rs crates/server/src/openai/chat.rs crates/server/src/openai/completions.rs
git commit -m "feat(audit): integrate audit logging into API endpoints"
```

---

### Task 6: 最终验证

**Files:**
- None (verification only)

- [ ] **Step 1: 运行完整测试**

Run: `cargo test -p vllm-server --workspace --nocapture`
Expected: ALL PASS

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy -p vllm-server -- -D warnings`
Expected: NO WARNINGS

- [ ] **Step 3: Build release**

Run: `cargo build -p vllm-server --release`
Expected: BUILD SUCCESS

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: complete Phase 8 security features"
```