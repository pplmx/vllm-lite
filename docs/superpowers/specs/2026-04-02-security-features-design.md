# 安全特性设计

## 概述

为 vLLM-lite 添加生产环境所需的安全特性：API Key 环境变量支持、改进的 Rate Limiting、审计日志和 TLS/SSL 支持。

## 1. API Key 环境变量支持

### 配置

支持两种方式配置 API Keys：

1. **配置文件** (`config.yaml`):
   ```yaml
   auth:
     api_keys:
       - key1
       - key2
   ```

2. **环境变量** (逗号分隔):
   ```bash
   export VLLM_API_KEYS="key1,key2,key3"
   ```

优先级：环境变量 > 配置文件

### 实现

- 在 `AppConfig::load()` 中读取 `VLLM_API_KEYS` 环境变量
- 解析逗号分隔的 keys，追加到配置的 api_keys
- 空环境变量时使用配置文件值

## 2. Rate Limiting 改进

### 配置

```yaml
auth:
  rate_limit_requests: 100      # 每窗口最大请求数
  rate_limit_window_secs: 60    # 时间窗口（秒）
```

### 实现

- 保持现有滑动窗口实现
- 全局统一限制（所有 API Key 共享配额）
- 超出限制返回 HTTP 429
- 响应头添加：
  - `X-RateLimit-Limit`: 允许的请求数
  - `X-RateLimit-Remaining`: 剩余请求数
  - `X-RateLimit-Reset`: 窗口重置时间戳

### 优化

- 使用更高效的数据结构（`DashMap` 替代 `HashMap`）
- 添加后台清理过期记录（可选）

## 3. 审计日志

### 配置

```yaml
auth:
  audit_log_enabled: true
  audit_log_path: "/var/log/vllm/audit.json"
```

### 记录内容

每个请求记录：
```json
{
  "timestamp": "2026-04-02T10:30:00.123Z",
  "request_id": "req_abc123",
  "method": "POST",
  "endpoint": "/v1/chat/completions",
  "status": 200,
  "latency_ms": 150,
  "prompt_tokens": 100,
  "completion_tokens": 50,
  "api_key_hash": "sha256:a1b2c3...",
  "client_ip": "192.168.1.1",
  "user_agent": "openai-python/1.0"
}
```

### 实现

- 创建 `AuditLogger` 模块
- 异步写入文件（不阻塞请求）
- 可选：发送到远程日志收集器
- API Key 使用 SHA256 哈希存储（不记录原始 key）

## 4. TLS/SSL 支持 (Rustls)

### 配置

```yaml
server:
  tls_cert: "/path/to/cert.pem"
  tls_key: "/path/to/key.pem"
```

### 实现

- 使用 `rustls` crate（纯 Rust，无需 OpenSSL）
- 新增 `enable_tls()` 方法
- 同时支持 HTTP 和 HTTPS：
  - HTTP: `http://host:port`
  - HTTPS: `https://host:port`（当配置了证书时自动启用）

### 证书格式

- PEM 格式（支持 Let's Encrypt）
- 自签名证书用于测试

## 文件变更

| 文件 | 变更 |
|------|------|
| `crates/server/src/config.rs` | 添加 TLS 和审计日志配置 |
| `crates/server/src/auth.rs` | 改进 Rate Limiter，添加审计日志 |
| `crates/server/src/main.rs` | TLS 初始化 |
| `crates/server/src/audit.rs` | 新建审计日志模块 |

## 验证

- 单元测试：Rate Limiter、审计日志格式化
- 集成测试：API Key 认证、HTTPS 连接