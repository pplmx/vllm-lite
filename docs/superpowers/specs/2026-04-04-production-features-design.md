# Production Features Design (Simplified)

**Date**: 2026-04-04
**Status**: Approved (Simplified)
**Goal:** Complete production-ready features following vLLM upstream approach - auth and rate limiting in-server, everything else external.

---

## 1. 现有实现 (Already Implemented)

vllm-lite 已经实现了以下生产就绪特性，与 vLLM 上游一致：

### 1.1 认证 (Authentication)

- **API Key**: `Authorization: Bearer <key>` 方式
- **配置方式**: config 文件、env var、命令行
- **实现**: `crates/server/src/auth.rs`

```rust
// 当前实现 (已足够)
pub struct AuthMiddleware {
    api_keys: Arc<Vec<String>>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
}
```

### 1.2 Rate Limiting

- **per-key 限流**: 每个 API key 独立计数
- **滑动窗口**: 基于时间的请求限制
- **配置**: `rate_limit_requests`, `rate_limit_window_secs`

### 1.3 监控 (Monitoring)

- **Prometheus**: `/metrics` 端点
- **指标**: 延迟、吞吐量、batch size、KV cache 使用率、prefix cache hit rate

### 1.4 健康检查

- `/health` - 服务健康状态
- `/ready` - 就绪检查

---

## 2. 不需要实现 (External)

按照 vLLM 上游做法，以下特性由外部组件处理：

| 特性 | 原因 |
|------|------|
| TLS/HTTPS | 由 nginx/负载均衡处理 |
| 多租户 | 如需要可使用 API Key 区分 |
| 审计日志 | 外部日志系统收集 |
| JWT 认证 | API Key 已足够 |
| 告警 | 监控系统处理 |
| Quota 管理 | 外部计费系统处理 |

---

## 3. 架构

```
server/
├── auth.rs          # ✅ 已有: API Key 认证 + Rate Limiting
├── config.rs        # ✅ 已有: 认证配置
├── metrics.rs       # ✅ 已有: Prometheus 指标
└── main.rs          # ✅ 已有: /health, /metrics 端点
```

---

## 4. 验收标准

- [x] API Key 认证工作正常
- [x] Rate Limiting 工作正常
- [x] /metrics 端点返回 Prometheus 格式指标
- [x] /health 端点工作正常
- [x] 配置从文件/env/命令行加载
