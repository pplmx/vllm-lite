# 安全、可靠性、可观测性与生产就绪度

## 1. 总体判断

项目具备结构化日志、健康检查、Prometheus 类型、认证/授权/TLS/审计模块以及部署清单，
但“实现存在”与“生产启用”之间有明显断层。当前默认配置不适合直接暴露到不可信网络。

## 2. SEC-01：安全模块未接入主路由

**事实**

- `crates/server/src/main.rs:101-108` 仅在 API key 非空时挂载简单 auth。
- `AuthConfig::default()` 的 key 列表为空，因此默认无认证。
- JWT、RBAC、size limit、correlation ID、audit、TLS 均有实现，但 main 未挂载。
- `/debug/metrics`、`/debug/kv-cache`、`/debug/trace`、`/shutdown` 未受 admin 保护。
- RBAC 从可伪造的 `X-User-Role` header 获取角色，未绑定可信 JWT claims。

**根因**

安全能力以独立模块和单元测试交付，缺少唯一的 production router builder 与真实组装
测试；本地开发便利性被隐式当成默认安全策略。

**影响**

未授权用户可消耗推理资源、读取内部状态或触发引擎 shutdown；若错误接入现有 RBAC，
还可能引入权限提升。

**严重性 / 优先级**：Critical / P0  
**复杂度**：中  
**收益**：极高

**方案**

1. Safe-by-default：无 credential 时拒绝启动，`--insecure` 显式用于本地。安全最佳，
   但改变现有开发体验。
2. 保留本地默认开放，但非 loopback bind 时强制认证。兼顾体验，规则更复杂。
3. 将 TLS、WAF、限流交给 Envoy/Ingress，应用只验证可信身份。生产常见，但必须把
   信任边界、header 清洗和直连防护写清楚。

推荐建立单一 `build_router(config)`，固定 middleware 顺序并以真实 app state 做 e2e。

## 3. REL-01：无界入口和不可靠 token 交付

**事实**

- `crates/server/src/api.rs:14-19` 明确使用 `mpsc::UnboundedSender`。
- `crates/server/src/main.rs:91` 创建无界 engine channel。
- `crates/core/src/scheduler/batch.rs:57-60` 使用 `try_send`，失败被忽略。
- `backpressure.rs` 标记 dead code，未接入主路径。

**根因**

Engine actor 接口最初以简化调用为目标，没有定义 admission control、队列容量、等待
预算和客户端消费速度之间的契约。

**影响**

请求突发时消息和 scheduler queue 可无限增长；慢客户端可能导致 token 静默丢失；
系统不会给调用者明确 overload 信号。

**严重性 / 优先级**：High / P0  
**复杂度**：中  
**收益**：极高

**建议设计**

- 有界 admission queue，满载返回 429 或 503，并带 `Retry-After`。
- 对 waiting/running/token-buffer 分别设置容量和指标。
- token channel 不能静默丢弃：可等待、取消该序列或明确终止流。
- 客户端断开必须发送 `CancelRequest`，释放 scheduler 和 KV 资源。
- 建立每租户/模型的并发、prompt token 和 output token 配额。

## 4. 输入边界与 API 安全

当前 chat validation 主要检查 model/messages 非空，缺少可靠的 prompt/context 上限；
body size middleware 虽有测试但未挂载。建议同时在三层限制：

1. HTTP body byte limit：防止 JSON 分配攻击。
2. tokenization 后 context limit：返回 OpenAI 风格 `context_length_exceeded`。
3. scheduler admission budget：按估算 KV、最大输出和并发拒绝超预算请求。

服务端未发现接受用户 URL 并主动出站请求的主要路径，因此 SSRF 不是当前首要风险。
模型路径由部署者提供，仍应 canonicalize 并限制到允许根目录，但优先级低于公开端点和
DoS。

## 5. OBS-01：指标数据源不可信

**事实**

- `main.rs:113` 创建独立 `EnhancedMetricsCollector` 供 `/metrics` 使用。
- Engine/Scheduler 维护另一套真实运行指标。
- health details 通过 Engine message 获取指标，而 `/metrics` 使用独立 collector。

**根因**

HTTP exporter 与 engine actor 分别演进，没有指定指标单一真相源。

**影响**

Prometheus 可能抓到零值或与健康端点矛盾的值，导致容量、SLO、告警和事故诊断失真。

**严重性 / 优先级**：Critical / P0  
**复杂度**：低–中  
**收益**：极高

**方案**

1. Engine 与 server 共享同一个线程安全 collector。读取快，但要控制热路径锁开销。
2. `/metrics` 通过 Engine snapshot message 获取。边界清晰，但抓取会进入 actor 队列。
3. Engine 定期发布不可变 snapshot，HTTP 无锁读取。复杂度中，长期扩展最好。

最低指标集应包括 queue depth、admission rejection、running/waiting sequences、
prefill/decode tokens、TTFT、TPOT、batch size、KV 使用率/命中率/驱逐、取消、错误、
token channel backpressure 和模型 forward 时间。

## 6. 日志与追踪

结构化 tracing、JSON 文件输出和生命周期日志是优势。但 correlation middleware 未挂载，
request ID 不能贯穿 HTTP、scheduler、model 和 token stream；文档仍提及已移除的
OpenTelemetry feature。

建议：

- 统一 request_id/seq_id/model/phase/error_code 字段。
- correlation ID 由边界验证或生成，通过 `tracing::Span` 传播。
- 禁止记录 prompt、token、API key、JWT 和模型私有路径等敏感内容。
- 先完善结构化 span，再接 OTLP；不要仅添加依赖而没有 trace topology。

## 7. 健康检查与关停

**问题**

- readiness 由静态布尔值初始化，不反映模型加载、GPU OOM 或 engine thread 状态。
- HTTP 支持 graceful shutdown，但 engine thread 未 join。
- 关停没有先置 ready=false、停止 admission、drain 在途请求和超时强退的阶段。
- `/shutdown` 只通知 engine，HTTP 仍可能继续接收。

**目标流程**

```text
SIGTERM/admin request
  -> readiness=false
  -> stop accepting new inference
  -> cancel or drain queued requests
  -> wait in-flight with deadline
  -> flush metrics/logs
  -> shutdown engine and join thread
  -> exit
```

## 8. 部署阻断项

### Docker

- Dockerfile builder 为 Rust 1.82，低于 MSRV 1.88。
- HEALTHCHECK 调用不存在的 `vllm-server --health-check`。
- build 未一致使用 `--locked`。

### Compose

- build target 指向不存在/未命名的 `runtime` stage。
- Prometheus 挂载引用不存在的 `monitoring/prometheus.yml`，实际文件在 `config/`。
- load test 路径不存在。

### Helm/Kubernetes

- Chart 使用 `MODEL_PATH`，CLI 要求 `VLLM_MODEL`。
- 默认 NetworkPolicy 关闭，缺少 API key/TLS secret 的完整示例。
- HPA 若依赖失真的 `/metrics`，扩缩容决策也不可信。

这些是 P0，因为它们使官方部署路径直接失败或产生错误安全预期。修复后应增加：

- `docker build` + container health smoke。
- `docker compose config` 和服务启动 smoke。
- `helm lint`、`helm template`、kind/minikube smoke。
- 非 root、read-only FS、drop capabilities、secret mount 和 NetworkPolicy 检查。

## 9. TLS 与 CORS

TLS 配置模块存在，但 main 仍使用明文 `TcpListener`。生产更推荐 Ingress/Envoy terminate
TLS，应用只监听受保护网络；若支持裸机，则需完整 rustls 接线和证书 reload 策略。

仓库无实际 CORS layer。对于服务端 SDK 不是问题；若支持浏览器直连，应使用显式
allowlist，禁止默认 `*` 与 credential 组合。

## 10. Batch 与 Embeddings 的生产语义

- Batch API 没有 worker 或持久化，进程重启丢失且状态无法正常推进。短期应返回 501
  或标记 experimental，而不是创建永不完成的 job。
- Embeddings 调用模型 `embed()`，但并非所有 causal LM checkpoint 都能提供质量可用、
  归一化和维度稳定的 embedding。应维护支持矩阵并在模型加载时校验 capability。

## 11. 生产门槛

在以下条件完成前，不建议使用“生产就绪”：

1. 前缀缓存和 sampling 正确性修复。
2. 默认认证、admin 隔离、body/context limit。
3. 有界 admission、取消传播和 token 不丢失。
4. 统一真实指标、动态 readiness 和完整 shutdown。
5. Docker/Compose/Helm 可执行 smoke test。
6. 至少一个真实 GPU checkpoint 的持续回归。
7. 明确兼容矩阵、容量基准、升级/回滚和事故手册。
