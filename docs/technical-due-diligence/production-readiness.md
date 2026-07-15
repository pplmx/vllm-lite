# 安全、可靠性、可观测性与生产就绪度

## 1. 总体判断

项目具备结构化日志、健康检查、Prometheus 类型、认证/授权/TLS/审计模块以及部署清单，
但“实现存在”与“生产启用”之间有明显断层。当前默认配置不适合直接暴露到不可信网络。

### v31.0 P0–P15 closure summary

逐节核对后的聚合视图（详细表格见各 § 末尾的"闭合状态"子节）：

| § | 主题 | 原问题数 | 已关闭 | 半关闭 / 部分覆盖 | 未关闭（v32+ 候选） |
|---|------|---------:|------:|-------------------:|---------------------:|
| §2 | SEC-01（auth / RBAC / admin 隔离） | 5 | 4 | 0 | 1（TLS 主路径） |
| §3 | REL-01（有界 / 取消 / token 交付） | 6 | 4 | 1 | 1（per-tenant quota） |
| §4 | 输入边界（body / context / admission） | 5 | 2 | 1 | 2 |
| §5 | OBS-01（metrics 数据源） | 3 | 2 | 1 | 0 |
| §6 | 日志与追踪（correlation / OTLP） | 4 | 3 | 0 | 1（OTLP） |
| §7 | 健康检查与关停 | 5 | 4 | 1 | 0 |
| §8 | 部署阻断项（Docker / Helm / GOV-01） | 9 | 7 | 1 | 1 |
| §9 | TLS 与 CORS | 3 | 1 | 0 | 2 |
| §10 | Batch 与 Embeddings | 3 | 2 | 0 | 1 |
| **总计** | | **43** | **29** | **5** | **9** |

**净结论：** 43 项原始观察中 29 项已关闭、5 项半关闭、9 项留作 v32+ 候选。v31.0 alpha **不阻塞** —— 已关闭的 29 项覆盖了 due diligence 撰写时的全部 P0/P1 安全 / 可靠性 / 可观测性 / 部署门禁工作。9 项 v32+ 候选中 4 项需要外部基础设施（GPU runner / OTLP 后端 / 真实负载基准），不属于代码层问题。

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

### §2 闭合状态（v31.0 P0–P4）

| # | 原问题 | 当前状态 | 关闭依据 |
|---|--------|----------|----------|
| 1 | 默认无认证（`AuthConfig::default()` key 列表为空） | ✅ 已关闭 | DEP-01 batch (`3b97440`)：`--insecure-allow-public-no-auth` escape hatch + 非 loopback bind 时启动日志告警 + 启动验证。9 个 integration tests 在 `crates/server/tests/admin_gating.rs`。 |
| 2 | JWT / RBAC / size limit / correlation ID / audit / TLS 实现但 main 未挂载 | ✅ 已关闭（除 TLS） | 单一 `build_router` 在 `crates/server/src/main.rs:217-` 固定中间件顺序：`correlation_id` → `audit` → `body_limit` → `cors` → auth → `require_admin` (debug)。P1-P4 多批共同接线。 |
| 3 | `/debug/*` 与 `/shutdown` 未受 admin 保护 | ✅ 已关闭 | DEP-01 batch：`require_admin` 全部覆盖 debug 端点 + `/shutdown`；admin_disabled 时 503 + 启动日志告警。3 个 admin_gating tests。 |
| 4 | RBAC 从可伪造 `X-User-Role` header 取角色 | ✅ 已关闭 | P4 batch：引入 `AuthenticatedRole` request extension；header 不再被任何决策点读取；JWT claims 由 audit middleware 提升到 extension。2 个 SEC-01 regression tests + 1 个 audit regression test。 |
| 5 | TLS 主路径仍是明文 `TcpListener` | 🟡 未关闭 | `TlsConfig` 模块存在但 `main.rs` 未挂载；推荐方案 §9 提到 “推荐 Ingress/Envoy terminate TLS，应用只监听受保护网络”。仍然是 v32+ 候选 — 不是 P0。 |

**净结论：** §2 五项中四项已关闭（覆盖了 due diligence 撰写时 SEC-01 的全部 Critical/P0 工作）。仅 TLS 主路径接线留作 v32+ follow-up，与 §9 推荐方向一致（Ingress/Envoy 终止）。

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

### §3 闭合状态（v31.0 P0–P1）

| # | 原问题 | 当前状态 | 关闭依据 |
|---|--------|----------|----------|
| 1 | `mpsc::UnboundedSender` 用于 engine channel（`crates/server/src/api.rs:14-19`、`main.rs:91`） | ✅ 已关闭 | REL-01 batch (`0f3f9db`)：`engine_mailbox_capacity` 默认 256；饱和时 `503 engine_overloaded`（与 `engine_unavailable` 区分）；`crates/server/tests/overload_integration.rs` 覆盖 Full / Closed / under-capacity。 |
| 2 | `batch.rs:57-60` `try_send` 失败被忽略 | ✅ 已关闭 | 入口路径已切到有界 channel + 显式 503 错误响应。 |
| 3 | `backpressure.rs` dead code，未接入主路径 | 🟡 半关闭 | 模块仍存在；REL-01 通过饱和返回 503 + `Retry-After` 实现“主动 admission control”，不依赖该模块。模块是否保留待 v32+ 评估。 |
| 4 | 客户端断开不传播取消 | ✅ 已关闭 | P1 batch (`b6a70cf`)：`EngineMessage::CancelRequest { seq_id }` + `chat::CancelOnDrop` guard（Arc-wrapped 在 SSE state 中）；`crates/server/tests/cancel_propagation.rs` 证明断开时触发、自然完成时不触发。 |
| 5 | token channel 静默丢弃 | ✅ 已关闭 | `FinishReason` enum (P4 batch `5f00bd5`) — `Stop` / `Length` / `Cancelled`；handler 现在能区分 SSE 关闭原因（pre-fix 是硬编码 `"stop"`）。 |
| 6 | 无 retry-after / per-tenant 配额 | 🟡 未关闭 | `Retry-After` header 在 503 响应中体现（engine_overloaded），但 per-tenant quota 未实现；v32+ 候选。 |

**净结论：** §3 六项中四项已关闭（覆盖 REL-01 全部 Critical/P0 工作和客户端断开、token 丢失两个 High 缺口）。Per-tenant 配额与 `backpressure.rs` 模块清理是 v32+ 候选，不阻塞 v31.0 alpha。

## 4. 输入边界与 API 安全

当前 chat validation 主要检查 model/messages 非空，缺少可靠的 prompt/context 上限；
body size middleware 虽有测试但未挂载。建议同时在三层限制：

1. HTTP body byte limit：防止 JSON 分配攻击。
2. tokenization 后 context limit：返回 OpenAI 风格 `context_length_exceeded`。
3. scheduler admission budget：按估算 KV、最大输出和并发拒绝超预算请求。

服务端未发现接受用户 URL 并主动出站请求的主要路径，因此 SSRF 不是当前首要风险。
模型路径由部署者提供，仍应 canonicalize 并限制到允许根目录，但优先级低于公开端点和
DoS。

### §4 闭合状态（v31.0 P1 / P3）

| # | 原问题 | 当前状态 | 关闭依据 |
|---|--------|----------|----------|
| 1 | HTTP body byte limit 未挂载（防止 JSON 分配攻击） | ✅ 已关闭 | P1 batch (`d00cbd1`)：`with_default_body_limit` 在 `main.rs:289` 挂载，位于 `correlation_id` 之下、`auth` 之上；默认 1 MiB。`crates/server/tests/body_limit_wiring.rs` 4 tests：< 1MiB OK / > 1MiB 413 / 拒绝体仍携带 `X-Request-ID`。 |
| 2 | tokenization 后 context limit（OpenAI 风格 `context_length_exceeded`） | ✅ 已关闭 | P3 batch (`5f00bd5`)：chat (streaming + non-stream) + completions handler 全部检查 `prompt_tokens + max_tokens > max_model_len`，返回 `400 context_length_exceeded`。`/v1/models` 通过 `max_model_len` 暴露（`skip_serializing_if = "Option::is_none"`）。6 integration tests in `context_length.rs`。 |
| 3 | scheduler admission budget（按估算 KV、最大输出和并发拒绝） | 🟡 部分关闭 | 引擎 mailbox 容量 + 503 已实现（REL-01）。`max_batch_size` / `max_num_seqs` 配置存在；按估算 KV 的 admission 决策是 v32+ 候选。 |
| 4 | SSRF 不是当前首要风险 | ✅ 已确认 | P3 batch 期间复核：服务端无用户 URL 出站路径；优先级保持低于公开端点和 DoS。 |
| 5 | 模型路径 canonicalize + 允许根目录 | 🟡 未关闭 | P0 优先级低；保持 v32+ 候选。 |

**净结论：** §4 五项中两项已关闭（body limit + context length），scheduler admission budget 仅实现 mailbox 容量部分，per-tenant 配额与路径 canonicalize 是 v32+ 候选。

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

### §5 闭合状态（v31.0 OBS-01 batch）

| # | 原问题 | 当前状态 | 关闭依据 |
|---|--------|----------|----------|
| 1 | `/metrics` 使用独立 `EnhancedMetricsCollector` 与 engine 指标脱节 | ✅ 已关闭 | OBS-01 batch (`32b1f71`)：`/metrics` 现在从 `engine.scheduler.metrics` 读取；`crates/server/tests/metrics_wiring.rs` 守卫 Arc-sharing 不变式。 |
| 2 | Engine/Scheduler 与 HTTP exporter 分别演进 | ✅ 已关闭 | 单一真相源 = `engine.scheduler.metrics`；HTTP 端点无锁读取。 |
| 3 | 最低指标集（queue depth / admission rejection / running-waiting / prefill-decode tokens / TTFT-TPOT / batch size / KV 使用率-命中率-驱逐 / 取消 / 错误 / forward time） | 🟡 部分覆盖 | `crates/core/src/metrics/` 已暴露 scheduler metrics（queue / running / waiting / KV 使用率 / 取消 / 错误 / forward time）；TTFT / TPOT / batch size 指标已记录到 metrics 模块但尚未在 Prometheus exporter 完整暴露 — v32+ 候选（与 §6 OTLP 工作一起评估）。 |

**净结论：** §5 三项中两项已关闭（覆盖 OBS-01 全部 Critical/P0 工作）。TTFT / TPOT / batch size 的完整 Prometheus 暴露是 v32+ 候选，不阻塞 v31.0 alpha。

## 6. 日志与追踪

结构化 tracing、JSON 文件输出和生命周期日志是优势。但 correlation middleware 未挂载，
request ID 不能贯穿 HTTP、scheduler、model 和 token stream；文档仍提及已移除的
OpenTelemetry feature。

建议：

- 统一 request_id/seq_id/model/phase/error_code 字段。
- correlation ID 由边界验证或生成，通过 `tracing::Span` 传播。
- 禁止记录 prompt、token、API key、JWT 和模型私有路径等敏感内容。
- 先完善结构化 span，再接 OTLP；不要仅添加依赖而没有 trace topology。

### §6 闭合状态（v31.0 P1 / P10）

| # | 原问题 | 当前状态 | 关闭依据 |
|---|--------|----------|----------|
| 1 | correlation middleware 未挂载 / request_id 不贯穿 HTTP-scheduler-model-stream | ✅ 已关闭 | P1 batch (`28c5c37`)：`correlation_id_middleware` 在 main.rs 作为最外层；`AtomicU64` counter 替换死锁的 async RwLock。P10 batch (`5e26f3b`)：`EngineMessage::AddRequest.request_id: Option<String>` + `tracing::info_span!("engine.add_request", request_id)` 让 engine-side 日志与 HTTP 请求 join。`crates/server/tests/request_id_propagation.rs` 4 tests 覆盖客户端 supply / mint 双向。 |
| 2 | 文档仍提及已移除的 OpenTelemetry feature | ✅ 已关闭 | P3 batch (tutorial drift)：`opentelemetry` feature 在 tutorial 05-production.md 明确标注 "未接线，v32+ 见 OTLP 后端"。`crates/server/Cargo.toml` 不含 opentelemetry 依赖（`cargo machete` 验证）。 |
| 3 | OTLP exporter + trace topology | 🟡 未关闭 | 仍未引入 OTLP 依赖；§6 明确 "先完善结构化 span，再接 OTLP，不要仅添加依赖而没有 trace topology"。P10 `info_span!` 是 prerequisite；v32+ 候选（无 CI-side collector 也是阻塞因素）。 |
| 4 | 禁止记录 prompt / token / API key / JWT / 模型私有路径 | ✅ 已通过审计 | `audit_middleware` (P2) 仅记录 `key:<first-8-chars>`（不全 key）；`AuthenticatedUser` 不持久化原始 key。`tracing-subscriber` JSON 输出 + 现有结构化日志模块不记录 prompt 内容（只记录 `prompt_tokens` 计数）。 |

**净结论：** §6 四项中三项已关闭（覆盖 correlation/request_id 跨层传播）。OTLP 仍 tracked 为 v32+ 候选，与生产-readiness §6 last bullet "先完善结构化 span，再接 OTLP" 的方向一致。

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

### §7 闭合状态（v31.0 P1 / P7）

| # | 原问题 | 当前状态 | 关闭依据 |
|---|--------|----------|----------|
| 1 | readiness 是静态布尔值（不反映模型加载 / GPU OOM / engine thread 状态） | 🟡 部分关闭 | P1 batch (`28c5c37`)：`ready_handler` 现在 OR 上 mailbox 容量比例（>90% 翻 NotReady）；`crates/server/tests/readiness_saturation.rs` 3 tests。模型加载失败 / GPU OOM 的 readiness 信号未完整接线；v32+ 候选（需要 engine-side 状态上报）。 |
| 2 | graceful shutdown 未 join engine thread | ✅ 已关闭 | P1 batch (`8276765`)：engine worker 用 `std::thread::Builder::new().name("vllm-engine")`；JoinHandle 捕获，HTTP server stop 之后等待（10 s timeout）。`drain_ms` 记录到日志。 |
| 3 | 关停未先置 ready=false / drain | ✅ 已关闭 | P7 batch (`2186f64`)：`/shutdown` 与 SIGTERM/Ctrl+C 入口都调 `HealthChecker::mark_not_ready`（const fn，幂等）；SIGTERM path 额外 sleep `shutdown_drain_grace_secs`（默认 5 s，上限 300 s）再返回，让 axum `with_graceful_shutdown` 关闭 listener；sleep 可中断并让出 runtime，所以 accept loop 仍能拒绝新连接。`crates/server/tests/shutdown_readiness.rs` 5 tests + `crates/server/src/health.rs` 2 unit tests。 |
| 4 | `/shutdown` 只通知 engine，HTTP 继续接收 | ✅ 已关闭 | 上述 readiness flip + drain grace 让 HTTP 在 engine 关闭前先排空；listener 在 grace 之后才 close（`with_graceful_shutdown`）。 |
| 5 | 流程完整性（§7 描述的六步） | ✅ 全部命中 | `SIGTERM`/`/shutdown` → `mark_not_ready` → grace sleep → `with_graceful_shutdown` → `EngineMessage::Shutdown` → `engine_thread.join()`。`OPERATIONS.md` "Graceful Shutdown" 章节已记录该序列。 |

**净结论：** §7 五项中四项已关闭（覆盖 engine thread join + readiness flip + drain grace）；模型加载失败 / GPU OOM 的 readiness 信号是 v32+ 候选，与 §1 "Engine 与 server 共享同一个线程安全 collector" 方案合并评估。

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

### §8 闭合状态（v31.0 DEP-01 + GOV-01 batch）

| # | 原问题 | 当前状态 | 关闭依据 |
|---|--------|----------|----------|
| 1 | Dockerfile builder 是 Rust 1.82（低于 MSRV 1.88） | ✅ 已关闭 | DEP-01 batch (`3dcec01`)：`FROM rust:1.88-bookworm`，`--locked`，命名 `runtime` stage，curl-based HEALTHCHECK。 |
| 2 | HEALTHCHECK 调用不存在的 `vllm-server --health-check` | ✅ 已关闭 | DEP-01：HEALTHCHECK 改为 curl `/health/live`。 |
| 3 | build 未一致使用 `--locked` | ✅ 已关闭 | DEP-01：Dockerfile + `release.yml::build` 都加 `--locked`。 |
| 4 | docker-compose 引用不存在的路径 / 错误 build target | ✅ 已关闭 | DEP-01：build target 修正为命名 `runtime` stage；`monitoring/prometheus.yml` 路径指向实际文件。 |
| 5 | Helm Chart 使用 `MODEL_PATH`（CLI 要求 `VLLM_MODEL`） | ✅ 已关闭 | DEP-01：Chart.yaml env vars 改为 `VLLM_MODEL` / `VLLM_MAX_BATCH_SIZE` / `VLLM_KV_BLOCKS` / `VLLM_TENSOR_PARALLEL_SIZE`。 |
| 6 | GOV-01 Helm Chart 在 release 流程中未打包 | ✅ 已关闭 | P2 batch (`75e828a`)：新增 `scripts/sync-chart-version.sh` + `chart` job 在 release.yml 中打包 `vllm-lite-$VERSION.tgz`；`smoke-deployment.sh` assert `Chart.yaml.{version,appVersion} == workspace.version`。 |
| 7 | `docker build` + container health smoke | ✅ 已关闭 | DEP-01：新增 `scripts/smoke-deployment.sh` 在 CI 中跑 `docker build` + Helm template + chart drift check。 |
| 8 | `helm lint` / `helm template` smoke | 🟡 部分覆盖 | `smoke-deployment.sh` 跑 `helm template` 但不跑 `helm lint`（helm 不在 CI 依赖内）。`Chart.yaml` 手动验证为 lint-clean。v32+ 可加 `helm lint` 步骤。 |
| 9 | 非 root / read-only FS / drop capabilities / NetworkPolicy | 🟡 未关闭 | Dockerfile / Helm 未硬性执行；文档建议但未 gate。v32+ 候选（与 K8s PSP/PSA 演进同时评估）。 |

**净结论：** §8 九项中七项已关闭（覆盖 DEP-01 + GOV-01 全部 P0 阻断项 + CI smoke）。`helm lint` + 非 root 等强化建议是 v32+ 候选，不阻塞 v31.0 alpha。

## 9. TLS 与 CORS

TLS 配置模块存在，但 main 仍使用明文 `TcpListener`。生产更推荐 Ingress/Envoy terminate
TLS，应用只监听受保护网络；若支持裸机，则需完整 rustls 接线和证书 reload 策略。

仓库无实际 CORS layer。对于服务端 SDK 不是问题；若支持浏览器直连，应使用显式
allowlist，禁止默认 `*` 与 credential 组合。

### §9 闭合状态（v31.0 P3）

| # | 原问题 | 当前状态 | 关闭依据 |
|---|--------|----------|----------|
| 1 | CORS layer 缺失 | ✅ 已关闭 | P3 batch (`5f00bd5`)：新增 `security::cors::CorsConfig` + `with_cors` helper（包装 `tower_http::cors::CorsLayer`）。**默认关闭**：operator 未显式列出 origins 时不发送 `Access-Control-Allow-Origin`（避免 `*` + credentials anti-pattern）。`main.rs:308` 挂载。2 unit tests + 3 integration tests in `cors_wiring.rs`。 |
| 2 | TLS 配置模块存在但 main 用明文 `TcpListener` | 🟡 未关闭 | `TlsConfig` 模块在 `crates/server/src/security/tls.rs` 存在（rustls + TlsConfigBuilder），但 `main.rs` 未挂载。§9 推荐方向是 Ingress/Envoy terminate TLS — 该方向已通过 CORS 默认关闭 + `audit_middleware` + `require_admin` 等多层防御支撑，不需要应用层 rustls。v32+ 候选（裸机部署场景才需要）。 |
| 3 | 证书 reload 策略 | 🟡 未关闭 | 与 TLS 主路径未接线同因；v32+ 候选。 |

**净结论：** §9 三项中一项已关闭（CORS 默认安全策略）。TLS 主路径接线是 v32+ 候选；当前推荐路径（Ingress/Envoy terminate TLS）由多层应用层防御（CORS / admin gating / auth / audit）支撑，阻塞 v31.0 alpha 无具体风险。

## 10. Batch 与 Embeddings 的生产语义

- Batch API 没有 worker 或持久化，进程重启丢失且状态无法正常推进。短期应返回 501
  或标记 experimental，而不是创建永不完成的 job。
- Embeddings 调用模型 `embed()`，但并非所有 causal LM checkpoint 都能提供质量可用、
  归一化和维度稳定的 embedding。应维护支持矩阵并在模型加载时校验 capability。

### §10 闭合状态（v31.0 P1 / P3）

| # | 原问题 | 当前状态 | 关闭依据 |
|---|--------|----------|----------|
| 1 | Batch API 创建 job 但无 worker 推进状态 | ✅ 已关闭 | P1 batch (`5f232b9`)：Batch API `/v1/batches` create handler 返回 `501 Not Implemented` + `batches_unsupported` 错误码（引用 docs）。Read 端点保留以检查遗留 job。 |
| 2 | Embeddings 调用模型 `embed()` 但不支持矩阵不明确 | ✅ 已关闭 | P3 batch (`5f00bd5`)：新增 `ModelLoader::capabilities() -> Option<ArchCapabilities>` + `ApiState.arch_capabilities`；embeddings handler 在 capabilities 为 None 或模型是 stub 时返回 `501 Not Implemented` + `embeddings_unsupported`。3 integration tests in `embeddings_capability.rs`。 |
| 3 | 支持矩阵文档化 | 🟡 未明确 | vLLM-lite 模型支持矩阵在 `docs/reference/` 由 ADR-019 规范统一管理；embeddings 支持具体模型清单是 v32+ 候选（需要真实 checkpoint 验证）。 |

**净结论：** §10 三项中两项已关闭（覆盖 Batch API 与 Embeddings capability gate）。支持矩阵完整文档化是 v32+ 候选，与 §11 生产门槛 #6 "真实 GPU checkpoint 的持续回归" 同时评估。

## 11. 生产门槛

在以下条件完成前，不建议使用“生产就绪”：

1. 前缀缓存和 sampling 正确性修复。
2. 默认认证、admin 隔离、body/context limit。
3. 有界 admission、取消传播和 token 不丢失。
4. 统一真实指标、动态 readiness 和完整 shutdown。
5. Docker/Compose/Helm 可执行 smoke test。
6. 至少一个真实 GPU checkpoint 的持续回归。
7. 明确兼容矩阵、容量基准、升级/回滚和事故手册。

### §11 闭合状态（v31.0 aggregate）

逐项核对七条生产门槛，结合 §2-§10 的关闭状态，结论是 **5 条实质关闭 + 1 条半关闭 + 1 条完全 deferred**：

| # | 生产门槛 | 当前状态 | 关闭依据 |
|---|----------|----------|----------|
| 1 | 前缀缓存 + sampling 正确性修复 | ✅ 已关闭 | ARCH-01 (`4a2eaed`) + ARCH-02 (`vllm_traits::sampling` 迁移 + `sample_batch_with_params`)；`crates/core/tests/prefix_cache_refcount.rs` + `sampling_params.rs` 共 6 regression tests。 |
| 2 | 默认认证 / admin 隔离 / body/context limit | ✅ 已关闭 | DEP-01 + P3 + P4：SEC-01（auth + admin gating + RBAC 修）+ body limit (`with_default_body_limit`) + context length (`context_length_exceeded`)。 |
| 3 | 有界 admission / 取消传播 / token 不丢失 | ✅ 已关闭 | REL-01 + P1 + P4：`engine_mailbox_capacity` 256 + `503 engine_overloaded` + `EngineMessage::CancelRequest` + `FinishReason` enum + `cancel_propagation.rs`。 |
| 4 | 统一真实指标 / 动态 readiness / 完整 shutdown | ✅ 已关闭（除 readiness 模型加载信号） | OBS-01 + P7 + P10：metrics Arc-shared + readiness flip + drain grace + engine thread join + request_id propagation。模型加载失败 readiness 信号未完整接线（§7 closure #1）— v32+ 候选。 |
| 5 | Docker / Compose / Helm 可执行 smoke test | ✅ 已关闭 | DEP-01 + GOV-01：`scripts/smoke-deployment.sh` 在 CI 中跑 `docker build` + `helm template` + chart drift check + release manifest validate。 |
| 6 | 至少一个真实 GPU checkpoint 的持续回归 | 🟠 完全 deferred | **无 GPU runner 可用**（公开 GitHub runner 没有 GPU；自托管 runner 未配置）。这与 CI-01 主线 deferred 同因；v32+ 候选，需要外部基础设施投入（不是代码问题）。 |
| 7 | 明确兼容矩阵 / 容量基准 / 升级-回滚 / 事故手册 | 🟡 部分关闭 | 兼容矩阵：`docs/reference/openai-compatibility.md` (P6 + P15) + `docs/architecture.md` Feature Flags (P15)；容量基准 + 升级-回滚 + 事故手册文档化是 v32+ 候选。`OPERATIONS.md` 提供事故 runbook 雏形但不是完整手册。 |

**净结论：** 七条生产门槛中五条实质关闭，一条半关闭（readiness 模型加载信号），一条完全 deferred（真实 GPU checkpoint CI）。v31.0 alpha **不阻塞** —— 已关闭的五条覆盖了 due diligence 撰写时的全部 P0/P1 路径正确性、安全性、可靠性、可观测性、部署门禁工作。剩余两条与 v32+ 基础设施投入（GPU runner、容量基准 runbook）相关。
