# ROI 导向演进路线图

## 1. 排序原则

1. 正确性和安全性优先于性能与功能数量。
2. 先修复真实生产链路，再继续增加未接线模块。
3. 用端到端验收替代“文件/类型已存在”的完成定义。
4. 先建立可复现基线，再做复杂 kernel 或分布式投资。
5. 每个阶段都应减少文档与实现差距。

## 2. 立即改进：1–2 周

目标：停止错误宣传，修复低成本高风险问题，建立可信基线。

### P0 正确性与可靠性

1. 修复 prefix cache retain/release；在完成前可默认禁用共享。
2. 对非 greedy 采样参数返回明确错误，或接入统一 sampler 的最小版本。
3. 将 Engine 入口改为有界 channel，满载返回标准 overload 错误。
4. 禁止 token `try_send` 静默失败；客户端断开触发取消。

### P0 安全与可观测性

5. 保护或禁用 debug/shutdown；非 loopback 默认要求认证。
6. 在真实 router 挂载 body limit、correlation ID 和基础审计。
7. 让 `/metrics` 读取真实 engine snapshot；修正 readiness。

### P0 文档与部署

8. 修复 README 模型参数、自动下载、认证和性能声明。
9. Docker 升级 Rust 1.88、修复 HEALTHCHECK、使用 `--locked`。
10. 修复 Compose target/path 与 Helm `VLLM_MODEL`。
11. Batch API 返回 501/experimental，直到执行器存在。
12. 填写 CoC/SECURITY 联系方式，统一示例端口。

**复杂度**：约 2–4 工程人周，可拆为 6–10 个小 PR。  
**预期收益**：将“不可安全试用”提升到“可验证的受限单机 alpha”。

## 3. 短期演进：1–2 个月

目标：形成可持续的单机正确性与发布闭环。

### 推理语义

1. 重构 `ModelBackend` 输出 logits，由 Engine/Sampler 统一 token 决策。
2. 支持 temperature、top-p、top-k、stop、seed 和正确 finish_reason。
3. 合并 legacy draft 与 resolver 路径，实现 sampling-aware speculative verification。
4. 建立 block manager，统一 prefix、preemption、cancel、finish 的生命周期。

### 生产链路

5. 完成 admission control、per-request budget、context limit 和 drain shutdown。
6. 明确 API 兼容矩阵，使用官方 OpenAI SDK 做 contract tests。
7. 建立统一指标 snapshot、request span 和基础 SLO dashboard。
8. 为 auth/RBAC 明确信任边界；JWT claims 注入角色，禁止用户伪造 header。

### 工程闭环

9. 对齐 CI 和 `just ci` feature matrix。
10. 每周运行固定小模型 GPU/checkpoint 测试。
11. 恢复 benchmark workflow，发布可复现 CPU/GPU 基线。
12. 统一 SemVer，完成 binary/image/Chart release automation。
13. 建立 `rust-toolchain.toml`、Dependabot 和 SBOM。

**复杂度**：约 2–3 名工程师持续 1–2 月。  
**预期收益**：达到“可用于受控环境和有限流量的单机 beta”。

## 4. 中期架构演进：3–6 个月

目标：把调度层能力转化为真实 GPU 吞吐。

### 数据面

1. 定义 `KernelBackend` 与 reference/optimized 双实现。
2. 优先实现 batched decode：ragged metadata、paged KV 读取、批量 logits。
3. 再实现 chunked prefill、prefill/decode 公平调度和 continuous batching。
4. 接入或开发 fused RMSNorm/RoPE/attention/MLP kernel。
5. 建立 CUDA Graph shape cache，并评估与 speculative 的组合。
6. 通过 profiler 定位 memcpy、allocation、kernel launch 和锁竞争。

### 模型与量化

7. 为每个“支持”架构建立真实 checkpoint conformance suite。
8. Stub 架构移出支持矩阵。
9. 明确 GGUF/量化是加载后反量化还是运行时量化；以显存和吞吐衡量。
10. 收缩 public API，将 kernel 与 scheduler internals 改为 experimental/internal。

### 可靠性

11. 进行 24h soak、突发流量、慢客户端、取消风暴、OOM 和模型加载失败测试。
12. 定义 SLO：成功率、TTFT、TPOT、queue delay、OOM/restart rate。

**复杂度**：4–6 名具备 CUDA/ML 系统经验的工程师，风险较高。  
**预期收益**：从“Rust 原型”进化为“具备竞争力的单机推理引擎”。

## 5. 长期愿景：6–24 个月

长期方向应由用户和基准数据选择，而不是同时追求所有成熟项目能力。

### 建议定位

优先成为：

> 模块化、类型安全、可嵌入、对单机/边缘场景友好的 Rust 推理运行时。

这比正面复制 vLLM 的全部 Python/CUDA/分布式生态更有差异化。

### 可选战略

#### 路线 A：单机高性能运行时

- 多 GPU tensor parallel、成熟量化、广泛模型兼容。
- 类似 llama.cpp 的可嵌入和跨平台优势，加上 vLLM 式调度。
- 投资集中，最符合当前模块化基础。

#### 路线 B：生产 serving 平台

- 多租户、模型热切换、autoscaling、rolling upgrade、quota、完整 OpenAI 兼容。
- 需要显著增加控制面、SRE 和安全投入。

#### 路线 C：分布式研究平台

- KV transfer、prefill/decode disaggregation、NCCL TP/PP、容错调度。
- 技术吸引力高，但在单机数据面成熟前 ROI 最低。

推荐先走 A，在明确用户需求后选择性增加 B；C 应设置明确投资门槛。

## 6. 暂缓事项

以下工作当前不应优先：

- 在采样和 KV 生命周期未正确前增加更多模型架构。
- 在单机 batched kernel 未成熟前建设完整多节点 MESI/KV 协议。
- 为提高覆盖率数字而给简单 getter 添加大量低价值测试。
- 在指标数据源错误时建设复杂 dashboard。
- 在版本未统一时发布多个 crates。
- 为保持“兼容”而继续静默接受无效 OpenAI 参数。

## 7. 目标架构

```text
API Gateway / Ingress
  -> Auth + Quota + Request Limits
  -> Admission Controller (bounded)
  -> Engine Actor
      -> Scheduler
      -> Block Manager
      -> Sampler
      -> Execution Planner
          -> Reference Backend
          -> CUDA Kernel Backend
  -> Reliable Token Stream + Cancellation

Shared snapshots:
  Metrics / Readiness / Tracing / Audit
```

关键不变量：

- 每个 KV block 有唯一可审计的生命周期。
- 每个请求参数要么生效，要么明确拒绝。
- 队列和缓存均有容量上限与过载语义。
- token 不静默丢失，断连不继续浪费计算。
- capability 由真实 checkpoint 和持续测试证明。
- 文档、发布物和版本来自单一真相源。

## 8. 里程碑退出标准

### Alpha

- P0 正确性、安全、部署阻断全部关闭。
- README quickstart、Docker 和一个真实模型可运行。

### Beta

- OpenAI compatibility matrix 稳定。
- GPU checkpoint、取消、过载、指标和 release 进入持续验证。
- 有公开、可复现的性能基线。

### 1.0 候选

- Stable API 和迁移策略明确。
- 关键模型矩阵、长稳测试和安全审查通过。
- 至少两名维护者能独立发布与处理事故。
- 性能、正确性与生产声明均有持续证据。

## 9. 成功指标

- Prefix cache 与禁用缓存输出一致性：100%。
- 非默认采样参数 contract tests：100% 通过。
- 过载时内存有界，拒绝响应可预测。
- 客户端取消到 GPU 停止工作的传播延迟有明确上限。
- `/metrics` 与内部 snapshot 一致。
- 官方 Docker/Helm smoke 成功率 100%。
- GPU 性能基线按 commit 可追踪，无未解释显著回归。
- release 版本在 Cargo、tag、image、Chart、CHANGELOG 中完全一致。
- 核心目录至少两名可审批维护者。
