<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.75+-blue.svg" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<h1 align="center">vLLM-lite 路线图</h1>

<p align="center">
  高效、易用的 LLM 推理引擎 · 基于 Rust + Candle
</p>

---

## 📊 总体进展

| Phase | 名称         | 状态      |
| ----- | ------------ | --------- |
| 1     | 核心推理引擎 | ✅ 已完成 |
| 2     | 生成质量     | ✅ 已完成 |
| 3     | I/O 优化     | ✅ 已完成 |
| 4     | 性能优化     | ✅ 已完成 |
| 5     | 生产就绪     | ✅ 已完成 |
| 6     | 多模型支持   | ✅ 已完成 |
| 7     | API 完善     | ✅ 已完成 |
| 8     | 安全与监控   | ✅ 已完成 |

---

## ✅ Phase 1: 核心推理引擎

| 功能                 | 描述                           |
| -------------------- | ------------------------------ |
| Continuous Batching  | 连续批处理，动态调度           |
| Paged KV Cache       | 分页 KV 缓存，LRU 淘汰         |
| Prefix Caching       | 前缀缓存 (完整匹配 + 前缀命中) |
| Speculative Decoding | 投机解码架构                   |

---

## ✅ Phase 2: 生成质量

### 2.1 Beam Search

- [x] Beam Search 实现
- [x] Beam size 配置
- [x] Length penalty
- [x] Early stopping

### 2.2 Sampling 增强

- [x] Temperature 控制
- [x] Top-K Sampling
- [x] Top-P (Nucleus) Sampling
- [x] Repeat Penalty

### 2.3 多候选

- [x] 多候选结果返回
- [x] 候选评分

---

## ✅ Phase 3: I/O 优化

### 3.1 流式输出

- [x] Server-Sent Events (SSE)
- [x] 首 Token 快速返回
- [x] 流式 Tokenizer

### 3.2 Tokenizer 集成

- [x] HuggingFace Tokenizer
- [x] TikToken 支持
- [x] tokenizer.json 加载

### 3.3 请求验证

- [x] 最大 Token 限制
- [x] Prompt 长度验证
- [x] Min/Max tokens 验证

---

## ✅ Phase 4: 性能优化

### 4.1 量化支持

- [x] FP16 支持
- [x] INT8 Weight-Only 量化
- [x] INT8 KV Cache
- [x] 量化校准工具

### 4.2 计算优化

- [x] Flash Attention 框架
- [x] Tiled Attention
- [x] CUDA Graph 框架

### 4.3 调度优化

- [x] Prefill/Decode 分离 (PD 分离)
- [x] Chunked Prefill
- [x] 动态 Batch Size
- [x] 优先级调度

### 4.4 分布式

- [x] 多 GPU 张量并行
- [ ] Pipeline 并行 🚧
- [ ] 分布式 KV Cache 🚧

---

## ✅ Phase 5: 生产就绪

### 5.1 监控

- [x] Metrics 收集 (`/v1/stats`)
- [x] Prometheus 导出 (`/metrics`)
- [x] Grafana Dashboard
- [x] 延迟/吞吐量追踪

### 5.2 日志

- [x] 结构化日志 (JSON)
- [x] 日志级别控制
- [x] 请求 ID 追踪

### 5.3 可靠性

- [x] 健康检查 (`/health`)
- [x] 就绪检查 (`/ready`)
- [x] 优雅关闭
- [x] 请求超时
- [x] 错误重试

### 5.4 配置管理

- [x] CLI 参数 (`--config=`)
- [x] 配置文件 (YAML)
- [x] 环境变量
- [x] 配置验证

---

## ✅ Phase 6: 多模型支持

### 6.1 支持的模型

| 模型                 | 架构                   | 状态 |
| -------------------- | ---------------------- | ---- |
| Qwen2.5-0.5B         | GQA + RoPE             | ✅   |
| Qwen3-0.6B           | GQA + RoPE + QK-Norm   | ✅   |
| DeepSeek-R1-8B       | GQA + MoE              | ✅   |
| Qwen3.5-0.8B (Mamba) | Mamba SSM + Attention  | ✅   |
| Llama                | GQA + RMSNorm          | ✅   |
| Mistral              | Sliding Window + GQA   | ✅   |
| Gemma4               | Hybrid Attention       | ✅   |
| Mixtral              | Sparse MoE (8 experts) | ✅   |

### 6.2 模型特性

- [x] `tie_word_embeddings` 支持
- [x] `q_norm` / `k_norm` 支持
- [x] RoPE YARN scaling
- [x] 自定义 `head_dim` 支持

### 6.3 架构注册系统

- [x] `Architecture` trait 动态抽象
- [x] `ArchitectureRegistry` 注册表
- [x] 新架构 3 步添加流程
- [x] 每个架构独立模块和测试

### 6.4 模型管理

- [x] `--model` CLI 参数
- [x] 模型热插拔 (通过 CLI 参数)
- [x] 模型版本管理 (通过 HF Hub)

---

## ✅ Phase 7: API 完善

### 7.1 OpenAI 兼容 API

- [x] `/v1/chat/completions`
- [x] `/v1/completions`
- [x] `/v1/embeddings`
- [x] Streaming 支持

### 7.2 Batch API

- [x] `POST /v1/batches` - 创建任务
- [x] `GET /v1/batches` - 列出任务
- [x] `GET /v1/batches/:id` - 查询状态
- [x] `GET /v1/batches/:id/results` - 获取结果

### 7.3 运维端点

- [x] `/health` - 健康检查
- [x] `/metrics` - Prometheus 指标
- [x] `/shutdown` - 关闭服务

---

## ✅ Phase 8: 安全与监控

### 8.1 认证与安全

- [x] API Key 认证 (Bearer token)
- [x] Rate Limiting (per-key)
- [x] 请求验证

### 8.2 监控

- [x] Prometheus 指标
- [x] 延迟/吞吐量追踪
- [x] 调度器指标

### 8.3 日志

- [x] 结构化日志
- [x] 请求 ID 追踪

> **注**: TLS/SSL、多租户隔离、审计日志等企业特性由外部组件处理 (nginx、监控系统)

---

## ✅ Phase 9: 架构重构 (2026-04-19)

| 功能 | 描述 | 状态 |
| ---- | ---- | ---- |
| Cargo.toml 优化 | tokio features 精确化, release profile 优化 | ✅ |
| Feature Flags | cuda/gguf/real_weights 可选 | ✅ |
| 共享组件层 | attention, mlp, norm, positional 子模块 | ✅ |
| TransformerBlock | 基类设计, 支持组合模式 | ✅ |
| core→model 解耦 | vllm-core 对 vllm-model 可选依赖 | ✅ |
| 文档完善 | ADR, SPEC 更新 | ✅ |

### 成果

- **代码行数**: ~800+ 行重复代码移除
- **组件目录**: 4 个新子模块 (attention, mlp, norm, positional)
- **Feature Flags**: 4 个新可选特性
- **测试数**: 849 tests pass

详细设计: [docs/superpowers/specs/2026-04-19-architecture-refactor-design.md](docs/superpowers/specs/2026-04-19-architecture-refactor-design.md)

---

## ✅ Phase 10: 性能优化 (2026-04-26)

| 功能 | 描述 | 状态 |
| ---- | ---- | ---- |
| FlashAttention V2 | 在线 softmax 算法实现 | ✅ |
| CUDA Graph 优化 | 图池化、warmup、invalidation | ✅ |
| Chunked Prefill | 动态 chunk 大小、内存优化 | ✅ |
| 基准测试套件 | attention/scheduler benchmarks | ✅ |

### 成果

- **FlashAttention V2**: 块级处理，在线 softmax 计算
- **CUDA Graph**: 图池化策略减少重复捕获
- **Chunked Prefill**: 32k+ 上下文无 OOM
- **测试数**: 337+ model tests, 145+ core tests

详细设计: [.planning/phases/](.planning/phases/)

---

## 🔮 长期愿景

| 目标 | 描述                | 状态 |
| ---- | ------------------- | ---- |
| 🚧   | 更多模型架构支持    | 新架构可快速添加 |
| 🚧   | 移动端/边缘部署优化 | 依赖轻量级后端 |
| 🚧   | WebAssembly 支持    | 依赖 WASM 编译目标 |
| 🚧   | 在线微调接口        | 未来规划 |

### 架构扩展性

新的模型架构只需 3 步即可添加:

```rust
// 1. 创建 arch.rs
pub struct NewModel;
impl Architecture for NewModel { ... }

// 2. 创建 register.rs
pub fn register(registry: &ArchitectureRegistry) {
    registry.register::<NewModel>();
}

// 3. 更新 register_all_archs()
registry::register::<NewModel>(registry);
```

详细文档: [docs/superpowers/specs/2026-04-18-multi-model-architecture-design.md](docs/superpowers/specs/2026-04-18-multi-model-architecture-design.md)

---

## 🤝 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解如何参与。

---

<p align="center">MIT License</p>
