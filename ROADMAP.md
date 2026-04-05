# vLLM-lite Roadmap

> vLLM-lite: 高效、易用的 LLM 推理引擎

## 概述

vLLM-lite 是一个用 Rust 构建的高效 LLM 推理引擎，目标是为研究和生产提供快速、可扩展的推理能力。

---

## Phase 1: 核心推理引擎 ✅ (已完成)

- [x] Continuous Batching - 连续批处理
- [x] Paged KV Cache - 分页 KV 缓存
- [x] Prefix Caching - 前缀缓存 (完整匹配 + 前缀命中)
- [x] Speculative Decoding - 投机解码

---

## Phase 2: 生成质量 ✅ (已完成)

### 2.1 Beam Search

- [x] Beam Search 实现 (`BeamSequence`)
- [x] Beam size 配置
- [x] Length penalty
- [x] Early stopping

### 2.2 Sampling 增强

- [x] Temperature 控制
- [x] Top-K Sampling
- [x] Top-P (Nucleus) Sampling
- [x] Repeat Penalty

### 2.3 多候选

- [x] 返回多个候选结果 (Beam Search)
- [x] 候选评分

---

## Phase 3: I/O 优化 ✅ (已完成)

### 3.1 流式输出

- [x] Server-Sent Events (SSE)
- [x] 首 token 快速返回
- [x] 流式 tokenizer

### 3.2 Tokenizer 集成

- [x] HuggingFace Tokenizer 支持
- [x] TikToken 支持
- [x] tokenizer.json 加载

### 3.3 请求验证

- [x] 最大 token 限制
- [x] Prompt 长度验证
- [x] Min/Max tokens 验证

---

## Phase 4: 性能优化 ✅ (已完成)

### 4.1 量化支持

- [x] FP16 支持
- [x] INT8 量化 (Weight-Only)
- [x] INT8 KV Cache
- [x] 量化校准工具 (QuantizationCalibrator)

### 4.2 计算优化

- [x] Flash Attention 框架 + 软件 fallback
- [x] Tiled Attention
- [x] CUDA Graph 框架

### 4.3 调度优化

- [x] Prefill/Decode 分离 (PD 分离)
- [x] Chunked Prefill
- [x] 动态 Batch Size
- [x] 优先级调度

### 4.4 分布式

- [x] 多 GPU 张量并行
- [ ] Pipeline 并行
- [ ] 分布式 KV Cache

---

## Phase 5: 生产就绪 ✅ (已完成)

### 5.1 监控

- [x] Metrics 收集 (/v1/stats)
- [x] Prometheus 导出 (/metrics)
- [x] Grafana Dashboard
- [x] 延迟/吞吐量追踪

### 5.2 日志

- [x] 结构化日志 (JSON)
- [x] 日志级别控制
- [x] 请求 ID 追踪

### 5.3 可靠性

- [x] 健康检查端点 (/health)
- [x] 就绪检查 (/ready)
- [x] 优雅关闭
- [x] 请求超时
- [x] 错误重试 (客户端参数)

### 5.4 配置管理

- [x] CLI 参数 (--config=)
- [x] 配置文件 (YAML)
- [x] 环境变量 (VLLM_HOST, VLLM_PORT, etc.)
- [x] 配置验证

---

## Phase 6: 多模型支持 ✅ (已完成)

### 6.1 模型加载

- [x] ModelLoader 权重加载
- [x] Qwen2.5-0.5B 支持
- [x] Qwen3-0.6B 支持
- [x] DeepSeek-R1-8B 支持
- [x] Qwen3.5-0.8B (Mamba) 支持
- [x] tie_word_embeddings 支持
- [x] q_norm/k_norm 支持
- [x] RoPE YARN scaling 支持

### 6.2 模型管理

- [x] --model CLI 参数
- [ ] 模型注册表
- [ ] 模型热插拔
- [ ] 模型版本管理

---

## Phase 7: API 完善 ✅ (已完成)

### 7.1 OpenAI 兼容 API

- [x] /v1/chat/completions
- [x] /v1/completions
- [x] /v1/embeddings (占位实现)
- [x] Streaming 支持

### 7.2 Batch API

- [x] POST /v1/batches - 创建批量任务
- [x] GET /v1/batches - 列出任务
- [x] GET /v1/batches/:id - 查询状态
- [x] GET /v1/batches/:id/results - 获取结果

### 7.3 运维端点

- [x] /health - 健康检查
- [x] /metrics - Prometheus 指标

---

## Phase 8: 安全与监控 ✅ (已完成)

参考 vLLM 上游实现 - 认证和限流由服务器自身处理，其他企业特性由外部组件（nginx、监控系统）负责。

### 8.1 认证与安全

- [x] API Key 认证 (Bearer token)
- [x] Rate Limiting (per-key)
- [x] 健康检查 (/health, /ready)
- [x] 请求验证 (max tokens)

### 8.2 监控

- [x] Prometheus 指标 (/metrics)
- [x] 延迟/吞吐量追踪
- [x] 调度器指标

### 8.3 日志

- [x] 结构化日志
- [x] 请求 ID 追踪

### 8.4 不需要 (外部处理)

- TLS/SSL → 由 nginx/负载均衡处理
- 多租户隔离 → 如需要使用 API Key 区分
- 审计日志 → 由外部日志系统收集
- 告警 → 由监控系统处理

---

## 长期愿景

- [ ] 支持更多模型架构 (Llama, Mistral, Gemma, etc.)
- [ ] 移动端/边缘部署优化
- [ ] WebAssembly 支持
- [ ] 自主学习/在线微调接口

---

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解如何参与。

## 许可证

MIT License
