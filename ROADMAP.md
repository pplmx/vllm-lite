# vLLM-lite Roadmap

> vLLM-lite: 高效、易用的 LLM 推理引擎

## 概述

vLLM-lite 是一个用 Rust 构建的高效 LLM 推理引擎，目标是为研究和生产提供快速、可扩展的推理能力。

---

## Phase 1: 核心推理引擎 (已完成 ✅)

- [x] Continuous Batching - 连续批处理
- [x] Paged KV Cache - 分页 KV 缓存
- [x] Prefix Caching - 前缀缓存 (完整匹配 + 前缀命中)
- [x] Speculative Decoding - 投机解码

---

## Phase 2: 生成质量

### 2.1 Beam Search
- [ ] 标准 Beam Search
- [ ] Beam size 配置
- [ ] Length penalty
- [ ] Early stopping

### 2.2 Sampling 增强
- [ ] Temperature 控制
- [ ] Top-K Sampling
- [ ] Top-P (Nucleus) Sampling
- [ ] Repeat Penalty (Presence/Frequency)
- [ ] Seed 随机种子
- [ ] Logit bias

### 2.3 多候选
- [ ] 返回多个候选结果
- [ ] 候选评分

---

## Phase 3: I/O 优化

### 3.1 流式输出
- [ ] Server-Sent Events (SSE)
- [ ] 首 token 快速返回
- [ ] 流式 tokenizer

### 3.2 Tokenizer 集成
- [ ] HuggingFace Tokenizer 支持
- [ ] TikToken 支持
- [ ] tokenizer.json 加载

### 3.3 请求验证
- [ ] 最大 token 限制
- [ ] Prompt 长度验证
- [ ] Min/Max tokens 验证

---

## Phase 4: 性能优化

### 4.1 量化支持
- [x] FP16 支持
- [x] INT8 量化 (Weight-Only)
- [x] INT8 KV Cache
- [x] 量化校准工具

### 4.2 计算优化
- [x] Flash Attention 集成 (框架 + 软件 fallback)
- [x] Tiled Attention
- [ ] CUDA Graph

### 4.3 调度优化
- [x] Prefill/Decode 分离 (PD 分离)
- [x] Chunked Prefill
- [x] 动态 Batch Size
- [x] 优先级调度

### 4.4 分布式
- [ ] 多 GPU 支持 (张量并行)
- [ ] Pipeline 并行
- [ ] 分布式 KV Cache

---

## Phase 5: 生产就绪 ✅

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

## Phase 6: 多模型支持

### 6.1 模型管理
- [ ] 模型注册表
- [ ] 模型热插拔
- [ ] 模型版本管理

### 6.2 路由
- [ ] 请求路由 (按模型名)
- [ ] 负载均衡
- [ ] A/B Testing

### 6.3 多模型推理
- [ ] 动态模型切换
- [ ] 模型池

---

## Phase 7: API 完善

### 7.1 OpenAI 兼容
- [ ] /v1/completions
- [ ] /v1/chat/completions
- [ ] /v1/embeddings
- [ ] StreamChat API

### 7.2 扩展
- [ ] Batch API (异步批量)
- [ ] Function Calling
- [ ] Tool Use

---

## Phase 8: 企业特性

### 8.1 安全
- [ ] API Key 认证
- [ ] Rate Limiting
- [ ] 请求审计日志
- [ ] TLS/SSL

### 8.2 多租户
- [ ] Tenant 隔离
- [ ] Quota 管理
- [ ] 使用统计

---

## 长期愿景

- [ ] 支持更多模型架构 (Llama, Mistral, Gemma, etc.)
- [ ] 移动端/边缘部署优化
- [ ] WebAssembly 支持
- [ ] 自主学习/在线微调接口

---

## 优先级顺序

```
Phase 2 (生成质量)
    ↓
Phase 3 (I/O 优化)
    ↓
Phase 5 (生产就绪)
    ↓
Phase 4 (性能优化)
    ↓
Phase 6-8 (多模型/企业)
```

---

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解如何参与。

## 许可证

MIT License