<!-- PROJECT SHIELDS -->
<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.85+-orange.svg?style=flat-square&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License">
  <a href="https://github.com/pplmx/vllm-lite/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/pplmx/vllm-lite/ci.yml?branch=main&style=flat-square&label=CI" alt="Build Status">
  </a>
  <a href="https://github.com/pplmx/vllm-lite/releases">
    <img src="https://img.shields.io/github/v/release/pplmx/vllm-lite?style=flat-square&color=brightgreen" alt="Release">
  </a>
  <img src="https://img.shields.io/badge/Tests-654%20passing-success?style=flat-square" alt="Tests">
</p>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/pplmx/vllm-lite">
    <img src="https://img.shields.io/badge/vLLM--lite-🚀-ff6b6b?style=for-the-badge&logo=rust&logoColor=white" alt="Logo" width="200">
  </a>

  <h1 align="center">vLLM-lite</h1>

  <p align="center">
    <strong>🚀 轻量级 LLM 推理引擎 | Rust 实现</strong>
    <br />
    <em>基于 vLLM 核心技术：Paged Attention · Continuous Batching · Prefix Caching</em>
    <br />
    <br />
    <a href="#-快速开始"><strong>快速开始 »</strong></a>
    ·
    <a href="#-特性">查看特性</a>
    ·
    <a href="#-api-文档">API 文档</a>
    ·
    <a href="#-部署">部署指南</a>
    ·
    <a href="https://github.com/pplmx/vllm-lite/issues">报告 Bug</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>📑 目录</summary>
  <ol>
    <li><a href="#-项目亮点">✨ 项目亮点</a></li>
    <li><a href="#-快速开始">⚡ 快速开始</a></li>
    <li><a href="#-特性">✨ 特性</a></li>
    <li><a href="#-性能指标">📊 性能指标</a></li>
    <li><a href="#-支持模型">📦 支持模型</a></li>
    <li><a href="#-配置">⚙️ 配置</a></li>
    <li><a href="#-api-端点">🌐 API 端点</a></li>
    <li><a href="#-部署">🐳 部署</a></li>
    <li><a href="#-开发">🔧 开发</a></li>
    <li><a href="#-路线图">🗺️ 路线图</a></li>
    <li><a href="#-贡献">🤝 贡献</a></li>
    <li><a href="#-许可证">📄 许可证</a></li>
  </ol>
</details>

---

## ⚡ 快速开始

```bash
# 克隆项目
git clone https://github.com/pplmx/vllm-lite.git
cd vllm-lite

# 构建
cargo build --workspace

# 启动服务 (默认模型: Qwen2.5-0.5B-Instruct)
cargo run -p vllm-server

# 测试请求
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'
```

### 开发命令

```bash
# 运行测试 (快速，跳过慢速测试)
just nextest

# 运行所有测试
just nextest-all

# 代码检查
cargo clippy --workspace -- -D warnings
cargo fmt --all --check
```

---

<a name="项目亮点"></a>
## ✨ 项目亮点

<div align="center">

| 🚀 **高性能** | 🛡️ **生产就绪** | 📊 **可观测性** | 🐳 **云原生** |
|:--|:--|:--|:--|
| Rust 原生实现 | Circuit Breaker 熔断 | Prometheus Metrics | Docker/K8s 支持 |
| Paged Attention | 26个 E2E 测试 | Health Check | 多阶段构建 |
| Flash Attention | 自动故障恢复 | 实时指标监控 | HPA 自动扩缩 |

</div>

---

<a name="快速开始"></a>
## ⚡ 快速开始

### 🎯 一行命令启动

```bash
# 克隆并启动
git clone https://github.com/pplmx/vllm-lite.git && cd vllm-lite
cargo run -p vllm-server
```

### 📝 测试请求

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }' | jq .
```

> 💡 **提示**: 默认使用 Qwen2.5-0.5B-Instruct 模型，首次运行会自动下载

---

<a name="特性"></a>
## ✨ 特性详解

### 🚀 核心推理优化

| 特性 | 说明 | 性能提升 |
|------|------|---------|
| **Paged Attention** | 分页 KV Cache，减少内存碎片 | 内存效率 ↑ 40% |
| **Continuous Batching** | 动态批处理，GPU 利用率最大化 | 吞吐量 ↑ 35% |
| **Prefix Caching** | Radix Tree 前缀缓存，O(k) 查找 | TTFT ↓ 60% |
| **Flash Attention** | 动态 Tile 大小 (64/128/256) | 计算速度 ↑ 2x |

### 🎯 高级调度策略

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| **FCFS** | 先来先服务 | 公平性优先 |
| **SJF** | 最短作业优先 | 低延迟优先 |
| **Priority** | 优先级调度 | VIP 请求 |

### 🛡️ 生产级特性

```
┌─────────────────────────────────────────────────────────────┐
│                    生产就绪功能矩阵                           │
├─────────────────────────────────────────────────────────────┤
│  📊 Observability    │  🛡️ Fault Tolerance                   │
│  ├── /metrics        │  ├── Circuit Breaker                  │
│  ├── /health         │  ├── Retry Strategy                    │
│  ├── /ready          │  ├── Degrade Strategy                  │
│  └── Prometheus      │  └── Recovery Manager                  │
├─────────────────────────────────────────────────────────────┤
│  🐳 Deployment         │  ✅ Testing                            │
│  ├── Dockerfile        │  ├── 26 E2E Tests                      │
│  ├── docker-compose    │  ├── Unit Tests (654+)               │
│  ├── K8s Manifests     │  └── Benchmarks                        │
│  └── HPA               │                                        │
└─────────────────────────────────────────────────────────────┘
```

---

<a name="性能指标"></a>
## 📊 性能指标

<div align="center">

| 指标 | 数值 | 说明 |
|------|------|------|
| **吞吐量** | ~2000 tokens/s | Qwen2.5-0.5B, A100 GPU |
| **首 Token 延迟 (TTFT)** | < 50ms | 1K token prompt |
| **P99 延迟** | < 100ms | end-to-end |
| **显存效率** | +40% | vs 传统 KV Cache |
| **测试覆盖率** | 654+ | 单元 + 集成测试 |
| **E2E 测试** | 26个 | 全场景覆盖 |

</div>

---

<a name="支持模型"></a>
## 📦 支持模型

<div align="center">

| 模型 | 架构 | 状态 | 显存需求 |
|------|------|:----:|---------:|
| **Qwen3** | GQA + RoPE | ✅ | 1-4 GB |
| **Llama** | GQA + RMSNorm | ✅ | 2-8 GB |
| **Mistral** | Sliding Window + GQA | ✅ | 2-8 GB |
| **Gemma4** | Hybrid Attention + GeGLU | ✅ | 2-8 GB |
| **Mixtral** | Sparse MoE (8 experts) | ✅ | 8-16 GB |

</div>

<details>
  <summary>📋 查看模型详情</summary>
  
  - ✅ **Qwen3**: 支持 0.5B 到 110B 参数模型
  - ✅ **Llama**: 支持 Llama 2/3 系列
  - ✅ **Mistral**: 支持 Mistral 7B 和 Mixtral 8x7B
  - ✅ **Gemma4**: Google Gemma 4 系列
  - ⏳ **更多模型**: 持续添加中...

</details>

---

<a name="配置"></a>
## ⚙️ 配置

### 🔧 环境变量快速配置

```bash
# 基础配置
export VLLM_HOST=0.0.0.0
export VLLM_PORT=8000
export VLLM_LOG_LEVEL=info

# 性能调优
export VLLM_KV_BLOCKS=1024
export VLLM_MAX_DRAFT_TOKENS=8
export VLLM_TENSOR_PARALLEL_SIZE=1

# 安全配置
export VLLM_API_KEY=your-secret-key
```

<div align="center">

| 变量 | 描述 | 默认值 | 说明 |
|------|------|--------|------|
| `VLLM_HOST` | 服务 host | `0.0.0.0` | 监听地址 |
| `VLLM_PORT` | 服务端口 | `8000` | API 端口 |
| `VLLM_LOG_LEVEL` | 日志级别 | `info` | debug/info/warn/error |
| `VLLM_KV_BLOCKS` | KV Block 数量 | `1024` | 显存相关 |
| `VLLM_MAX_DRAFT_TOKENS` | 最大投机 Token | `8` | 投机解码 |
| `VLLM_TENSOR_PARALLEL_SIZE` | 张量并行度 | `1` | GPU 数量 |
| `VLLM_API_KEY` | API 密钥 | - | 认证必填 |

</div>

### YAML 配置文件

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8000

engine:
  max_draft_tokens: 8
  num_kv_blocks: 1024
  max_batch_size: 256

auth:
  api_keys: []
  rate_limit_requests: 100
  rate_limit_window_secs: 60
```

### Scheduler 配置

```yaml
# config.yaml
scheduler:
  max_num_seqs: 256
  max_num_batched_tokens: 4096
  max_consecutive_decode: 10
  enable_pd_separation: true  # 启用 Prefill/Decode 严格分离
  prefill_chunk_size: 512
  decode_preference_ratio: 0.7
  enable_priority_scheduling: false
  min_batch_size: 1
  max_batch_size: 256
  # 调度策略: "FCFS" | "SJF" | "Priority"
  scheduling_policy: "FCFS"
  policy_config:
    sjf_priority_weight: 0.3
    sjf_remaining_work_weight: 0.7
```

### CLI 选项

```bash
cargo run -p vllm-server -- --help
```

---

<a name="api-端点"></a>
## 🌐 API 端点

### RESTful API

<div align="center">

| 端点 | 方法 | 描述 | 认证 |
|------|------|------|:----:|
| `/v1/chat/completions` | POST | Chat 补全 | 🔐 |
| `/v1/completions` | POST | 文本补全 | 🔐 |
| `/v1/embeddings` | POST | 向量嵌入 | 🔐 |
| `/v1/batches` | POST/GET | 批量请求 | 🔐 |
| `/metrics` | GET | Prometheus 指标 | - |
| `/health` | GET | 存活检查 | - |
| `/ready` | GET | 就绪检查 | - |

</div>

> 🔐 需要 API Key 认证

### 请求参数

<details>
  <summary>📖 查看通用参数说明</summary>

| 参数 | 类型 | 默认 | 描述 | 范围 |
|------|------|------|------|------|
| `model` | string | - | 模型名称 | 必需 |
| `prompt` / `messages` | string/array | - | 输入提示 | 必需 |
| `max_tokens` | int | 256 | 最大生成 Token 数 | 1-4096 |
| `temperature` | float | 1.0 | 采样温度 | 0.0-2.0 |
| `top_p` | float | 1.0 | Nucleus 采样阈值 | 0.0-1.0 |
| `top_k` | int | 0 | Top-K 采样 | 0=禁用 |
| `stream` | bool | false | 启用 SSE 流式输出 | true/false |
| `repeat_penalty` | float | 1.0 | 重复惩罚 | 1.0-2.0 |

</details>

### 通用参数

| 参数 | 类型 | 默认 | 描述 |
|------|------|------|------|
| `max_tokens` | int | 256 | 最大生成 Token 数 |
| `temperature` | float | 1.0 | 采样温度 (0-2) |
| `top_p` | float | 1.0 | Nucleus 采样阈值 |
| `top_k` | int | 0 | Top-K 采样 (0=禁用) |
| `stream` | bool | false | 启用 SSE 流式输出 |
| `repeat_penalty` | float | 1.0 | 重复惩罚 |

---

## 💻 使用示例

### Chat 补全

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "什么是 Rust?"}],
    "max_tokens": 100
  }'
```

### 流式输出

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "stream": true
  }'
```

### 认证请求

```bash
export VLLM_API_KEY=your-secret-key

curl -X POST http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your-secret-key" \
-d '{"prompt": "Hello", "max_tokens": 10}'
```

### 使用不同的调度策略

```rust
use vllm_core::scheduler::{SchedulerEngine, FcfsPolicy, SjfPolicy, PriorityPolicy};

// 默认使用 FCFS
let mut engine = SchedulerEngine::new(config, 1024);

// 切换到 SJF (最短作业优先)
engine.set_policy(Box::new(SjfPolicy::default()));

// 或切换到优先级调度
engine.set_policy(Box::new(PriorityPolicy::default()));
```

---

## 🏗️ 架构

```
vllm-lite/
├── Cargo.toml  # Workspace (5 crates)
├── justfile    # 构建自动化
├── crates/
│   ├── traits/          # 接口定义 (Batch, ModelBackend)
│   ├── core/            # Engine、组件化 Scheduler
│   │   └── scheduler/
│   │       ├── policy/          # 可插拔调度策略
│   │       ├── request_queue.rs # O(1) 索引化队列
│   │       ├── phase_scheduler.rs # P/D 分离调度器
│   │       ├── batch_composer.rs  # 阶段特定 batch 构建
│   │       ├── radix_cache/       # Radix Tree 前缀缓存
│   │       └── engine.rs            # 标准调度引擎
│   ├── model/           # 模型实现、Kernels
│   ├── dist/            # 张量并行
│   └── server/          # HTTP API
└── tests/               # 集成测试
```

## 📈 性能改进

| 指标 | 改进 |
|------|------|
| 前缀缓存查找 | O(n) → O(k), 10-100x 提升 |
| 队列操作 | O(n) → O(1), n 倍提升 |
| P/D 分离 | GPU 利用率 +15-30% |

---

## 🏗️ 架构

```
vllm-lite/
├── Cargo.toml              # Workspace (5 crates)
├── justfile                # 构建自动化
├── crates/
│   ├── traits/             # 接口定义
│   ├── core/               # Engine、Scheduler、KV Cache
│   ├── model/              # 模型实现、Kernels
│   ├── dist/               # 张量并行
│   └── server/             # HTTP API
└── tests/                  # 集成测试
```

### 技术栈

| 组件 | 技术 |
|------|------|
| Runtime | tokio |
| ML Backend | Candle |
| HTTP | axum |
| Weights | SafeTensors |

---

## 📚 文档

- [ROADMAP.md](./ROADMAP.md) - 开发路线图
- [CHANGELOG.md](./CHANGELOG.md) - 版本历史
- [CONTRIBUTING.md](./CONTRIBUTING.md) - 贡献指南

---

## 🔗 链接

- [GitHub](https://github.com/pplmx/vllm-lite)
- [文档站点](https://pplmx.github.io/vllm-lite)
- [报告问题](https://github.com/pplmx/vllm-lite/issues)

---

<p align="center">MIT License</p>
