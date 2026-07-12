<!-- PROJECT SHIELDS -->
<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.88+-orange.svg?style=flat-square&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License">
  <a href="https://github.com/pplmx/vllm-lite/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/pplmx/vllm-lite/ci.yml?branch=main&style=flat-square&label=CI" alt="Build Status">
  </a>
  <a href="https://github.com/pplmx/vllm-lite/releases">
    <img src="https://img.shields.io/github/v/release/pplmx/vllm-lite?style=flat-square&color=brightgreen" alt="Release">
  </a>
  <img src="https://img.shields.io/badge/Tests-1235%20passing-success?style=flat-square" alt="Tests">
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

# 启动服务 — 你必须先准备一个本地模型目录（含 tokenizer.json 和权重），
# 然后通过 --model 或 VLLM_MODEL 指向它。vLLM-lite 不会自动下载模型。
# 例：使用 Qwen2.5-0.5B-Instruct（自行从 Hugging Face 下载到 ./models/qwen2.5-0.5b）
cargo run -p vllm-server -- --model ./models/qwen2.5-0.5b

# 测试请求
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'
```

> ⚠️ **诚实声明**：CLI 要求显式 `--model <PATH>`（或 `VLLM_MODEL`），
> 启动时不会下载权重。如果某个示例/教程声称无参启动或自动下载，
> 那就是 bug，请报告给我们。

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

| 🚀 **高性能**   | 🛡️ **生产就绪**      | 📊 **可观测性**    | 🐳 **云原生**   |
| :-------------- | :------------------- | :----------------- | :-------------- |
| Rust 原生实现   | JWT / RBAC / 审计    | Structured Logging | Docker/K8s 支持 |
| Paged Attention | 请求体大小限制       | Prometheus Metrics | 多阶段构建      |
| Flash Attention | 优雅关闭 / 健康检查  | Health / Ready     | HPA 自动扩缩    |

</div>

---

<a name="快速开始"></a>

## 🚀 安装与启动

### 🎯 一行命令启动

```bash
# 克隆 + 构建 + 启动（需要本地已有模型目录；参见上面的快速开始）
git clone https://github.com/pplmx/vllm-lite.git && cd vllm-lite
cargo build --workspace
cargo run -p vllm-server -- --model ./models/qwen2.5-0.5b
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

> 💡 **提示**: 需要先自行准备模型目录（含 `tokenizer.json` 与权重），
> 并通过 `--model` 或 `VLLM_MODEL` 指向。vLLM-lite 不会自动下载。

> 📘 **新手教程**: 第一次接触 vllm-lite？请按顺序阅读 [教程 1: 环境搭建与构建](./docs/tutorial/01-setup.md) 到 [教程 5: 生产部署](./docs/tutorial/05-production.md)。

---

<a name="特性"></a>

## ✨ 特性详解

### 🚀 核心推理优化

> ⚠️ **诚实声明**：下表的百分比来自历史/受控基准，**不是当前实现的
> 持续回归值**。当前批次调度仍是逐序列 `forward`（见
> `docs/technical-due-diligence/architecture-performance.md` 的
> PERF-01）；真实数字应由 release 阶段的稳定基准产生并写入
> `docs/perf/`。我们保留这些目标数字作为长期方向，但请勿作为当下
> 部署选型的依据。

| 特性                    | 说明                           | 目标性能（受控基准，非当前实测） |
| ----------------------- | ------------------------------ | --------------------------------- |
| **Paged Attention**     | 分页 KV Cache，减少内存碎片    | 内存效率 ↑ 40%（目标）            |
| **Continuous Batching** | 动态批处理，GPU 利用率最大化   | 吞吐量 ↑ 35%（目标）              |
| **Prefix Caching**      | Radix Tree 前缀缓存，O(k) 查找 | TTFT ↓ 60%（目标）                |
| **Flash Attention**     | 动态 Tile 大小 (64/128/256)    | 计算速度 ↑ 2x（目标）             |

### 🎯 高级调度策略

| 策略         | 描述         | 适用场景   |
| ------------ | ------------ | ---------- |
| **FCFS**     | 先来先服务   | 公平性优先 |
| **SJF**      | 最短作业优先 | 低延迟优先 |
| **Priority** | 优先级调度   | VIP 请求   |

### 🛡️ 生产级特性

```text
┌─────────────────────────────────────────────────────────────┐
│                    生产就绪功能矩阵                           │
├─────────────────────────────────────────────────────────────┤
│  📊 Observability    │  🛡️ Fault Tolerance                   │
│  ├── /metrics        │  ├── Circuit Breaker                  │
│  ├── /health         │  ├── Retry Strategy                    │
│  ├── /ready          │  ├── Degrade Strategy                  │
│  ├── Prometheus      │  └── Recovery Manager                  │
│  └── Structured Logs │                                        │
├─────────────────────────────────────────────────────────────┤
│  🐳 Deployment         │  ✅ Testing                            │
│  ├── Dockerfile        │  ├── \`just nextest\` 输出当前测试数 │
│  ├── docker-compose    │  ├── 集成测试覆盖 OpenAI 路由         │
│  ├── K8s Manifests     │  ├── Fuzz + Mutation + Proptest      │
│  └── HPA               │  └── Criterion benchmarks              │
└─────────────────────────────────────────────────────────────┘
```

### 📝 结构化日志系统

vLLM-lite 提供 5 级结构化日志，支持控制台美化输出和 JSON 文件输出：

| 级别      | 用途      | 示例                               |
| --------- | --------- | ---------------------------------- |
| **ERROR** | 系统失败  | 配置错误、模型加载失败             |
| **WARN**  | 降级/回退 | CUDA Graph 禁用、tokenizer 回退    |
| **INFO**  | 生命周期  | 启动、请求开始/结束                |
| **DEBUG** | 内部流程  | 批处理、调度决策、内存分配         |
| **TRACE** | 详细调试  | Token 生成、KV Cache、Attention 层 |

**日志输出示例**：

```bash
# 控制台输出（彩色美化）
2026-04-19 22:30:00 [INFO] vllm_server::main: Starting vllm-lite
2026-04-19 22:30:05 [INFO] vllm_server::openai: Request started (request_id=req_ABC123, prompt_tokens=150)
2026-04-19 22:30:06 [DEBUG] vllm_core::scheduler: Batch built (batch_size=4, phase=Prefill)
2026-04-19 22:30:06 [INFO] vllm_server::openai: Request completed (request_id=req_ABC123, output_tokens=42, duration_ms=1234)

# 文件输出（JSON 格式）
{"timestamp":"2026-04-19T22:30:06.123Z","level":"INFO","target":"vllm_server::openai","message":"Request completed","request_id":"req_ABC123","output_tokens":42,"duration_ms":1234}
```

**启用不同日志级别**：

```bash
# 默认 info 级别
cargo run -p vllm-server

# 启用 debug 日志
RUST_LOG=debug cargo run -p vllm-server

# 启用 trace 日志（详细调试）
RUST_LOG=trace cargo run -p vllm-server

# 启用文件日志
cargo run -p vllm-server -- --log-dir ./logs
```

---

<a name="性能指标"></a>

## 📊 性能指标

> ⚠️ **诚实声明**：吞吐量 / TTFT / P99 / 显存效率的数字来自历史内部
> 测量，**当前实现尚未接入持续 GPU 基准**（CI-01 仍 deferred）。
> 真实数字应以 `docs/perf/` 下的 release 阶段基准产物为准。本节保留
> 数字仅为长期目标参考；不要将其作为当前部署的吞吐/延迟承诺。

<div align="center">

| 指标                     | 数值（目标，非当前实测） | 说明                              |
| ------------------------ | ------------------------ | --------------------------------- |
| **吞吐量**               | ~2000 tokens/s（目标）   | Qwen2.5-0.5B, A100 GPU（参考）    |
| **首 Token 延迟 (TTFT)** | < 50ms（目标）           | 1K token prompt（参考）           |
| **P99 延迟**             | < 100ms（目标）          | end-to-end（参考）                |
| **显存效率**             | +40%（目标）             | vs 传统 KV Cache（参考）          |
| **测试数**               | `just nextest`           | 跑一次即得当前数；不要硬编码到文档 |
| **Checkpoint 测试**      | `just nextest-checkpoint`| 需模型权重，默认 ignored          |

</div>

---

<a name="支持模型"></a>

## 📦 支持模型

<div align="center">

| 模型              | 架构                     |  状态   | 显存需求 |
| ----------------- | ------------------------ | :-----: | -------: |
| **Llama**         | GQA + RMSNorm            |   ✅    |   2-8 GB |
| **Llama 4**       | MoE + Hybrid Attention   | 🟡 StubArchitecture | 16-64 GB |
| **Mistral**       | Sliding Window + GQA     |   ✅    |   2-8 GB |
| **Mistral Small** | Grouped Query + Sliding  | 🟡 StubArchitecture |   4-8 GB |
| **Qwen3**         | GQA + MLA + RoPE         |   ✅    |   1-4 GB |
| **Qwen3.5**       | Mamba SSM Hybrid         |   ✅    |   1-4 GB |
| **Gemma3**        | GQA + GeLU               | 🟡 StubArchitecture |   2-4 GB |
| **Gemma4**        | Hybrid Attention + GeGLU |   ✅    |   2-8 GB |
| **Mixtral**       | Sparse MoE (8 experts)   |   ✅    |  8-16 GB |
| **Phi-4**         | GQA + RoPE               | 🟡 StubArchitecture |   4-8 GB |

</div>

<details>
  <summary>📋 查看模型详情</summary>

- ✅ **Qwen3**: 支持 0.5B 到 110B 参数模型
- ✅ **Qwen3.5**: Mamba SSM Hybrid 架构
- ✅ **Llama**: 支持 Llama 2/3 系列
- ✅ **Llama 4 / Gemma3 / Mistral Small / Phi-4**: 通过 `StubArchitecture` 注册（检测权重，拒绝推理直至完整实现）
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

| 变量                        | 描述           | 默认值    | 说明                  |
| --------------------------- | -------------- | --------- | --------------------- |
| `VLLM_HOST`                 | 服务 host      | `0.0.0.0` | 监听地址              |
| `VLLM_PORT`                 | 服务端口       | `8000`    | API 端口              |
| `VLLM_LOG_LEVEL`            | 日志级别       | `info`    | debug/info/warn/error |
| `VLLM_KV_BLOCKS`            | KV Block 数量  | `1024`    | 显存相关              |
| `VLLM_MAX_DRAFT_TOKENS`     | 最大投机 Token | `8`       | 投机解码              |
| `VLLM_TENSOR_PARALLEL_SIZE` | 张量并行度     | `1`       | GPU 数量              |
| `VLLM_API_KEY`              | API 密钥       | -         | 认证必填              |

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

| 端点                       | 方法     | 描述            | 认证  |
| -------------------------- | -------- | --------------- | :---: |
| `/v1/models`               | GET      | 模型列表        |   -   |
| `/v1/chat/completions`     | POST     | Chat 补全       |  🔐   |
| `/v1/completions`          | POST     | 文本补全        |  🔐   |
| `/v1/embeddings`           | POST     | 向量嵌入        |  🔐   |
| `/v1/batches`              | POST/GET | 批量请求        |  🔐   |
| `/v1/batches/{id}`         | GET      | 批量状态        |  🔐   |
| `/v1/batches/{id}/results` | GET      | 批量结果        |  🔐   |
| `/metrics`                 | GET      | Prometheus 指标 |   -   |
| `/health`                  | GET      | 存活检查        |   -   |
| `/health/live`             | GET      | K8s liveness    |   -   |
| `/health/ready`            | GET      | K8s readiness   |   -   |
| `/health/details`          | GET      | 详细健康状态    |   -   |
| `/ready`                   | GET      | 就绪检查        |   -   |
| `/debug/metrics`           | GET      | 调试指标快照    |   -   |
| `/debug/kv-cache`          | GET      | KV cache 状态   |   -   |
| `/debug/trace`             | GET      | 追踪状态        |   -   |
| `/shutdown`                | GET      | 优雅关闭        |   -   |

</div>

> 🔐 需要 API Key 认证

### 请求参数

<details>
  <summary>📖 查看通用参数说明</summary>

| 参数                  | 类型         | 默认  | 描述              | 范围       |
| --------------------- | ------------ | ----- | ----------------- | ---------- |
| `model`               | string       | -     | 模型名称          | 必需       |
| `prompt` / `messages` | string/array | -     | 输入提示          | 必需       |
| `max_tokens`          | int          | 256   | 最大生成 Token 数 | 1-4096     |
| `temperature`         | float        | 1.0   | 采样温度          | 0.0-2.0    |
| `top_p`               | float        | 1.0   | Nucleus 采样阈值  | 0.0-1.0    |
| `top_k`               | int          | 0     | Top-K 采样        | 0=禁用     |
| `stream`              | bool         | false | 启用 SSE 流式输出 | true/false |
| `repeat_penalty`      | float        | 1.0   | 重复惩罚          | 1.0-2.0    |

</details>

### 通用参数

| 参数             | 类型  | 默认  | 描述                |
| ---------------- | ----- | ----- | ------------------- |
| `max_tokens`     | int   | 256   | 最大生成 Token 数   |
| `temperature`    | float | 1.0   | 采样温度 (0-2)      |
| `top_p`          | float | 1.0   | Nucleus 采样阈值    |
| `top_k`          | int   | 0     | Top-K 采样 (0=禁用) |
| `stream`         | bool  | false | 启用 SSE 流式输出   |
| `repeat_penalty` | float | 1.0   | 重复惩罚            |

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
use std::sync::Arc;
use vllm_core::scheduler::SchedulerEngine;
use vllm_core::scheduler::policy::{FcfsPolicy, SjfPolicy, PriorityPolicy};
use vllm_core::metrics::EnhancedMetricsCollector;

// 默认使用 FCFS
let metrics = Arc::new(EnhancedMetricsCollector::new());
let mut engine = SchedulerEngine::new(config, 1024, metrics);

// 切换到 SJF (最短作业优先)
engine.set_policy(Box::new(SjfPolicy::default()));

// 或切换到优先级调度
engine.set_policy(Box::new(PriorityPolicy::default()));
```

---

## 🏗️ 架构

> 完整架构图与请求生命周期见 **[docs/architecture.md](./docs/architecture.md)**。

```text
vllm-lite/
├── Cargo.toml              # Workspace (6 crates)
├── justfile                # 构建自动化
├── crates/
│   ├── traits/             # 接口定义 (ModelBackend, Batch, Kernel traits)
│   ├── core/               # Engine、Scheduler、KV Cache、Metrics
│   ├── model/              # 模型实现、Loader、Quantize、Kernels
│   │   ├── arch/           # Architecture trait + Registry + StubArchitecture
│   │   ├── components/     # 共享组件层 (attention, mlp, norm, positional, ssm)
│   │   ├── loader/         # ModelLoader、format detection
│   │   ├── paged_tensor/   # 物理 KV cache (tensor_store, quantization)
│   │   ├── quantize/       # GGUF Q4_K_M + StorageTensor
│   │   ├── causal_lm/      # CausalLM head + block wrapper + hybrid dispatcher
│   │   ├── llama/          # Llama 架构
│   │   ├── mistral/        # Mistral 架构
│   │   ├── mixtral/        # Mixtral 架构 (Sparse MoE)
│   │   ├── qwen3/          # Qwen2/3 架构 (GQA + MLA)
│   │   ├── qwen3_5/        # Qwen3.5 架构 (Mamba SSM Hybrid)
│   │   ├── gemma4/         # Gemma 4 架构
│   │   └── kernels/        # GPU Kernels (flash_attention, cuda_graph)
│   ├── dist/               # 多节点原语 (multi-node, feature-gated)
│   ├── server/             # HTTP API (OpenAI 兼容 + 安全 + 批处理)
│   └── testing/            # 测试工具 (BatchBuilder、fixtures、stubs)
```

集成测试位于 `crates/*/tests/`（非顶层 `tests/` 目录）。

### 技术栈

| 组件       | 技术        |
| ---------- | ----------- |
| Runtime    | tokio       |
| ML Backend | Candle      |
| HTTP       | axum        |
| Weights    | SafeTensors |

### 共享组件层

核心组件提取到 `components/` 子模块，实现代码复用：

| 组件       | 文件                     | 描述                                |
| ---------- | ------------------------ | ----------------------------------- |
| Attention  | `components/attention/`  | GqaAttention, paged/tiled attention |
| MLP        | `components/mlp/`        | SwiGLU feed-forward                 |
| Norm       | `components/norm/`       | RMSNorm, LayerNorm                  |
| Positional | `components/positional/` | RoPE, MRoPE                         |

### Feature Flags

| Feature         | Crate              | 描述                                                |
| --------------- | ------------------ | --------------------------------------------------- |
| `cuda-graph`    | core, server       | CUDA Graph 捕获/回放（经 `CudaGraphExecutor` trait） |
| `cuda`          | model              | Candle CUDA 支持                                    |
| `gguf`          | model              | GGUF 模型加载                                       |
| `multi-node`    | core, model, testing | 启用 `vllm-dist`（分布式 KV + gRPC）              |
| `full`          | model              | `cuda` + `gguf`                                     |
| `candle`        | traits             | 启用 Candle 核心类型                                |
| `kernels`       | traits             | 启用 kernel trait 定义                              |

Note: Tokenizer (`tiktoken`, `tokenizers`) 始终启用。`vllm-dist` 非 default-member，需 `--features multi-node` 构建。

---

## 📈 性能改进

| 指标         | 改进                      |
| ------------ | ------------------------- |
| 前缀缓存查找 | O(n) → O(k), 10-100x 提升 |
| 队列操作     | O(n) → O(1), n 倍提升     |
| P/D 分离     | GPU 利用率 +15-30%        |
| 编译优化     | fat LTO, panic=abort      |

---

## 📚 文档

- [docs/architecture.md](./docs/architecture.md) - 系统架构（单一真相源）
- [OPERATIONS.md](./OPERATIONS.md) - 部署与故障排查
- [.planning/v31.0-MASTER-PLAN.md](./.planning/v31.0-MASTER-PLAN.md) - 当前开发路线图
- [CHANGELOG.md](./CHANGELOG.md) - 版本历史
- [CONTRIBUTING.md](./CONTRIBUTING.md) - 贡献指南
- [docs/adr/](./docs/adr/) - 架构决策记录 (19 篇 ADR)
- [Tutorials](./docs/tutorial/01-setup.md) - 新贡献者教程（从 clone 到 serving）

---

## 🔗 链接

- [GitHub](https://github.com/pplmx/vllm-lite)
- [文档站点](https://pplmx.github.io/vllm-lite)
- [报告问题](https://github.com/pplmx/vllm-lite/issues)

---

<p align="center">MIT License</p>
