<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.75+-blue.svg" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <a href="https://github.com/pplmx/vllm-lite/actions/workflows/ci.yml"><img src="https://github.com/pplmx/vllm-lite/actions/workflows/ci.yml/badge.svg" alt="Build"></a>
</p>

<h1 align="center">vLLM-lite</h1>

<p align="center">
  <strong>轻量级 LLM 推理引擎 | Rust 实现</strong><br>
  基于 vLLM 核心技术：Paged Attention、Continuous Batching、Prefix Caching
</p>

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

## ✨ 特性

| 特性 | 说明 |
|------|------|
| 🚀 | 高性能 Rust 实现 |
| 🎯 | Continuous Batching + Decode 优先调度 |
| 💾 | Paged KV Cache (LRU 淘汰 + 内存池) |
| 🔍 | Block Hash 前缀缓存 |
| ⚡ | Flash Attention (动态 Tile: 64/128/256) |
| 🔗 | 融合 Attention + MLP Kernel |
| 🔄 | 流式 Token 生成 (SSE) |
| 📡 | OpenAI 兼容 HTTP API |
| 🖥️ | CUDA GPU 加速 (via Candle) |
| 📊 | 实时 Metrics 收集 |
| 🔐 | API Key 认证 |
| ⏱️ | 请求限流 |

---

## 📦 支持模型

| 模型 | 架构 | 状态 |
|------|------|------|
| Qwen3 | GQA + RoPE | ✅ |
| Llama | GQA + RMSNorm | ✅ |
| Mistral | Sliding Window + GQA | ✅ |
| Gemma4 | Hybrid Attention + GeGLU | ✅ |
| Mixtral | Sparse MoE (8 experts) | ✅ |

---

## ⚙️ 配置

### 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `VLLM_HOST` | 服务 host | `0.0.0.0` |
| `VLLM_PORT` | 服务端口 | `8000` |
| `VLLM_LOG_LEVEL` | 日志级别 | `info` |
| `VLLM_MAX_DRAFT_TOKENS` | 最大投机 Token 数 | `8` |
| `VLLM_KV_BLOCKS` | KV Block 数量 | `1024` |
| `VLLM_TENSOR_PARALLEL_SIZE` | 张量并行度 | `1` |
| `VLLM_API_KEY` | API 密钥 | - |

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

### CLI 选项

```bash
cargo run -p vllm-server -- --help
```

---

## 🌐 API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/chat/completions` | POST | Chat 补全 |
| `/v1/completions` | POST | 文本补全 |
| `/v1/embeddings` | POST | 向量嵌入 |
| `/v1/batches` | POST/GET | 批量请求 |
| `/metrics` | GET | Prometheus Metrics |
| `/health` | GET | 健康检查 |

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
