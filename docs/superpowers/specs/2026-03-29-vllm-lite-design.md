# vLLM-lite — Rust 推理引擎设计文档

## 1. Overview

用 Rust 实现一个轻量级 LLM 推理引擎，聚焦 vLLM 的核心创新：continuous batching + paged KV cache。

学习目标：理解 vLLM 的调度器设计、KV cache 管理、批处理策略。

### 目标

- Continuous batching（动态加入/移除请求）
- Paged KV cache（分页管理显存）
- OpenAI-compatible HTTP API
- GPU 推理（Candle 后端）

### 非目标

- 训练
- 分布式推理（多节点）
- 量化支持（后续再加）
- DeepSeek MLA（后续再加）

---

## 2. 技术选型

| 维度 | 选择 | 理由 |
|------|------|------|
| 语言 | Rust | 学习目标 |
| ML 后端 | Candle | 纯 Rust，CUDA 支持，无 FFI |
| 模型 | Qwen3 初始，后续 DeepSeek | 热门开源模型 |
| 权重格式 | SafeTensors | HuggingFace 标准，Candle 原生支持 |
| HTTP 框架 | axum | tokio 生态，简洁 |
| 并发 | tokio + 单线程 GPU worker | vLLM 原始架构 |
| 量化 | 暂不支持 | 聚焦核心调度逻辑 |

---

## 3. 整体架构

### Crate 结构

```
vllm-lite/
├── Cargo.toml              # workspace root
├── crates/
│   ├── core/               # scheduler, kv_cache, types, sampling, engine
│   │   ├── Cargo.toml      # deps: tokio, thiserror, parking_lot
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types.rs
│   │       ├── scheduler.rs
│   │       ├── kv_cache.rs
│   │       ├── engine.rs
│   │       ├── sampling.rs
│   │       ├── error.rs
│   │       └── metrics.rs
│   ├── model/              # Model trait + Qwen3 + Candle backend
│   │   ├── Cargo.toml      # deps: core, candle-core, candle-nn, candle-transformers, safetensors
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── kv_cache.rs         # PagedKvCache (Candle Tensor)
│   │       ├── loader.rs           # SafeTensors 加载
│   │       └── qwen3/
│   │           ├── mod.rs
│   │           ├── config.rs
│   │           ├── attention.rs
│   │           ├── mlp.rs
│   │           ├── block.rs
│   │           └── model.rs
│   └── server/             # axum HTTP (OpenAI-compatible)
│       ├── Cargo.toml      # deps: core, model, axum, tokio, serde, tokenizers
│       └── src/
│           ├── main.rs
│           ├── api/
│           │   ├── completions.rs
│           │   ├── chat.rs
│           │   └── models.rs
│           └── tokenizer.rs
└── docs/
    └── superpowers/
        └── specs/
```

核心原则：**core 不依赖 Candle**。KV block 分配/映射在 core，实际 tensor 数据在 model crate。

### 并发模型

```
┌──────────────────────────────────────────────────┐
│  tokio runtime (server crate)                     │
│                                                   │
│  axum HTTP ──mpsc──► Engine Worker (blocking)     │
│       ▲                     │                     │
│       │                  mpsc channel             │
│       └─────────────────────┘                     │
│               response channel                    │
└──────────────────────────────────────────────────┘
```

- **HTTP 线程** (tokio): 接收请求，放入 mpsc channel，等待响应（SSE streaming）
- **Engine Worker** (`spawn_blocking`): 单线程循环：收请求 → build_batch → forward → sample → update → 发回 token
- **为什么单 worker**: GPU 是序列化的，多线程争抢无收益。vLLM 原始设计也是单进程调度。

### 数据流

```
Request (HTTP)
  → Tokenize (server 层，tokenizers crate)
  → Request { id, prompt_tokens, max_tokens, sampling_params }
  → mpsc channel
  → Engine Worker
    → Scheduler.add_request()
    → Scheduler.build_batch() → Batch
    → Model.forward(Batch) → BatchOutput
    → Sample → TokenId
    → Scheduler.update()
    → Response via per-request channel
  → Detokenize (server 层)
  → SSE stream (HTTP)
```

---

## 4. Core Types

```rust
// crates/core/src/types.rs

pub type TokenId = u32;
pub type SeqId = u64;
pub type BlockId = usize;

pub struct Request {
    pub id: SeqId,
    pub prompt: Vec<TokenId>,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
}

pub struct SamplingParams {
    pub temperature: f32,   // 0.0 = greedy
    pub top_k: usize,       // 0 = disabled
    pub top_p: f32,         // 1.0 = disabled
    pub repetition_penalty: f32,
}

pub struct Sequence {
    pub id: SeqId,
    pub tokens: Vec<TokenId>,
    pub kv_blocks: Vec<BlockId>,
    pub num_computed_tokens: usize,
    pub status: Status,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Status {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

pub struct Batch {
    pub seq_ids: Vec<SeqId>,
    pub input_tokens: Vec<Vec<TokenId>>,
    pub positions: Vec<Vec<usize>>,
    pub kv_block_maps: Vec<Vec<BlockId>>,
}

pub struct BatchOutput {
    pub seq_ids: Vec<SeqId>,
    pub next_tokens: Vec<TokenId>,
}
```

关键设计：
- `input_tokens: Vec<Vec<TokenId>>` — 保留序列边界，不同序列可能处于不同阶段
- `positions: Vec<Vec<usize>>` — 显式传位置，支持 RoPE
- `num_computed_tokens` — 跟踪已完成 prefill 的 token 数，为 chunked prefill 准备

---

## 5. Scheduler

```rust
// crates/core/src/scheduler.rs

pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    kv_allocator: BlockAllocator,
    max_num_seqs: usize,           // 最大并发序列数 (如 256)
    max_num_batched_tokens: usize, // 每步最大 token 数 (如 4096)
}
```

### 调度循环 (每步)

1. 从 waiting 队列取序列加入 running（不超过 max_num_seqs）
2. `build_batch()`:
   - 收集所有 running 序列
   - Waiting/Prefilling 序列: 传待处理的 prompt tokens（分块）
   - Decoding 序列: 只传最后一个 token
   - 检查 max_num_batched_tokens 上限
3. 模型 forward 后 `update()`:
   - 追加生成的 token
   - 检查终止条件（max_tokens、EOS）
   - Finished 的序列移出，释放 KV blocks

### 调度策略

```
decode 优先，prefill 填充剩余容量
```

- Decode 延迟敏感（用户在等 token），优先处理
- Prefill 是 compute-bound，可分块，延迟容忍度高
- `max_num_batched_tokens` 限制总计算量，避免 OOM

---

## 6. KV Cache

### Block 分配 (core crate)

```rust
// crates/core/src/kv_cache.rs

pub const BLOCK_SIZE: usize = 16;

pub struct BlockAllocator {
    num_blocks: usize,
    free_list: Vec<BlockId>,
    block_ref_counts: Vec<usize>,
}

impl BlockAllocator {
    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>>;
    pub fn free(&mut self, blocks: &[BlockId]);
    pub fn can_allocate(&self, seq: &Sequence) -> bool;
}

pub struct LogicalTokenBlock {
    pub block_id: BlockId,
    pub num_tokens: usize,
}
```

- core 只管 block id 分配和映射，不存 tensor 数据
- 引用计数支持 prefix sharing（copy-on-write）
- BLOCK_SIZE = 16：管理开销 vs 显存浪费的折中

### Paged KV Cache (model crate)

```rust
// crates/model/src/kv_cache.rs

pub struct PagedKvCache {
    key_cache: Vec<Vec<Option<Tensor>>>,   // [layer][block_id] → Tensor
    value_cache: Vec<Vec<Option<Tensor>>>,
    block_size: usize,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    device: Device,
}

impl PagedKvCache {
    pub fn append(&mut self, layer: usize, block_id: usize,
                  slot: usize, key: &Tensor, value: &Tensor) -> Result<()>;

    pub fn gather(&self, layer: usize, block_ids: &[BlockId])
        -> Result<(Tensor, Tensor)>;
}
```

### Attention (Paged)

```rust
// crates/model/src/qwen3/attention.rs

impl Attention {
    pub fn forward(&self, hidden: &Tensor, positions: &[usize],
                   kv_cache: &mut PagedKvCache, block_map: &[BlockId])
        -> Result<Tensor>
    {
        // 1. Q = hidden @ Wq, K = hidden @ Wk, V = hidden @ Wv
        // 2. RoPE on Q, K (positions)
        // 3. kv_cache.append(layer, block_id, slot, k, v)
        // 4. kv_cache.gather(layer, block_map) → 拼接所有 block 的 KV
        // 5. Attention(Q, K, V)
        // 6. Output @ Wo
    }
}
```

---

## 7. Engine + Token 生成

### 主循环

```rust
// crates/core/src/engine.rs

pub struct Engine {
    scheduler: Scheduler,
    model: Box<dyn Model>,
}

impl Engine {
    pub fn step(&mut self) -> Vec<(SeqId, TokenId)> {
        let batch = self.scheduler.build_batch();
        if batch.is_empty() { return vec![]; }
        let output = self.model.forward(&batch)?;
        let tokens = self.sample(&batch, &output);
        self.scheduler.update(&batch.seq_ids, &tokens);
        batch.seq_ids.into_iter().zip(tokens).collect()
    }

    pub fn run(&mut self, msg_rx: mpsc::UnboundedReceiver<EngineMessage>) {
        loop {
            while let Ok(msg) = msg_rx.try_recv() { /* handle */ }
            let outputs = self.step();
            for (seq_id, token) in outputs {
                self.scheduler.send_token(seq_id, token);
            }
        }
    }
}
```

### Prefill vs Decode

```
Prefill (序列刚进来):
  input_tokens = [完整 prompt 或分块]
  positions = [0, 1, ..., n-1]
  → 一次性算出所有 KV cache

Decode (后续每步):
  input_tokens = [最后一个 token]
  positions = [当前长度]
  → 只算一个 token 的 KV，append
```

### Chunked Prefill

长 prompt 分多个 step 完成，每个 step 可同时处理 decode 序列（混合 batch）：

```
Step 1: prefill [0..1024]
Step 2: prefill [1024..2048] + decode seq_A, seq_B
Step 3: prefill [2048..3072] + decode seq_A, seq_B, seq_C
Step 4: prefill [3072..4096] → 转 Decoding
```

---

## 8. Sampling

```rust
// crates/core/src/sampling.rs

pub fn sample(logits: &[f32], params: &SamplingParams) -> TokenId {
    if params.temperature == 0.0 {
        return greedy_sample(logits);
    }

    // 1. Temperature scaling
    // 2. Top-k filtering
    // 3. Top-p (nucleus) filtering
    // 4. Softmax + multinomial sampling
}
```

实现顺序：
1. Phase 1: `greedy_sample`（argmax）
2. Phase 2: + `temperature` + `top_k`
3. Phase 3: + `top_p` + `repetition_penalty`

---

## 9. HTTP API (OpenAI-compatible)

### Endpoints

| Method | Path | 说明 |
|--------|------|------|
| POST | /v1/completions | 文本补全 |
| POST | /v1/chat/completions | 对话补全 |
| GET | /v1/models | 列出可用模型 |
| GET | /metrics | 运行时指标 |

### /v1/completions

Request:
```json
{
  "model": "qwen3",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true
}
```

Streaming Response (SSE):
```
data: {"id":"...","choices":[{"text":" there"}]}\n\n
data: {"id":"...","choices":[{"text":" was"}]}\n\n
data: [DONE]\n\n
```

### /v1/chat/completions

Request:
```json
{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 100,
  "stream": true
}
```

Streaming Response (SSE):
```
data: {"id":"...","choices":[{"delta":{"content":"Hi"}}]}\n\n
data: [DONE]\n\n
```

### Tokenizer

- 用 `tokenizers` crate 加载 HuggingFace `tokenizer.json`
- 放在 server 层（core 不依赖具体 tokenizer）
- Chat template: 按 Qwen3 格式拼接 messages

---

## 10. 错误处理

```rust
// crates/core/src/error.rs

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("KV cache OOM: need {needed} blocks, have {available}")]
    KvCacheOom { needed: usize, available: usize },

    #[error("Sequence {id} not found")]
    SeqNotFound { id: SeqId },

    #[error("Model forward failed: {0}")]
    ModelError(String),

    #[error("Sampling failed: {0}")]
    SamplingError(String),
}
```

- 核心路径返回 `Result`，不 panic
- KV OOM：拒绝新请求或抢占低优先级序列
- `thiserror` 定义错误类型，`anyhow` 用于上层传播

---

## 11. 可观测性

```rust
// crates/core/src/metrics.rs

pub struct Metrics {
    pub num_requests_running: AtomicUsize,
    pub num_requests_waiting: AtomicUsize,
    pub tokens_per_second: AtomicF64,
    pub kv_cache_usage: AtomicF64,
    pub prefill_time_ms: AtomicU64,
    pub decode_time_ms: AtomicU64,
}
```

暴露为 `GET /metrics` (JSON)。

---

## 12. 依赖汇总

| Crate | 关键依赖 |
|-------|---------|
| core | tokio, thiserror, parking_lot |
| model | core, candle-core, candle-nn, candle-transformers, safetensors |
| server | core, model, axum, tokio, serde, tokenizers, eventsource-stream |

---

## 13. 实现路线

### Phase 1: 最小可运行版本

- [ ] workspace 搭建，3 crate 编译通过
- [ ] core types 定义
- [ ] 单序列调度（无 batching）
- [ ] FakeModel（随机输出）验证端到端
- [ ] greedy sampling
- [ ] 基础 HTTP API（非 streaming）

### Phase 2: 基本 Batching

- [ ] 多序列批处理
- [ ] decode-priority 调度策略
- [ ] streaming token 回传（SSE）
- [ ] max_num_batched_tokens 限制

### Phase 3: Continuous Batching

- [ ] 动态加入/移除序列
- [ ] chunked prefill
- [ ] 混合 batch（prefill + decode 同步执行）

### Phase 4: Paged KV Cache

- [ ] BlockAllocator + 引用计数
- [ ] PagedKvCache (Candle Tensor)
- [ ] Paged attention 实现
- [ ] KV block 复用

### Phase 5: 真实模型

- [ ] Qwen3 模型实现（Candle）
- [ ] SafeTensors 权重加载
- [ ] RoPE 实现
- [ ] 端到端推理验证

### Phase 6: 完善

- [ ] chat completions endpoint
- [ ] top-k / top-p sampling
- [ ] metrics endpoint
- [ ] 错误处理完善
