# vLLM-lite Session Handoff

> 新 session 可直接读取本文恢复进度。最后更新：2026-06-26  
> Git：`main` @ `54af5ad` (Wave 1 全部完成：11 commits；Phase 0–5 + Qwen3.5 Hybrid 收敛)

---

## 下一优先级（2026-06-26，Wave 1 完成）

**Wave 1: 文档同步 + dead_code 审计** ✅ 完成（11 commits）

| Task | Commit | 描述 |
|------|--------|------|
| 1 | `4344b77` | 同步 MODEL-ARCHITECTURE-REFACTOR.md |
| 2 | `1f91686` | 同步 PHASE-5-QWEN35-HYBRID.md |
| 3 | `20ce45e` | 同步 STATE/PROJECT/ROADMAP（amended） |
| 4 | `6fd37b2` | CHANGELOG 补 Phase 5 |
| 5a | `5d963a7` | dead_code 审计（vllm-core） |
| 5b | `fa7350a` | dead_code 审计（vllm-model） |
| 1.5d | `2eb1340` | cfg-gate cuda-graph 引入（unblock clippy -D warnings） |
| 5b+ | `16908c1` | RoPE/MRoPE audit correction |
| 5a+ | `54af5ad` | HashRouter audit completion |

**后续 Wave:** Wave 2 (adaptive draft depth) → Wave 3 (Dependabot) → Wave 4 (warmup) → Wave 5 (benchmarks)

**Wave 1 spec/plan:**
- Spec: `docs/superpowers/specs/2026-06-26-wave1-doc-sync-design.md` (commit `d42b151`)
- Plan: `docs/superpowers/plans/2026-06-26-wave1-doc-sync.md` (commit `dba9f5e`)

---

## 项目定位

Rust 实现的轻量 LLM 推理引擎，核心能力对标 vLLM：

- Continuous Batching
- Paged KV Cache
- Prefix Caching
- Speculative Decoding
- OpenAI 兼容 HTTP API

**工作区路径：** `/workspace/vllm-lite`

---

## 近期 Commit 脉络（新 → 旧）

| Commit | 说明 |
|--------|------|
| `5c7cb1c` | `PagedDecoderBlock` trait；Gemma4 sliding window mask；Mixtral MoE softmax top-k |
| `0e55d1f` | Mixtral 真实 paged KV forward + MoE |
| `f03c62d` | Gemma4 真实 paged KV forward |
| `417471e` | dead_code 审计；`kv_quantization` loader→Architecture 贯通 |
| `3f41d32` | 删除 `StandardBlock`；`create_block` 的 `todo!()` 改 `Err` |
| 更早 | Engine 简化、core↔model 解耦、测试基础设施、Llama/Mistral forward |

---

## Crate 结构

```text
vllm-traits  → 接口 (ModelBackend, Batch, SeqId)
vllm-core    → Engine, Scheduler, KV cache 逻辑, metrics
vllm-model   → 各架构实现 + PagedKvCache 物理层
vllm-server  → OpenAI 兼容 HTTP API
```

---

## 架构现状

### Engine

- 单一入口 `Engine::step()`
- `Arc<Mutex<Box<dyn ModelBackend>>>`
- 已删除 legacy hash PrefixCache、重复 metrics

### 各架构 Forward 状态

| 架构 | Paged KV | 说明 |
|------|----------|------|
| Llama | ✅ | `RopeGqaDecoderBlock` + `forward_with_paged_kv` |
| Mistral | ✅ | 同上 |
| Qwen3 | ✅ | `TransformerBlock` 包装 `RopeGqaDecoderBlock` |
| Gemma4 | ✅ | 自定义 attention/block；sliding window mask 已加 |
| Mixtral | ✅ | `RopeGqaAttention` + `MixtralSparseMoe`；softmax top-k 路由 |

**已无「只加载权重、forward 返回 token 0」的 stub 架构。**

### 共享抽象（`5c7cb1c`）

```rust
// crates/model/src/components/decoder_block.rs
pub trait PagedDecoderBlock {
    fn forward_prefill(...);
    fn forward_decode(...);
}
```

**实现者：** `RopeGqaDecoderBlock`、`Gemma4Block`、`MixtralBlock`、Qwen3 `TransformerBlock`

- `causal_lm::forward_with_paged_kv` + `run_decoder_layers` 统一 prefill/decode loop
- Gemma4/Mixtral 已删除各自重复的 `run_decoder_layers`
- `RmsNorm` / `LnLayerNorm` 已实现 `candle_core::Module`，可接入 `forward_with_paged_kv`

---

## 关键设计决策

- **无 backward 兼容** — fresh project，可大刀阔斧改
- **Block size:** 16 tokens
- **KV cache：** 写入 expanded heads（`PagedKvCache::new(..., num_heads, ...)`，不是 `num_kv_heads`）
- **Paged attention 张量布局：** `[batch, num_heads, seq, head_dim]`（见 `compute_paged_attention` / `paged_attention`）
- **无 git worktree** — 见 `AGENTS.md`
- **Commit 格式：** `<type>(<scope>): <subject>`
- **仅在用户明确要求时 commit**

---

## 已知差距 / 下一批优先级

### Wave 1 已收口（2026-06-26）

- ✅ MoE 向量化（`mixtral/sparse_moe.rs` expert-grouped batching + scatter_add）
- ✅ Gemma4 sliding window mask（`compute_attention` 已对齐）
- ✅ `TransformerBlock: PagedDecoderBlock` + metadata
- ✅ Phase 0–5 架构重构（`decc8c8` ~ `52f77ce` + Wave 1 docs sync）
- ✅ dead_code 审计：31 处全部处置（22 stub 注释 + 5 删除 + 2 keep 注释 + 2 redundant-annotation 修复）

### 中价值（Wave 2+ 处理）

- **Dependabot** — GitHub 报 5 个漏洞（1 high, 4 moderate），未处理 → Wave 3
- **Mistral/Llama final norm shape 风险** — `from_weights` 用 `Linear` + 1D 权重（Gemma4 已改 `RmsNorm`）；若遇 shape 问题需统一 → Wave 2+ 视情
- **Flash 真 CUDA kernel** — 需 GPU 环境 → 延后至 Wave 5 验收后

### 低优先级

7. Gemma4 sliding window 大 window 下行为等价 full attention（测试 window=512，未专门测小 window）
8. Qwen3.5 / Gemma3 / 其他 arch 的 clippy / dead_code 清理

---

## 重要文件路径

| 区域 | 路径 |
|------|------|
| 共享 causal LM | `crates/model/src/causal_lm.rs` |
| Paged decoder trait | `crates/model/src/components/decoder_block.rs` |
| RopeGqa paged KV 参考 | `crates/model/src/components/attention/rope_gqa.rs` |
| Paged KV 物理层 | `crates/model/src/paged_tensor/tensor_store.rs` |
| Gemma4 | `crates/model/src/gemma4/{model,block,attention}.rs` |
| Mixtral | `crates/model/src/mixtral/{model,block,sparse_moe}.rs` |
| Architecture 注册 | `crates/model/src/arch/mod.rs` |
| Loader + kv_quantization | `crates/model/src/loader/builder.rs` |
| Engine | `crates/core/src/engine.rs` |
| Scheduler | `crates/core/src/scheduler/` |
| 开发指南 | `AGENTS.md`, `CLAUDE.md` |

---

## 验证命令

```bash
cd /workspace/vllm-lite

# 架构专项
cargo test -p vllm-model --lib gemma4
cargo test -p vllm-model --lib mixtral

# 全量测试（跳过 #[ignore] 慢测）
just nextest

# 完整 CI（fmt → clippy → doc → nextest）
just ci
# 最近一次：1016 passed, 42 skipped
```

---

## 环境备注

- `/models/` 有 Qwen 等权重；**无 Gemma4 权重**（仅单元测试 tiny config）
- 慢测标 `#[ignore]`，默认 `just nextest` 跳过；全量用 `just nextest-all`

---

## 新 Session 起手 Prompt（可复制）

```text
继续 vllm-lite 开发。请先读 .planning/PHASE-5-QWEN35-HYBRID.md（Wave 1 任务表）。
当前 main @ e192492，Phase 0–4 + Gemma4→CausalLm 已完成。
下一项：Phase 5 Wave 1 — 拆分 qwen3_5/hybrid.rs，零行为变更。
直接在 main 推进，不需要 PR。用中文回复。
```
