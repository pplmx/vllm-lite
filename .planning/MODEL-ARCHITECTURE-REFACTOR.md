# Model 层架构重构计划（生产化 / Rustic）

> **创建日期:** 2026-06-12  
> **最后更新:** 2026-06-26（Phase 0–5 全部完成；进入 Wave 1 收口 + Wave 2 增量）  
> **目标:** 消除重复、统一推理抽象、区分生产级与 stub 架构，使 `vllm-model` 可维护、可扩展、可安全部署  
> **关联文档:** [`docs/architecture.md`](../docs/architecture.md)、[`.planning/STATE.md`](./STATE.md)

---

## 执行进度总览

| Phase | 名称 | 状态 | 进度 | 目标 PR/Commit |
|-------|------|------|------|----------------|
| **Pre** | 近期已完成（GDN / Hybrid cache） | ✅ 完成 | 100% | `26520ab` 及之前 |
| **0** | Capability 与 Stub 隔离 | ✅ 完成 | 4/4 | `11a4ad4` |
| **1** | 泛型 CausalLm + BlockWrapper | ✅ 完成 | 6/6 | `8043a02` |
| **2** | 统一 Attention Core + MLP | ✅ 完成 | 5/5 | `9dcfd10` |
| **3** | DecoderLayer 统一 Loop + Config | ✅ 完成 | 5/5 | `cbafe75` |
| **4** | Qwen3 瘦身 + Stub 实现或移除 | ✅ 完成 | 4/4 | `88f54df` |
| **4b** | Gemma4 → CausalLm | ✅ 完成 | 1/1 | `e192492` |
| **5** | Qwen3.5 Hybrid 收敛 | ✅ 完成 | 4/4 waves | `decc8c8` / `73dab5e` / `52f77ce` |

**整体进度:** `[██████████] Phase 0–5 全部完成；进入 Wave 1 文档收口 + Wave 2–5 spec decode 增量`

**图例:** ✅ 完成 · 🔄 进行中 · ⬜ 未开始 · ⏸ 暂停 · ❌ 取消

---

## 设计原则（Rustic）

1. **One loop** — 所有架构共用一条 `run_layers` 解码循环  
2. **One attention core** — paged KV write/read 只实现一次  
3. **One causal LM shell** — `CausalLm<L>` 替代多份 `model.rs`  
4. **Explicit stubs** — 占位架构必须可识别，禁止 silent 返回 token 0  
5. **Thin arch/** — registry 只做 detect、remap keys、`create_model`  
6. **组件在 components/** — 架构目录只组合，不重新实现 attention/MLP  

---

## 成熟度基线（2026-06-12）

| 档位 | 架构 | 推理 | 备注 |
|------|------|------|------|
| **A 生产级** | Llama, Mistral, Mixtral, Gemma4, Qwen3 | `CausalLm` + paged KV | 可 serving |
| **A 生产级（hybrid + spec）** | Qwen3.5 | `HybridLm` + `run_layers` + GDN state | 主路径 + 自投机 OK；`PRODUCTION_SPECULATIVE` 自 `52f77ce` |
| **C Stub** | Gemma3, Llama4, Phi4, MistralSmall | 返回 0；loader 默认拒绝（`--allow-stub` 可 override） | Phase 0 + 4.4 Option C |

---

## Pre：近期已完成 ✅

> 2026-06-12 前已在 `main` 落地，作为本计划的起点。

- [x] Qwen3.5 Gated DeltaNet 实现（`gated_delta.rs`）
- [x] GDN decode state cache + parity 测试
- [x] Hybrid block 暴露 prefill/decode API
- [x] `Qwen35HybridModel::forward_with_cache` + Engine 参数接通
- [x] `attention35.rs` 抽取 + Full 层 paged KV
- [x] 删除 `LinearAttentionForMamba`、遗留 `Qwen35Model` stub
- [x] Flash 接入 RopeGqa / GqaAttention；dead_code 一批修复

**验证命令（已通过）:**

```bash
just nextest
cargo clippy -p vllm-model -- -D warnings
```

---

## Phase 0：Capability 与 Stub 隔离

**目标:** 防止 stub 架构在生产路径被误用；启动/加载时可观测。

**风险:** 低 | **预估:** 1 PR | **依赖:** 无

### 任务清单

- [x] **0.1** 在 `vllm_traits` 或 `crates/model/src/arch/mod.rs` 定义 `ArchCapabilities`：

  ```rust
  pub struct ArchCapabilities {
      pub inference: bool,      // 真实 forward
      pub paged_kv: bool,
      pub weight_load: bool,
      pub speculative: bool,    // 支持 forward_to_layer / forward_logits
  }
  ```

- [x] **0.2** `Architecture` trait 增加 `fn capabilities(&self) -> ArchCapabilities`
- [x] **0.3** 为每个已注册架构填写 capability 表（A/B/C 档见上）
- [x] **0.4** Loader / server 启动时：若 `inference == false` 则 warn 或 error（可配置 `--allow-stub`）

### 验收标准

- [x] Gemma3 加载时日志明确提示 stub（默认 error；`--allow-stub` 时 warn）
- [ ] 集成测试：stub 架构 `forward()` 不 silently 通过 golden test（或标记 `#[ignore]` + 文档）
- [x] `just ci` 绿（待本次验证）

### 进度记录

| 日期 | 动作 | Commit |
|------|------|--------|
| 2026-06-12 | Phase 0：`ArchCapabilities` + loader/server `--allow-stub` | `11a4ad4` |

---

## Phase 1：泛型 CausalLm + BlockWrapper

**目标:** 合并 Llama/Mistral（及同类）重复的 `model.rs` + `block.rs` + `*BlockWrapper`。

**风险:** 低–中 | **预估:** 2–3 PR | **依赖:** Phase 0（可选）

### 任务清单

- [x] **1.1** 新增 `crates/model/src/causal_lm/model.rs`：

  ```rust
  pub struct CausalLm<B, N, H> {
      embed_tokens, layers: Vec<B>, norm: N, lm_head: H,
      kv_cache, device, config,
  }
  ```

- [x] **1.2** 实现 `CausalLm::forward_with_cache` 委托现有 `forward_with_paged_kv`
- [x] **1.3** 泛型 `BlockWrapper<B: PagedDecoderBlock>` 替代 7 份 arch wrapper boilerplate
- [x] **1.4** `LlamaModel` / `MistralModel` / `MixtralModel` 改为 `CausalLm` type alias
- [x] **1.5** 删除/合并 `llama/block.rs` 与 `mistral/block.rs` 重复工厂 → `components/decoder_block/factory.rs`
- [x] **1.6** Mixtral/Gemma4/Qwen3 arch wrapper 迁移到泛型 `BlockWrapper`

### 验收标准

- [x] Llama + Mistral 测试无回归
- [x] 删除重复代码（llama/mistral block 工厂 + model shell + arch wrapper）
- [x] 生产 arch 文件显著瘦身（llama arch ~110 行，mistral ~70 行）

### 进度记录

| 日期 | 动作 | Commit |
|------|------|--------|
| 2026-06-12 | Phase 1：`CausalLm` + `BlockWrapper` + shared factory | 8043a02 |

---

## Phase 2：统一 Attention Core + MLP

**目标:** 合并 `RopeGqaAttention`、`Gemma4Attention`、`Attention35WithRoPE` 的 paged KV 路径。

**风险:** 中 | **预估:** 2–3 PR | **依赖:** Phase 1 推荐

### 任务清单

- [x] **2.1** 新增 `components/attention/paged_gqa.rs`：
  - `project_qkv` / `write_prefill_kv` / `read_decode_kv` / `compute_attention`
- [x] **2.2** RoPE 插件：标准 RoPE / MRoPE / none（Gemma sliding 用 mask 插件）— `QkRotaryEmb` trait
- [x] **2.3** `RopeGqaAttention` 重构为 core + flash 路径（保留 `run_attention_fn`）
- [x] **2.4** `Gemma4Attention` / `Attention35WithRoPE` 变薄 wrapper
- [x] **2.5** 删除 `MLP35`，Qwen3.5 改用 `SwiGLU`；评估 `GeGLU` 是否并入 `components/mlp`（暂缓）

### 验收标准

- [x] Gemma4 / Qwen3.5 / RopeGqa parity 测试通过（1033 nextest）
- [x] attention 相关重复逻辑减少（-105 行 net）
- [ ] Full 层可选走 flash（`AttentionConfig.use_fused`）— 已有 `run_attention_fn`

### 进度记录

| 日期 | 动作 | Commit |
|------|------|--------|
| 2026-06-12 | Phase 2：`paged_gqa` + attention refactor + MLP35→SwiGLU | 9dcfd10 |

---

## Phase 3：DecoderLayer 统一 Loop + Config

**目标:** 消除 `run_decoder_layers` vs `run_hybrid_layers` 分裂；统一配置类型。

**风险:** 中–高 | **预估:** 2 PR | **依赖:** Phase 2 推荐

### 任务清单

- [x] **3.1** 定义 `DecoderLayer` trait + `LayerCtx`（KV + 可选 `GatedDeltaState`）
- [x] **3.2** 单一 `run_layers(ctx, layers, hidden)` 替代 hybrid 专用 loop
- [x] **3.3** `HybridBlock` 实现 `DecoderLayer`；删除 `run_hybrid_layers`
- [x] **3.4** `ModelConfig` / `Qwen3Config` 统一：`ModelHyperparams` + `From<ModelConfig>`
- [x] **3.5** Qwen3.5 `embed()` / `forward_to_layer()` 接入 cache 路径

### 验收标准

- [x] Qwen3.5 hybrid 集成测试 prefill+decode 仍 parity
- [x] 仅一条 layer loop 被 production 路径使用（`causal_lm::run_layers`）
- [x] 移除 `hybrid.rs` 顶部 `#![allow(clippy::all)]`

### 进度记录

| 日期 | 动作 | Commit |
|------|------|--------|
| 2026-06-12 | Phase 3.1–3.3：`LayerCtx` + `run_layers` + hybrid 接入 | 3fa4543 |
| 2026-06-12 | Phase 3.4–3.5：`ModelHyperparams` + hybrid embed/forward_to_layer | `cbafe75` |

---

## Phase 4：Qwen3 瘦身 + Stub 处置

**目标:** `qwen3/model.rs` 降至 ~400 行；stub 架构要么实现要么移出默认 registry。

**风险:** 中 | **预估:** 2+ PR | **依赖:** Phase 1–3

### 任务清单

- [x] **4.1** Qwen3 TP 逻辑下沉到 `qwen3/tp.rs`（`new_with_tp`）
- [x] **4.2** Qwen3 `forward_to_layer` 基于 `CausalLm::run_layers_upto` + `DecoderLayer`
- [x] **4.3** 修复 `ModelBackend::forward_with_cache` default（返回 Err，不再假 logits）
- [x] **4.4** Stub 架构（Gemma3/Llama4/Phi4/MistralSmall）→ **Option C**：
  - 保留 `register_all_archs` 用于 detect
  - `ModelLoader` / server 默认拒绝 stub（`--allow-stub` / `VLLM_ALLOW_STUB`）

### 验收标准

- [x] `qwen3/model.rs` < 500 行（当前 ~52 行）
- [x] Speculative decoding 测试不运行在 stub 上（core 测试仅用 FakeModel）
- [x] 架构 capability 文档与代码一致（见 `arch/capabilities.rs` + 成熟度基线表）

### 进度记录

| 日期 | 动作 | Commit |
|------|------|--------|
| 2026-06-12 | Phase 4：Qwen3→CausalLm、TP 下沉、embed 修复、trait 默认实现 | `88f54df` |

---

## 目标目录结构（Phase 1–3 完成后）

```text
crates/model/src/
├── causal_lm/
│   ├── mod.rs              # embed, run_layers, forward_with_paged_kv, greedy_sample
│   └── model.rs            # CausalLm<B, N, H>
├── components/             # 唯一实现层（已有，继续收敛）
│   ├── attention/
│   │   ├── paged_gqa.rs    # NEW: 共享 KV loop
│   │   ├── rope_gqa.rs
│   │   └── ...
│   ├── mlp/
│   └── decoder_block.rs
├── layers/                 # NEW: 组合 components
│   ├── rope_gqa_block.rs
│   ├── gemma4_block.rs
│   └── hybrid_block.rs
├── arch/                   # 薄 registry
└── qwen3_5/                # 逐步迁入 layers/ + causal_lm
```

---

## 验证清单（每 Phase 必跑）

```bash
just fmt-check
cargo clippy --workspace -- -D warnings
just nextest
# 可选：单架构冒烟
cargo test -p vllm-model --lib llama
cargo test -p vllm-model --lib qwen3_5
cargo test -p vllm-model --lib gemma4
```

---

## 风险与决策记录

| ID | 决策 | 理由 | 日期 |
|----|------|------|------|
| D1 | Phase 0 先于大规模合并 | 避免 stub 在 refactor 中混入生产路径 | 2026-06-12 |
| D2 | `CausalLm` 用泛型而非 trait object | 与现有 `Vec<LlamaBlock>` 零成本抽象一致 | 2026-06-12 |
| D3 | Stub 策略选 **Option C** | 保留 detect/registry；loader 默认拒绝，需 `--allow-stub` | 2026-06-12 |
| D4 | Qwen3Config 短期保留 | 避免一次性破坏 Qwen3.5 weight remap | 2026-06-12 |

---

## 会话接续说明

新 session 恢复步骤：

1. 读本文 **执行进度总览** 表  
2. 找到第一个非 ✅ Phase，读其 **任务清单** 未勾选项  
3. 跑 **验证清单** 确认基线绿  
4. 完成后更新：勾选 `- [ ]` → `- [x]`，填写 **进度记录** 表与 **最后更新** 日期  

**Git 基线:** `main` @ `88f54df`（Phase 0–4 完成，领先 origin 17 commits）

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-12 | 初版：基于全仓架构 review 制定 Phase 0–4 + Pre 已完成项 |
| 2026-06-12 | Phase 4 完成：Qwen3 瘦身、stub Option C、规划文档同步 |
| 2026-06-12 | Gemma4 迁 CausalLm（`e192492`） |
| 2026-06-12 | Phase 5 规划：`.planning/PHASE-5-QWEN35-HYBRID.md`（4 waves） |
