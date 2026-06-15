# Phase 5：Qwen3.5 Hybrid 收敛计划

> **创建日期:** 2026-06-12  
> **状态:** 🔄 Wave 2 完成（待 commit）  
> **前置:** Phase 0–4 完成（`main` @ `e192492`+）  
> **关联:** `.planning/MODEL-ARCHITECTURE-REFACTOR.md`、`.planning/SESSION-HANDOFF.md`

---

## 1. 背景与目标

Phase 1–4 已将标准 decoder-only 架构（Llama / Mistral / Mixtral / Gemma4 / Qwen3）收敛到泛型 `CausalLm<B, Norm, Head>`。**Qwen3.5 hybrid 仍是唯一例外**：1176 行的 `hybrid.rs` 单体文件，混合 GDN（线性注意力）与 Full Attention 层，并维护 per-sequence 的 GDN 状态。

### 1.1 当前状态（2026-06-12）

| 模块 | 行数 | 职责 |
|------|------|------|
| `qwen3_5/hybrid.rs` | **1176** | Block 定义 + 权重加载 + ModelBackend + 测试 |
| `qwen3_5/gated_delta.rs` | 580 | GDN 核心（prefill/decode + state） |
| `qwen3_5/attention35.rs` | 267 | Full 层 MRoPE + paged GQA |
| `qwen3_5/arch.rs` | 178 | detect + weight remap + create_model |
| `causal_lm/layer_loop.rs` | ~140 | `DecoderLayer` / `run_layers`（已接入 hybrid） |

**已完成的收敛（Phase 3）：**

- `HybridBlock` 实现 `DecoderLayer`，走 `run_layers` / `run_layers_upto`
- `embed()` / `forward_to_layer()` 已接入 KV + GDN aux 路径
- Full 层 attention 已用 `paged_gqa` 共享 core

**仍未收敛：**

| 问题 | 影响 |
|------|------|
| `hybrid.rs` 单体过大 | 难 review、难测试、难复用 |
| `HybridBlock` 双路径 API | legacy `forward_prefill/decode` + `DecoderLayer` 重复 |
| 无 `HybridLm` shell | `ModelBackend` 与 forward 逻辑手写 ~200 行 |
| `layer_loop` → `qwen3_5::gated_delta` | `causal_lm` 反向依赖架构 crate 内部 |
| GDN 维度硬编码 `(16, 4, 2)` | `new` / `from_weights` 与真实 checkpoint 可能不一致 |
| `ArchCapabilities::HYBRID`（speculative: false） | self-speculation 未对 Qwen3.5 验收 |
| `Qwen3Config` 与 `ModelConfig` 双轨 | loader 需 `From` 转换，增加认知负担 |

### 1.2 Phase 5 目标

1. **可维护性：** `hybrid.rs` 拆分为 ≤4 个文件，model shell ≤150 行  
2. **架构一致：** 引入 `HybridLm`（或等价 shell），与 `CausalLm` 共享 batch/embed/forward_to_layer 辅助函数  
3. **依赖卫生：** `causal_lm` 不再 `use qwen3_5::*`  
4. **正确性：** GDN 状态在多 seq、prefill→decode、partial layer 场景有回归测试  
5. **可选升级：** 验收通过后 `ArchCapabilities` 升为 `PRODUCTION_SPECULATIVE`

### 1.3 非目标（本 Phase 不做）

- 将 GDN 层替换为通用 `components/ssm.rs` 的 MambaBlock（算法不同）  
- 实现 Qwen3.5 vision / multimodal 路径  
- Stub 架构（Gemma3/Llama4/Phi4）真实 forward  
- CUDA Graph / flash 新优化（除非拆分 PR 中顺手修复明显 bug）

---

## 2. 设计决策

### D5-1：`HybridLm` 而非扩展 `CausalLm`

**决策：** 新建 `causal_lm/hybrid_lm.rs`，不强行把 GDN state 塞进 `CausalLm`。

**理由：**

- `CausalLm::forward_with_cache` 无 `seq_id` 参数；hybrid 必须用 `HashMap<SeqId, GatedDeltaState[]>`  
- `lm_head: Option<Linear>` + tied embedding fallback 与 `CausalLm` 的 `Head: Module` 泛型冲突  
- Norm 类型为 `LayerNorm`（带 bias），与现有 `RmsNorm` / `LnLayerNorm` 分支不同  

**接口草案：**

```rust
pub struct HybridLm<B, Norm> {
    config: Qwen3Config,           // 短期保留；Wave 3 再评估 ModelConfig 统一
    embed_tokens: Embedding,
    layers: Vec<B>,
    norm: Norm,
    lm_head: Option<Linear>,
    kv_cache: PagedKvCache,
    gdn_states: HashMap<SeqId, Vec<Option<GatedDeltaState>>>,
    device: Device,
}

impl<B, Norm> HybridLm<B, Norm>
where
    B: DecoderLayer + Send + Sync,
    Norm: Module + Send + Sync,
{
    pub fn forward_with_cache(
        &mut self,
        seq_id: SeqId,
        tokens: &[TokenId],
        ...
    ) -> Result<(Tensor, usize)>;
}
```

`ModelBackend` impl 委托现有 `forward_batch` / `logits_to_vector` / `greedy_sample_token`（与 `CausalLm` 相同）。

### D5-2：`GatedDeltaState` 上移到 `components/`

**决策：** 将 `GatedDeltaState`（及可选 `GatedDeltaConfig`）移到 `components/gated_delta/` 或 `components/ssm/gated_delta.rs`。

**理由：** `LayerAuxMut::Gdn` 在 `causal_lm/layer_loop.rs`，不应依赖 `qwen3_5` 模块。

**迁移顺序：** 先 re-export 保持 API 稳定 → 改 import → 删旧路径。

### D5-3：删除 `HybridBlock` legacy forward 方法

**决策：** Wave 1 末尾只保留 `DecoderLayer` 作为 production 路径；`HybridBlock::forward_prefill(..., gdn_state)` 等降为 `pub(crate)` 或直接内联到 `DecoderLayer` impl。

### D5-4：Speculative 验收门槛

**决策：** 仅在以下条件 **全部** 满足后，将 `Qwen35Architecture::capabilities()` 从 `HYBRID` 改为 `PRODUCTION_SPECULATIVE`：

1. `forward_to_layer` 与 full `forward` 在 tiny config 上 token 一致（同 seed、同输入）  
2. GDN state 在 draft 步进后不污染 target 完整 forward（或文档化已知限制并实现 state 隔离）  
3. `SelfSpeculativeModel<Qwen35HybridModel>` 集成测试通过（可 `#[ignore]` slow）

若 GDN 层在 partial forward 下语义不明确，**保持 HYBRID**，在 capabilities 文档中说明限制。

---

## 3. 目标目录结构（Phase 5 完成后）

```text
crates/model/src/
├── causal_lm/
│   ├── mod.rs
│   ├── model.rs              # CausalLm（不变）
│   ├── hybrid_lm.rs          # NEW: HybridLm + ModelBackend
│   ├── layer_loop.rs         # LayerAuxMut 引用 components::GatedDeltaState
│   └── ...
├── components/
│   └── gated_delta/          # NEW（或 ssm/gated_delta.rs）
│       ├── mod.rs            # GatedDeltaState, GatedDeltaConfig
│       └── net.rs            # 从 qwen3_5/gated_delta.rs 迁入
└── qwen3_5/
    ├── arch.rs               # 薄 registry（不变）
    ├── block/
    │   ├── mod.rs            # HybridBlock enum
    │   ├── linear.rs         # LinearAttentionBlock
    │   └── full.rs           # FullAttentionBlock35
    ├── model.rs              # type alias + new/from_weights（~80 行）
    ├── weights.rs            # from_weights 层循环 + key 解析
    ├── config.rs             # LayerType 解析（从 Qwen3Config）
    ├── attention35.rs        # 保留（或后续并入 components/attention）
    └── hybrid.rs             # 删除或仅 re-export
```

---

## 4. 任务分解（4 个 Wave）

### Wave 1：文件拆分（低风险，无行为变更）

**预估：** 1 commit | **风险：** 低

| ID | 任务 | 验收 |
|----|------|------|
| 5.1.1 | 新建 `qwen3_5/block/linear.rs`、`full.rs`、`mod.rs`，迁出 block 定义 | 编译通过，block 单测不变 |
| 5.1.2 | 新建 `qwen3_5/weights.rs`，迁出 `from_weights` 层加载循环 | arch `create_model` 路径不变 |
| 5.1.3 | 新建 `qwen3_5/config.rs`，迁出 `LayerType` + `parse_layer_types` | layer type 单测通过 |
| 5.1.4 | `hybrid.rs` → `model.rs`（仅 `Qwen35HybridModel` + ModelBackend） | `hybrid.rs` 删除或 `pub use` |
| 5.1.5 | 测试迁至 `qwen3_5/model_tests.rs` 或各 block 模块 | `cargo test -p vllm-model qwen3_5` 全绿 |

**Wave 1 完成标志：** `hybrid.rs` 不存在或 ≤20 行 re-export；最大单文件 <500 行。

---

### Wave 2：`HybridLm` shell + 依赖解耦（中风险）

**预估：** 1–2 commit | **风险：** 中 | **依赖：** Wave 1

| ID | 任务 | 验收 |
|----|------|------|
| 5.2.1 | 新建 `causal_lm/hybrid_lm.rs`，提取 `forward_with_cache` / `ModelBackend` 公共逻辑 | 与 Wave 1 前行为 bit-identical（现有单测） |
| 5.2.2 | `Qwen35HybridModel = HybridLm<HybridBlock, LayerNorm>` type alias | `model.rs` <150 行 |
| 5.2.3 | 移动 `GatedDeltaState` → `components/gated_delta/` | `layer_loop.rs` 无 `qwen3_5` import |
| 5.2.4 | 删除 `HybridBlock` 对外 legacy forward API | 仅 `DecoderLayer` 路径 |
| 5.2.5 | 提取 `apply_lm_head(embed, lm_head, hidden)` 共享 helper | 消除 embed tie 重复 |

**Wave 2 完成标志：** `rg 'qwen3_5' crates/model/src/causal_lm/` 无匹配（layer_loop 清洁）。

---

### Wave 3：配置与权重加载硬化（中风险）

**预估：** 1 commit | **风险：** 中 | **依赖：** Wave 2

| ID | 任务 | 验收 |
|----|------|------|
| 5.3.1 | 从 `Qwen3Config` / `TextConfig` 读取 GDN 维度，移除硬编码 `(16,4,2)` | 新增 config 字段单测 |
| 5.3.2 | 统一 norm / lm_head key 解析（与 `CausalLm::load_lm_head` 对齐） | weight load 单测 |
| 5.3.3 | 评估 `FullAttentionBlock35` 是否可薄包装 `RopeGqaDecoderBlock` + MRoPE | 可选；parity 测试通过才合并 |
| 5.3.4 | 文档化 `remap_qwen35_weight_keys` 与 HF checkpoint 格式 | arch.rs 注释 + 测试 fixture |

---

### Wave 4：Speculative 验收与 Capability 升级（高风险）

**预估：** 1–2 commit | **风险：** 高 | **依赖：** Wave 2

| ID | 任务 | 验收 |
|----|------|------|
| 5.4.1 | 单测：`run_layers_upto` vs `run_layers` full forward logits 一致（Full-only tiny model） | 新测试绿 |
| 5.4.2 | 单测：GDN 层 prefill→decode 状态不丢失（已有 parity 扩展） | 新测试绿 |
| 5.4.3 | 单测：multi-seq GDN state 隔离（seq A 不影响 seq B） | 新测试绿 |
| 5.4.4 | 集成：`SelfSpeculativeModel` + `Qwen35HybridModel` draft 生成 | 通过或 documented skip |
| 5.4.5 | 若 5.4.1–5.4.4 通过：`ArchCapabilities::PRODUCTION_SPECULATIVE` | loader 日志 tier 更新 |

**已知风险（Wave 4）：** GDN 递归状态在 `forward_to_layer(upto_layer < gdn_layer)` 时可能语义不完整；若无法保证，保持 `HYBRID` 并在 `self_spec.rs` 对 Qwen3.5 禁用或 warn。

---

## 5. 验收标准（Phase 5 整体）

```bash
just fmt-check
cargo clippy -p vllm-model -- -D warnings
just nextest
cargo test -p vllm-model --lib qwen3_5
```

| 指标 | 目标 |
|------|------|
| `qwen3_5/` 最大单文件 | < 500 行 |
| `Qwen35HybridModel` shell（model.rs） | < 150 行 |
| `causal_lm` → `qwen3_5` 依赖 | 0 |
| 现有 Qwen3.5 单测 | 无回归 |
| nextest 全量 | 1036+ passed |

---

## 6. 风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| GDN state 与 speculative partial layer 不兼容 | 高 | 中 | Wave 4 先测后升 capability；不通过则文档化 |
| 权重 key 变体遗漏 | 中 | 高 | 保留现有 key fallback 链；加 fixture 测试 |
| 文件拆分引入 visibility 回归 | 低 | 中 | Wave 1 纯移动，零逻辑变更 |
| `GatedDeltaNet` 迁移破坏 parity | 中 | 高 | re-export 过渡；gated_delta 单测先行 |

---

## 7. 执行顺序与 Commit 策略

```text
Wave 1  →  refactor(qwen3_5): split hybrid.rs into block/model/weights modules
Wave 2  →  refactor(model): add HybridLm shell and decouple layer_loop from qwen3_5
Wave 3  →  fix(qwen3_5): read GDN dims from config, unify weight loading
Wave 4  →  test(qwen3_5): speculative parity and capability upgrade (if green)
```

每个 Wave 独立 commit，Wave 内不混合行为变更与纯移动。

---

## 8. 会话接续

1. 读本文 **Wave 1** 任务表  
2. `just nextest` 确认基线绿（当前 1036 passed）  
3. 从 **5.1.1** 开始，完成后更新下表  

### 进度记录

| 日期 | Wave | 动作 | Commit |
|------|------|------|--------|
| 2026-06-12 | — | Phase 5 规划文档创建 | — |
| 2026-06-15 | 1 | 拆分 hybrid.rs → block/model/weights/config | `decc8c8` |
| 2026-06-15 | 2 | HybridLm shell + GatedDelta 上移 components | `73dab5e` |
| 2026-06-12 | 3 | GDN 维度从 config 读取；统一 norm/lm_head 加载 | 待提交 |
| 2026-06-12 | 4 | speculative parity 测试 + `PRODUCTION_SPECULATIVE` | 待提交 |

---

## 9. 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-12 | 初版：基于 Phase 4 完成后的 codebase 分析 |
