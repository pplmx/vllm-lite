# Wave 1: 文档同步 + dead_code 审计实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `.planning/` 文档与 `main @ d42b151+` 实际状态对齐；清理 refactor 后残留的 `#[allow(dead_code)]`。

**Architecture:** 纯文档/注释工作，零行为变更。每个 commit 独立可回滚，验证通过再做下一个。

**Tech Stack:** Markdown, ripgrep, cargo clippy

**基线 commit:** `d42b151`（spec 已落地）

**前置验证:**

```bash
cd /workspace/vllm-lite
just fmt-check       # 必须绿
just nextest         # 必须 ≥ 1036 passed
git log --oneline -1 # 应为 d42b151
```

---

## Task 1: 同步 MODEL-ARCHITECTURE-REFACTOR.md

**文件:**
- Modify: `.planning/MODEL-ARCHITECTURE-REFACTOR.md`

- [ ] **Step 1: 更新顶部日期与整体进度行**

```text
// 替换前（第 4 行附近）:
> **最后更新:** 2026-06-12（Phase 4 完成；Phase 5 规划见 `.planning/PHASE-5-QWEN35-HYBRID.md`）

// 替换后:
> **最后更新:** 2026-06-26（Phase 0–5 全部完成；进入 Wave 1 收口 + Wave 2 增量）
```

```text
// 替换前:
**整体进度:** `[██████████░] Phase 0–4 + Gemma4 完成；Phase 5 待执行`

// 替换后:
**整体进度:** `[██████████] Phase 0–5 全部完成；进入 Wave 1 文档收口 + Wave 2–5 spec decode 增量`
```

- [ ] **Step 2: 更新执行进度总览 Phase 5 行**

```text
// 替换前:
| **5** | Qwen3.5 Hybrid 收敛 | ⬜ 未开始 | 0/4 waves | [规划](./PHASE-5-QWEN35-HYBRID.md) |

// 替换后:
| **5** | Qwen3.5 Hybrid 收敛 | ✅ 完成 | 4/4 waves | `decc8c8` / `73dab5e` / `52f77ce` |
```

- [ ] **Step 3: 更新成熟度基线 Qwen3.5 行注释**

```text
// 替换前:
| **A- 生产级（hybrid）** | Qwen3.5 | `run_layers` + GDN state | 主路径 OK；[Phase 5](./PHASE-5-QWEN35-HYBRID.md) 收敛中 |

// 替换后:
| **A 生产级（hybrid + spec）** | Qwen3.5 | `HybridLm` + `run_layers` + GDN state | 主路径 + 自投机 OK；`PRODUCTION_SPECULATIVE` 自 `52f77ce` |
```

- [ ] **Step 4: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat .planning/MODEL-ARCHITECTURE-REFACTOR.md
# 预期: 1 file changed, ~6 insertions, ~3 deletions

git add .planning/MODEL-ARCHITECTURE-REFACTOR.md
git commit -m "docs(planning): sync arch-refactor phase 5 completion"
```

预期 commit hash: 与 `52f77ce` 之后的新 commit。

---

## Task 2: 同步 PHASE-5-QWEN35-HYBRID.md

**文件:**
- Modify: `.planning/PHASE-5-QWEN35-HYBRID.md`

- [ ] **Step 1: 更新顶部状态行**

```text
// 替换前（第 4 行）:
> **状态:** 🔄 Wave 2 完成（待 commit）

// 替换后:
> **状态:** ✅ 完成（2026-06-26；Wave 1–4 全部合并至 main）
```

- [ ] **Step 2: 更新 §8 进度记录表**

```text
// 替换前:
| 2026-06-15 | 1 | 拆分 hybrid.rs → block/model/weights/config | `decc8c8` |
| 2026-06-15 | 2 | HybridLm shell + GatedDelta 上移 components | `73dab5e` |
| 2026-06-12 | 3 | GDN 维度从 config 读取；统一 norm/lm_head 加载 | 待提交 |
| 2026-06-12 | 4 | speculative parity 测试 + `PRODUCTION_SPECULATIVE` | 待提交 |

// 替换后:
| 2026-06-15 | 1 | 拆分 hybrid.rs → block/model/weights/config | `decc8c8` |
| 2026-06-15 | 2 | HybridLm shell + GatedDelta 上移 components | `73dab5e` |
| 2026-06-15 | 3 | GDN 维度从 config 读取；统一 norm/lm_head 加载 | `52f77ce` |
| 2026-06-15 | 4 | speculative parity 测试 + `PRODUCTION_SPECULATIVE` 升级 | `52f77ce`（同 Wave 3 commit） |
| 2026-06-26 | 1.5 | 文档同步：状态、引用、CHANGELOG 补录 | `d42b151` + 本 Wave 后续 commit |
```

- [ ] **Step 3: §9 变更日志追加**

```text
// 在文末追加:
| 2026-06-26 | Wave 1 收口：状态标完成；同步 Wave 3–4 commit；CHANGELOG 补录 |
```

- [ ] **Step 4: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat .planning/PHASE-5-QWEN35-HYBRID.md
# 预期: 1 file changed, ~8 insertions, ~5 deletions

git add .planning/PHASE-5-QWEN35-HYBRID.md
git commit -m "docs(qwen3_5): mark Phase 5 waves complete"
```

---

## Task 3: 同步 STATE.md / PROJECT.md / ROADMAP.md

**文件:**
- Modify: `.planning/STATE.md`
- Modify: `.planning/PROJECT.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: 更新 STATE.md 时间戳与位置**

```text
// 替换前:
last_updated: "2026-05-13T05:00:00.000Z"
last_activity: 2026-05-13

// 替换后:
last_updated: "2026-06-26T00:00:00.000Z"
last_activity: 2026-06-26
```

```text
// 替换前（Current Position 段）:
Phase: 4 of 4 (Phase 17.4: Speculative Warmup & Metrics)
Plan: 24 plans completed (17.1-A through 17.4-H)
Status: ALL PHASES COMPLETE
Last activity: 2026-05-13 — Phase 17.4 verified complete

// 替换后:
Wave: 1 of 5 (Wave 1: 文档同步 + dead_code 审计)
Status: Wave 1 in progress; Wave 2–5 in pipeline
Last activity: 2026-06-26 — Wave 1 spec 落地 (`d42b151`)
```

- [ ] **Step 2: 更新 PROJECT.md v17.0 active 区**

```text
// 替换前（v17.0 Active 段第 100–108 行）:
- [ ] **SPEC-ENG-01**: Engine integration of speculative decoding (`step_speculative`)
- [ ] **SPEC-ENG-02**: Seamless fallback between speculative and non-speculative paths
- [ ] **SPEC-BENCH-01**: Real hardware benchmark suite (throughput, latency, P50/P95/P99)
- [ ] **SPEC-BENCH-02**: Baseline comparison vs non-speculative inference
- [ ] **SPEC-ADAPT-01**: Adaptive draft depth based on real-time acceptance rates
- [ ] **SPEC-ADAPT-02**: Acceptance rate monitoring and dynamic adjustment
- [ ] **SPEC-WARM-01**: Speculative warmup (prefill draft KV cache before decode)
- [ ] **SPEC-MULTI-01**: External draft model support (smaller model as drafter)
- [ ] **SPEC-MULTI-02**: Draft model lifecycle management (load/unload/swap)

// 替换后:
- [x] **SPEC-ENG-01**: Engine integration — `step_speculative_inner` (commit `52f77ce`)
- [x] **SPEC-ENG-02**: Seamless fallback — parity tests in `qwen3_5/speculative_tests.rs`
- [ ] **SPEC-BENCH-01**: Real hardware benchmark suite → Wave 5
- [ ] **SPEC-BENCH-02**: Baseline comparison vs non-speculative inference → Wave 5
- [ ] **SPEC-ADAPT-01**: Adaptive draft depth → Wave 2
- [ ] **SPEC-ADAPT-02**: Acceptance rate monitoring → Wave 2
- [ ] **SPEC-WARM-01**: Speculative warmup → Wave 4
- [ ] **SPEC-MULTI-01**: External draft model support → deferred to v18.0
- [ ] **SPEC-MULTI-02**: Draft model lifecycle management → deferred to v18.0
```

- [ ] **Step 3: 更新 PROJECT.md Last updated**

```text
// 替换前:
*Last updated: 2026-05-13 — v17.0 milestone started*

// 替换后:
*Last updated: 2026-06-26 — Wave 1 收口；Wave 2–5 spec decode 增量在 pipeline*
```

- [ ] **Step 4: ROADMAP.md 加一行 Qwen3.5 注释**

```text
// 替换前（第 156 行附近）:
| Qwen3.5-0.8B (Mamba) | Mamba SSM + Attention  | ✅   |

// 替换后:
| Qwen3.5-0.8B (Mamba) | Mamba SSM + Attention  | ✅ + spec |

// 同时在表格下方加注:
> 2026-06-26 更新：Qwen3.5 自 `52f77ce` 起 capability 升 `PRODUCTION_SPECULATIVE`，支持自投机解码。Phase 5 重构已完成（`MODEL-ARCHITECTURE-REFACTOR.md`）。
```

- [ ] **Step 5: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat .planning/STATE.md .planning/PROJECT.md ROADMAP.md
# 预期: 3 files changed, ~25 insertions, ~10 deletions

cargo check --workspace  # 确认 markdown 改动没意外影响
git add .planning/STATE.md .planning/PROJECT.md ROADMAP.md
git commit -m "docs(planning): reflect speculative engine integration in v17 status"
```

---

## Task 4: CHANGELOG 补录 Phase 5

**文件:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 在 [Unreleased] → Refactored 段补条目**

定位：在 `### Refactored` → `#### Architecture Refactoring` 子段之后追加新子段：

```text
// 在 CHANGELOG.md 第 55 行附近（`### Added (Phase 4)` 之前）追加:

#### Architecture Refactor Phase 5 (Qwen3.5 Hybrid 收敛, 2026-06-15)

- Split `qwen3_5/hybrid.rs` (1176 lines) into `block/` + `model.rs` + `weights.rs` + `config.rs`
- Introduce `HybridLm<B, Norm>` shell paralleling `CausalLm<B, N, H>`
- Move `GatedDeltaState` from `qwen3_5::gated_delta` to `components::gated_delta`
- Remove `causal_lm → qwen3_5` reverse dependency (`rg 'use qwen3_5' crates/model/src/causal_lm/` → 0 matches)
- GDN dims now read from `Qwen3Config` (no more hardcoded `(16, 4, 2)`)
- `Qwen35Architecture::capabilities()` upgraded to `PRODUCTION_SPECULATIVE`
- Speculative parity tests in `model_tests.rs` (124 lines) + `speculative_tests.rs` (285 lines)

Refs: `decc8c8`, `73dab5e`, `52f77ce`
```

- [ ] **Step 2: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat CHANGELOG.md
# 预期: 1 file changed, ~12 insertions

git add CHANGELOG.md
git commit -m "docs(core): add Phase 5 refactor entries to CHANGELOG"
```

---

## Task 5a: dead_code 审计 — `vllm-traits` + `vllm-core`

**文件:**
- Audit: `crates/traits/src/**/*.rs`
- Audit: `crates/core/src/**/*.rs`

- [ ] **Step 1: 列出审计范围内的 dead_code**

```bash
cd /workspace/vllm-lite
rg '#\[allow\(dead_code\)\]' crates/traits/ crates/core/ -n
# 预期: 列出当前作用域内的所有 allow(dead_code) 项（数量 X，记录到分类表）
```

- [ ] **Step 2: 逐条分类**

按 spec §1.5 的分类表格式记录每条：

```text
| 文件:行 | 分类 | 动作 | 备注 |
|--------|------|------|------|
| crates/core/src/foo.rs:42 | 保留 | 补 // audited: 注释 | feature flag 守卫 |
| crates/core/src/bar.rs:88 | 删除 | 移除 allow + 删代码 | refactor 后无 caller |
| ... |
```

每行读上下文判断，必要时用 `git log -p -- crates/X/Y.rs` 查历史。

- [ ] **Step 3: 应用变更**

对每条分类为"删除"的：移除 `#[allow(dead_code)]` 行 + 删除对应代码 + 跑 `cargo build -p vllm-core` 验证。

对每条分类为"保留"的：在 allow 行下方补：

```rust
#[allow(dead_code)] // audited 2026-06-26 (Wave 1): <理由>
```

- [ ] **Step 4: 验证 + commit**

```bash
cd /workspace/vllm-lite
cargo build -p vllm-traits -p vllm-core
cargo clippy -p vllm-traits -p vllm-core -- -D warnings
just nextest -p vllm-traits -p vllm-core

git add crates/traits/ crates/core/
git commit -m "chore(core): audit and classify dead_code allow attributes (Wave 1.5a)"
```

---

## Task 5b: dead_code 审计 — `vllm-model`

**文件:**
- Audit: `crates/model/src/**/*.rs`（按子目录：components/ → 各 arch/ → kernels/ → loader/）

- [ ] **Step 1: 列出审计范围**

```bash
cd /workspace/vllm-lite
rg '#\[allow\(dead_code\)\]' crates/model/ -n | wc -l
# 记录总数到分类表
```

- [ ] **Step 2: 逐目录分类**

按目录分批，每批单独审：

```bash
# 子目录扫描
rg '#\[allow\(dead_code\)\]' crates/model/src/components/ -n
rg '#\[allow\(dead_code\)\]' crates/model/src/llama/ crates/model/src/mistral/ crates/model/src/qwen3/ crates/model/src/qwen3_5/ crates/model/src/gemma4/ crates/model/src/mixtral/ -n
rg '#\[allow\(dead_code\)\]' crates/model/src/kernels/ crates/model/src/loader/ crates/model/src/paged_tensor/ -n
```

每条按 spec §1.5 三分类（保留/删除/stub）。注意：

- `arch/{gemma3,llama4,phi4,mistral_small}.rs` 中的项应为 "stub" 类（Option C）
- `components/` 中的项应为 "保留" 类（feature flag 守卫或 API 预留）
- 各 arch 的项可能是 refactor 后遗留的 dead 引用 → "删除" 类

- [ ] **Step 3: 应用变更（同 5a 模式）**

- [ ] **Step 4: 验证 + commit**

```bash
cd /workspace/vllm-lite
cargo build -p vllm-model
cargo clippy -p vllm-model -- -D warnings
just nextest -p vllm-model

git add crates/model/
git commit -m "chore(model): audit and classify dead_code allow attributes (Wave 1.5b)"
```

---

## Task 5c: dead_code 审计 — `vllm-server` + `vllm-dist`

**文件:**
- Audit: `crates/server/src/**/*.rs`
- Audit: `crates/dist/src/**/*.rs`

- [ ] **Step 1: 列出审计范围**

```bash
cd /workspace/vllm-lite
rg '#\[allow\(dead_code\)\]' crates/server/ crates/dist/ -n
```

- [ ] **Step 2: 逐条分类**

- `vllm-server`：多为 HTTP handler 入口保留项或 feature flag 守卫
- `vllm-dist`：tensor parallel 相关，可能是 "保留" 或 "删除"（若未启用）

- [ ] **Step 3: 应用变更**

- [ ] **Step 4: 验证 + commit**

```bash
cd /workspace/vllm-lite
cargo build -p vllm-server -p vllm-dist
cargo clippy -p vllm-server -p vllm-dist -- -D warnings
just nextest -p vllm-server -p vllm-dist

git add crates/server/ crates/dist/
git commit -m "chore(server,dist): audit and classify dead_code allow attributes (Wave 1.5c)"
```

---

## Task 6: 刷新 SESSION-HANDOFF.md

**文件:**
- Modify: `.planning/SESSION-HANDOFF.md`

- [ ] **Step 1: 更新 Git 行**

```text
// 替换前（第 4 行）:
> Git：`main` @ `e192492`+（Phase 0–4 + Gemma4→CausalLm 完成）

// 替换后:
> Git：`main` @ `085089e` + Wave 1 spec `d42b151`（Phase 0–5 + Qwen3.5 Hybrid 收敛完成）
```

- [ ] **Step 2: 替换"下一优先级"段**

定位第 8–18 行，替换为：

```text
## 下一优先级（2026-06-26）

**Wave 1: 文档同步 + dead_code 审计** — [`docs/superpowers/specs/2026-06-26-wave1-doc-sync-design.md`](../../docs/superpowers/specs/2026-06-26-wave1-doc-sync-design.md) | [`Wave 1 计划`](../../docs/superpowers/plans/2026-06-26-wave1-doc-sync.md)

| Task | 内容 | 状态 |
|------|------|------|
| 1 | 同步 MODEL-ARCHITECTURE-REFACTOR.md | ⬜ |
| 2 | 同步 PHASE-5-QWEN35-HYBRID.md | ⬜ |
| 3 | 同步 STATE/PROJECT/ROADMAP 三件套 | ⬜ |
| 4 | CHANGELOG 补 Phase 5 | ⬜ |
| 5a–c | dead_code 审计（traits/core/model/server/dist） | ⬜ |
| 6 | 刷新本 HOFF | ⬜ |

**后续 Wave:** Wave 2 (adaptive draft) → Wave 3 (Dependabot) → Wave 4 (warmup) → Wave 5 (benchmarks)
```

- [ ] **Step 3: 已知差距段清理**

```text
// 替换前（"已知差距"段第 113–124 行附近）:
// 整个段重写为：

### 已完成（Wave 0 + 1 收口）

- ✅ MoE 向量化（`mixtral/sparse_moe.rs`）
- ✅ Gemma4 sliding window mask（`compute_attention`）
- ✅ `TransformerBlock: PagedDecoderBlock` + metadata
- ✅ Phase 0–5 架构重构（commit `decc8c8` ~ `52f77ce`）

### 待 Wave 1.5 处置

- `#[allow(dead_code)]` 30 处审计（分类表见 spec §1.5）

### 中优先级（Wave 2+）

- Flash 真 CUDA kernel（需 GPU 环境）
- Mistral/Llama final norm shape 风险（`from_weights`）
- Dependabot 5 个漏洞（1 high, 4 moderate）
```

- [ ] **Step 4: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat .planning/SESSION-HANDOFF.md
# 预期: 1 file changed, ~30 insertions, ~25 deletions

git add .planning/SESSION-HANDOFF.md
git commit -m "docs(planning): refresh SESSION-HANDOFF for Wave 1 status"
```

---

## 收口验证

所有 8 个 commit 完成后：

```bash
cd /workspace/vllm-lite

# 1. 全量 CI
just ci

# 2. 文档一致性
echo "--- commit 引用交叉检查 ---"
git log --oneline -10
# 与各 .planning/*.md 中引用的 commit hash 对照

# 3. dead_code 数量
echo "--- dead_code 审计后剩余数量 ---"
rg '#\[allow\(dead_code\)\]' crates/ -c | sort -t: -k2 -nr | head -10

# 4. 文档日期一致性
echo "--- .planning/ 文档日期 ---"
rg 'last_updated|最后更新|2026-06-26' .planning/ -l

# 5. 测试基线
just nextest 2>&1 | tail -3
```

**Wave 1 完成标志：**

- ✅ `just ci` 全绿
- ✅ `just nextest` ≥ 1036 passed（无回归）
- ✅ 30 处 allow 全部出现在分类表中，每条有明确动作
- ✅ `.planning/` 文档日期统一为 `2026-06-26` 或更新
- ✅ `rg 'use qwen3_5' crates/model/src/causal_lm/` 仍为 0（验证 refactor 维持）

---

## 错误处理 / 风险

| 风险 | 缓解 |
|------|------|
| 文档日期笔误 | Task 末尾统一 grep 验证 |
| dead_code 误删导致编译失败 | 每 commit 前 `cargo build` 验证 |
| 30 处 allow 计数变化 | 若审计中发现漏数（如新加 allow），补到分类表，不擅自扩大范围 |
| Markdown 格式破坏 | `just fmt-check` 不验证 .md，但可肉眼检查 diff |
| SESSION-HANDOFF 引用错 commit hash | Task 6 Step 1 后 grep 验证 |

---

## 不做（明确边界）

- **不**修复 Dependabot 漏洞（独立风险评估）
- **不**实现 stub 架构（Option C 决策）
- **不**做 SPEC-ADAPT / WARM / BENCH（后续 Wave）
- **不**重写 `run_attention_fn` flash 路径
- **不**更新 `docs/adr/`（架构决策无变更）
- **不**调整 `Cargo.toml` 依赖版本

---

## 自审

- **Spec 覆盖:** ✅ 任务 1–6 对应 spec §任务 1.1–1.6；提交策略 §提交策略 表完整对应
- **占位符扫描:** ✅ 无 TBD/TODO；每处变更都有具体 before/after
- **类型一致性:** ✅ 仅 markdown + 注释，无代码类型变更
- **范围:** ✅ 8 个 commit，单次会话可完成

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-26 | 初版：基于 `docs/superpowers/specs/2026-06-26-wave1-doc-sync-design.md` |
