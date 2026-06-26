# Wave 1: 文档同步 + dead_code 审计设计

**日期**: 2026-06-26
**状态**: 🔄 待审
**作者**: AI 协作
**基线**: `main @ 085089e` (Phase 0–5 + Qwen3.5 Hybrid 收敛已完成)

---

## 背景

近期项目完成了大量架构重构（Phase 0–5，见 `.planning/MODEL-ARCHITECTURE-REFACTOR.md`），但规划文档与代码实际状态出现偏差：

- `MODEL-ARCHITECTURE-REFACTOR.md` 仍标 Phase 5 为"未开始"
- `PHASE-5-QWEN35-HYBRID.md` 进度表只填到 Wave 2，状态栏写"Wave 2 完成（待 commit）"但 Wave 3、4 已合并
- `STATE.md` v17.0 active items 未反映 SPEC-ENG-01/02 已落地
- `ROADMAP.md` / `PROJECT.md` 未提及 Qwen3.5 已升 `PRODUCTION_SPECULATIVE`
- 30 处 `#[allow(dead_code)]` 部分源于 refactor 前的旧约束，未逐一审计是否仍适用

Wave 1 不引入新功能，纯属"让规划系统恢复可信"——是后续 SPEC-ADAPT / SPEC-WARM 等增量工作的前置。

---

## 目标

1. **规划文档与代码同步**：`.planning/` 下任何里程碑/阶段描述与 `main` 实际 commit 对齐
2. **CHANGELOG 完整**：补录 Phase 5 重构条目
3. **dead_code 透明化**：30 处 allow 标记全部审过，分类处置（保留+注释 / 删除 / 转 stub）

**非目标：**

- Dependabot 漏洞修复（→ Wave 3）
- SPEC-ADAPT-01/02（→ Wave 2）
- SPEC-WARM-01（→ Wave 4）
- SPEC-BENCH-01/02（→ Wave 5）
- 4 个 stub 架构（Gemma3/Llama4/Phi4/MistralSmall）的实现/移除决策（已 Option C，仅文档标注）
- `vllm-dist` 张量并行相关 dead_code（专项审查需独立 spec）

---

## 设计

### 任务 1.1 — 架构重构总览文档同步

**文件:** `.planning/MODEL-ARCHITECTURE-REFACTOR.md`

变更：

| 区域 | 变更 |
|------|------|
| "最后更新"日期 | `2026-06-12` → `2026-06-26` |
| "整体进度"行 | `[██████████░] Phase 0–4 + Gemma4 完成；Phase 5 待执行` → `[██████████] Phase 0–5 全部完成；进入 Wave 1 收口 + Wave 2 增量` |
| 执行进度总览 Phase 5 行 | `⬜ 未开始 0/4 waves` → `✅ 完成 4/4 waves`，补 commit hash `decc8c8` / `73dab5e` / `52f77ce` |
| 成熟度基线 Qwen3.5 | `A- 生产级（hybrid）` 注释补"`PRODUCTION_SPECULATIVE` 自 `52f77ce` 起" |
| 新增段落 | "Wave 1: 文档同步与 dead_code 审计"链接到本 spec |

### 任务 1.2 — Phase 5 专项文档同步

**文件:** `.planning/PHASE-5-QWEN35-HYBRID.md`

变更：

| 区域 | 变更 |
|------|------|
| 顶部状态 | `🔄 Wave 2 完成（待 commit）` → `✅ 完成（2026-06-26）` |
| §8 进度记录表 | 补 `Wave 3 → 52f77ce` 行；`Wave 4 → 52f77ce (含 parity 测试)` 行（合并到 Wave 3 commit） |
| §9 变更日志 | 补 `2026-06-26: Wave 3–4 完成，capability 升级 PRODUCTION_SPECULATIVE` |

### 任务 1.3 — STATE / PROJECT / ROADMAP 三件套同步

#### `.planning/STATE.md`

| 区域 | 变更 |
|------|------|
| milestone / milestone_name | 保留 `v17.0` / `Production Speculative Decoding` |
| last_updated | `2026-05-13` → `2026-06-26` |
| last_activity | `2026-05-13` → `2026-06-26` |
| Current Position | `Phase: 4 of 4 ... ALL PHASES COMPLETE` → 改为 `Wave 1 of 5: 文档同步 + dead_code 审计` |
| Accumulated Context → Active SPEC | 列已落地：SPEC-ENG-01/02（`step_speculative_inner`）；未落地：SPEC-ADAPT-01/02、WARM-01、BENCH-01/02、MULTI-01/02（deferred v18） |

#### `.planning/PROJECT.md`

| 区域 | 变更 |
|------|------|
| Current Milestone "Goal" | 加一句：`Wave 1 已落地文档同步；engine integration 已完成；adaptive/warmup/benchmark 待 Wave 2/4/5` |
| v17.0 Active 区 | `SPEC-ENG-01` / `SPEC-ENG-02` 标 `[x]`（带 commit `52f77ce`） |
| Context 段 v17.0 注释 | 补 "Wave 1 收口 (2026-06-26); Wave 2–5 in pipeline" |
| Last updated | `2026-05-13` → `2026-06-26` |

#### `ROADMAP.md`

| 区域 | 变更 |
|------|------|
| Phase 6 多模型表 | Qwen3.5 行状态保持 ✅；注释补 `+ PRODUCTION_SPECULATIVE (Wave 4)` |
| 长期愿景区 | 加 "近期: Wave 2–5 spec decode 收尾（v17 关闭）" |

### 任务 1.4 — CHANGELOG 补录

**文件:** `CHANGELOG.md`

在 `[Unreleased]` → `Refactored` 段补：

```text
#### Architecture Refactor Phase 5 (Qwen3.5 Hybrid 收敛)

- Split `qwen3_5/hybrid.rs` (1176 lines) into `block/` + `model.rs` + `weights.rs` + `config.rs`
- Introduce `HybridLm<B, Norm>` shell paralleling `CausalLm`
- Move `GatedDeltaState` from `qwen3_5::gated_delta` to `components::gated_delta`
- Remove `causal_lm → qwen3_5` reverse dependency
- GDN dims now read from `Qwen3Config` (no more hardcoded `(16, 4, 2)`)
- `Qwen35Architecture::capabilities()` upgraded to `PRODUCTION_SPECULATIVE`
- Speculative parity tests (`model_tests.rs`, `speculative_tests.rs`)

Refs: `decc8c8`, `73dab5e`, `52f77ce`
```

### 任务 1.5 — dead_code 审计

**目标:** 30 处 `#[allow(dead_code)]` 全部有明确归属（保留+文档 / 删除 / 转 stub 标注）。

**执行步骤：**

1. **清单：** `rg '#\[allow\(dead_code\)\]' crates/ -n` 列出全部 30 处
2. **分类表（每条 1 行）：**

   | 文件:行 | 分类 | 动作 | 备注 |
   |--------|------|------|------|
   | `crates/X/src/Y.rs:N` | 保留 | 补 `// audited: <理由>` 注释 | 例如：`#[cfg(feature = "gguf")]` 守的入口 |
   | `crates/X/src/Y.rs:N` | 删除 | 移除 allow + 删除代码 | refactor 后已无 caller |
   | `crates/X/src/Y.rs:N` | stub | 改为 `Option C` 文档化 stub | 4 个 stub 架构残留 |

3. **提交流水：** 按目录分批，每批 1 个 commit：
   - 1.5a: `vllm-traits` + `vllm-core` 审计
   - 1.5b: `vllm-model` 审计（按 arch 子目录再分）
   - 1.5c: `vllm-server` + `vllm-dist` 审计
4. **每批验证：** `cargo clippy --workspace -- -D warnings` + `just nextest`

**审计完成标志：**

- 30 处全部出现在分类表中
- `rg '#\[allow\(dead_code\)\]' crates/` 数量：保留类（含新注释） + 删除类（已移除） = 30
- 测试 1036+ passed，无回归

### 任务 1.6 — SESSION-HANDOFF 更新

**文件:** `.planning/SESSION-HANDOFF.md`

| 区域 | 变更 |
|------|------|
| 顶部 "Git" 行 | `main @ e192492+` → `main @ 085089e (Wave 1 of 5 in progress)` |
| "下一优先级" | 改为 Wave 1 任务表（5 项），Wave 2–5 仅一行概述 |
| 近期 Commit 脉络 | 补 `decc8c8` / `73dab5e` / `52f77ce` / `085089e` |
| 已知差距 | 删已完成项（MoE 向量化、Gemma4 sliding、TransformerBlock trait） |

---

## 提交策略

| # | Commit | 范围 | 类型 |
|---|--------|------|------|
| 1 | `docs(planning): sync arch-refactor phase 5 completion` | MODEL-ARCHITECTURE-REFACTOR.md | docs |
| 2 | `docs(qwen3_5): mark Phase 5 waves complete` | PHASE-5-QWEN35-HYBRID.md | docs |
| 3 | `docs(planning): reflect speculative engine integration in v17 status` | STATE.md + PROJECT.md + ROADMAP.md | docs |
| 4 | `docs(core): add Phase 5 refactor entries to CHANGELOG` | CHANGELOG.md | docs |
| 5a | `chore(core): audit and classify dead_code allow attributes` | crates/{traits,core}/ | chore |
| 5b | `chore(model): audit and classify dead_code allow attributes` | crates/model/ | chore |
| 5c | `chore(server,dist): audit and classify dead_code allow attributes` | crates/{server,dist}/ | chore |
| 6 | `docs(planning): refresh SESSION-HANDOFF for Wave 1 status` | SESSION-HANDOFF.md | docs |

每个 commit 独立可回滚，验证通过后再做下一个。

---

## 错误处理

| 风险 | 缓解 |
|------|------|
| 文档日期/链接笔误 | commit 后 `just fmt-check`（不直接验证 markdown，但防格式问题） |
| dead_code 误删导致编译失败 | 5a/5b/5c 每批 commit 前 `cargo build -p <crate>` 验证 |
| 文档与代码仍然不同步 | Wave 1 末尾 `git log --oneline -30` 与文档引用交叉检查 |
| SESSION-HANDOFF 引用旧 commit hash 误导 | Wave 1 末尾人工对照一次 |

---

## 测试 / 验证

每 commit 后必跑：

```bash
just fmt-check
cargo clippy --workspace -- -D warnings
just nextest
```

Wave 1 末尾全量：

```bash
# 文档一致性
git log --oneline -10  # 与 doc 中 commit hash 对照
rg '#\[allow\(dead_code\)\]' crates/ -c  # 应 = 保留类数

# 代码正确性
just ci  # = fmt-check + clippy + doc + nextest
```

成功标准：

- `just ci` 全绿
- `just nextest` ≥ 1036 passed（无回归）
- 30 处 allow 全部出现在分类表中，每条有明确动作
- `.planning/` 文档日期统一为 `2026-06-26` 或更新

---

## 不做（明确）

- **不**修复 Dependabot 漏洞（独立 PR，需要 bump 主版本号或选择替换库）
- **不**实现 stub 架构（已有 Option C 决策）
- **不**做 SPEC-ADAPT / SPEC-WARM / SPEC-BENCH（后续 Wave）
- **不**重写现有 `run_attention_fn` flash 路径
- **不**更新 `docs/adr/`（如有架构决策变更，需独立 ADR PR）

---

## 风险与决策记录

| ID | 决策 | 理由 | 日期 |
|----|------|------|------|
| D1 | Wave 1 不做 Dependabot | 1 high vuln 可能需 Rust 工具链升级或库替换；独立风险评估更稳 | 2026-06-26 |
| D2 | dead_code 按目录分批 commit | 5a/5b/5c 单独可回滚，避免大批 churn | 2026-06-26 |
| D3 | SESSION-HANDOFF 仅小改 | 不重写整体结构，仅更新"下一优先级"段 | 2026-06-26 |
| D4 | ROADMAP.md 改动最小 | ROADMAP 是面向用户的"完成态快照"，不应频繁变更 | 2026-06-26 |

---

## 会话接续

Wave 1 完成后，下一 session 应：

1. 读 `.planning/WAVE-2-PLAN.md`（由 writing-plans skill 生成）
2. 跑 `just ci` 确认基线
3. 按 Wave 2 任务表从 SPEC-ADAPT-01 acceptance rate monitor 开始

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-26 | 初版：基于 Wave 0 规划对话（`085089e` 基线） |
