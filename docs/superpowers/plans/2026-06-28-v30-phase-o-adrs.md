# v30.0 Phase O: ADR + Design Docs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 沉淀 v30 测试决策为 ADR,补 2 篇设计文档(回顾 v24 重构后的 kv_cache 演进、scheduler 演进)。

**Architecture:**
- 3 个新 ADR 沿用现有模板(`docs/adr/ADR-NNN-*.md`)
- 2 篇设计文档按现有 `docs/superpowers/specs/YYYY-MM-DD-topic-design.md` 命名
- 每个 ADR / design doc 通过 self-review(无 TBD、无矛盾)后才 commit
- `docs/adr/README.md` 索引更新

**Tech Stack:** markdown,现有的 ADR / spec 模板

**前置依赖:** 不阻塞任何 phase,可最早启动

**关联:**
- 上游:v24.0 重构、v28/v29 测试基础
- 下游:Phase N(文档可能引用 ADR)、Phase P(tutorial 可能引用设计文档)

---

## File Structure

| File | Change | Sub-phase |
|------|--------|-----------|
| `docs/adr/ADR-016-proptest-strategy.md` (NEW) | proptest 选型与 properties 选取标准 | O-1 |
| `docs/adr/ADR-017-fuzz-strategy.md` (NEW) | cargo-fuzz 选型 + target 选取 + corpus 策略 | O-1 |
| `docs/adr/ADR-018-mutation-testing.md` (NEW) | cargo-mutants 选型 + scope 限定 + 生存者处置 | O-1 |
| `docs/adr/README.md` | 索引追加 ADR-016/017/018 | O-1 |
| `docs/superpowers/specs/2026-XX-XX-kv-cache-evolution.md` (NEW) | v24 重构后 kv_cache (radix_cache) 模块设计回顾 | O-2 |
| `docs/superpowers/specs/2026-XX-XX-scheduler-evolution.md` (NEW) | v24 重构后 scheduler 模块设计回顾 | O-2 |
| `CHANGELOG.md` | v30.0 Phase O 条目 | O-2 |

---

## Sub-phase Plan(待 Phase K 完成后展开为 bite-sized tasks)

### O-1: 新增 3 个 ADR (1 task per ADR)
- O-1.1: ADR-016 proptest-strategy
- O-1.2: ADR-017 fuzz-strategy
- O-1.3: ADR-018 mutation-testing
- O-1.4: 更新 `docs/adr/README.md` 索引

### O-2: 新增 2 篇设计文档 + CHANGELOG (2 tasks)
- O-2.1: kv-cache 演进设计回顾
- O-2.2: scheduler 演进设计回顾 + CHANGELOG 更新

---

## ADR 内容大纲

### ADR-016-proptest-strategy.md
- Context:v28 引入 proptest,已覆盖 4 个组件
- Decision:proptest 1.11 + workspace dev-dep + `#[cfg(test)] mod tests {}` 嵌入源文件
- Rationale:与 cargo test 无缝集成、arbitrary 自动生成、shrinking 友好
- Consequences:generator 维护成本(后续 M phase 文档化标准)
- Alternatives 考虑:quickcheck、arbitrary-framework、custom fuzz

### ADR-017-fuzz-strategy.md
- Context:v29 引入 cargo-fuzz,已 3 个 target
- Decision:cargo-fuzz 0.13 + libFuzzer + ` ```no_run ``` 注释解释每个 target 覆盖的 parser
- Rationale:libFuzzer 是 Rust 生态最成熟的 fuzz 引擎,corpus 持久化方案成熟
- Consequences:CI 需 nightly toolchain(L phase 解决)
- Alternatives 考虑:arbitrary 配合 proptest、AFL++、honggfuzz-rs

### ADR-018-mutation-testing.md
- Context:v30 K 引入 cargo-mutants
- Decision:cargo-mutants 25.x + scope 限定 vllm-core 核心模块 + 本地运行(不进 CI)
- Rationale:行业标准、scope 限定避免长耗时、CI 评估推迟到 v31
- Consequences:本地 mutation scan 需 30-60 分钟(开发者工作站)
- Alternatives 考虑:mutagen、cargo-mutagen、strazar/japanese-modern

---

## 已知风险

- **设计回顾文档容易写得像 CHANGELOG** — 必须聚焦"设计决策回顾",而非 commit-by-commit
- **ADR 数量增加让新人 onboarding 变长** — README 索引需分组(架构 / 测试 / 安全)
- **设计文档可能与现有 15 个 ADR 重复** — 写前先 grep 现有 ADR,避免重复

---

## 验证清单

- [ ] 3 个新 ADR 通过 self-review
- [ ] 2 篇设计文档通过 self-review
- [ ] `docs/adr/README.md` 索引更新
- [ ] CHANGELOG 反映 v30.0 Phase O 完成
- [ ] `just ci` 全绿

---

## 待 Phase K 完成后展开为详细 bite-sized plan

**当前 stub 不含可执行细节,仅为 phase scope 与执行顺序。**
