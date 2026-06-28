# v30.0 Phase N: 文档覆盖 + API Examples

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 文档覆盖率从 97.8% 推至 ≥99%,所有公开 API 配 ` ```no_run ` 可运行示例,`cargo doc --no-deps` 0 warnings。

**Architecture:**
- 分两层推进:第一层补 doc comment(覆盖率),第二层补 example(可达性)
- 每个 crate 的 `lib.rs` 顶部加 crate-level tour(模块索引 + 5 分钟 quick start + 链接到关键 example)
- example 全部使用 ` ```no_run ` (不需要 doctest 编译,但保留代码可读性);少量关键路径用 ` ``` ` 完整 doctest
- 修复 broken intra-doc links(常见于 v24 重构后的模块重命名)

**Tech Stack:** `scripts/doc_coverage.sh`(已存在)、rustdoc、`cargo doc --no-deps`

**前置依赖:**
- v23.0 已 ship 文档基线(97.8%)
- v24-v25 已修复大量 lint 警告(docs 已稳定)
- 不阻塞 K,但 L/M 完成后才能完整对比 baseline

**关联:**
- 上游:v23 文档基线、Phase O ADR(引用本文档策略)
- 下游:Phase P(tutorial 引用 rustdoc)

---

## File Structure

| File | Change | Sub-phase |
|------|--------|-----------|
| `scripts/doc_coverage.sh` | 增加 "user-visible" 区分(`#[doc(hidden)]` 排除) | N-1 |
| 全部 pub struct/enum/trait 模块 | 补 /// doc + module-level 链接 | N-2 |
| 全部 pub fn/method | 补 ` ```no_run ` 示例 | N-3 |
| 各 crate `lib.rs` | 顶部 crate-level tour | N-2 |
| `CHANGELOG.md` | v30.0 Phase N 条目 | N-4 |

---

## Sub-phase Plan(待 Phase K 完成后展开为 bite-sized tasks)

### N-1: doc coverage 工具增强 (1 task)
- N-1.1: 区分 "real coverage" vs `#[doc(hidden)]` vs macro 生成,定义 user-visible 覆盖率指标

### N-2: pub 类型 doc (3-4 tasks)
- N-2.1: `vllm-core` pub types 补 /// doc
- N-2.2: `vllm-model` pub types 补 /// doc
- N-2.3: `vllm-traits` / `vllm-server` pub types 补 /// doc
- N-2.4: 各 crate `lib.rs` 顶部 crate-level tour

### N-3: API examples (3-4 tasks)
- N-3.1: `vllm-core` 关键 fn/method 示例
- N-3.2: `vllm-model` 关键 fn/method 示例
- N-3.3: `vllm-server` 关键 handler 示例
- N-3.4: `vllm-traits` trait 方法示例

### N-4: 验证 + CHANGELOG (1 task)
- N-4.1: `cargo doc --no-deps --document-private-items --workspace --all-features` 0 warnings,doc coverage ≥99%

---

## 已知风险

- **剩余 2.2% 中可能含 macro 生成代码**(如 `#[derive]` 的字段)— 不计入 user-visible
- **example 代码可能引用过期 API** — 写 example 后跑 `cargo test --doc` 验证可编译
- **crate-level tour 可能与 README 重复** — tour 强调 "代码视角",README 强调 "用户视角",分工明确
- **broken intra-doc links** — 需要 grep `[name](` 形式的 doc 链接并修复

---

## 验证清单

- [ ] `scripts/doc_coverage.sh` 输出 ≥99% pub items documented
- [ ] `cargo doc --no-deps --document-private-items --workspace --all-features` 0 warnings
- [ ] 各 crate `lib.rs` 顶部 tour 可读且链接有效
- [ ] 关键 fn/method 有可读 ` ```no_run ` 示例
- [ ] CHANGELOG 反映 v30.0 Phase N 完成

---

## 待 Phase K 完成后展开为详细 bite-sized plan

**当前 stub 不含可执行细节,仅为 phase scope 与执行顺序。**
