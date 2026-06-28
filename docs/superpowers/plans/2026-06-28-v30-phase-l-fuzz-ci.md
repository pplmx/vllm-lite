# v30.0 Phase L: Fuzz CI 集成 + Corpus 持久化

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 v29.0 已建立的 3 个 fuzz targets 接入 GitHub Actions CI,通过短期 fuzz-smoke + corpus 持久化 + crash 归档形成 PR 反馈闭环。新增 target(via Phase M)完成后,自动纳入 CI 运行。

**Architecture:**
- 新增 `.github/workflows/fuzz.yml` — push/PR 触发,运行 `just fuzz-smoke`(10s × N targets)
- corpus 通过 `actions/cache` 持久化,key 含 target hash 与 cargo-fuzz 版本
- crash 发现时,artifact 自动上传到 GitHub Actions run,PR 评论提示
- fuzz-smoke 用伪随机 seed(`-seed=1`)减少 flaky;corpus 模式 (`-corpus=...`)确保跨 PR 复用

**Tech Stack:** GitHub Actions, `actions/cache`, cargo-fuzz 0.13.x

**前置依赖:** v29.0 fuzz 已 ship(✅),Phase K 不阻塞 L 启动

**关联:**
- 上游:v29.0 fuzz targets、`justfile` `fuzz-*` targets
- 下游:Phase M(新 fuzz targets 会被 L 自动纳入)、Phase P(tutorial 引用 fuzz-smoke 流程)

---

## File Structure

| File | Change | Sub-phase |
|------|--------|-----------|
| `.github/workflows/fuzz.yml` (NEW) | CI workflow,fuzz-smoke on push/PR | L-1 |
| `.github/workflows/fuzz-nightly.yml` (NEW) | nightly long-run fuzz(15min × N) | L-2 |
| `docs/fuzz.md` (NEW) | fuzz 文档(corpus、crash 处置、本地运行) | L-3 |
| `justfile` | 新增 `fuzz-repro` crash 复现 target | L-3 |
| `CHANGELOG.md` | v30.0 Phase L 条目 | L-3 |

不修改任何 Rust 源代码 — L 阶段只引入 CI 与文档。

---

## Sub-phase Plan(待 Phase K 完成后展开为 bite-sized tasks)

### L-1: GitHub Actions fuzz-smoke workflow (3 tasks)
- L-1.1: 安装 nightly toolchain + cargo-fuzz
- L-1.2: 跑 `just fuzz-smoke` 在 PR 上下文
- L-1.3: 上传 crash artifacts

### L-2: Corpus 持久化 + nightly long-run (2-3 tasks)
- L-2.1: 配置 `actions/cache` key/correlation
- L-2.2: nightly workflow 跑 15min × N targets
- L-2.3: crash 检测 + issue 自动创建

### L-3: 文档 + 维护脚本 (2 tasks)
- L-3.1: 写 `docs/fuzz.md`(corpus 管理、crash repro、本地 long-run)
- L-3.2: `just fuzz-repro CRASH_FILE` 命令 + CHANGELOG

---

## 已知风险

- **GitHub Actions time budget**:free tier 2000 分钟/月。`fuzz-smoke` 10s × 3 targets ≈ 30s + build time ≈ 5 分钟/次,远低于预算。
- **nightly toolchain 拉取**:首次跑会下载 nightly toolchain(约 500MB)。可用 `Swatinem/rust-cache` 缓存。
- **crash artifact 可能含敏感数据**: 默认上传,但需要在 docs 里明确 policy。

---

## 验证清单

- [ ] `.github/workflows/fuzz.yml` 通过 GitHub Actions Linter 校验
- [ ] 在 PR 中触发 fuzz-smoke job 并完成
- [ ] crash artifact 上传测试(可用 `cargo +nightly fuzz run` 配合人工构造输入触发)
- [ ] `docs/fuzz.md` 含本地 long-run 指南
- [ ] `just fuzz-repro` 在已知 crash 上可复现
- [ ] CHANGELOG 反映 v30.0 Phase L 完成
- [ ] `just ci` 全绿(无源码改动,应当已绿)

---

## 待 Phase K 完成后展开为详细 bite-sized plan

每个 task 应包含:
- 实际 yaml 配置块
- 实际 justfile 行
- 实际命令 + 预期输出
- commit message

**当前 stub 不含可执行细节,仅为 phase scope 与执行顺序。**
