# v30.0 Phase P: Tutorial & Onboarding

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新贡献者从 clone 到 serving 模型 < 30 分钟。提供 5 篇教程:`01-setup.md` 到 `05-production.md`。

**Architecture:**
- tutorial 放在 `docs/tutorial/` 目录
- 每篇 tutorial 引用真实测试作为可执行示例(测试本身是活文档)
- 一个端到端集成测试镜像 tutorial 步骤(可作为 regression 测试)
- `CONTRIBUTING.md` 与 `README.md` 加入 tutorial 引用

**Tech Stack:** markdown, 现有 `crates/server/tests/integration.rs` 等集成测试

**前置依赖:**
- Phase N(rustdoc 已稳定,避免 tutorial 引用过期 API)
- Phase M(覆盖扩充后,tutorial 中的命令更稳定)
- 不阻塞 K / L / O

**关联:**
- 上游:v23+ 文档基线、所有 phase 的可执行示例
- 下游:无(v30 终点 phase)

---

## File Structure

| File | Change | Sub-phase |
|------|--------|-----------|
| `docs/tutorial/01-setup.md` (NEW) | 环境准备(Rust、CUDA 可选、clone、build) | P-1 |
| `docs/tutorial/02-load-model.md` (NEW) | 用 ModelLoader 加载测试模型 | P-1 |
| `docs/tutorial/03-inference.md` (NEW) | 跑通最简 inference(镜像 integration test) | P-1 |
| `docs/tutorial/04-customize.md` (NEW) | 添加 scheduler / sampling 策略的 hook 点 | P-2 |
| `docs/tutorial/05-production.md` (NEW) | 部署到本地 / k8s(引用 `k8s/`) | P-2 |
| `crates/server/tests/tutorial_e2e.rs` (NEW) | 镜像 tutorial 步骤的端到端测试 | P-2 |
| `CONTRIBUTING.md` | 加入 tutorial 引用 | P-2 |
| `README.md` | quick start 改用 tutorial 链接 | P-2 |
| `CHANGELOG.md` | v30.0 Phase P 条目 | P-2 |

---

## Sub-phase Plan(待 Phase K 完成后展开为 bite-sized tasks)

### P-1: Tutorial 前 3 篇 (3 tasks)
- P-1.1: `01-setup.md` — 环境准备
- P-1.2: `02-load-model.md` — 加载模型(引用 `crates/server/tests/integration.rs::test_load_simple_model`)
- P-1.3: `03-inference.md` — 跑通最简 inference(引用最简 integration test)

### P-2: Tutorial 后 2 篇 + 镜像测试 + CONTRIBUTING (5 tasks)
- P-2.1: `04-customize.md` — 添加自定义策略
- P-2.2: `05-production.md` — 部署(引用 `k8s/` 与 `docker-compose.yml`)
- P-2.3: `crates/server/tests/tutorial_e2e.rs` — 镜像测试
- P-2.4: 更新 CONTRIBUTING.md + README.md
- P-2.5: CHANGELOG 更新

---

## Tutorial 内容大纲

### 01-setup.md
- Rust 1.85+ 安装(rustup)
- CUDA(可选,11.8+ for Qwen3 / 12.x for newest)
- clone + `cargo build --workspace`
- 跑 `just nextest` 验证 baseline

### 02-load-model.md
- 选测试模型(checkpoint_loading_tests 用的最小模型)
- `ModelLoader::builder(Device::Cpu).with_model_dir(...).with_kv_blocks(1024).build()?`
- 处理可能的错误(`EngineError` 子类型)

### 03-inference.md
- 用 `engine.add_request(...)`
- 跑 `engine.step()` 一次
- 验证 output tokens 数量符合预期
- 与 `crates/server/tests/integration.rs` 中最简用例对照

### 04-customize.md
- 实现 `Policy` trait 的扩展点
- 加到 `ArchitectureRegistry`
- proptest 验证 invariant
- 引用 v28 proptest 用法

### 05-production.md
- `docker-compose.yml` 本地起服务
- `k8s/` manifest 部署到集群
- Prometheus + Grafana 接入(引用 `docs/grafana/`)
- 常见 troubleshooting

---

## 已知风险

- **教程命令在干净环境不一定 work** — P-2.3 端到端测试是验证手段
- **教程可能引用 v24 重构前的 API** — N phase 必须先完成(因此 P 在 Wave 2)
- **教程写得过细反而难维护** — 限制每篇 ≤ 200 行
- **k8s manifest 可能陈旧** — 教程只引用,不重复内容

---

## 验证清单

- [ ] 5 篇 tutorial 在干净 Linux 环境可走通
- [ ] `crates/server/tests/tutorial_e2e.rs` 通过
- [ ] CONTRIBUTING.md / README.md 链接正确
- [ ] CHANGELOG 反映 v30.0 Phase P 完成
- [ ] `just ci` 全绿

---

## 待 Phase K 完成后展开为详细 bite-sized plan

**当前 stub 不含可执行细节,仅为 phase scope 与执行顺序。**
