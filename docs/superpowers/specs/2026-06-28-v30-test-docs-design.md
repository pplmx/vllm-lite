# v30.0 测试生态完善 + 文档提升 — 设计文档

**日期:** 2026-06-28
**作者:** vLLM-lite Team
**状态:** Approved
**前置里程碑:** v23.0 Audit Remediation (shipped 2026-06-28), v24.0-v29.0 (已完成)

---

## 1. Overview

v23.0 完成了"功能正确 + 默认 clippy 干净"的基线。v24-v29 在此基础上分别推进了**重构、lint、安全、性能、property-based、fuzz**等维度。本里程碑聚焦两个长期被低估的工程化要素:**测试质量主动验证**与**文档可达性**,把 vllm-lite 从"测试覆盖率高"推向"测试能持续发现 bug、文档能支撑新人 onboarding"。

### 目标

- **测试主动验证:** 引入 mutation testing,验证现有测试对逻辑变更的敏感度,确保 1212 个测试不仅覆盖代码路径,还能在逻辑错误时失败
- **Fuzz 持续运行:** 把 v29 的 fuzz targets 接入 CI,通过短期 fuzz-smoke + corpus 持久化 + crash 归档形成反馈闭环
- **覆盖盲点补齐:** 新增 tokenizer / GGUF / HTTP / Batch JSON 等 fuzz targets;为 sampling / eviction / priority queue 补充 proptest
- **文档覆盖到 99%+:** 当前 97.8% → 99%+,所有公开 API 配 ` ```no_run ` 可运行示例
- **测试策略沉淀:** 3 个测试相关 ADR + 2 篇设计文档,把"为什么这样测"固化
- **Tutorial & Onboarding:** 新贡献者 30 分钟内可走通 model load → inference → deploy

### 非目标

- 不引入新模型架构、新推理算法、新功能特性(NMC-01/02/03、OPS-05 仍为 v31+ 候选)
- 不动 v27.0 已固化的性能基线(除非 mutation 测试直接暴露 perf regression)
- 不做 breaking API change
- 不重写 crate 间依赖图
- 不引入需要 nightly-only 功能的稳定 API(可允许 CI/benches 内部使用 nightly)

---

## 2. 当前基线测量

### 2.1 状态指标

| 指标 | 当前值 | 目标值 | 来源 |
|------|--------|--------|------|
| Test 总数 | 1212 | ≥1300 | `cargo test --workspace` |
| Doc coverage | 97.8% | ≥99% | `scripts/doc_coverage.sh` |
| Cargo doc warnings | 0 | 0 | `cargo doc --no-deps` |
| Clippy (default) | 0 | 0 | `cargo clippy --workspace` |
| Fuzz targets | 3 (`app_config_yaml`, `safetensors_header`, `qwen3_config_json`) | ≥7 | `fuzz/fuzz_targets/` |
| proptest 模块 | 4 (RadixTree, BlockAllocator, RequestQueue, BatchComposer) | ≥7 | `crates/core/src/**/tests.rs` |
| Fuzz CI 集成 | ❌ 无 | ✅ fuzz-smoke on PR | `.github/workflows/` |
| Mutation testing | ❌ 无 | ✅ 接入 cargo-mutants | N/A |
| ADRs | 15 | ≥18 | `docs/adr/` |
| Tutorial docs | ❌ 无 | ✅ 完整路径 | `docs/tutorial/` |

### 2.2 Fuzz targets 现状

| Target | 模块路径 | 输入 | 已运行次数 |
|--------|---------|------|-----------|
| `app_config_yaml` | `crates/server/src/config.rs` | UTF-8 bytes → `serde_saphyr::from_str::<AppConfig>` | 751k |
| `safetensors_header` | `safetensors 0.7` crate | bytes → `SafeTensors::deserialize` | 8.77M |
| `qwen3_config_json` | `crates/model/src/qwen3/config.rs` | bytes → `serde_json::from_slice::<Qwen3Config>` | 8.13M |

### 2.3 proptest 现状

| 模块 | properties | 关键发现 |
|------|-----------|---------|
| RadixTree | 3 | 插入+查找 round-trip |
| BlockAllocator | 3 | LIFO 复用、容量边界 |
| RequestQueue | 4 | FIFO 顺序、phase index 一致性 |
| BatchComposer | 7 | **发现 `compose_decode_batch` 在空 token 时 `tokens_len - 1` 下溢 bug**,已修复 |

---

## 3. 架构:6-Phase 结构

### Phase K — Mutation Testing 接入 (2-3 plans)

**目标:** 用 `cargo-mutants` 在核心模块跑 mutation testing,验证现有测试对逻辑变更的敏感度。

**范围限定:**
- `crates/core/src/scheduler/` (engine, batch_composer, policy, memory)
- `crates/core/src/kv_cache/` (block_allocator, prefix_cache)
- `crates/core/src/sampling.rs`
- **不跑:** model 层(计算密集)、server 层(IO 密集)、CUDA-only 代码

**计划:**
- K-1: 接入 `cargo-mutants` (workspace dev-dep),生成 baseline report
- K-2: 处置 top-N 高生存 mutation(每个 module 跑完后分析),补充针对性测试
- K-3: 维护脚本(`just mutants MODULE`)+ 文档(mutation testing 选型说明)

**成功标准:**
- 核心模块 mutation score ≥ 70%(industry baseline: 60-80%)
- 任何"equal" 生存者(逻辑等价 mutation)被显式标注或排除
- `just mutants` 一键复现

### Phase L — Fuzz CI 集成 + Corpus 持久化 (2-3 plans)

**目标:** 把 v29 的 fuzz 接入 CI 流程,确保 PR 不引入 panic 路径。

**计划:**
- L-1: `.github/workflows/fuzz.yml` — push/PR 触发,运行 `just fuzz-smoke` (10s × N targets),失败即报错
- L-2: Corpus 持久化策略 — GitHub Actions cache (key 含 target hash),crash 自动上传 artifact
- L-3: 文档 `docs/fuzz.md` + justfile 增量(失败快速复现命令 `just fuzz-repro CRASH_FILE`)

**成功标准:**
- CI fuzz-smoke job 稳定运行(<5 分钟)
- Crash 出现时自动归档 + 通知 PR
- Corpus 在 cache 中累积,跨 PR 复用

**注意:** GitHub Actions free tier 2000 分钟/月。fuzz-smoke 每次约 30-60s × N targets,远低于预算。

### Phase M — 测试覆盖扩充 (3-4 plans)

**目标:** 补齐 fuzz 与 proptest 覆盖盲点。

**M-1: 新增 fuzz targets (4 个):**
- `tokenizer_decode`: 输入 → `tiktoken-rs` decode,捕获非法 token
- `gguf_header`: bytes → GGUF magic + metadata parsing
- `openai_http_request`: bytes → `serde_json::from_slice::<ChatCompletionRequest>`
- `batch_json_input`: bytes → `BatchRequest` 解析

**M-2: 新增 proptest 模块 (3 个):**
- `SamplingStrategy` (top_k / top_p / temperature 组合不变性: 输出长度 == 输入长度、token id ∈ vocab)
- `EvictionPolicy` (LRU/priority invariant: 驱逐后总 block 数守恒、priority 排序保持)
- `PriorityQueue` (heap invariant: pop 顺序严格按 priority 降序、长度守恒)

**成功标准:**
- Fuzz targets 总数 ≥ 7
- proptest 模块总数 ≥ 7
- 每个新 fuzz target 跑 60s 无 crash

### Phase N — 文档覆盖 + Examples (3-4 plans)

**目标:** doc coverage 97.8% → 99%+,所有公开 API 配 ` ```no_run ` 示例。

**计划:**
- N-1: 跑 `scripts/doc_coverage.sh json`,分析剩余 2.2% 缺文档项,分类(真缺 / `#[doc(hidden)]` / macro 生成)
- N-2: 给所有 pub struct/enum/trait 加 module-level 描述 + link 到 ADR
- N-3: 给所有 pub fn/method 加 ` ```no_run ` 示例(不含断言,只展示典型用法)
- N-4: 修剩余 cargo doc warnings(主要是 broken intra-doc links)

**成功标准:**
- `scripts/doc_coverage.sh` 输出 ≥ 99% pub items documented
- `cargo doc --no-deps` 0 warnings
- 每个 crate 的 `lib.rs` 顶部有 crate-level tour(example list + quick start)

### Phase O — ADR + Design Docs (2 plans)

**目标:** 沉淀 v30 测试决策,补 2 篇设计文档。

**O-1: 新增 3 个 ADR:**
- `ADR-016-proptest-strategy.md` — 为什么选 proptest、覆盖哪些组件、properties 选取标准
- `ADR-017-fuzz-strategy.md` — 为什么选 cargo-fuzz、target 选取标准、corpus 管理
- `ADR-018-mutation-testing.md` — 为什么选 cargo-mutants、scope 限定、生存者处置策略

**O-2: 新增 2 篇设计文档:**
- `docs/superpowers/specs/2026-XX-XX-kv-cache-evolution.md` — v24 重构后 kv_cache 模块的设计回顾
- `docs/superpowers/specs/2026-XX-XX-scheduler-evolution.md` — v24 重构后 scheduler 模块的设计回顾

**成功标准:**
- 3 个新 ADR 在 `docs/adr/README.md` 索引中
- 2 篇设计文档通过 self-review(无 TBD、无矛盾)
- 在 PR 中至少 1 名 reviewer 确认 ADR 决策可执行

### Phase P — Tutorial & Onboarding (2 plans)

**目标:** 新贡献者从 clone 到 serving 模型 < 30 分钟。

**P-1: `docs/tutorial/` 完整路径:**
- `01-setup.md` — 环境准备(Rust 1.85+、可选 CUDA、clone、build)
- `02-load-model.md` — 用 `ModelLoader` 加载一个测试模型(checkpoint_loading_tests 镜像)
- `03-inference.md` — 跑通 `crates/server/tests/integration.rs` 中最简用例
- `04-customize.md` — 添加新 scheduler 策略 / sampling 策略的 hook 点
- `05-production.md` — 部署到本地或 k8s(引用 `k8s/` 已有 manifest)

**P-2: 同步更新:**
- `CONTRIBUTING.md` 加入 tutorial 引用
- `README.md` quick start 章节改用 tutorial 链接
- 一个端到端集成测试镜像 tutorial 步骤(可作为 regression 测试)

**成功标准:**
- 一个不熟悉 vllm-lite 的开发者可按 tutorial 走通
- tutorial 中的命令在干净环境可执行
- 端到端镜像测试通过

---

## 4. 依赖图与波次结构

```text
                    Wave 1 (并行启动)
                  ┌─────────┬─────────┬─────────┐
                  │  K      │  N      │  O      │
                  │ mutation│ docs+ex │  ADR    │
                  └────┬────┴────┬────┴────┬────┘
                       │         │         │
                       ▼         ▼         ▼
                    Wave 2 (并行启动)
                  ┌─────────┬─────────┬─────────┐
                  │  L      │  M      │  P      │
                  │ fuzz CI │ coverage│tutorial │
                  └─────────┴─────────┴─────────┘
```

**Wave 1 (K, N, O) 并行:**
- K (mutation): 独立,只依赖 `cargo-mutants` 安装
- N (docs): 独立,可基于 v23/v24/v25 已完成 doc 改进继续
- O (ADR): 独立,基于 K/L/M 之前的决策

**Wave 2 (L, M, P) 并行:**
- L (fuzz CI): 依赖 v29 的 fuzz 基础设施;M 的新 target 在 L 完成后增量纳入
- M (coverage expansion): 独立,可先行
- P (tutorial): 依赖 N 的 rustdoc 改进(避免 tutorial 引用过期 API 文档)

**Phase 内每个 plan 都遵循 v23.0 模式:** 探索 → 实施 → 验证 → 文档

---

## 5. 质量门

### 5.1 每个 phase 完成门

- `just ci` 全绿(fmt + clippy + doc-check + nextest)
- Doc coverage 不下降(最终目标 99%+)
- CHANGELOG.md 更新对应 phase 条目
- `.planning/STATE.md` 进度同步
- 至少 1 个 commit 含 phase 完成标记(如 `test(mutant): Phase K complete`)

### 5.2 v30 全局门

| 指标 | 目标 |
|------|------|
| 6 phases | 全部 ship |
| Test 总数 | ≥1300(v29 基线 1212 + 新增 ≥88) |
| Cargo doc warnings | 0 |
| ADRs | ≥18(原 15 + 新增 3) |
| Tutorial 路径 | 5 篇,端到端测试通过 |
| Mutation score (核心模块) | ≥70% |
| Fuzz targets | ≥7 |
| proptest 模块 | ≥7 |

### 5.3 验证清单

- `just ci` 通过
- `just fuzz-smoke` 通过(全部 target 不 panic)
- `just mutants` 在核心模块跑完一轮(<1h)
- `cargo doc --no-deps --document-private-items --workspace --all-features` 0 warnings
- 端到端集成测试镜像 tutorial 步骤,通过

---

## 6. 关键风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| `cargo-mutants` 跑完一轮耗时过长(>1h) | Phase K 难推进 | 限定核心模块,分批跑;接受 phase 内长 plan |
| GitHub Actions free tier 时间不足跑 fuzz | Phase L 受阻 | 10-30s 短期 fuzz-smoke;corpus 增量;可后续加 self-hosted runner |
| Doc coverage 卡在 98.5%(剩余是 macro 生成 / `#[doc(hidden)]`) | Phase N 目标难达 | 区分"真实缺文档"与"#[doc(hidden)]",定义新的"用户可见覆盖率"指标 |
| Tutorial 写得过细反而无法维护 | Phase P 失效 | Tutorial 引用真实测试作为可执行示例,测试本身是活文档 |
| Fuzz 在 CI 上 flaky(time-based panic) | Phase L 误报 | 用伪随机种子固定;`-seed=1` 或 corpus 模式;失败时保存 artifact |
| Mutation testing 暴露 v23 已 ship 的代码 bug | Phase K 范围蔓延 | 仅记录为 follow-up,不强制在 K 内修复(避免 K 变 phase B) |
| proptest 维护成本(generator 容易过时) | Phase M 长期成本 | ADR-016 记录 generator 选取标准;每个 proptest 模块加注释解释 invariant |

---

## 7. 不在范围内 / 未来工作

- **GPU-side mutation/fuzz testing** — 留待 v32+;当前 CPU-only
- **Coverage-guided fuzz with SanCov in production** — 留待 v32+
- **Crash reproduction automation** — 当前依赖手动 reproduce,后续可加 fuzz-replay CI
- **Tutorial 视频化** — 留待 docs 稳定后
- **NMC-01 长上下文、OPS-05 多节点复活** — 仍为 v31+ 候选

---

## 8. 时间预估

| Phase | 预估 commits | 估算工时 |
|-------|-------------|---------|
| K | 3-5 | 中(等待 mutation 运行) |
| L | 3-5 | 中(CI 配置调试) |
| M | 5-7 | 中(fuzz target 编写 + 调试) |
| N | 6-8 | 高(API 数量大) |
| O | 4-5 | 中(ADR review) |
| P | 4-5 | 中(tutorial 编写 + 镜像测试) |
| **合计** | **25-35 commits** | **~3-5 个工作日纯执行** |

---

## 9. 参考

- v23.0 Audit Remediation: `.planning/STATE.md`
- v29.0 Fuzz Testing: `fuzz/` 目录,CHANGELOG.md
- v28.0 Property-Based Testing: `crates/core/src/**/tests.rs`,CHANGELOG.md
- 现有 fuzz justfile targets: `justfile:fuzz-build`、`fuzz-smoke`、`fuzz TARGET`、`fuzz-list`
- Doc coverage 工具: `scripts/doc_coverage.sh`
- CI workflow 模板: `.github/workflows/ci.yml`
