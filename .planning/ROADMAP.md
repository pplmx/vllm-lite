# Roadmap: vllm-lite

## Milestones

- ✅ **v16.0 Speculative Decoding** — Phases 16.1-16.4 (shipped 2026-04-28)
- ✅ **v17.0 Production Speculative Decoding** — Phases 17.1-17.4 (shipped 2026-05-13)
- ✅ **v18.0 Multi-Model Speculative Decoding** — Phases 18.1-18.4 + Phase 19 gap closure (shipped 2026-06-27)
- ✅ **v19.0 Codebase Health Audit** — Phases 20-24 (shipped 2026-06-27; analysis-only, no code changes; see `.planning/audit/` for deliverables)
- ✅ **v20.0 Codebase Remediation** — Phases 25-30 (shipped 2026-06-27; BACKLOG.md-driven; 6 sub-phases v20.1-v20.6; 48/48 requirements; 1144 tests pass; clippy/fmt clean; doc coverage 97.8%)
- 🚧 **v21.0 P2/P3 Backlog Cleanup** — Phases 31-35 (Phase 31 complete 2026-06-27; 4 sub-phases remaining v21.2-v21.5; 42 requirements: 9 ML done + 11 API + 8 NAM + 4 DOC + 6 P3 + 4 FINAL pending; ~71h estimated)

## Phases

- [x] **Phase 18.1: Draft Registry + External Loading** - 引入 DraftModelRegistry 与外部 draft model 加载
- [x] **Phase 18.2: Lifecycle + Memory Budget** - 卸载时回收 KV cache,执行 VRAM 预算
- [x] **Phase 18.3: Request Routing + Fallback** - 请求级 draft 路由,失败回退到 self-spec
- [x] **Phase 18.4: Integration Tests + Benchmarks** - E2E 验证 + 多 draft 性能基线
- [x] **Phase 19: Wire v18.0 into Engine step loop** - DraftResolver 接入 + HTTP exporter + ServerDraftLoader
- [x] **Phase 20: Architecture Audit** - crate 依赖图 / 模块边界 / 循环依赖 / 分层一致性 / 测试架构
- [x] **Phase 21: Naming Audit** - 文件 / 类型 / 方法 / 变量 / 模块命名一致性 + 语义清晰度
- [x] **Phase 22: Comments + Documentation Audit** - doc 覆盖率 / 模块文档 / 过期注释 / 外部文档 / ADR
- [x] **Phase 23: API + Error Handling Audit** - API 一致性 / 错误类型 / 错误人体工学 / trait 设计 / 弃用卫生
- [x] **Phase 24: Synthesis + Remediation Backlog** - 跨维度综合 + P0/P1/P2 优先级清单 + v20.0+ 迁移路线图
- [x] **Phase 25: P0 Critical Fixes** (v20.1) - 消除 vllm-model→vllm-dist 依赖 + ModelError enum + 8 个 trait object-safe + CudaGraphError thiserror
- [x] **Phase 26: Module Tree Restoration** (v20.2) - orphan 模块挂回 (kv_cache_fp8 + debug) + stage-info 重命名 + 3 个测试文件迁移 + vllm-dist feature-gate
- [x] **Phase 27: Error Handling Standardization** (v20.3) - 13 个 error enum 用 thiserror + Result<_,String> 消除 + mutex-poison 修复 + EngineError 4 新变体 + anyhow 边界
- [x] **Phase 28: Documentation Coverage Push** (v20.4) - workspace doc 7.6%→≥60% + 776 个 pub item /// + 121 个文件 //! + README 修复
- [x] **Phase 29: External Docs + ADRs** (v20.5) - README/AGENTS.md 调和 + 12 个新 ADR
- [x] **Phase 30: Naming + Final Polish** (v20.6) - 7 P1 + 19 P2 命名 + #[deprecated] 卫生 + 注释清理 + 最终验证 (test pass + clippy + fmt)
- [x] **Phase 31: Module Layout Reorganization** (v21.1) - draft_registry 拆分 + engine/speculative 子树 + qwen3_config 下沉 + attention/util 提取 + TensorParallelError 迁移 + test_fixtures 重定位
- [ ] **Phase 32: API Consistency** (v21.2) - builder 约定文档化 + #[source] 补 + Box<dyn Error> 替换 + 22 个 builder 引入 + FallbackStrategy sync/async 拆分 + 错误 context 携带
- [ ] **Phase 33: Naming Consistency** (v21.3) - flash_v3 重命名 + NodeInfo 评估 + AGENTS.md 命名约定文档化 + 非 tensor 单字母变量重命名
- [ ] **Phase 34: External Doc Fixes** (v21.4) - DeepSeek 修正 + vllm-dist ADR + Phase 5 Wave 4 ref + PROJECT.md Key Decisions 交叉链接
- [ ] **Phase 35: P3 Actionable + Final Verification** (v21.5) - P3 actionable 收尾 + MIGRATING.md + CircuitBreakerError 变体 + FINAL gates

## Phase Details

### ✅ v18.0 Multi-Model Speculative Decoding (Phases 18.1-18.4 + 19) — SHIPPED 2026-06-27

**Milestone Goal:** 兑现 v17 延期的 MULTI-01/02/03,引入外部 draft model(可与 target 不同架构/尺寸),实现请求级 draft 路由、并发 target + draft 显存预算与运行时回退语义。Self-spec 路径(v17)保持为基线回退。

#### Phase 18.1: Draft Registry + External Loading

**Goal**: Engine 加载并注册与 target 不同架构/尺寸的外部 draft model,weights 懒加载
**Depends on**: Phase 17.4 (v17.0 baseline shipped)
**Requirements**: MMLT-01, MMLT-02, MMLT-03, LIFE-01
**Success Criteria** (what must be TRUE):

  1. Engine 可通过 DraftModelRegistry 加载与 target 不同架构/尺寸的 draft model 实例
  2. 每个外部 draft 拥有独立 ModelBackend 实例,KV cache block IDs 与 target 隔离,无状态泄漏
  3. Draft weights 懒加载 — 首个使用该 draft 的请求触发实际加载,无冷启动惩罚
  4. Registry 暴露 register / load / unload 运行时操作,可在 Engine 启动后调用
  5. 异构 draft(target=Llama + draft=Qwen 等)与 target 共存无冲突

**Plans**: 1 plan (18.1)

Plans:

- [x] 18.1: DraftModelRegistry + lazy loading — 12 unit tests + 3 engine integration tests

#### Phase 18.2: Lifecycle + Memory Budget

**Goal**: Draft model 可安全卸载并触发 KV 回收,VRAM 预算在加载/运行时均受约束
**Depends on**: Phase 18.1
**Requirements**: LIFE-02, LIFE-03, MEM-01, MEM-02, MEM-03
**Success Criteria** (what must be TRUE):

  1. 卸载 draft model 通过 MemoryManager 释放其 KV cache blocks,无 orphan blocks 残留
  2. Registry 跟踪每 draft 的引用计数,refcount 归零时自动卸载
  3. Engine 在加载前计算总 VRAM 预算 = target weights + target KV cache + N concurrent drafts
  4. 加载时权重尺寸估算来自 model loader metadata,无需触发完整加载
  5. 运行时 KV cache 增长被追踪,超出预算时引擎拒绝加载并返回结构化错误

**Plans**: 1 plan (18.2)

Plans:

- [x] 18.2: MemoryBudget + refcount lifecycle + auto-unload — 25 tests across budget, registry, allocator, engine (254 total)

#### Phase 18.3: Request Routing + Fallback

**Goal**: 请求按需选择 draft model,draft 失败时优雅降级到 self-spec 或非推测式解码
**Depends on**: Phase 18.2
**Requirements**: RTE-01, RTE-02, RTE-03, FALL-01, FALL-02
**Success Criteria** (what must be TRUE):

  1. Request 可通过 SamplingParams 或 Request 结构指定 `draft_model_id`
  2. Scheduler 在 batch 组合时将每个请求路由到正确的 draft model 实例
  3. 同一 batch 内允许多种 draft 并存(mixed routing)
  4. 外部 draft 加载失败自动回退到 self-spec 路径(v17 baseline),用户无感知
  5. 运行时 draft 推理错误对该请求优雅降级为非推测式 decode,不影响其他请求

**Plans**: 1 plan (18.3)

Plans:

- [x] 18.3: DraftResolver + per-request routing + fallback — 7 resolver tests + 2 collector tests (263 total)

#### Phase 18.4: Integration Tests + Benchmarks

**Goal**: 多 draft 推测式解码路径的端到端验证与多 draft 性能基线
**Depends on**: Phase 18.3
**Requirements**: (无新增 — 验证 Phase 18.1-18.3 实现)
**Success Criteria** (what must be TRUE):

  1. 集成测试覆盖完整生命周期: register → lazy load → route → unload
  2. 显存预算执行测试覆盖 accept / refuse 边界(刚好超预算、远超预算、并发 N draft)
  3. 多 draft batch routing 测试验证异构请求流(target + draft_A + draft_B 共存)
  4. Fallback 路径对加载期与运行时失败各有显式测试
  5. 基准对比 single-target-only 与 target+draft 吞吐,VRAM 预算在指标中可见

**Plans**: 1 plan (18.4)

Plans:

- [x] 18.4: Integration tests (14) + criterion bench (3 configs) — validates full v18.0 pipeline end-to-end

#### Phase 19: Wire v18.0 into Engine step loop + HTTP exporter

**Goal**: 修复 v18.0 审计中识别的 3 个 gap — DraftResolver 接入 Engine step 循环 / FALL-02 sticky 语义生效 / 5 个 v18.0 counter 通过 /metrics 暴露,并补全 ServerDraftLoader 用于生产加载
**Depends on**: Phase 18.4
**Requirements**: RTE-02 (resolver wired), RTE-03 (mixed routing live), FALL-02 (sticky degraded_draft live)
**Success Criteria** (what must be TRUE):

  1. `Engine::step_speculative_inner` 通过 `DraftResolver::resolve` 按请求分发 draft,而不是单 draft 全局路径
  2. 同 batch 内多个请求携带不同 `draft_model_id` 时,各 seq 独立 resolve,无状态泄漏
  3. FALL-02 运行时错误 → `Sequence.degraded_draft = true`,后续 step 跳过该 seq 的 draft 尝试(sticky)
  4. 5 个 v18.0 counter(draft resolutions / load failures / runtime errors)通过 /metrics 端点暴露
  5. `ServerDraftLoader` 包装 `ModelLoader`,server 启动时按声明加载 draft model,失败时按 FALL-01 静默回退

**Plans**: 1 plan (19)

Plans:

- [x] 19: DraftResolver wiring + FALL-02 sticky + HTTP exporter + ServerDraftLoader — 9 integration tests + 1 phase + 2 #[ignore]d awaiting Engine::step() hang fix

---

### ✅ v19.0 Codebase Health Audit (Phases 20-24) — SHIPPED 2026-06-27

**Milestone Goal:** 对 vllm-lite 整个 codebase 做多维度深度审计(架构 / 命名 / 注释文档 / API + 错误处理),产出 P0/P1/P2 优先级清单与具体修复建议。**本 milestone 不执行任何代码修改**,清单用于指导后续 milestone(v20.0+)的重构工作。

**Audit Execution Model:**

Each audit phase produces:

1. A detailed report at `.planning/audit/{dimension}/REPORT.md`
2. A summary table at `.planning/audit/{dimension}/SUMMARY.md` (P0/P1/P2 prioritized)
3. Raw inventory data (file lists, naming tables, doc-coverage stats) where applicable

Synthesis phase (Phase 24) reads all four dimension reports and produces:

- `.planning/audit/SYNTHESIS.md` — cross-cutting findings
- `.planning/audit/BACKLOG.md` — P0/P1/P2 remediation backlog with impact / cost / suggested-phase columns
- `.planning/audit/MIGRATION-ROADMAP.md` — proposed v20.0+ phase breakdown (advisory only)

#### Phase 20: Architecture Audit

**Goal**: 验证 7-crate 架构的依赖方向、模块边界、循环依赖、分层一致性与测试架构健康度,产出 `.planning/audit/architecture/REPORT.md` 与 `SUMMARY.md`
**Depends on**: Phase 19 (v18.0 shipped; v19.0 starting point)
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04, ARCH-05
**Success Criteria** (what must be TRUE):

  1. `.planning/audit/architecture/REPORT.md` 包含 crate 依赖图,明确记录 `traits ← core ← {model, server, dist}` 方向,任何反向依赖被标记为 P0/P1/P2;并伴随 `.planning/audit/architecture/SUMMARY.md` 用 P0/P1/P2 表格汇总所有架构问题
  2. REPORT.md 中每个 crate 的所有模块都有 single-responsibility 一句话陈述;任何 God module(行数 / 公开项超过阈值)被显式标记
  3. REPORT.md 包含基于 `cargo metadata` 的循环依赖扫描结果;任何循环依赖被以严重度分级列出
  4. REPORT.md 包含 layering consistency 矩阵(行 = crate,列 = 允许的入向依赖),任何越层 import 被标记
  5. REPORT.md 包含测试架构审计:unit / integration / bench 的目录边界、`vllm-testing` crate 的复用情况、以及共享测试 fixture 的卫生度

**Plans**: 1 plan (20)

Plans:

- [x] 20-01: Architecture audit — 1 subagent dispatch producing REPORT.md + SUMMARY.md (no code changes)

#### Phase 21: Naming Audit

**Goal**: 识别文件 / 类型 / 方法 / 变量 / 模块五个维度的命名不一致与语义不清问题,产出 `.planning/audit/naming/REPORT.md` 与 `SUMMARY.md`
**Depends on**: Phase 20 (并行可选,但报告顺序固定为 architecture → naming → docs → api → synthesis)
**Requirements**: NAME-01, NAME-02, NAME-03, NAME-04, NAME-05
**Success Criteria** (what must be TRUE):

  1. `.planning/audit/naming/REPORT.md` 包含文件命名审计表,显式列出所有含 stage-info 命名的文件(如 `17_*.rs`、`18_*.rs`),以及与目录 / 模块名不匹配的文件;并伴随 `.planning/audit/naming/SUMMARY.md` 用 P0/P1/P2 表格汇总所有命名问题
  2. REPORT.md 包含 type 命名审计表,覆盖所有 `pub struct` / `pub enum` / `pub trait`,标注 PascalCase 合规与冗余后缀(如 `Info`、`Data`、`Manager` 当语义重复时)
  3. REPORT.md 包含 function / method 命名审计,覆盖所有 `pub fn`,标注 snake_case 合规、动词一致性(get / set / with / is / has 前缀使用规范)
  4. REPORT.md 包含变量命名审计(对 top-N 大文件全部变量做单字母 / 一致性扫描),列出所有单字母变量(非循环索引)与同名异义 / 异名同义的对
  5. REPORT.md 包含 module 命名审计:模块名与文件名一致性表 + 嵌套深度一致性表

**Plans**: 1 plan (21)

Plans:

- [x] 21-01: Naming audit — 1 subagent dispatch producing REPORT.md + SUMMARY.md (no code changes)

#### Phase 22: Comments + Documentation Audit

**Goal**: 度量 doc-comment 覆盖率、模块文档完整度、过期注释与外部文档漂移,产出 `.planning/audit/docs/REPORT.md` 与 `SUMMARY.md`
**Depends on**: Phase 21 (reports sequential; data independent)
**Requirements**: DOCS-01, DOCS-02, DOCS-03, DOCS-04, DOCS-05
**Success Criteria** (what must be TRUE):

  1. `.planning/audit/docs/REPORT.md` 包含 doc-comment 覆盖率表(每个 crate 的 `pub` 项总数 / 有 `///` 的数量 / 覆盖率 %);未达 80% 目标的 crate 显式列出未覆盖项;并伴随 `.planning/audit/docs/SUMMARY.md` 用 P0/P1/P2 表格汇总所有文档问题
  2. REPORT.md 包含 module-level 文档审计表:每个 `.rs` 文件顶部是否有 `//!` 或同等上下文注释;缺失模块被列出
  3. REPORT.md 包含 stale comment / TODO 扫描:所有 `// TODO` / `// FIXME` / `// XXX` 及指向已删除代码或过期 API 的 docstring,带 file:line 与建议处理
  4. REPORT.md 包含外部文档审计:根 README、AGENTS.md、.planning/PROJECT.md / REQUIREMENTS.md / STATE.md / ROADMAP.md 与当前 codebase 的一致性核对,任何事实性偏差被列出
  5. REPORT.md 包含 ADR 现状盘点:`docs/adr/` 已有 ADR 列表 + 已识别但未文档化的架构决策(基于代码注释 / commit 历史推断);缺失 ADR 被列出

**Plans**: 1 plan (22)

Plans:

- [x] 22-01: Comments + documentation audit — 1 subagent dispatch producing REPORT.md + SUMMARY.md (no code changes)

#### Phase 23: API + Error Handling Audit

**Goal**: 审计公开 API 一致性、错误类型完整度、错误人体工学、trait 设计与弃用卫生,产出 `.planning/audit/api/REPORT.md` 与 `SUMMARY.md`
**Depends on**: Phase 22 (reports sequential; data independent)
**Requirements**: API-01, API-02, API-03, API-04, API-05
**Success Criteria** (what must be TRUE):

  1. `.planning/audit/api/REPORT.md` 包含公开 API 清单(每个 crate 的 `pub` 项),标注函数签名一致性(参数顺序、可变性、生命周期)与 builder 模式使用情况;并伴随 `.planning/audit/api/SUMMARY.md` 用 P0/P1/P2 表格汇总所有 API 问题
  2. REPORT.md 包含 error 类型审计:每个 crate 的 error enum / struct,thiserror 使用率、变体覆盖度、错误消息质量(含相关 ID / file / line)
  3. REPORT.md 包含 error ergonomics 审计:`Result<T>` 传播路径中缺失的 `From` impl、缺失的 `.context()` / `.with_context()`、过度 `unwrap()` / `expect()` 使用
  4. REPORT.md 包含 trait 设计审计:每个公开 trait 的 object safety、async/sync 一致性、default method 使用、`dyn Trait` 兼容性
  5. REPORT.md 包含 deprecation 卫生审计:所有 `#[deprecated]` 项 + 缺失迁移路径的项;内部已删除但仍被外部文档提及的项

**Plans**: 1 plan (23)

Plans:

- [x] 23-01: API + error handling audit — 1 subagent dispatch producing REPORT.md + SUMMARY.md (no code changes)

#### Phase 24: Synthesis + Remediation Backlog

**Goal**: 消费 Phase 20-23 的四个 dimension 报告,产出跨维度综合、P0/P1/P2 优先级清单与 v20.0+ 迁移路线图,作为审计 milestone 的最终交付
**Depends on**: Phase 20, Phase 21, Phase 22, Phase 23 (MUST wait for all four dimension REPORT.md files to exist)
**Requirements**: SYNTH-01, SYNTH-02, SYNTH-03
**Success Criteria** (what must be TRUE):

  1. 开始前必须验证四个 dimension 报告全部存在:`.planning/audit/{architecture,naming,docs,api}/REPORT.md`;任一缺失则 SYNTH 不可启动(在 SYNTHESIS.md 顶部记录输入清单 + 完整性校验)
  2. `.planning/audit/SYNTHESIS.md` 包含跨维度综合:将 ARCH/NAME/DOCS/API 发现按 root cause 重新归类(如 "命名不一致" 可能是 DOCS drift 也可能是 ARCH 模块边界不清的表象),识别重复问题与协同修复机会
  3. `.planning/audit/BACKLOG.md` 包含 P0/P1/P2 优先级 backlog:每行含 ID / 描述 / 来源 dimension / 影响范围 / 修复成本估算(小时) / 建议阶段(v20.0+ 哪个 phase);P0 项必须先列出
  4. `.planning/audit/MIGRATION-ROADMAP.md` 包含 v20.0+ 迁移建议:将 backlog 项分组到提议的未来 phase(如 "Phase 25: 文件重命名 + 文档补全" / "Phase 26: 错误类型统一" / "Phase 27: 模块边界重构"),明确依赖关系
  5. milestone 结束时 `.planning/audit/` 目录树结构完整(每个 dimension 含 REPORT.md + SUMMARY.md,根目录含 SYNTHESIS.md / BACKLOG.md / MIGRATION-ROADMAP.md),且 `git diff --stat` 显示仅 `.planning/` 与 `.planning/audit/` 下有变更 — 验证本 milestone 是纯分析、未触发任何代码改动

**Plans**: 1 plan (24)

Plans:

- [x] 24-01: Cross-dimensional synthesis + backlog + v20.0+ migration roadmap — consumes Phases 20-23 outputs, produces SYNTHESIS.md / BACKLOG.md / MIGRATION-ROADMAP.md

---

### ✅ v20.0 Codebase Remediation (Phases 25-30) — SHIPPED 2026-06-27

**Milestone Goal:** 执行 v19.0 审计产出的修复 backlog,按 MIGRATION-ROADMAP.md 提议顺序执行 6 个子 phase(v20.1→v20.6),系统化恢复 codebase 健康度。**这是 v19.0 审计清单的修复 milestone**,所有改动基于 `.planning/audit/BACKLOG.md` 的 100 个 finding。

**Remediation Execution Model:**

Each remediation phase produces:

1. Code changes addressing its assigned requirements (P0 fixes, module tree, errors, docs, ADRs, naming)
2. New unit / integration tests verifying the fix (≥1 test per requirement)
3. Updated `AGENTS.md` / `Cargo.toml` / `README.md` reflecting any convention or scope changes
4. Phase summary capturing before/after metrics (test count, doc coverage %, clippy warnings, deprecated markers)

**Constraints (apply to ALL v20.0 phases):**

- All 287+ existing tests must remain green throughout (FINAL-01 invariant)
- Public API removals require `#[deprecated(since, note)]` markers + migration path (DEP-01/02)
- `vllm-dist` is feature-gated (not removed) — multi-node work is future (MT-07 + EXT-07 outcome)
- `cargo clippy --workspace --all-targets -- -D warnings` must remain clean after each phase
- No new features, no performance optimization, no architectural redesign — strictly backlog execution

**Phase dependency graph:**

```text
25 (P0 critical fixes)
   │
   ├─→ 27 (error handling standardization)
   │      └─ depends on 25 (ModelError must be enum first; From impls for new variants)
   │
   └─→ 26 (module tree + vllm-dist feature-gate)
          └─ depends on 25 (vllm-dist feature-gate decision from P0-01)

28 (doc coverage) ──→ 29 (external docs + ADRs)
                          └─ depends on 28 (/// docs must exist before ADRs reference them)

30 (naming + polish) — independent of 25-29; runs last by convention
```

**Parallel execution possible:** 25 → (26 ‖ 28) → (27 + 29) → 30.

#### Phase 25: P0 Critical Fixes (v20.1)

**Goal**: 消除 5 个 P0 架构 / API / 错误违规 — `vllm-model → vllm-dist` 边、`vllm-core → vllm-model` 边、`ModelError` struct、`CudaGraphError` 手写 impl、8 个 non-object-safe trait
**Depends on**: v19.0 baseline (Phase 24 shipped)
**Requirements**: P0-01, P0-02, P0-03, P0-04, P0-05
**Success Criteria** (what must be TRUE):

  1. `cargo tree -p vllm-model --no-default-features` 不再显示 `vllm-dist` 依赖边;`--features multi-node` 时边重新出现 — 验证 P0-01
  2. `cargo build -p vllm-core --no-default-features` 不依赖 `vllm-model`;`--features cuda-graph` 时依赖建立 — 验证 P0-02
  3. `ModelError` 改为 `pub enum`(可通过 `cargo doc --document-private-items | grep "enum ModelError"` 或 rustdoc JSON 验证);所有现有 `match ModelError` 站点迁移到 enum 变体;vllm-testing 新增 compile-fail 测试验证变体穷尽性 — 验证 P0-03
  4. 8 个之前 non-object-safe 的 trait(`Architecture` × 12 用法、`FlashAttention` × 2、`DraftLoader`、`PipelineStage`、`AllReduce`、`QkRotaryEmb`、`FormatLoader`、`Quantization` 之一)均能 `dyn Trait` 编译;新增 `crates/testing/tests/dyn_safety.rs` 含 ≥8 compile-only 测试 — 验证 P0-04
  5. `CudaGraphError` 改为 `#[derive(thiserror::Error)]`;手写 `Display`/`Error` impl 删除(~14 LOC);通过 `From<OldStrError>` 保留 semver 兼容 — 验证 P0-05
  6. 所有 287+ pre-Phase-25 测试继续通过;`cargo clippy --workspace --all-targets -- -D warnings` 清洁

**Rollback criteria** (Phase 25 是高风险架构变更,必须可回滚):

- **Pre-Phase-25 baseline** — 开始前捕获最近 passing 的 commit SHA + `cargo tree -p vllm-model` 输出快照 + `cargo test --workspace` 全绿基线
- **Failure triggers** — 任一即触发回滚:(a) 集成测试失败, (b) `cargo build --workspace` 默认 features 失败, (c) >2 个 P0 需求测试 regress, (d) `cargo clippy --workspace -- -D warnings` 报错
- **Rollback action** — `git revert` 所有 Phase 25 提交(单 revert 链);恢复 pre-Phase-25 Cargo.toml 依赖图;通过 `pub type ModelError = OldModelError` 类型别名恢复 `ModelError` struct;所有新公开类型加 `#[deprecated(since = "0.X.0", note = "...")]`
- **Backward-compat preservation** — Phase 25 中移除的任何 public API 项 MUST 保留 ≥1 个 minor 版本的 `#[deprecated]` 标记或类型别名;新公开类型在首个 stable release 必须标注 deprecated-with-migration-note

**Plans**: TBD

Plans:

- [ ] 25-01: Eliminate `vllm-model → vllm-dist` edge (feature-gate) — ARCH-F-11
- [ ] 25-02: Feature-gate `vllm-core → vllm-model` (cuda-graph) — ARCH-F-12
- [ ] 25-03: Convert `ModelError` struct → enum + migrate match sites — API-F-01
- [ ] 25-04: Convert `CudaGraphError` to thiserror + From shim — API-F-03
- [ ] 25-05: Make 8 traits object-safe (or split + associated types) + add `dyn_safety.rs` — API-F-02

#### Phase 26: Module Tree Restoration + vllm-dist Feature-Gate (v20.2)

**Goal**: 把 2 个 orphan 模块(`kv_cache_fp8.rs` 289 LOC、`debug.rs` 175 LOC)挂回模块树;重命名 stage-info 文件 `engine_v18_wiring.rs`;迁移 3 个 src/ 内的测试文件到 tests/;feature-gate `vllm-dist` (1,600 LOC) 防止默认构建
**Depends on**: Phase 25 (P0-01 已解决 `vllm-model → vllm-dist` 边,MT-07 的 feature-gate 决策顺承)
**Requirements**: MT-01, MT-02, MT-03, MT-04, MT-05, MT-06, MT-07
**Success Criteria** (what must be TRUE):

  1. `kv_cache_fp8.rs`(289 LOC)在 `crates/model/src/components/mod.rs` 中以 `pub mod kv_cache_fp8;` 声明;`cargo build -p vllm-model` 编译包含之;`cargo doc -p vllm-model` 列出 kv_cache_fp8 符号 — 验证 MT-01
  2. `debug.rs`(175 LOC)在 `crates/server/src/lib.rs` 中以 `pub mod debug;` 声明;`cargo doc -p vllm-server` 列出 debug 类型 — 验证 MT-02
  3. `crates/core/tests/engine_v18_wiring.rs` 重命名为 `engine_wiring.rs`;所有 `mod` 声明与 import 站点更新;`git log --follow` 显示 rename history — 验证 MT-03
  4. 3 个测试文件(`qwen3/model_tests.rs`、`qwen3_5/model_tests.rs`、`qwen3_5/speculative_tests.rs`)从 `src/` 迁至 `crates/model/tests/`;新 test 注册文件按需添加;原有测试计数保留 — 验证 MT-04/05/06
  5. `vllm-dist` (~1,600 LOC) feature-gated behind `--features multi-node`;`cargo build --workspace`(默认 features)不编译 vllm-dist;`cargo build --workspace --features multi-node` 编译之 — 验证 MT-07
  6. `cargo test --workspace` 通过,测试数 ≥287(v19.0 基线);orphan 模块挂回可能暴露并修复先前 dead-code 状态隐藏的编译错误(这是 feature,不是 bug)

**Plans**: TBD

Plans:

- [ ] 26-01: Wire `kv_cache_fp8.rs` + `debug.rs` into module trees — MT-01, MT-02
- [ ] 26-02: Rename `engine_v18_wiring.rs` → `engine_wiring.rs` + update refs — MT-03
- [ ] 26-03: Migrate 3 src/ test files to tests/ — MT-04, MT-05, MT-06
- [ ] 26-04: Feature-gate `vllm-dist` behind `--features multi-node` — MT-07

#### Phase 27: Error Handling Standardization (v20.3)

**Goal**: 制定并实施项目级 error conventions — 13 个 error enum 统一用 thiserror、消除 10 处 `Result<_,String>` 反模式、25+ 处 mutex-poison `.expect()` 替换、4 个 `EngineError` 新变体、anyhow 边界、跨 crate From impl 完整
**Depends on**: Phase 25 (`ModelError` 必须先 enum 化,否则 new variants 的 From impls 无法建立)
**Requirements**: ERR-01, ERR-02, ERR-03, ERR-04, ERR-05, ERR-06, ERR-07
**Success Criteria** (what must be TRUE):

  1. 生产代码(测试 fixtures 除外) `Result<_, String>` 出现次数 = 0 — 通过 `rg "Result<.*String>" crates/*/src --type rust | wc -l` 验证返回 0 — 验证 ERR-01
  2. workspace 内 13 个 error enum 全部 `#[derive(thiserror::Error)]`;手写 `Display`/`Error` impl 仅在 trivial wrapper 中保留,数量 ≤5 — 验证 ERR-02
  3. 25+ 处 mutex-poison `.expect()` 替换为 `.lock().context("acquiring X mutex")?` 或等价错误传播;生产代码中残留 `.expect()` ≤5 且每处均有 `///` 文档说明为何保留 — 验证 ERR-03
  4. `EngineError` enum 新增 4 个变体:`Timeout { op: String, ms: u64 }`、`Cancelled { request_id: u64 }`、`ResourceExhausted { resource: String }`、`BackendUnavailable { backend: String }`;每个变体有 enum-variant 编译测试覆盖 — 验证 ERR-05
  5. 顶层 server binary(`vllm-server`) `main.rs` 与 CLI entry points 使用 `anyhow::Result`;结构化上下文通过 `.context()` / `.with_context()`;`?` 传播在 binary 边界终止 — 验证 ERR-06
  6. 跨 crate error 路径均有显式 `From` impl;`cargo build --workspace` 不再需要 non-error 模块中的 `Into::<EngineError>` workaround — 验证 ERR-04
  7. 所有 287+ 测试通过;新增 ≥10 个变体覆盖测试 + ≥5 个 context propagation 测试 — 验证 ERR-07

**Plans**: TBD

Plans:

- [ ] 27-01: Convert 13 error enums to `thiserror::Error` derive — ERR-02
- [ ] 27-02: Eliminate `Result<_, String>` (10 sites) — ERR-01
- [ ] 27-03: Replace 25+ mutex `.expect()` with `.context()?` — ERR-03
- [ ] 27-04: Add 4 `EngineError` variants (`Timeout`/`Cancelled`/`ResourceExhausted`/`BackendUnavailable`) — ERR-05
- [ ] 27-05: Add cross-crate `From` impls — ERR-04
- [ ] 27-06: Adopt `anyhow` for server boundary error reporting — ERR-06
- [ ] 27-07: Add `.context()` to error propagation paths — ERR-07

#### Phase 28: Documentation Coverage Push (v20.4)

**Goal**: 把 workspace doc 覆盖率从 7.6% 提升到 ≥60%(per-crate 目标 ≥80%);补 776 个 `pub` item 的 `///`;补 121 个文件的 `//!` 模块级文档;修复 README broken example + crate count + architecture 列表
**Depends on**: v19.0 baseline (与 Phase 25-27 独立,可并行)
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04, DOC-05, DOC-06, DOC-07
**Success Criteria** (what must be TRUE):

  1. workspace doc 覆盖率 ≥60%(`pub` item 数 / 有 `///` 的 item 数);提交新覆盖度测量脚本 `scripts/doc_coverage.sh` 输出 per-crate 表 — 验证 DOC-01
  2. 776 个未文档化 `pub` item 新增 `///` doc comment;per-crate 目标:`traits` 0%→80%、`dist` 2.7%→80%、`server` 4.9%→80%、`model` 8.5%→80%、`core` 9.0%→80%、`testing` 12.9%→80% — 验证 DOC-02
  3. 121 个 source file 新增 `//!` 模块级文档(232 个 source file 的 52%);剩余文件由首个 `pub` item 的 `///` 作为模块介绍 — 验证 DOC-03
  4. `README.md` 中 `SchedulerEngine::new(config, 1024)` 错误示例修正为 v19 engine.rs 实际 3-arg 签名;`mdbook` 与 GitHub preview 渲染正常 — 验证 DOC-04
  5. `README.md` "Supported Architectures" 表列出全部 10 个 registered architectures(Llama / Mistral / Qwen2/3 / Qwen3.5 / Gemma4 / Mixtral / Gemma3 / Phi-4 / Llama 4 / Mistral Small),通过 `rg "register.*Architecture" crates/model/src/ | wc -l` 验证 ≥10 — 验证 DOC-05
  6. `README.md` crate count 从 "7 crates" 修正为 "6 crates";与 `Cargo.toml [workspace] members` 一致 — 验证 DOC-06
  7. `AGENTS.md` "Architecture" 章节与当前 crate 结构 reconcile;file:line 引用经验证存在 — 验证 DOC-07
  8. `cargo doc --workspace --no-deps` 构建无 warning;覆盖度报告附加到 phase summary

**Plans**: TBD

Plans:

- [ ] 28-01: Add coverage measurement script + baseline report — DOC-01 (mechanism)
- [ ] 28-02: Backfill `///` on `traits` crate (0% → 80%, 14 items) — DOC-02
- [ ] 28-03: Backfill `///` on `dist` crate (2.7% → 80%, 36 items) — DOC-02
- [ ] 28-04: Backfill `///` on `server` crate (4.9% → 80%, 65 items) — DOC-02
- [ ] 28-05: Backfill `///` on `model` crate (8.5% → 80%, 170 items) — DOC-02
- [ ] 28-06: Backfill `///` on `core` crate (9.0% → 80%, 99 items) — DOC-02
- [ ] 28-07: Backfill `///` on `testing` crate (12.9% → 80%, 12 items) — DOC-02
- [ ] 28-08: Add `//!` to 121 source files — DOC-03
- [ ] 28-09: Fix README code example + architecture list + crate count — DOC-04, DOC-05, DOC-06
- [ ] 28-10: Reconcile AGENTS.md Architecture section — DOC-07

#### Phase 29: External Documentation + ADRs (v20.5)

**Goal**: 调和 README / AGENTS.md / Cargo.toml 的事实性陈述;新建 12 个 ADR 记录 v15.0-v20.0 的 tribal knowledge 决策;验证 `.planning/PROJECT.md` Core Value 是否漂移
**Depends on**: Phase 28(`///` docs 必须先存在,否则 ADRs 中引用的 type 找不到文档)
**Requirements**: EXT-01, EXT-02, EXT-03, EXT-04, EXT-05, EXT-06, EXT-07, EXT-08, EXT-09, EXT-10, EXT-11, EXT-12
**Success Criteria** (what must be TRUE):

  1. README.md / AGENTS.md / Cargo.toml claims reconciled — 无 crate count / architecture count / feature flag table 矛盾;通过 cross-grep 测试验证 — 验证 EXT-01
  2. 12 个 ADR 新建于 `docs/adr/`,遵循 ADR-001/002 的 Context / Decision / Rationale / Consequences / Alternatives Considered 模板:
     - ADR-003: Self-speculation 1/8 layer ratio (v16.0)
     - ADR-004: FP8 E4M3 format for KV cache (v15.0)
     - ADR-005: KV cache split across 3 locations (v1.0)
     - ADR-006: Speculative decoding architecture overview (v16.0)
     - ADR-007: Per-request draft routing RTE-01..03 (v18.0)
     - ADR-008: Why `vllm-dist` is feature-gated (v20.1 outcome)
     - ADR-009: FP8 quantizer orphan module decision (v20.2 outcome)
     - ADR-010: CUDA graph feature gating strategy (v10.1)
     - ADR-011: Cross-crate error type boundaries (v20.3 outcome)
     - ADR-012..014: 2+ ADRs from v15.0-v18.0 tribal knowledge(TLS choice / K8s Lease vs Raft / etc.) — 验证 EXT-02..11
  3. `.planning/PROJECT.md` "What This Is" + "Core Value" 与当前 codebase 比对;任何漂移记录并修正 — 验证 EXT-12
  4. `cargo doc --workspace --no-deps` 构建清洁;ADRs 从 `.planning/PROJECT.md` "Key Decisions" 表交叉链接

**Plans**: TBD

Plans:

- [ ] 29-01: Reconcile README.md / AGENTS.md / Cargo.toml claims — EXT-01
- [ ] 29-02: Create 9 ADR-003..011 (P0-P1 architecture decisions) — EXT-02..10
- [ ] 29-03: Create 2+ tribal-knowledge ADRs from v15.0-v18.0 — EXT-11
- [ ] 29-04: Verify + update `.planning/PROJECT.md` "What This Is" / "Core Value" — EXT-12

#### Phase 30: Naming + Final Polish + Verification (v20.6)

**Goal**: 应用 7 P1 + 19 P2 命名修复;为 v20.0 移除的 public API 加 `#[deprecated]` 标记 + 迁移路径;清理 stale comments 与 dead TODOs;运行 FINAL-01..04 全套验证(`cargo test` + `cargo clippy` + `cargo fmt` + 规划文档更新)
**Depends on**: v19.0 baseline (与 Phase 25-29 独立;按惯例最后跑以汇聚所有先期变更)
**Requirements**: NAM-01, NAM-02, DEP-01, DEP-02, CMT-01, CMT-02, FINAL-01, FINAL-02, FINAL-03, FINAL-04
**Success Criteria** (what must be TRUE):

  1. 7 个 P1 命名修复应用(per NAME-F-05..07):`data` 变量重命名(生产代码 31× 处)、`EmbeddingData` → `Embedding` 或文档化 `*Data` 后缀约定、AGENTS.md 正式化 verb policy(`get_`/`load_`/`read_`/`create_`/`build_`) — 验证 NAM-01
  2. 19 个 P2 命名一致性修复应用 per NAME-F-08..20(文件重命名 `flash_v3.rs` → `flash_attention_v3.rs`、AGENTS.md 补 `*Manager` 后缀文档、builder/build/get/load/create/read verb policy、async/sync split rationale、tensor-math 单字母变量豁免、test-file location 约定) — 验证 NAM-02
  3. 所有 v20.0 移除的 public API item 标记 `#[deprecated(since = "0.X.0", note = "use NewName instead")]`,带 `since` 与 `note` 字段(DEP-01);v20.0 新增 `#[deprecated]` 标记数 ≥5(终值记录到 phase summary)
  4. 迁移路径文档化:`MIGRATING.md`(新文件)或每个 deprecated item 的 inline `///` + `#[doc = "..."]` 迁移示例(DEP-02)
  5. Stale comments 解决:per DOCS-F-12 `quantize/gguf.rs:7` placeholder;per DOCS-F-13 "Phase 18.3 will drive this" 注释在 `draft_registry.rs:434` 与 `engine.rs:327`(CMT-01)
  6. Dead TODOs/FIXMEs(v19.0 审计发现)解决或标记 stale;终值 TODO/FIXME 计数 ≤3(per 项目优秀卫生基线)(CMT-02)
  7. **FINAL-01**: 所有 287+ 测试 post-remediation 通过 — `cargo test --workspace --all-features` 返回 0 失败;测试数 ≥287(目标 297+,含 Phase 25 新增 ≥8 个 `dyn_safety` 测试 + Phase 27 新增 ≥15 个 error variant / context 测试)
  8. **FINAL-02**: `cargo clippy --workspace --all-targets --all-features -- -D warnings` 清洁(FINAL-02)
  9. **FINAL-03**: `cargo fmt --all --check` 清洁(FINAL-03)
  10. **FINAL-04**: `.planning/PROJECT.md` 与 `.planning/STATE.md` 更新 v20.0 成果(Phase 25-30 结果、决策日志、未解决项移至 v20.7+ backlog)

**Plans**: TBD

Plans:

- [ ] 30-01: Apply 7 P1 naming fixes — NAM-01
- [ ] 30-02: Apply 19 P2 naming consistency fixes — NAM-02
- [ ] 30-03: Add `#[deprecated]` markers + migration paths — DEP-01, DEP-02
- [ ] 30-04: Resolve stale comments (gguf placeholder + "Phase 18.3" comments) — CMT-01
- [ ] 30-05: Resolve or mark-stale dead TODOs/FIXMEs — CMT-02
- [ ] 30-06: FINAL-01 — `cargo test --workspace --all-features` 287+ tests pass
- [ ] 30-07: FINAL-02 — `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [ ] 30-08: FINAL-03 — `cargo fmt --all --check` clean
- [ ] 30-09: FINAL-04 — Update `.planning/PROJECT.md` + `.planning/STATE.md` with v20.0 outcomes

---

## 🚧 v21.0 P2/P3 Backlog Cleanup (Phases 31-35) — IN PROGRESS 2026-06-27

**Milestone Goal:** Close the remaining 44 P2 + 13 P3 backlog from v19.0 audit (v20.0 already shipped 5 P0 + 38 P1). Target: 100% backlog closure preserving all v20.0 invariants (1144+ tests green, clippy/fmt clean, doc coverage ≥60% baseline). All changes must be backward-compatible via `#[deprecated]` markers for any public API removal; `vllm-dist` remains feature-gated.

**Source:** `.planning/audit/BACKLOG.md` (v19.0 100 findings), `.planning/audit/MIGRATION-ROADMAP.md` (deferred items promoted)

**Effort estimate:** ~71h (~2 working weeks, single engineer)

**Dependency graph:** Phase 31 → Phase 32 → Phase 33 → Phase 34 → Phase 35 (linear chain; engine splits in Phase 31 unblock Phase 32's FallbackStrategy relocation; Phase 32 API surface stabilizes before Phase 33 documents conventions; Phase 33 AGENTS.md updates precede Phase 34 PROJECT.md cross-links; Phase 35 depends on all)

### Phase 31: Module Layout Reorganization (v21.1)

**Goal**: Reorganize oversized God modules into focused sub-trees so contributors can navigate and modify the codebase without cross-cutting concerns; error types live at their semantic boundaries (not in convenient-but-wrong locations)

**Depends on**: Phase 30 (v20.6 baseline — 1144 tests green, clippy/fmt clean)

**Requirements**: ML-01, ML-02, ML-03, ML-04, ML-05, ML-06, ML-07, ML-08, ML-09

**Success Criteria** (what must be TRUE):

  1. `draft_registry.rs` (929 LOC) is decomposed into a `registry/` sub-tree (`loader.rs` + `lifecycle.rs` + thin `mod.rs`); each leaf file <300 LOC and focused on a single concern; all 1144+ tests pass without modification — validates ML-01
  2. `engine.rs` + `engine/speculative.rs` are unified into a single `engine/speculative/` sub-tree with consistent organization; no duplicate re-exports; engine's speculative path remains the single source of truth for that behavior — validates ML-02
  3. `qwen3_config.rs` lives at `qwen3/config.rs`; `attention/mod.rs` utilities (180+ LOC) are extracted to `attention/util.rs` leaving `mod.rs` focused on re-exports — validates ML-03, ML-04
  4. `TensorParallelError` lives in `vllm-dist::error` with a re-export from `vllm-traits`; all callers reference the canonical path (no duplicate definitions); `vllm-dist` remains feature-gated — validates ML-06
  5. `vllm-testing` is either split into a `vllm-testkit` + `vllm-harness` lemon pair OR documented why the split is infeasible; server tests consume `vllm-testing` exports (no local `test_fixtures.rs`); unused exports verified or removed — validates ML-05, ML-07, ML-08, ML-09

**Plans**: TBD

Plans:

- [ ] 31-01: Split `draft_registry.rs` into `registry/{loader,lifecycle}.rs` + re-export shim — ML-01
- [ ] 31-02: Collapse `engine.rs` + `engine/speculative.rs` into `engine/speculative/` sub-tree — ML-02
- [ ] 31-03: Move `qwen3_config.rs` → `qwen3/config.rs`; extract `attention/mod.rs` utilities → `attention/util.rs` — ML-03, ML-04
- [ ] 31-04: Decide vllm-testing lemon pair (split or document); execute chosen path — ML-05
- [ ] 31-05: Move `TensorParallelError` to `vllm-dist::error`; re-export from `vllm-traits`; migrate callers — ML-06
- [ ] 31-06: Move `crates/server/src/test_fixtures.rs` into `vllm-testing`; migrate server tests; verify/remove unused exports — ML-07, ML-08, ML-09

### Phase 32: API Consistency (v21.2)

**Goal**: Make API surface uniform — typed errors throughout, ergonomic builders, structured error context, sync/async trait splits where the runtime requires it; document conventions so future API additions don't regress

**Depends on**: Phase 31 (engine splits relocate `FallbackStrategy` to canonical home)

**Requirements**: API-01, API-02, API-03, API-04, API-05, API-06, API-07, API-08, API-09, API-10, API-11

**Success Criteria** (what must be TRUE):

  1. AGENTS.md documents builder-vs-struct-literal convention and common trait re-export patterns at crate roots (e.g., `vllm_core::prelude`); new contributors have unambiguous guidance for adding APIs — validates API-01, API-07
  2. Error chains preserved end-to-end: `DraftRegistryError::LoadFailed` carries `#[source]` for the underlying error; `From<candle_core::Error>` for `EngineError` enables `?` propagation from model layer to engine without manual conversion — validates API-02, API-09
  3. `Box<dyn Error>` is eliminated from `model` lib crate; `predictive_batching.rs` no longer panics on mutex poison (parking_lot or sync helper for all 8 sites) — validates API-03, API-04
  4. 22 new builders exist where only `Default` previously did; each follows `::builder()` + `with_*` + `build()` pattern; object-safe traits (`DraftVerifier`, `SchedulerObserver`) gain `Default` impls; `dyn Trait` compile-only tests cover every public trait — validates API-05, API-06, API-10
  5. `FallbackStrategy` is split into sync + async traits so callers pick based on runtime requirements; engine errors carry `request_id` / `seq_id` as structured fields for log correlation — validates API-08, API-11

**Plans**: TBD

Plans:

- [ ] 32-01: Document builder/struct-literal convention + crate-root re-exports in AGENTS.md — API-01, API-07
- [ ] 32-02: Add `#[source]` to `DraftRegistryError::LoadFailed`; add `From<candle_core::Error>` for `EngineError` — API-02, API-09
- [ ] 32-03: Replace 2 `Box<dyn Error>` in `model` lib with typed errors — API-03
- [ ] 32-04: Replace 8 `Mutex::lock().unwrap()` in `predictive_batching.rs` with parking_lot or sync helper — API-04
- [ ] 32-05: Introduce 22 builders where only `Default` exists; verify pattern across crates — API-05
- [ ] 32-06: Add `Default` impls for `DraftVerifier`, `SchedulerObserver`; add `dyn Trait` compile-only tests — API-06, API-10
- [ ] 32-07: Split `FallbackStrategy` into sync + async traits; migrate callers — API-08
- [ ] 32-08: Carry `request_id`/`seq_id` in error context (structured fields) — API-11

### Phase 33: Naming Consistency (v21.3)

**Goal**: Bring remaining naming drift to the documented standards established in Phase 30 (NAM-02) — single-letter variables confined to tensor-math, suffix conventions enforced, AGENTS.md updated with remaining ambiguities

**Depends on**: Phase 32 (API surface stable so AGENTS.md documentation refers to stable symbols)

**Requirements**: NAM-01, NAM-02, NAM-03, NAM-04, NAM-05, NAM-06, NAM-07, NAM-08

**Success Criteria** (what must be TRUE):

  1. `flash_v3.rs` is renamed `flash_attention_v3.rs` (matches V2 naming pattern); all `use` paths updated; no dangling references — validates NAM-01
  2. `NodeInfo` is either renamed to `NodeSummary`/`NodeMetadata` with a `#[deprecated]` alias OR a documented rationale exists in AGENTS.md for keeping the current name — validates NAM-03
  3. Non-tensor single-letter variables in scheduler and sampling code are renamed to descriptive names (e.g., `i` → `priority_a`, `random_threshold`); tensor-math exemption (`q`/`k`/`v`/`o`/`b`/`c`/`h`/`z`/`d`/`x`/`g`/`r`) is preserved and explicitly documented in AGENTS.md — validates NAM-06, NAM-07
  4. AGENTS.md documents the remaining naming conventions: `*Manager` suffix usage, `create_*` vs `build_*` policy, async/sync split rationale, test-file location convention — validates NAM-02, NAM-04, NAM-05, NAM-08

**Plans**: TBD

Plans:

- [ ] 33-01: Rename `flash_v3.rs` → `flash_attention_v3.rs`; update imports — NAM-01
- [ ] 33-02: Evaluate `NodeInfo` rename (decide + execute or document rationale) — NAM-03
- [ ] 33-03: Rename non-tensor single-letter variables in scheduler/sampling code — NAM-07
- [ ] 33-04: Update AGENTS.md with `*Manager` suffix, `create_*` vs `build_*`, async/sync split, tensor-math exemption, test-file location — NAM-02, NAM-04, NAM-05, NAM-06, NAM-08

### Phase 34: External Doc Fixes (v21.4)

**Goal**: Resolve the remaining external documentation inconsistencies discovered during v19.0 audit — stale phase references, missing ADRs, broken cross-links

**Depends on**: Phase 33 (AGENTS.md must be updated before PROJECT.md cross-links to ADR list)

**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04

**Success Criteria** (what must be TRUE):

  1. `REQUIREMENTS.md:53` no longer references DeepSeek unless `crates/model/src/deepseek/` exists; if removed, the directory must be restored OR the requirement text corrected — validates DOC-01
  2. A new ADR captures the vllm-dist investment vs deprecation decision rationale (why feature-gated vs removed vs retained), referencing the v20.1 decision and multi-node future — validates DOC-02
  3. `qwen3_5/speculative_tests.rs:1` no longer references "Phase 5 Wave 4" — uses current phase terminology (e.g., "Phase 18.4 speculative decoding tests" or a neutral phase-agnostic description) — validates DOC-03
  4. `.planning/PROJECT.md` "Key Decisions" table cross-links to ADRs by ID; navigating from a decision to its rationale works in one click — validates DOC-04

**Plans**: TBD

Plans:

- [ ] 34-01: Reconcile DeepSeek reference in REQUIREMENTS.md (remove or restore directory) — DOC-01
- [ ] 34-02: Create ADR for vllm-dist investment vs deprecation decision — DOC-02
- [ ] 34-03: Reframe "Phase 5 Wave 4" reference in `qwen3_5/speculative_tests.rs:1` — DOC-03
- [ ] 34-04: Cross-link `.planning/PROJECT.md` "Key Decisions" table to ADRs by ID — DOC-04

### Phase 35: P3 Actionable + Final Verification (v21.5)

**Goal**: Resolve remaining actionable P3 findings and verify all v21.0 invariants hold — 100% backlog closure with no test regression, no clippy/fmt regression, and no doc coverage regression

**Depends on**: Phase 31, 32, 33, 34 (all prior phases must be complete before FINAL gates run)

**Requirements**: P3-01, P3-02, P3-03, P3-04, P3-05, P3-06, FINAL-01, FINAL-02, FINAL-03, FINAL-04

**Success Criteria** (what must be TRUE):

  1. All P3 actionable items resolved: dead `crates/traits/tests/mod.rs` removed; `gemma4/attention.rs` non-test `.unwrap()` replaced with graceful error propagation; `CircuitBreakerError::HalfOpenRejected(u32)` variant added; `CudaGraphError::Clone` derive decision verified and documented (keep with rationale or remove) — validates P3-01, P3-02, P3-04, P3-06
  2. `MIGRATING.md` skeleton exists at repo root with v15.0 → v21.0 versioned changelog (backfill of intermediate versions may be deferred); `model` crate production `unwrap()` count is re-verified post-v21.0 changes (≤baseline from v20.6) — validates P3-03, P3-05
  3. **FINAL-01**: `cargo test --workspace --all-features` returns ≥1144 tests passing (no regression from v20.6 baseline)
  4. **FINAL-02**: `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean (no new warnings)
  5. **FINAL-03**: `cargo fmt --all --check` clean (no formatting drift)
  6. **FINAL-04**: `.planning/PROJECT.md` "Current Milestone", "What This Is", and "Key Decisions" sections updated with v21.0 outcomes; `.planning/STATE.md` reflects Phase 31-35 completion; v21.0 100% backlog closure status declared

**Plans**: TBD

Plans:

- [ ] 35-01: Remove dead `crates/traits/tests/mod.rs`; fix `gemma4/attention.rs` non-test unwrap — P3-01, P3-02
- [ ] 35-02: Add `HalfOpenRejected(u32)` variant to `CircuitBreakerError`; verify `CudaGraphError::Clone` derive decision — P3-04, P3-06
- [ ] 35-03: Create `MIGRATING.md` skeleton with v15.0 → v21.0 versioned changelog — P3-03
- [ ] 35-04: Re-verify `model` crate production `unwrap()` count post-v21.0; document baseline — P3-05
- [ ] 35-05: FINAL-01 — `cargo test --workspace --all-features` ≥1144 tests pass
- [ ] 35-06: FINAL-02 — `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [ ] 35-07: FINAL-03 — `cargo fmt --all --check` clean
- [ ] 35-08: FINAL-04 — Update `.planning/PROJECT.md` + `.planning/STATE.md` with v21.0 outcomes (100% backlog closure declared)

---

## Progress

**Execution Order:**
v19.0: 20 → 21 → 22 → 23 → 24 (SHIPPED 2026-06-27)
v20.0: 25 → (26 ‖ 28) → (27 + 29) → 30 (parallel where independent; SHIPPED 2026-06-27)
v21.0: 31 → 32 → 33 → 34 → 35 (linear chain; engine splits unblock API refactor; AGENTS.md updates precede doc cross-links)

| Phase                                              | Milestone | Plans Complete | Status      | Completed    |
| -------------------------------------------------- | --------- | -------------- | ----------- | ------------ |
| 18.1 Draft Registry + External Loading             | v18.0     | 1/1            | Complete    | 2026-06-27   |
| 18.2 Lifecycle + Memory Budget                     | v18.0     | 1/1            | Complete    | 2026-06-27   |
| 18.3 Request Routing + Fallback                    | v18.0     | 1/1            | Complete    | 2026-06-27   |
| 18.4 Integration Tests + Benchmarks                | v18.0     | 1/1            | Complete    | 2026-06-27   |
| 19 Wire v18.0 into Engine step loop                | v18.0     | 1/1            | Complete    | 2026-06-27   |
| 20 Architecture Audit                              | v19.0     | 1/1            | Complete    | 2026-06-27   |
| 21 Naming Audit                                    | v19.0     | 1/1            | Complete    | 2026-06-27   |
| 22 Comments + Documentation Audit                  | v19.0     | 1/1            | Complete    | 2026-06-27   |
| 23 API + Error Handling Audit                      | v19.0     | 1/1            | Complete    | 2026-06-27   |
| 24 Synthesis + Remediation Backlog                 | v19.0     | 1/1            | Complete    | 2026-06-27   |
| 25 P0 Critical Fixes (v20.1)                       | v20.0     | 5/5            | Complete    | 2026-06-27   |
| 26 Module Tree Restoration (v20.2)                 | v20.0     | 4/4            | Complete    | 2026-06-27   |
| 27 Error Handling Standardization (v20.3)          | v20.0     | 7/7            | Complete    | 2026-06-27   |
| 28 Documentation Coverage Push (v20.4)             | v20.0     | 10/10          | Complete    | 2026-06-27   |
| 29 External Docs + ADRs (v20.5)                    | v20.0     | 4/4            | Complete    | 2026-06-27   |
| 30 Naming + Final Polish (v20.6)                   | v20.0     | 9/9            | Complete    | 2026-06-27   |
| 31 Module Layout Reorganization (v21.1)            | v21.0     | 6/6            | Complete    | 2026-06-27   |
| 32 API Consistency (v21.2)                         | v21.0     | 0/8            | Not started | -            |
| 33 Naming Consistency (v21.3)                      | v21.0     | 0/4            | Not started | -            |
| 34 External Doc Fixes (v21.4)                      | v21.0     | 0/4            | Not started | -            |
| 35 P3 Actionable + Final Verification (v21.5)      | v21.0     | 0/8            | Not started | -            |

---

*Roadmap updated: 2026-06-27 — v20.0 phases defined (6 phases, 48/48 requirements mapped, BACKLOG-driven remediation, Phase 25 includes rollback criteria for P0 architectural changes, Phase 30 includes FINAL-01..04 verification gates)*

*Roadmap updated: 2026-06-27 — v21.0 P2/P3 Backlog Cleanup phases added (5 phases: 31-35; 42 requirements: 9 ML + 11 API + 8 NAM + 4 DOC + 6 P3 + 4 FINAL; 100% requirement coverage; ~71h estimated; linear dependency chain 31→32→33→34→35; preserves v20.0 invariants: 1144+ tests green, clippy/fmt clean, doc coverage ≥60% baseline, #[deprecated] for public API changes, vllm-dist feature-gated)*
