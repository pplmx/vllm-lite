# Roadmap: vllm-lite

## Milestones

- ✅ **v16.0 Speculative Decoding** — Phases 16.1-16.4 (shipped 2026-04-28)
- ✅ **v17.0 Production Speculative Decoding** — Phases 17.1-17.4 (shipped 2026-05-13)
- ✅ **v18.0 Multi-Model Speculative Decoding** — Phases 18.1-18.4 + Phase 19 gap closure (shipped 2026-06-27)
- ✅ **v19.0 Codebase Health Audit** — Phases 20-24 (shipped 2026-06-27; analysis-only, no code changes; see `.planning/audit/` for deliverables)

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

### 📋 v19.0 Codebase Health Audit (Phases 20-24) — PLANNED (analysis-only)

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

- [ ] 20-01: Architecture audit — 1 subagent dispatch producing REPORT.md + SUMMARY.md (no code changes)

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

- [ ] 21-01: Naming audit — 1 subagent dispatch producing REPORT.md + SUMMARY.md (no code changes)

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

- [ ] 22-01: Comments + documentation audit — 1 subagent dispatch producing REPORT.md + SUMMARY.md (no code changes)

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

- [ ] 23-01: API + error handling audit — 1 subagent dispatch producing REPORT.md + SUMMARY.md (no code changes)

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

- [ ] 24-01: Cross-dimensional synthesis + backlog + v20.0+ migration roadmap — consumes Phases 20-23 outputs, produces SYNTHESIS.md / BACKLOG.md / MIGRATION-ROADMAP.md

---

## Progress

**Execution Order:** 20 → 21 → 22 → 23 → 24
**Audit Dimension Reports Parallelism:** Phases 20-23 各自独立可并行 dispatch(只读 codebase),但 Phase 24 必须在 20-23 全部完成后执行。

| Phase                                              | Milestone | Plans Complete | Status      | Completed    |
| -------------------------------------------------- | --------- | -------------- | ----------- | ------------ |
| 18.1 Draft Registry + External Loading             | v18.0     | 1/1            | Complete    | 2026-06-27   |
| 18.2 Lifecycle + Memory Budget                     | v18.0     | 1/1            | Complete    | 2026-06-27   |
| 18.3 Request Routing + Fallback                    | v18.0     | 1/1            | Complete    | 2026-06-27   |
| 18.4 Integration Tests + Benchmarks                | v18.0     | 1/1            | Complete    | 2026-06-27   |
| 19 Wire v18.0 into Engine step loop                | v18.0     | 1/1            | Complete    | 2026-06-27   |
| 20 Architecture Audit                              | v19.0     | 0/1            | Not started | -            |
| 21 Naming Audit                                    | v19.0     | 0/1            | Not started | -            |
| 22 Comments + Documentation Audit                  | v19.0     | 0/1            | Not started | -            |
| 23 API + Error Handling Audit                      | v19.0     | 0/1            | Not started | -            |
| 24 Synthesis + Remediation Backlog                 | v19.0     | 0/1            | Not started | -            |

---

*Roadmap updated: 2026-06-27 — v19.0 phases defined (5 phases, 23/23 requirements mapped, analysis-only milestone, Phase 24 explicitly gates on Phases 20-23 outputs)*
