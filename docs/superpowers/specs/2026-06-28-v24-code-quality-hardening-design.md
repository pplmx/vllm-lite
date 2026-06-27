# v24.0 Code Quality Hardening — 设计文档

## 1. Overview

把 vllm-lite 推向**工程化、生产化、idiomatic Rust**。当前 v23.0 已 ship,但代码质量仍处于"功能正确 + 默认 clippy 干净"水平,距离生产级 Rust 项目(deny 级 lint + 显式 invariant + builder API + typed error + 紧致模块边界)还有显著差距。

本次 hardening 不引入新功能、不改模型行为、不动 scheduler 语义;只做**结构性、风格性、健壮性**层面的逐层推进。

### 目标

- 默认 clippy 之外的 deny 级 lint 全 workspace 通过
- 生产路径消除无注释 `unwrap()/expect()`,错误路径有 typed error + `#[source]` 链
- 公开 API 全部 builder 化(>2 optional field)、去 stringly-typed、统一 error 规范
- 模块边界收紧、单文件 ≤ 500 行(剔除 tests)
- 4 个 phase 全部 ship 后,新增功能可"自动"继承这些约束(因为有 CI 兜底)

### 非目标

- 性能优化(留给后续 PERF phase)
- 新功能、新模型架构支持
- 公开 API breaking change(需要破坏时走 deprecation)
- 分布式 / 多节点 / 量化扩展
- 重写 crate 间依赖图

---

## 2. 当前基线测量

### 2.1 Lint 状态

| 指标 | 命令 | 结果 |
|------|------|------|
| 默认 clippy | `cargo clippy --workspace` | 0 警告 |
| Pedantic clippy | `cargo clippy --workspace --all-targets --all-features -- -W clippy::pedantic` | **2659 警告** |
| Lint 配置 | 搜索 `#![deny(...)]` / `[lints.clippy]` | **未配置** |

### 2.2 Pedantic 警告分布(Top 15)

| 计数 | Lint | 性质 |
|------|------|------|
| 490 | `must_use_candidate` | 方法缺少 `#[must_use]` |
| 359 | `doc_markdown` | 文档缺反引号 |
| 283 | `missing_errors_doc` | 返回 `Result` 的公开函数缺 `# Errors` |
| 156 | `uninlined_format_args` | 可内联的 `format!` 参数 |
| 107 | `return_self_not_must_use` | 返回 `Self` 的方法缺 `#[must_use]` |
| 46 | `cast_precision_loss` | `usize as f32` |
| 35 | `missing_panics_doc` | 可能 panic 的函数缺 `# Panics` |
| 30 | `cast_precision_loss` (u64→f64) | 同上 |
| 27 | `format_push_string` | `format!` 追加到 String |
| 25 | `needless_pass_by_value` | 引用即可 |
| 24 | `cast_possible_truncation` (usize→u32) | |
| 17 | `unused_self` | |
| 17 | `redundant_closure_for_method_calls` | |
| 16 | `missing_must_use` | 函数级 |
| 15 | `cast_possible_wrap` / `cast_possible_truncation` (usize→i32) | |

### 2.3 unwrap/expect 分布(实际生产代码,2026-06-28 audit)

**重要修正**: 最初 spec 文档声称 "非测试 unwrap/expect 总数 787" 是基于 `rg --type rust -g '!**/tests/**'` 的粗略计数,实际包含了 743 处 inline `#[cfg(test)] mod tests` 块。真实生产代码 unwrap/expect 计数为 **60**(其中 3 处是 doc-comment 误报,实际为 **57**)。

完整 audit 见 `/tmp/phase_b_audit/SUMMARY.md`(2026-06-28)。

| 类别 | 计数 | % | 说明 |
|------|------|---|------|
| KEEP(test/benches) | 764 | 92.7% | 不需变更 |
| INVARIANT(生产) | 51 | 6.2% | 需加 `// invariant:` 注释 |
| CONVERT(生产) | 6 | 0.7% | 需转 typed error |
| N/A(doc 注释误报) | 3 | 0.4% | |
| **总计** | **824** | 100% | |

**CONVERT 集群(6 处真实风险)**:

| # | 文件:行 | 代码 | 修复 | 风险 |
|---|---------|------|------|------|
| 1 | `crates/model/src/kernels/cuda_graph/executor.rs:222` | `self.graphs.get(&graph_key).unwrap()` | `ok_or(GraphExecutionError::GraphNotFound(graph_key))?` | **真 bug**(race condition) |
| 2 | `crates/core/src/engine.rs:565` | `beams.into_iter().max_by(...).unwrap()` | `ok_or(EngineError::EmptyBeamList)?` | 低 |
| 3 | `crates/server/src/main.rs:242` | `tokenizer_path.to_str().unwrap()` | `?` 传播到 from_file | 低 |
| 4 | `crates/server/src/main.rs:324` | `TcpListener::bind(...).await.unwrap()` | log + exit | 低 |
| 5 | `crates/server/src/main.rs:330` | `axum::serve(...).await.unwrap()` | log + exit | 低 |
| 6 | `crates/server/src/openai/batch/handler.rs:42` | `get_job(...).await.unwrap()` | map 到 404 | 中 |

**INVARIANT 集群(51 处需注释,按模式分组)**:

| 模式 | 处数 | 主要文件 |
|------|------|---------|
| RwLock/Mutex `.expect("...poisoned")` | 21 | registry/lifecycle.rs(13), registry/loader.rs(4), arch/registry.rs, spec_dispatch/drafts.rs, main.rs |
| `SystemTime::now().duration_since(UNIX_EPOCH).unwrap()` | 8 | dist/grpc.rs, distributed_kv/cache.rs, distributed_kv/protocol.rs, openai/batch/*, security/correlation.rs |
| `Tensor::zeros((1,1),...).expect(...)` | 4 | gemma4/attention.rs(已有注释,可作参考) |
| Tensor 分配(非 1×1) | 3 | kv_cache.rs, qwen3_5/block/linear.rs, components/ssm.rs |
| Signal handler 安装 | 2 | main.rs |
| `.expect("duplicate draft id")`(程序错误) | 2 | engine.rs |
| `.expect("generate_per_seq_drafts called without draft_resolver")` | 1 | spec_dispatch/drafts.rs |
| `.expect("vram_budget_bytes validated")` | 1 | main.rs |
| 序列化已知良好结构 | 2 | openai/chat.rs |
| HashMap insert 后立即访问 | 1 | causal_lm/hybrid_lm.rs |
| Cargo-provided env vars | 2 | dist/build.rs |
| 其他单点 invariant | 4 | memory_budget.rs (u64::MAX), auth.rs, api.rs, circuit_breaker/strategy.rs |

---

## 3. 阶段划分

| Phase | 主题 | 范围 | 原子 commit 数(估) | 依赖 |
|-------|------|------|------|------|
| **A. Lint Baseline** | 建立 workspace lint 表,修 deny 级违规 | 全 workspace | 3-5 | — |
| **B. Unwrap Cleanup** | 消除生产路径 unwrap/expect | 全 workspace,hot path 优先 | 8-15 | A(便于 CI 把关) |
| **C. API Ergonomics** | builder 化、typed error、crate-root re-export | 公开 API 表面 | 6-12 | B(error 类型稳定) |
| **D. Module Boundaries** | 拆大文件、收紧可见性、模块重组 | core/model/server 为主 | 4-8 | A-C(代码风格已稳定) |

每个 phase 独立 ship、独立 revert。每 phase 末尾:
- 跑 `just ci` 全过
- 更新 `AGENTS.md` 的对应章节(若有新规范)
- 在 milestone roadmap 中登记 phase 完成

---

## 4. Phase A 详情:Lint Baseline

### 4.1 Workspace lint 表

添加到根 `Cargo.toml`:

```toml
[workspace.lints.clippy]
# === deny 层:正确性 / 性能 / 可疑代码 ===
correctness = { level = "deny", priority = -1 }
suspicious = { level = "deny", priority = -1 }
perf = { level = "deny", priority = -1 }

# === warn 层:文档、风格、可维护性 ===
pedantic = { level = "warn", priority = -1 }
nursery = { level = "warn", priority = -1 }
missing_errors_doc = "warn"
missing_panics_doc = "warn"
module_name_repetitions = "warn"
must_use_candidate = "warn"
return_self_not_must_use = "warn"
missing_const_for_fn = "warn"
uninlined_format_args = "warn"

# === 显式 allow:与项目风格冲突或低 ROI ===
cast_precision_loss = "allow"          # 模型维度 cast 常见
cast_possible_truncation = "allow"     # 同上
cast_possible_wrap = "allow"
cast_sign_loss = "allow"
similar_names = "allow"
too_many_lines = "allow"               # Phase D 才处理
too_many_arguments = "allow"
multiple_crate_versions = "allow"      # 依赖图遗留,单独治理
```

### 4.2 Per-crate 覆盖

每个 crate 的 `Cargo.toml` 添加:

```toml
[lints]
workspace = true
```

### 4.3 CI 接入

`just clippy` 已是 `cargo clippy --workspace --all-targets --all-features -- -D warnings`,无需改命令,只需 lint 表生效即可。

新增 `just clippy-pedantic`(可选,本地用):
```bash
cargo clippy --workspace --all-targets --all-features -- -W clippy::pedantic -W clippy::nursery 2>&1 | rg "warning:" | wc -l
```
把当前 pedantic 警告数登记为基线,在 phase 末尾核对减少量。

### 4.4 完成定义

- [ ] 根 `Cargo.toml` 有 `[workspace.lints.clippy]` 表
- [ ] 所有 6 个 crate 的 `Cargo.toml` 有 `[lints]` 节
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` 通过
- [ ] `just ci` 通过
- [ ] Phase A 不修复 pedantic warn(留给后续 phase 在触及文件时顺势修)

---

## 5. Phase B 详情:Unwrap Cleanup

### 5.1 现状(2026-06-28 audit 修正)

原始 spec 假设 787 生产 unwrap,实际 **60**(其中 3 处为 doc 误报,实为 **57**)。spec 目标 `≤160(-80%)` 已被满足。Phase B 实际工作大幅缩减。

### 5.2 分类规则(已由 audit 完成)

| 类别 | 处理 |
|------|------|
| 测试代码(`#[cfg(test)]` / `tests/` / benches) | **保留** unwrap(764 处,不动) |
| const / static 初始化 | **保留** + `// const invariant` 注释 |
| 不变量(RwLock/Mutex poisoned、SystemTime::now、Tensor 分配、`HashMap` insert 后 `get()` 等) | 保留 + 加 `// invariant: ...` 注释(51 处) |
| IO / 解析 / 外部依赖 / 类型转换失败 | **改为 typed error**,加 `#[source]`(6 处) |
| 数组/Vec 索引 | 优先 `.get(idx)` + typed error;必要时保留 + 注释 |
| 数学计算中的 `.sqrt()` 等 | 保留(已是精确语义) |

### 5.3 阶段重划(基于 audit)

| 子阶段 | 范围 | 工作量 |
|--------|------|--------|
| **B-1** | 修 `cuda_graph/executor.rs:222`(真 bug, race condition) | 1 处,~5 行 |
| **B-2** | 修剩余 5 处 CONVERT(engine.rs:565, main.rs ×3, handler.rs:42) | 5 处,~30-50 行 |
| **B-3a** | RwLock/Mutex `.expect("poisoned")` 集群加 `// invariant:` 注释 | 21 处,~25 行 |
| **B-3b** | `SystemTime::duration_since` 集群加注释 | 8 处,~10 行 |
| **B-3c** | Tensor/serialize/signal/env/misc 杂项加注释 | 22 处,~30 行 |
| **B-4** (可选) | 给 test module 加 `#![allow(clippy::unwrap_used)]` | 配置性,~10 行 |

### 5.4 错误规范

按 AGENTS.md 已有的 `Error Type Conventions`:
- 用 `#[derive(thiserror::Error)]`
- 每个 variant 有 `#[error("...")]`
- 包装其它 error 用 `#[source]`
- 跨 crate 转换通过 `From<E>` impl,放在 `error/mod.rs`
- `Box<dyn Error>` 禁止出现在公开 API

### 5.5 with_request_id 工具

对有请求上下文的 error enum(参考 `EngineError`)加 helper:

```rust
impl EngineError {
    pub fn with_request_id(self, req_id: u64) -> Self { ... }
}
```

### 5.6 完成定义

- [ ] B-1: `cuda_graph/executor.rs:222` 改为 typed error
- [ ] B-2: 其余 5 处 CONVERT 完成
- [ ] B-3: 51 处 INVARIANT 全部带 `// invariant:` 注释
- [ ] 每个新 error variant 至少 1 个负向单元测试
- [ ] `just ci` 通过
- [ ] AGENTS.md 增加"不变量注释规范"小节(如未覆盖)

---

## 6. Phase C 详情:API Ergonomics

### 6.1 审计清单

每 crate 公开 API 过一遍:

- [ ] **Builder 化**:struct 有 >2 个 `Option<T>` 字段 → 改为 builder(参考 `SpeculationConfig`、`BatchBuilder`、`ModelLoader`)
- [ ] **Stringly-typed**:用 `String` 表示枚举/分类 → 改为 typed enum 或 `&'static str`(仅配置字段允许)
- [ ] **Error 规范**:所有公开 error enum 都 `#[derive(thiserror::Error)]` + 每 variant 有 `#[error(...)]` + 必要时 `#[source]`
- [ ] **Crate-root re-export**:每个 crate `lib.rs` 末尾有 `pub use` 块(参考 AGENTS.md 已有约定)
- [ ] **禁止 `Box<dyn Error>`**:搜遍公开 API 签名,确保无
- [ ] **Object-safe trait `Default` impl**:trait object-safe 且常用于 `Arc<dyn Trait>` 时,提供 `Default`

### 6.2 Builder 化模式

```rust
// 之前
let cfg = SomeConfig { max_tokens: Some(100), top_p: Some(0.9), temperature: Some(0.7), stop: None };

// 之后
let cfg = SomeConfig::builder()
    .with_max_tokens(100)
    .with_top_p(0.9)
    .with_temperature(0.7)
    .build();
```

约定:
- `Config::builder()` / `ConfigBuilder::new()`
- `with_<field>(value)` 设值,无 `with_<field>_opt`(必要时提供)
- `build() -> Result<Config, ConfigError>` 或 `build() -> Config`(若 infallible)

### 6.3 完成定义

- [ ] 公开 struct 中 >2 optional 字段的 100% builder 化
- [ ] 无 `Box<dyn Error>` 出现在公开 API
- [ ] 每个 crate `lib.rs` 有 crate-root re-export 块
- [ ] `just ci` 通过
- [ ] AGENTS.md "API Conventions" 小节如有偏差则更新

---

## 7. Phase D 详情:Module Boundaries

### 7.1 触发条件

- 文件 > 500 行(剔除 `mod tests` 与 `#[cfg(test)]`)
- 模块 `pub` 项目过多(>5 个 `pub fn`/`pub struct` 未被 crate 外使用)
- 跨 crate 出现循环或反向依赖

### 7.2 重构动作

1. **拆分大文件**:把内聚子逻辑提到 `xxx/inner.rs` 或 `xxx/detail.rs`,主文件保留 facade
2. **收紧可见性**:`pub` → `pub(crate)` 除非确实对外
3. **集中 re-export**:`pub use` 在 `lib.rs` 集中,移除散落的 `pub mod foo`
4. **trait 抽象**:出现 ≥2 处相似实现时提取 trait
5. **新文件行数上限**:单文件新建时 ≤ 500 行(由 lint 检查或 code review 把关)

### 7.3 预期热点文件

待 Phase A/B/C 完成后实测,初步判断:
- `crates/model/src/components/attention/mla.rs`
- `crates/core/src/engine.rs`
- `crates/core/src/scheduler/engine.rs`
- `crates/server/src/openai/chat.rs`
- `crates/model/src/loader/builder.rs`

### 7.4 完成定义

- [ ] 所有 `.rs` 文件(剔除 tests) ≤ 500 行
- [ ] 每个 crate `lib.rs` 集中 re-export
- [ ] 无 `pub use crate::foo::bar::baz` 深度链(≤ 2 层)
- [ ] `just ci` 通过

---

## 8. 测试与验证

每个 phase 每个 commit 前:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
just ci
```

每个 phase 末尾:

```bash
# Phase A: pedantic 警告数基线登记
cargo clippy --workspace --all-targets --all-features -- -W clippy::pedantic 2>&1 | rg "warning:" | wc -l

# Phase B: unwrap 计数
rg -c "\.unwrap\(\)|\.expect\(" /workspace/vllm-lite/crates/ --type rust -g '!tests' -g '!target/**' | awk -F: '{sum+=$2} END {print sum}'

# Phase D: 大文件扫描
find /workspace/vllm-lite/crates -name '*.rs' -not -path '*/target/*' -not -path '*/tests/*' | xargs wc -l | awk '$1 > 500 {print}'
```

---

## 9. 风险与回退

### 9.1 风险

| 风险 | 缓解 |
|------|------|
| Phase A 的 deny lint 触发大量现有代码违规 | Phase A 实施前先 `cargo clippy -- -D warnings` 跑一遍,确认违规面;若过多,把 deny 改为 warn,逐 lint 推进 |
| Phase B 改动可能影响下游调用方 | 保持公开签名;内部 error 类型新增 variant 不破坏现有 match(因 `#[non_exhaustive]` 或默认分支兜底) |
| Phase C builder 化可能与现有调用方重复样板 | 保留旧 struct literal 入口一段时间(标注 `#[deprecated]`);或一次性替换所有调用点 |
| Phase D 拆分会改变 git blame | 用 `git log --follow` 跟踪;接受历史重置 |

### 9.2 回退策略

每 phase 独立 milestone(如 `v24.0-phase-a`, `v24.0-phase-b`...),失败可单独 revert。

---

## 10. 不在本次范围(后续)

- 性能优化(PERF phase)
- 进一步 nursery lint 治理
- 测试覆盖率提升到具体目标(%)
- fuzz test / property-based test 引入
- 安全审计(独立的 `security-review` skill)
- 文档站重建(独立 phase)

---

## 11. 元数据

- **作者**: opencode agent
- **日期**: 2026-06-28
- **milestone**: v24.0 Code Quality Hardening
- **预计总周期**: 4 phases × 1-2 周 = 4-8 周
- **关联文档**:
  - `AGENTS.md`(Code Style、API Conventions、Error Type Conventions、Sync vs Async Trait Splits)
  - `docs/superpowers/specs/2026-03-29-vllm-lite-design.md`(整体架构)
