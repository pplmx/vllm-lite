# Wave 2: Adaptive Speculative Decoding Counter Wire-up 设计

**日期**: 2026-06-26
**状态**: 🔄 待审
**基线**: `main @ a4886a7` (Wave 1 + 1.6 完成)
**关联**: `.planning/PROJECT.md` v17.0 active items (SPEC-ADAPT-01, SPEC-ADAPT-02)

---

## 背景

### 当前状态（2026-06-26 探索发现）

Wave 2 探索时发现：**v17.0 SPEC-ADAPT-01/02 的核心实现已全部就位**。`AdaptiveSpeculativeDecoder` + `DraftAccuracyTracker`（589 行 + 18 unit tests）、engine 集成、scheduler 动态 draft 读取、metrics 收集器、Prometheus export、server config/CLI — 全部已实现并 wired up。

代码与文档偏差同 Wave 1 模式：`PROJECT.md` 仍标 `[ ]`。

### 唯一真实 Gap

`crates/core/src/metrics/collector.rs:117` 已定义 `record_speculative_adjustment()`，`crates/core/src/metrics/exporter.rs:56-61` 已 export Prometheus counter `speculative_adjustments_total`，`crates/server/src/debug.rs:39` 已暴露 `/debug/metrics`。但**调用点缺失**：

```rust
// crates/core/src/speculative/adaptive.rs:158-180
fn maybe_adjust(&mut self) {
    // ... 计算 adjustment ...
    if adjustment != 0 {
        let new_max = (... + adjustment).clamp(...) as usize;
        if new_max != self.current_max_draft_tokens {
            tracing::info!("Adjusted max_draft_tokens: ...", ...);  // ← 仅 log
            self.current_max_draft_tokens = new_max;
            // ← 缺 metrics.record_speculative_adjustment()
        }
    }
    // ...
}
```

结果：`speculative_adjustments_total` Prometheus counter 永远是 0，无法在 Grafana 观察到 adaptive decoder 实际工作频率。

---

## 目标

1. **正确性**：让 Prometheus counter 在每次实际 draft depth 调整时递增
2. **可观测性**：运维通过 Grafana 验证 adaptive decoder 在生产中是否活跃
3. **文档同步**：让 `.planning/` 与 `main` 实际状态对齐（SPEC-ADAPT 标完成）
4. **API 一致性**：新增测试验证 `record_verification` 返回值正确性

**非目标：**

- 不重写 EWMA / deadband / cooldown 算法
- 不改 `AdaptiveDraftConfig` schema
- 不做 SPEC-WARM-01 / SPEC-BENCH-01/02 / SPEC-MULTI-01/02
- 不引入新 metrics
- 不改 server config / CLI

---

## 设计

### D2-1：`record_verification` 返回 `bool` adjustment 事件

**决策：** `AdaptiveSpeculativeDecoder::record_verification` 由 `()` 改为返回 `bool`：

```rust
/// Record verification results. Returns `true` if an actual adjustment
/// to `current_max_draft_tokens` was made (i.e., the value changed),
/// `false` otherwise (within deadband, or clamped to bound).
pub fn record_verification(&mut self, num_draft: usize, num_accepted: usize) -> bool {
    for i in 0..num_draft {
        let accepted = i < num_accepted;
        self.accuracy_tracker.record(accepted);
    }
    self.steps_since_adjustment += 1;
    if self.steps_since_adjustment >= self.config.cooldown_steps {
        self.maybe_adjust()  // returns bool now
    } else {
        false
    }
}

fn maybe_adjust(&mut self) -> bool {
    // ... existing computation ...
    if adjustment != 0 {
        let new_max = (self.current_max_draft_tokens as i32 + adjustment)
            .clamp(self.config.min_draft_tokens as i32, self.config.max_draft_tokens as i32)
            as usize;
        if new_max != self.current_max_draft_tokens {
            tracing::info!("Adjusted max_draft_tokens: {} -> {} (rate: {:.3}, target: {:.2}, threshold: {:.2})",
                self.current_max_draft_tokens, new_max, rate, target, threshold);
            self.current_max_draft_tokens = new_max;
            self.steps_since_adjustment = 0;
            true  // ← 新增
        } else {
            false  // clamped to bound, no actual change
        }
    } else {
        self.steps_since_adjustment = 0;
        false  // within deadband
    }
}
```

**理由：**
- `AdaptiveSpeculativeDecoder` 保持纯组件层（无 metrics 依赖）
- 现有 18 个测试零修改（Rust 允许忽略返回值）
- 与 engine 现有 `record_speculative_efficiency` 调用模式一致

### D2-2：engine caller 调 metrics counter

**决策：** `crates/core/src/engine/speculative.rs:93-97` 改为：

```rust
// Track accuracy in adaptive decoder
if let Some(ref mut decoder) = self.adaptive_decoder {
    let total_draft: usize = draft_outputs.iter().map(|d| d.len()).sum();
    let total_accepted: usize = accepted_counts.iter().sum();
    if decoder.record_verification(total_draft, total_accepted) {
        self.scheduler.metrics.record_speculative_adjustment();
    }
}
```

**理由：** engine 已有 `record_speculative_efficiency` (line 99-108) 和 `record_per_request_acceptance` (line 110-117) 的直接调用先例。

### D2-3：新单元测试覆盖返回值

**决策：** 在 `crates/core/src/speculative/adaptive.rs` 的 `mod tests` 中加 3 个测试：

1. `test_record_verification_returns_true_on_increase` — high acceptance + cooldown triggered → returns `true`
2. `test_record_verification_returns_true_on_decrease` — low acceptance + cooldown triggered → returns `true`
3. `test_record_verification_returns_false_within_deadband` — rate within target±threshold → returns `false`

**理由：** 锁定 D2-1 的契约。

### D2-4：文档同步（4 files）

| 文件 | 变更 |
|------|------|
| `.planning/PROJECT.md` | SPEC-ADAPT-01/02 标 `[x]`，加 commit 引用 |
| `.planning/STATE.md` | current_focus 改为 Wave 2；accumulated_context 更新 |
| `ROADMAP.md` | Phase 8 监控段补 adaptive metrics；总进度同步 |
| `CHANGELOG.md` | `[Unreleased]` 段补 adaptive wire-up 条目 |
| `.planning/SESSION-HANDOFF.md` | 下一优先级改为 Wave 3 (Dependabot)；Wave 2 标完成 |

---

## 目标目录结构（无变化）

Wave 2 不引入新文件，仅修改：
- `crates/core/src/speculative/adaptive.rs`（~30 行：return type + 3 个 test）
- `crates/core/src/engine/speculative.rs`（~2 行：调用 metrics）
- 4 个 doc 文件

---

## 任务分解

### Wave 2 Task 1：spec doc（本文件已写）

`docs/superpowers/specs/2026-06-26-wave2-adapt-spec.md`（本文件）

### Wave 2 Task 2：counter wire-up（1 commit）

修改：
- `crates/core/src/speculative/adaptive.rs`:
  - `record_verification` 返回类型 `()` → `bool`
  - `maybe_adjust` 返回类型 `()` → `bool`
  - 新增 3 个单元测试
- `crates/core/src/engine/speculative.rs`:
  - line 93-97：调用 `record_speculative_adjustment()`

验证：
- `cargo test -p vllm-core adaptive`
- `cargo test -p vllm-core speculative`
- `cargo clippy -p vllm-core --all-targets -- -D warnings`
- 现有 18 个 adaptive 测试无回归

Commit: `feat(speculative): wire up speculative_adjustments_total counter`

### Wave 2 Task 3：文档同步（4 commits）

3a: `docs(planning): mark SPEC-ADAPT-01/02 complete in PROJECT/STATE/ROADMAP`
3b: `docs(core): add adaptive speculative counter wire-up to CHANGELOG`
3c: `docs(planning): refresh SESSION-HANDOFF for Wave 2 status`

验证：
- `rg "SPEC-ADAPT" .planning/` 应匹配 `[x]` 形式
- `rg "Wave 2" .planning/` 应反映完成态
- `git diff` 应只触及文档文件

---

## 验证

### Wave 2 Task 2 验证

```bash
# 单元测试
cargo test -p vllm-core adaptive --lib
# 预期: 21 tests pass (18 现有 + 3 新增)

cargo test -p vllm-core speculative --lib
# 预期: 现有测试全过

# 集成测试
cargo test -p vllm-core step_speculative --lib
cargo test -p vllm-core engine --lib

# 端到端验证 counter 递增
# 写一个最小测试：mock engine，跑 5 次 record_verification (低接受率)
# 断言 metrics.speculative_adjustments_total > 0

# Clippy
cargo clippy --workspace --all-targets -- -D warnings
# 预期: 0 errors

# 全量
just nextest
# 预期: 1035 passed (1032 + 3 新), 46 skipped
```

### Wave 2 收口验证

```bash
# 文档一致性
rg "SPEC-ADAPT-01" .planning/PROJECT.md  # 应匹配 `[x]`
rg "Wave 2" .planning/SESSION-HANDOFF.md  # 应反映完成态

# Counter 导出验证
grep -c "speculative_adjustments_total" crates/core/src/metrics/exporter.rs
# 预期: >= 1 (export 行已存在)

# Commit 序列
git log --oneline -10
# 预期: spec + code + 3 docs commits
```

---

## 错误处理 / 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 现有 18 测试因 return type 变化失败 | 低 | 中 | Rust 允许忽略返回值；测试仅断言 decoder 字段不变 |
| engine 调用 metrics 时机错（deadband 内误计） | 低 | 低 | `record_verification` 已过滤"实际改变"vs"无变化"；counter 只在实际改变时递增 |
| Prometheus counter 重复递增 | 极低 | 低 | 唯一调用点：engine `step_speculative_inner` |
| 文档 commit 引用错 commit hash | 低 | 低 | 写 commit 时 `git log --oneline -1` 确认 |

---

## 不做（明确边界）

- ❌ 不重写 EWMA / deadband / cooldown 算法
- ❌ 不改 `AdaptiveDraftConfig` schema
- ❌ 不做 SPEC-WARM-01（Wave 4）
- ❌ 不做 SPEC-BENCH-01/02（Wave 5）
- ❌ 不做 SPEC-MULTI-01/02（v18.0）
- ❌ 不引入新 metrics
- ❌ 不改 server config / CLI
- ❌ 不改 metrics collector / exporter（仅调用已有 API）

---

## 风险与决策记录

| ID | 决策 | 理由 | 日期 |
|----|------|------|------|
| D2-1 | `record_verification` 返回 `bool` 而非注入 metrics | 保持 decoder 纯逻辑层；与现有调用模式一致 | 2026-06-26 |
| D2-2 | engine caller 调 metrics | 与 `record_speculative_efficiency` 先例一致 | 2026-06-26 |
| D2-3 | 加 3 个新单测 | 锁定 bool 返回契约 | 2026-06-26 |
| D2-4 | 4 个 doc files 同步 | Wave 1 同模式；`PROJECT.md` 是面向用户的真实状态 | 2026-06-26 |
| D2-5 | 任务分解 1+1+3=5 commits | spec + code + 3 docs batch | 2026-06-26 |

---

## 会话接续

Wave 2 完成后，下一 session 应：
1. 读 `.planning/WAVE-2-PLAN.md`（由 writing-plans 生成）
2. 跑 `just nextest` 确认基线
3. 从 Task 2 (counter wire-up) 开始
4. Task 3 (doc sync) 紧随

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-26 | 初版：基于 Wave 1 完成后对 SPEC-ADAPT 现状的探索 |
