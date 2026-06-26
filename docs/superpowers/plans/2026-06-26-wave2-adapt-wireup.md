# Wave 2: Adaptive Counter Wire-up 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Prometheus `speculative_adjustments_total` counter 在每次 adaptive draft depth 实际调整时递增；同步文档标记 SPEC-ADAPT-01/02 完成。

**Architecture:** `AdaptiveSpeculativeDecoder::record_verification` 由 `()` 改为 `bool` adjustment 事件；engine caller 据此调 `record_speculative_adjustment()`。Decoder 保持纯组件层。

**Tech Stack:** Rust, candle-core (deps 不变)

**基线 commit:** `9e564f6`（spec 已落地）

**前置验证:**

```bash
cd /workspace/vllm-lite
just nextest         # 必须 ≥ 1032 passed
cargo clippy --workspace --all-targets -- -D warnings  # 必须绿
git log --oneline -1 # 应为 9e564f6
```

---

## Task 1: Wire-up counter（1 commit）

**Files:**
- Modify: `crates/core/src/speculative/adaptive.rs`
- Modify: `crates/core/src/engine/speculative.rs`

- [ ] **Step 1: 先写 3 个失败的新单测（在 adaptive.rs 的 mod tests 末尾追加）**

```rust
// 在 crates/core/src/speculative/adaptive.rs 文件末尾的 mod tests 块内追加：

    #[test]
    fn test_record_verification_returns_true_on_increase() {
        let config = AdaptiveDraftConfig {
            min_draft_tokens: 2,
            max_draft_tokens: 8,
            target_acceptance_rate: 0.5,
            accuracy_window_size: 5,
            adjustment_step: 1,
            cooldown_steps: 1,
            ewma_alpha: 0.5,
            deadband_threshold: 0.1,
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);
        let before = decoder.current_max_draft_tokens();

        // 100% acceptance → deviation > threshold → should adjust up
        let adjusted = decoder.record_verification(5, 5);
        assert_eq!(
            adjusted, true,
            "100% acceptance should trigger adjustment (return true)"
        );
        assert!(
            decoder.current_max_draft_tokens() >= before,
            "max_draft_tokens should not decrease on high acceptance"
        );
    }

    #[test]
    fn test_record_verification_returns_true_on_decrease() {
        let config = AdaptiveDraftConfig {
            min_draft_tokens: 2,
            max_draft_tokens: 8,
            target_acceptance_rate: 0.7,
            accuracy_window_size: 5,
            adjustment_step: 1,
            cooldown_steps: 1,
            ewma_alpha: 0.5,
            deadband_threshold: 0.1,
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);

        // 0% acceptance → deviation < -threshold → should adjust down
        let adjusted = decoder.record_verification(5, 0);
        assert_eq!(
            adjusted, true,
            "0% acceptance should trigger adjustment (return true)"
        );
        assert!(
            decoder.current_max_draft_tokens() < 8,
            "max_draft_tokens should decrease below initial max on 0% acceptance"
        );
    }

    #[test]
    fn test_record_verification_returns_false_within_deadband() {
        let config = AdaptiveDraftConfig {
            min_draft_tokens: 2,
            max_draft_tokens: 8,
            target_acceptance_rate: 0.7,
            accuracy_window_size: 5,
            adjustment_step: 1,
            cooldown_steps: 1,
            ewma_alpha: 0.5,
            deadband_threshold: 0.5, // very wide deadband
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);
        let initial = decoder.current_max_draft_tokens();

        // 3/5 = 0.6 acceptance; deviation = |0.6 - 0.7| = 0.1 < 0.5 threshold
        let adjusted = decoder.record_verification(5, 3);
        assert_eq!(
            adjusted, false,
            "Within deadband should NOT trigger adjustment (return false)"
        );
        assert_eq!(
            decoder.current_max_draft_tokens(),
            initial,
            "max_draft_tokens should be unchanged within deadband"
        );
    }
```

- [ ] **Step 2: 运行新单测，预期失败（因为 return type 还是 `()`）**

```bash
cd /workspace/vllm-lite
cargo test -p vllm-core --lib adaptive::tests::test_record_verification_returns_true_on_increase 2>&1 | tail -20
# 预期: 编译失败，错误 "expected `()`, found `bool`" 或类似
# （type mismatch 因为测试期望 `adjusted == true` 但函数返回 `()`）
```

- [ ] **Step 3: 修改 `record_verification` 返回类型 + `maybe_adjust` 返回类型**

在 `crates/core/src/speculative/adaptive.rs` 中：

```rust
// 替换前（约 line 125）：
    /// Record verification results and potentially adjust
    pub fn record_verification(&mut self, num_draft: usize, num_accepted: usize) {
        // Record each draft token result
        for i in 0..num_draft {
            let accepted = i < num_accepted;
            self.accuracy_tracker.record(accepted);
        }

        // Check if we should adjust
        self.steps_since_adjustment += 1;
        if self.steps_since_adjustment >= self.config.cooldown_steps {
            self.maybe_adjust();
        }
    }

// 替换后：
    /// Record verification results and potentially adjust.
    /// Returns `true` if `current_max_draft_tokens` was actually changed,
    /// `false` if within deadband, clamped to bound, or cooldown not elapsed.
    pub fn record_verification(&mut self, num_draft: usize, num_accepted: usize) -> bool {
        // Record each draft token result
        for i in 0..num_draft {
            let accepted = i < num_accepted;
            self.accuracy_tracker.record(accepted);
        }

        // Check if we should adjust
        self.steps_since_adjustment += 1;
        if self.steps_since_adjustment >= self.config.cooldown_steps {
            self.maybe_adjust()
        } else {
            false
        }
    }
```

```rust
// 替换前（约 line 140）：
    /// Potentially adjust draft token count based on EWMA accuracy and deadband hysteresis
    fn maybe_adjust(&mut self) {

// 替换后：
    /// Potentially adjust draft token count based on EWMA accuracy and deadband hysteresis.
    /// Returns `true` if `current_max_draft_tokens` was actually changed.
    fn maybe_adjust(&mut self) -> bool {
```

```rust
// 替换前（约 line 158-180，maybe_adjust 内部）：
        if adjustment != 0 {
            let new_max = (self.current_max_draft_tokens as i32 + adjustment).clamp(
                self.config.min_draft_tokens as i32,
                self.config.max_draft_tokens as i32,
            ) as usize;

            if new_max != self.current_max_draft_tokens {
                tracing::info!(
                    "Adjusted max_draft_tokens: {} -> {} (rate: {:.3}, target: {:.2}, threshold: {:.2})",
                    self.current_max_draft_tokens,
                    new_max,
                    rate,
                    target,
                    threshold,
                );
                self.current_max_draft_tokens = new_max;
                self.steps_since_adjustment = 0;
            }
        } else {
            // Within deadband: reset cooldown to prevent stale accumulation
            self.steps_since_adjustment = 0;
        }
    }

// 替换后：
        if adjustment != 0 {
            let new_max = (self.current_max_draft_tokens as i32 + adjustment).clamp(
                self.config.min_draft_tokens as i32,
                self.config.max_draft_tokens as i32,
            ) as usize;

            if new_max != self.current_max_draft_tokens {
                tracing::info!(
                    "Adjusted max_draft_tokens: {} -> {} (rate: {:.3}, target: {:.2}, threshold: {:.2})",
                    self.current_max_draft_tokens,
                    new_max,
                    rate,
                    target,
                    threshold,
                );
                self.current_max_draft_tokens = new_max;
                self.steps_since_adjustment = 0;
                true
            } else {
                // Clamped to bound: adjustment would not change value
                false
            }
        } else {
            // Within deadband: reset cooldown to prevent stale accumulation
            self.steps_since_adjustment = 0;
            false
        }
    }
```

- [ ] **Step 4: 运行所有 adaptive 单测，预期 21 个全过（18 现有 + 3 新）**

```bash
cd /workspace/vllm-lite
cargo test -p vllm-core --lib adaptive 2>&1 | tail -5
# 预期: "test result: ok. 21 passed; 0 failed"
```

- [ ] **Step 5: 修改 engine caller 使用 bool 返回**

在 `crates/core/src/engine/speculative.rs` 中：

```rust
// 替换前（约 line 92-97）：
        // Track accuracy in adaptive decoder
        if let Some(ref mut decoder) = self.adaptive_decoder {
            let total_draft: usize = draft_outputs.iter().map(|d| d.len()).sum();
            let total_accepted: usize = accepted_counts.iter().sum();
            decoder.record_verification(total_draft, total_accepted);
        }

// 替换后：
        // Track accuracy in adaptive decoder and record adjustment events
        if let Some(ref mut decoder) = self.adaptive_decoder {
            let total_draft: usize = draft_outputs.iter().map(|d| d.len()).sum();
            let total_accepted: usize = accepted_counts.iter().sum();
            if decoder.record_verification(total_draft, total_accepted) {
                self.scheduler.metrics.record_speculative_adjustment();
            }
        }
```

- [ ] **Step 6: 全量验证**

```bash
cd /workspace/vllm-lite
cargo build -p vllm-core
cargo clippy -p vllm-core --all-targets -- -D warnings
cargo test -p vllm-core --lib adaptive
cargo test -p vllm-core --lib speculative
cargo test -p vllm-core --lib engine
cargo clippy --workspace --all-targets -- -D warnings
just nextest
```

预期：
- 全 build/clippy/test 干净
- `just nextest` ≥ 1032 passed（之前 1032；新 3 测试覆盖既有路径，无回归 = 1035 或同 1032 取决于不同 sequence 跑出的 unique path 数）

- [ ] **Step 7: Commit**

```bash
cd /workspace/vllm-lite
git add crates/core/src/speculative/adaptive.rs crates/core/src/engine/speculative.rs
git commit -m "feat(speculative): wire up speculative_adjustments_total counter

\`AdaptiveSpeculativeDecoder::record_verification\` now returns a \`bool\`
indicating whether current_max_draft_tokens was actually changed.
The engine caller in step_speculative_inner uses this to invoke
\`MetricsCollector::record_speculative_adjustment()\`, which feeds the
\`speculative_adjustments_total\` Prometheus counter (previously always 0).

Decoder stays metrics-agnostic; engine caller pattern mirrors the existing
\`record_speculative_efficiency\` / \`record_per_request_acceptance\` calls.

Adds 3 unit tests locking the bool return contract:
- increase on high acceptance
- decrease on low acceptance
- false within deadband

Refs: docs/superpowers/specs/2026-06-26-wave2-adapt-spec.md"
```

---

## Task 2: 同步 PROJECT.md / STATE.md / ROADMAP.md（1 commit）

**Files:**
- Modify: `.planning/PROJECT.md`
- Modify: `.planning/STATE.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: PROJECT.md v17.0 active 区更新**

```text
// 替换前（v17.0 Active 段）：
- [ ] **SPEC-ADAPT-01**: Adaptive draft depth based on real-time acceptance rates
- [ ] **SPEC-ADAPT-02**: Acceptance rate monitoring and dynamic adjustment

// 替换后：
- [x] **SPEC-ADAPT-01**: Adaptive draft depth — `AdaptiveSpeculativeDecoder` + EWMA + deadband + cooldown (commit `fe0XXXX` ← 填入 Task 1 commit hash)
- [x] **SPEC-ADAPT-02**: Acceptance rate monitoring — `record_per_request_acceptance` + Prometheus `speculative_adjustments_total` + `/debug/metrics` (commit `fe0XXXX`)
```

- [ ] **Step 2: PROJECT.md Last updated**

```text
// 替换前：
*Last updated: 2026-06-26 — Wave 1 收口；Wave 2–5 spec decode 增量在 pipeline*

// 替换后：
*Last updated: 2026-06-26 — Wave 2 SPEC-ADAPT counter wire-up + docs sync 完成；Wave 3 (Dependabot) 待启动*
```

- [ ] **Step 3: STATE.md current_focus + last_activity**

```text
// 替换前（last_updated / last_activity frontmatter）：
last_updated: "2026-06-26T00:00:00.000Z"
last_activity: 2026-06-26

// 替换为（保留日期，但 time 改为 Wave 2 时间，例如）：
last_updated: "2026-06-26T01:30:00.000Z"   // 或当前 UTC；用 `date -u +%FT%T.000Z` 获取
last_activity: 2026-06-26
```

```text
// 替换前（Current Position 段）：
Wave: 1 of 5 (Wave 1: 文档同步 + dead_code 审计)
Status: Wave 1 in progress; Wave 2–5 in pipeline

// 替换后：
Wave: 2 of 5 (Wave 2: SPEC-ADAPT counter wire-up + doc sync)
Status: Wave 2 in progress; Wave 3–5 in pipeline
```

```text
// 替换前（Project Reference 段）：
See: .planning/PROJECT.md (updated 2026-06-26)
**Current focus:** Wave 1 of 5 (文档同步 + dead_code 审计) — Wave 2–5 spec decode 增量在 pipeline

// 替换后：
See: .planning/PROJECT.md (updated 2026-06-26)
**Current focus:** Wave 2 of 5 (SPEC-ADAPT-01/02 counter wire-up + doc sync) — Wave 3–5 在 pipeline
```

- [ ] **Step 4: ROADMAP.md 补 adaptive 注释**

```text
// 在 ROADMAP.md Phase 5 生产就绪 / 监控 段（约 line 168-170）加一行 callout：

> 2026-06-26 更新：Wave 2 完成 `speculative_adjustments_total` Prometheus counter wire-up（`AdaptiveSpeculativeDecoder::record_verification` 返回 `bool` adjustment 事件）。SPEC-ADAPT-01/02 实现已完整；Wave 3 (Dependabot) 待启动。
```

- [ ] **Step 5: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat .planning/PROJECT.md .planning/STATE.md ROADMAP.md
# 预期: 3 files changed, ~12 insertions, ~6 deletions

cargo check --workspace  # sanity
git add .planning/PROJECT.md .planning/STATE.md ROADMAP.md
git commit -m "docs(planning): mark SPEC-ADAPT-01/02 complete in PROJECT/STATE/ROADMAP"
```

---

## Task 3: CHANGELOG 补 Wave 2 条目（1 commit）

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 在 `[Unreleased]` → `### Refactored` 段之前（或 `### Added` 段内）追加新条目**

定位：在 Wave 1 追加的 `#### Architecture Refactor Phase 5 (Qwen3.5 Hybrid 收敛, 2026-06-15)` 之后、`### Added (Phase 4)` 之前，追加新子段：

```markdown
#### Adaptive Speculative Decoding Counter Wire-up (Wave 2, 2026-06-26)

- `AdaptiveSpeculativeDecoder::record_verification` now returns `bool` adjustment event
- Engine `step_speculative_inner` calls `MetricsCollector::record_speculative_adjustment()` on actual adjustment
- `speculative_adjustments_total` Prometheus counter now correctly tracks adaptive decoder activity
- 3 new unit tests locking the bool return contract (high acceptance, low acceptance, deadband)
- Documentation: `SPEC-ADAPT-01` / `SPEC-ADAPT-02` marked complete in `.planning/PROJECT.md`

Refs: `docs/superpowers/specs/2026-06-26-wave2-adapt-spec.md`
```

（也可放在 `### Added` 段而非 `### Refactored`，视最终 CHANGELOG 结构美观而定。本计划选 Refactored 后是因为"counter wire-up"是行为微调而非新功能。）

- [ ] **Step 2: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat CHANGELOG.md
# 预期: 1 file changed, ~7 insertions

git add CHANGELOG.md
git commit -m "docs(core): add Wave 2 adaptive speculative counter wire-up to CHANGELOG"
```

---

## Task 4: 刷新 SESSION-HANDOFF.md（1 commit）

**Files:**
- Modify: `.planning/SESSION-HANDOFF.md`

- [ ] **Step 1: 更新顶部 Git 行**

```text
// 替换前：
> Git：`main` @ `54af5ad` (Wave 1 全部完成：11 commits；Phase 0–5 + Qwen3.5 Hybrid 收敛)

// 替换后：
> Git：`main` @ `<Wave 2 最新 commit>` (Wave 1 + 1.6 + 2 完成：共 18 commits)
```

（用 `git log --oneline -1` 获取实际 hash 后填入）

- [ ] **Step 2: 替换"下一优先级"段**

定位：找到当前"下一优先级"段（约 line 8-32）。整段替换为：

```markdown
## 下一优先级（2026-06-26，Wave 2 完成）

**Wave 1 + 1.6 + 2 全部完成（18 commits）**

| Wave | Commit | 描述 |
|------|--------|------|
| 1 | `d42b151` ~ `1499fcd` | 文档同步 + dead_code 审计（11 commits） |
| 1.6 | `a4886a7` | 清理 vllm-model pre-existing clippy（11 lints） |
| 2 | `9e564f6` + `<Task 1 hash>` + `<Task 2 hash>` + `<Task 3 hash>` + `<Task 4 hash>` | SPEC-ADAPT counter wire-up + docs sync |

**下一 Wave:** Wave 3 (Dependabot 5 漏洞：1 high, 4 moderate)
- 1 high 漏洞需要 bump 主版本号或选择替换库
- 需独立风险评估（不在 Wave 3 子任务范围内强行推进）

**Wave 2 spec/plan:**
- Spec: `docs/superpowers/specs/2026-06-26-wave2-adapt-spec.md` (commit `9e564f6`)
- Plan: `docs/superpowers/plans/2026-06-26-wave2-adapt-wireup.md` (本文件)
```

- [ ] **Step 3: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat .planning/SESSION-HANDOFF.md
# 预期: 1 file changed, ~25 insertions, ~20 deletions

cargo check --workspace  # sanity
git add .planning/SESSION-HANDOFF.md
git commit -m "docs(planning): refresh SESSION-HANDOFF for Wave 2 status"
```

---

## 收口验证

所有 5 commits 完成后（spec 已 + 4 new = 5）：

```bash
cd /workspace/vllm-lite

# 1. 全量 CI
just ci

# 2. Counter 导出验证
grep -c "speculative_adjustments_total" crates/core/src/metrics/exporter.rs
# 预期: 1 (export 行已存在)

# 3. 调用点验证
grep -n "record_speculative_adjustment" crates/core/src/engine/speculative.rs
# 预期: 1 行调用（在 step_speculative_inner 的 adaptive decoder 分支内）

# 4. 文档一致性
rg "SPEC-ADAPT-01" .planning/PROJECT.md
# 预期: 1 行匹配，包含 `[x]` (checked)

rg "Wave 2" .planning/SESSION-HANDOFF.md
# 预期: 反映完成态

# 5. 测试基线
just nextest 2>&1 | tail -3
# 预期: ≥ 1035 passed (1032 + 3 新), 46 skipped
```

**Wave 2 完成标志：**
- ✅ `just ci` 全绿
- ✅ `just nextest` ≥ 1035 passed（无回归 + 3 新）
- ✅ `record_speculative_adjustment` 在 engine 中确实被调用
- ✅ PROJECT.md SPEC-ADAPT-01/02 标 `[x]`
- ✅ CHANGELOG 补录 Wave 2 条目
- ✅ SESSION-HANDOFF 反映 Wave 2 完成

---

## 错误处理 / 风险

| 风险 | 缓解 |
|------|------|
| 新测试因现有 EWMA 行为差异失败 | Step 4 失败时检查 EWMA alpha / cooldown 配置是否正确 |
| Engine 调用 metrics 时机错 | Step 6 必须验证；counter 在 deadband 调用时**不**递增 |
| 文档 commit 引用错 commit hash | Step 1/2 时 `git log --oneline -1` 确认 |
| Markdown 渲染破坏 | `just fmt-check` 不验证 .md；肉眼检查 |

---

## 自审

- **Spec 覆盖:** ✅ D2-1 (bool 返回) → Task 1 Step 3；D2-2 (engine 调 metrics) → Task 1 Step 5；D2-3 (3 个新单测) → Task 1 Step 1；D2-4 (doc sync) → Tasks 2/3/4
- **占位符扫描:** ✅ 无 TBD/TODO；每处变更都有具体 before/after
- **类型一致性:** ✅ `record_verification` 返回 `bool` 在 Task 1 Step 3 定义，在 Step 5 engine caller 使用；在 Step 1 测试中也按 `bool` 断言
- **范围:** ✅ 5 commits（spec 已 1 + 4 new），单次会话可完成

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-26 | 初版：基于 `docs/superpowers/specs/2026-06-26-wave2-adapt-spec.md` |
