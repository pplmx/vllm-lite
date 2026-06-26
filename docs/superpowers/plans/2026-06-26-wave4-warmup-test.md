# Wave 4: SPEC-WARM-01 测试覆盖 + 文档同步实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 加 1 个 fast 单测锁定 `warmup_draft_kv` 在 prefill 时被调用；同步 PROJECT/STATE/ROADMAP/CHANGELOG/SESSION-HANDOFF。

**Architecture:** 内联 `CounterModel` wrapper（不动 `crates/testing/`），`warmup_draft_kv` 改 `pub(crate)` 让 test 可直接调。

**Tech Stack:** Rust, cargo (无新增依赖)

**基线 commit:** `55bc82d`（spec 已落地）

**前置验证:**

```bash
cd /workspace/vllm-lite
just nextest         # 必须 ≥ 1035 passed
cargo clippy --workspace --all-targets -- -D warnings  # 必须绿
git log --oneline -1 # 应为 55bc82d
```

---

## Task 1: CounterModel + fast 单测（1 commit）

**Files:**
- Modify: `crates/core/src/engine/speculative.rs`

- [ ] **Step 1: 改 `warmup_draft_kv` 可见性为 `pub(crate)`**

In `crates/core/src/engine/speculative.rs` line 10:

```rust
// Before:
    fn warmup_draft_kv(&mut self, batch: &Batch) -> Result<()> {

// After:
    pub(crate) fn warmup_draft_kv(&mut self, batch: &Batch) -> Result<()> {
```

- [ ] **Step 2: 在 mod tests 内加 CounterModel struct（紧接 inline FakeModel 之后，约 line 415）**

```rust
    /// Wrapper around FakeModel that counts forward/forward_logits invocations.
    /// Used to verify warmup_draft_kv calls draft model per sequence.
    struct CounterModel {
        inner: FakeModel,
        forward_count: std::sync::atomic::AtomicUsize,
    }

    impl CounterModel {
        fn new(token: TokenId) -> Self {
            Self {
                inner: FakeModel::new(token),
                forward_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }
        fn forward_count(&self) -> usize {
            self.forward_count
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl ModelBackend for CounterModel {
        fn forward(
            &mut self,
            seq_ids: &[SeqId],
            input_tokens: &[Vec<TokenId>],
            positions: &[Vec<usize>],
            kv_block_ids: &[Vec<usize>],
            num_computed_tokens: &[usize],
            is_prefill: &[bool],
        ) -> ModelResult<BatchOutput> {
            self.forward_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.forward(
                seq_ids,
                input_tokens,
                positions,
                kv_block_ids,
                num_computed_tokens,
                is_prefill,
            )
        }

        fn forward_logits(
            &mut self,
            seq_ids: &[SeqId],
            input_tokens: &[Vec<TokenId>],
            positions: &[Vec<usize>],
            kv_block_ids: &[Vec<usize>],
            num_computed_tokens: &[usize],
            is_prefill: &[bool],
        ) -> ModelResult<Vec<Vec<f32>>> {
            self.forward_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.forward_logits(
                seq_ids,
                input_tokens,
                positions,
                kv_block_ids,
                num_computed_tokens,
                is_prefill,
            )
        }

        fn embed(
            &mut self,
            input_tokens: &[Vec<TokenId>],
            positions: &[Vec<usize>],
        ) -> ModelResult<Vec<Vec<f32>>> {
            self.inner.embed(input_tokens, positions)
        }

        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }

        fn num_layers(&self) -> usize {
            self.inner.num_layers()
        }

        fn num_heads(&self) -> usize {
            self.inner.num_heads()
        }
    }
```

- [ ] **Step 3: 加 fast 单测 `test_warmup_draft_kv_invokes_draft_per_sequence`（紧接 CounterModel 之后）**

```rust
    /// Test Plan 17.4-A: warmup_draft_kv invokes draft model once per sequence.
    /// Fast unit test (no #[ignore]): directly constructs a Prefill batch and
    /// calls warmup_draft_kv to verify the contract independently of step().
    #[test]
    fn test_warmup_draft_kv_invokes_draft_per_sequence() {
        let target = FakeModel::new(42);
        let draft = CounterModel::new(42);
        let draft_count_before = draft.forward_count();
        let mut engine = super::super::Engine::new_boxed(
            Box::new(target),
            Some(Box::new(draft)),
        );
        engine.enable_speculative();

        // Construct a Prefill batch with 3 sequences.
        let batch = vllm_traits::types::Batch {
            seq_ids: vec![1, 2, 3],
            input_tokens: vec![vec![10, 20], vec![30], vec![40, 50, 60]],
            positions: vec![vec![0, 1], vec![0], vec![0, 1, 2]],
            kv_block_ids: vec![vec![0], vec![0], vec![0]],
            num_computed_tokens: vec![0, 0, 0],
            is_prefill: vec![true, true, true],
            phase: vllm_traits::BatchPhase::Prefill,
            total_tokens: 6,
            max_seq_len: 3,
        };

        // Execute warmup directly.
        engine
            .warmup_draft_kv(&batch)
            .expect("warmup_draft_kv should succeed");

        // Verify: draft model forward() called once per seq_id.
        let calls = draft.forward_count() - draft_count_before;
        assert_eq!(
            calls, 3,
            "warmup_draft_kv should invoke draft.forward() exactly once per seq_id (got {})",
            calls
        );
    }
```

- [ ] **Step 4: 运行新单测，预期通过**

```bash
cd /workspace/vllm-lite
cargo test -p vllm-core --lib engine::speculative::tests::test_warmup_draft_kv_invokes_draft_per_sequence 2>&1 | tail -10
# 预期: 1 passed
```

- [ ] **Step 5: 验证现有测试无回归 + clippy + nextest**

```bash
cd /workspace/vllm-lite
cargo build -p vllm-core
cargo clippy -p vllm-core --all-targets -- -D warnings
cargo test -p vllm-core --lib engine::speculative 2>&1 | tail -5
cargo clippy --workspace --all-targets -- -D warnings
just nextest
```

预期：
- 编译通过（含 CounterModel 实现）
- clippy 0 errors
- nextest ≥ 1036 passed（1035 + 1 新）

- [ ] **Step 6: Commit**

```bash
cd /workspace/vllm-lite
git add crates/core/src/engine/speculative.rs
git commit -m "test(engine): add fast unit test for speculative warmup_draft_kv

\`warmup_draft_kv\` previously had only \`#[ignore]\` e2e coverage. This
adds a fast unit test that directly constructs a Prefill batch and
verifies the draft model receives one forward() call per seq_id.

Changes:
- \`warmup_draft_kv\` visibility: \`fn\` -> \`pub(crate) fn\` for test access
- New \`CounterModel\` wrapper in engine test mod (delegates to inline
  FakeModel; counts forward/forward_logits calls via AtomicUsize)
- New test \`test_warmup_draft_kv_invokes_draft_per_sequence\`

Refs: docs/superpowers/specs/2026-06-26-wave4-warmup-test.md"
```

---

## Task 2: 同步 PROJECT.md / STATE.md / ROADMAP.md（1 commit）

**Files:**
- Modify: `.planning/PROJECT.md`
- Modify: `.planning/STATE.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: PROJECT.md v17.0 active 区更新**

In `.planning/PROJECT.md`, find the v17.0 Active section. Replace this 1 line:

```
- [ ] **SPEC-WARM-01**: Speculative warmup (prefill draft KV cache before decode)
```

With:
```
- [x] **SPEC-WARM-01**: Speculative warmup — `Engine::warmup_draft_kv` after prefill + `draft_kv_block_ids` tracking (commit `<Task 1 hash>`)
```

(Use `git log --oneline -1` to get actual hash.)

- [ ] **Step 2: PROJECT.md Last updated**

Find at end of file:
```
*Last updated: 2026-06-26 — Wave 2 SPEC-ADAPT counter wire-up + docs sync 完成；Wave 3 (Dependabot) 待启动*
```

Replace with:
```
*Last updated: 2026-06-26 — Wave 3 Dependabot 完成 + Wave 4 SPEC-WARM 测试覆盖；Wave 5 (SPEC-BENCH) 待启动*
```

- [ ] **Step 3: STATE.md current_focus + last_activity**

**Change A** (frontmatter):
```
// Replace date with current UTC (use `date -u +%FT%T.000Z`):
last_updated: "<current UTC timestamp>"
last_activity: 2026-06-26
```

**Change B** (Current Position section):
Find:
```
Wave: 2 of 5 (Wave 2: SPEC-ADAPT counter wire-up + doc sync)
Status: Wave 2 in progress; Wave 3–5 in pipeline
```
Replace with:
```
Wave: 4 of 5 (Wave 4: SPEC-WARM-01 测试覆盖 + doc sync)
Status: Wave 4 in progress; Wave 5 in pipeline
```

**Change C** (Project Reference section):
Find:
```
See: .planning/PROJECT.md (updated 2026-06-26)
**Current focus:** Wave 2 of 5 (SPEC-ADAPT-01/02 counter wire-up + doc sync) — Wave 3–5 在 pipeline
```
Replace with:
```
See: .planning/PROJECT.md (updated 2026-06-26)
**Current focus:** Wave 4 of 5 (SPEC-WARM-01 测试覆盖 + doc sync) — Wave 5 (SPEC-BENCH) 在 pipeline
```

- [ ] **Step 4: ROADMAP.md 补 Wave 4 callout**

In `ROADMAP.md`, find Phase 5 / 监控 section. Add callout after Wave 3's callout (search for "Wave 3 (Dependabot) 待启动" to find location):

```markdown
> 2026-06-26 更新：Wave 4 完成 SPEC-WARM-01 测试覆盖（`Engine::warmup_draft_kv` 改 `pub(crate)` + `CounterModel` wrapper + fast 单测 `test_warmup_draft_kv_invokes_draft_per_sequence`）。Wave 5 (SPEC-BENCH real-hardware benchmarks) 待启动。
```

If exact location unclear, insert at end of Phase 5 section before `## Phase 6:` heading.

- [ ] **Step 5: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat .planning/PROJECT.md .planning/STATE.md ROADMAP.md
# 预期: 3 files changed, ~10 insertions, ~6 deletions

cargo check --workspace  # sanity
git add .planning/PROJECT.md .planning/STATE.md ROADMAP.md
git commit -m "docs(planning): mark SPEC-WARM-01 complete in PROJECT/STATE/ROADMAP"
```

---

## Task 3: CHANGELOG 补 Wave 4 条目（1 commit）

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 在 `[Unreleased]` 段追加 Wave 4 子段**

Locate the `#### Adaptive Speculative Decoding Counter Wire-up (Wave 2, 2026-06-26)` subsection. Insert AFTER it (BEFORE `### Added (Phase 4)`):

```markdown
#### Speculative Warmup Test Coverage (Wave 4, 2026-06-26)

- `Engine::warmup_draft_kv` visibility relaxed from `fn` to `pub(crate) fn` for test access
- New `CounterModel` wrapper in `engine::speculative::tests` mod (counts forward/forward_logits calls via AtomicUsize)
- New fast unit test `test_warmup_draft_kv_invokes_draft_per_sequence` verifies draft model receives exactly N forward() calls for N-seq Prefill batch
- Documentation: `SPEC-WARM-01` marked complete in `.planning/PROJECT.md`

Refs: `docs/superpowers/specs/2026-06-26-wave4-warmup-test.md`

```

(Include trailing blank line before `### Added (Phase 4)`.)

- [ ] **Step 2: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat CHANGELOG.md
# 预期: 1 file changed, ~8 insertions

git add CHANGELOG.md
git commit -m "docs(core): add Wave 4 speculative warmup test to CHANGELOG"
```

---

## Task 4: 刷新 SESSION-HANDOFF.md（1 commit）

**Files:**
- Modify: `.planning/SESSION-HANDOFF.md`

- [ ] **Step 1: 更新顶部 Git 行**

Find:
```
> Git：`main` @ `4d58829` (Wave 1 + 1.6 + 2 + 3 全部完成：21 commits)
```

Replace with (use `git log --oneline -1` for actual hash):
```
> Git：`main` @ `<Wave 4 latest hash>` (Wave 1 + 1.6 + 2 + 3 + 4 全部完成：25 commits)
```

- [ ] **Step 2: 替换"下一优先级"段**

Find the section `## 下一优先级（2026-06-26，Wave 3 完成）`. Replace the entire section with:

```markdown
## 下一优先级（2026-06-26，Wave 4 完成）

**Wave 1 + 1.6 + 2 + 3 + 4 全部完成（25 commits）**

| Wave | Commit 范围 | 描述 |
|------|------------|------|
| 1 | `d42b151` ~ `1499fcd` | 文档同步 + dead_code 审计（11 commits） |
| 1.6 | `a4886a7` | 清理 vllm-model pre-existing clippy（11 lints） |
| 2 | `9e564f6` ~ `b5c587e` | SPEC-ADAPT counter wire-up + docs sync（5 commits） |
| 3 | `c93ba5e` ~ `2240065` | Dependabot bumps + SECURITY.md audit history（4 commits） |
| 4 | `55bc82d` ~ `<end>` | SPEC-WARM-01 测试覆盖 + doc sync（4 commits） |

**下一 Wave:** Wave 5 (SPEC-BENCH-01/02 real-hardware benchmarks)
- 多数 metrics 已就位（`speculative_acceptance_rate`, `efficiency`, `per_request_acceptance`, `adjustments_total`）
- 需 benchmark harness（`criterion` / 自定义）+ 报告模板
- 部分项目需 GPU 环境，可能仅文档化

**Wave 4 spec/plan:**
- Spec: `docs/superpowers/specs/2026-06-26-wave4-warmup-test.md` (commit `55bc82d`)
- Plan: `docs/superpowers/plans/2026-06-26-wave4-warmup-test.md` (本文件)
```

(Use `git log --oneline` to confirm hashes; adjust if drift.)

Also clean up the "已知差距" section if it references SPEC-WARM-01 as TODO:

Find:
```
### 中价值（Wave 2+ 处理）
```

Check if SPEC-WARM-01 is mentioned there; if yes, mark complete with ✅. (May not be present; verify by reading the file.)

- [ ] **Step 3: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat .planning/SESSION-HANDOFF.md
# 预期: 1 file changed, ~20 insertions, ~10 deletions

cargo check --workspace  # sanity
git add .planning/SESSION-HANDOFF.md
git commit -m "docs(planning): refresh SESSION-HANDOFF for Wave 4 status"
```

---

## 收口验证

所有 5 commits 完成后（spec 已 + 4 new = 5）：

```bash
cd /workspace/vllm-lite

# 1. 全量 CI
just ci

# 2. 新单测覆盖
cargo test -p vllm-core --lib engine::speculative::tests::test_warmup_draft_kv_invokes_draft_per_sequence 2>&1 | tail -3
# 预期: 1 passed

# 3. 文档一致性
rg "SPEC-WARM-01" .planning/PROJECT.md
# 预期: 1 行匹配，包含 [x]

rg "Wave 4" .planning/SESSION-HANDOFF.md
# 预期: 反映完成态

# 4. 测试基线
just nextest 2>&1 | tail -3
# 预期: ≥ 1036 passed (1035 + 1 新)
```

**Wave 4 完成标志：**
- ✅ `just ci` 全绿
- ✅ `just nextest` ≥ 1036 passed（+1 新单测）
- ✅ `warmup_draft_kv` 单测覆盖锁定调用契约
- ✅ PROJECT.md SPEC-WARM-01 标 [x]
- ✅ CHANGELOG 补 Wave 4 条目
- ✅ SESSION-HANDOFF 反映 Wave 4 完成

---

## 错误处理 / 风险

| 风险 | 缓解 |
|------|------|
| CounterModel 编译失败（漏实现某方法） | Step 5 cargo build 验证；可对照 ModelBackend trait 添加缺失方法 |
| Batch 字段类型错误 | 严格按 Step 3 代码构造；Batch 在 `vllm_traits::types` 中 |
| `pub(crate)` 暴露不期望 API | 仅 crate 内可见；不影响公开 API |
| 测试通过但 `step()` 中 warmup 实际未触发 | 已由现有 `#[ignore]` 测试覆盖 e2e；fast 单测锁 warmup_draft_kv 内部行为 |

---

## 自审

- **Spec 覆盖:** ✅ D4-1 (CounterModel) → Task 1 Step 2；D4-2 (pub(crate)) → Task 1 Step 1；D4-3 (新测试) → Task 1 Step 3；D4-4 (doc sync) → Tasks 2/3/4
- **占位符扫描:** ✅ 无 TBD/TODO；每处有具体 before/after
- **类型一致性:** ✅ CounterModel 委托所有方法给 inner FakeModel；Batch 字段类型与现有测试一致
- **范围:** ✅ 4 commits（spec 已 1 + 1 code + 3 docs），单次会话可完成

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-26 | 初版：基于 `docs/superpowers/specs/2026-06-26-wave4-warmup-test.md` |
