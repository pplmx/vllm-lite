# Wave 4: SPEC-WARM-01 测试覆盖 + 文档同步设计

**日期**: 2026-06-26
**状态**: 🔄 待审
**基线**: `main @ 2240065`（Wave 1 + 1.6 + 2 + 3 全部完成）
**关联**: `.planning/PROJECT.md` v17.0 SPEC-WARM-01

---

## 背景

### 现状（2026-06-26 代码探索发现）

**SPEC-WARM-01 实现已完整**：

| 组件 | 位置 | 状态 |
|------|------|------|
| `warmup_draft_kv` 方法 | `crates/core/src/engine/speculative.rs:8-33` | ✅ |
| `step_speculative_inner` 调用 warmup | `speculative.rs:46-51`（prefill 后调用） | ✅ |
| `draft_kv_block_ids` tracking | `crates/core/src/speculative/self_spec.rs:17,29,49` | ✅ |
| `clear_draft_kv` per-seq reset | `self_spec.rs:53-58` | ✅ |
| Plan 注释引用 | `17.4-A`, `17.4-E` 已写在源码 | ✅ |
| 端到端 `#[ignore]` 测试 | `test_batched_draft_generation`, `test_logit_verification_exact_match` | ✅ 但慢 |

### 唯一 Gap：缺 fast unit test

`warmup_draft_kv` 本身没有 fast 单测。`#[ignore]` 测试可端到端验证，但不在默认 `just nextest` 中。每次 refactor 都可能误删 `warmup_draft_kv` 调用而无 fast 反馈。

### 文档同步 Gap（同 Wave 2 模式）

- `PROJECT.md` SPEC-WARM-01 仍标 `[ ]`
- `STATE.md` / `ROADMAP.md` 未反映 Wave 4
- `CHANGELOG.md` 无 Wave 4 条目
- `SESSION-HANDOFF.md` Wave 4 状态缺失

---

## 目标

1. **添加 1 个 fast unit test** 锁定 `warmup_draft_kv` 在 prefill 时被调用
2. **同步文档**：PROJECT.md / CHANGELOG.md / SESSION-HANDOFF.md
3. **不引入 scope creep**：不动 warmup_draft_kv 自身逻辑

**非目标：**

- 不重写 warmup_draft_kv 实现
- 不改 FakeModel 的 public API（`crates/testing/src/mocks/mod.rs`）
- 不加 #[ignore] 端到端测试（已有）
- 不改 self_spec 的 draft_kv_block_ids 逻辑
- 不动 scheduler.batch.rs 的 prefill 触发条件

---

## 设计

### D4-1：内联 `CounterModel` wrapper（在 engine test module）

**决策：** 在 `crates/core/src/engine/speculative.rs` 的现有 `mod tests` 内（line ~335 附近，已有 inline `FakeModel`）新增 `CounterModel` 结构：

```rust
struct CounterModel {
    inner: FakeModel,
    forward_count: AtomicUsize,
}

impl CounterModel {
    fn new(token: TokenId) -> Self {
        Self {
            inner: FakeModel::new(token),
            forward_count: AtomicUsize::new(0),
        }
    }
    fn forward_count(&self) -> usize {
        self.forward_count.load(std::sync::atomic::Ordering::Relaxed)
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
    ) -> Result<BatchOutput> {
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
    ) -> Result<Vec<Vec<f32>>> {
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
    // embed / forward_with_cache 委托给 inner
}
```

**理由：**
- Wrapper 而非修改现有 FakeModel：保持 `crates/testing/src/mocks/mod.rs` 不变（避免影响其他 crate 的测试）
- `AtomicUsize` 而非 `Cell<usize>`：因为 `&mut self` 在 forward 中无法在 test 断言里读取；用 atomic 允许松散读
- 仅在 engine 的 inline test mod 中可见：`#[cfg(test)] mod tests`

### D4-2：`warmup_draft_kv` 可见性

**决策：** 将 `fn warmup_draft_kv(&mut self, batch: &Batch) -> Result<()>` 改为 `pub(crate) fn warmup_draft_kv(...)`。

**理由：**
- 当前是 `fn`（私有），test 无法直接调用
- `pub(crate)` 允许 test 通过 `super::super::Engine` 访问，但不影响 crate 外 API
- 比 `#[cfg(test)] pub` 更标准

### D4-3：新 fast unit test

**决策：** 在 `mod tests` 中添加 `test_warmup_draft_kv_invokes_draft_per_sequence`：

```rust
#[test]
fn test_warmup_draft_kv_invokes_draft_per_sequence() {
    // Setup: engine with CounterModel as draft
    let target = FakeModel::new(42);
    let draft = CounterModel::new(42);
    let draft_count_before = draft.forward_count();
    let mut engine = super::super::Engine::new_boxed(
        Box::new(target),
        Some(Box::new(draft)),
    );
    engine.enable_speculative();

    // Build a batch with 3 sequences (Prefill phase)
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

    // Execute warmup
    engine.warmup_draft_kv(&batch).expect("warmup should succeed");

    // Verify: draft model forward() called once per sequence in batch
    let calls = draft.forward_count() - draft_count_before;
    assert_eq!(
        calls, 3,
        "warmup_draft_kv should invoke draft.forward() once per seq_id"
    );
}
```

**理由：**
- 不需要完整 `add_request` + `step` 流程；直接构造 `Batch` 调 `warmup_draft_kv`
- 测试 warmup_draft_kv **内部行为**（forward 调用次数），不依赖 step() 流程
- 与现有 `#[ignore]` 端到端测试互补：fast 单测 + 慢 e2e

### D4-4：文档同步（4 files）

| 文件 | 变更 |
|------|------|
| `.planning/PROJECT.md` | SPEC-WARM-01 标 `[x]`，加 commit 引用 |
| `.planning/STATE.md` | current_focus 改为 Wave 4 |
| `ROADMAP.md` | 补 Wave 4 callout |
| `CHANGELOG.md` | `[Unreleased]` 段补 Wave 4 条目 |
| `.planning/SESSION-HANDOFF.md` | 下一优先级改为 Wave 5 (SPEC-BENCH) |

---

## 目标目录结构（无变化）

Wave 4 仅修改：
- `crates/core/src/engine/speculative.rs`（~80 行：pub(crate) + CounterModel + 1 test）
- 5 个 doc 文件

---

## 任务分解

### Wave 4 Task 1：spec doc（本文件）

### Wave 4 Task 2：counter + test（1 commit）

```bash
# 修改 crates/core/src/engine/speculative.rs:
# 1. fn warmup_draft_kv -> pub(crate) fn warmup_draft_kv
# 2. 在 mod tests 内加 CounterModel struct + impl ModelBackend
# 3. 加 test_warmup_draft_kv_invokes_draft_per_sequence

cargo test -p vllm-core --lib engine::speculative::tests::test_warmup_draft_kv_invokes_draft_per_sequence
# 预期: 1 passed

cargo clippy -p vllm-core --all-targets -- -D warnings
just nextest

git commit -m "test(engine): add fast unit test for speculative warmup_draft_kv"
```

### Wave 4 Task 3：PROJECT/STATE/ROADMAP 同步（1 commit）

### Wave 4 Task 4：CHANGELOG（1 commit）

### Wave 4 Task 5：SESSION-HANDOFF 刷新（1 commit）

---

## 验证

### Task 2 验证

```bash
cargo test -p vllm-core --lib engine::speculative::tests 2>&1 | tail -5
# 预期: 1 个新 test passed；现有 #[ignore] 测试不跑（默认）

cargo clippy --workspace --all-targets -- -D warnings
# 预期: 0 errors

just nextest
# 预期: ≥ 1036 passed (1035 + 1 new), 46 skipped
```

### 收口验证

```bash
# 文档一致性
rg "SPEC-WARM-01" .planning/PROJECT.md
# 预期: 1 行匹配，包含 [x]

rg "Wave 4" .planning/SESSION-HANDOFF.md
# 预期: 反映完成态

# Test 覆盖
cargo test -p vllm-core warmup_draft_kv
# 预期: 1 个新 test 通过
```

---

## 错误处理 / 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| `pub(crate)` 改动暴露不期望的 API | 极低 | 低 | 仅测试访问；不增加公开 API 表面 |
| CounterModel 实现遗漏某些 ModelBackend 方法导致编译失败 | 中 | 中 | 委托所有方法到 inner FakeModel；编译验证 |
| Batch 构造错误（字段类型不匹配） | 低 | 低 | 直接参考现有测试构造方式 |
| 现有 #[ignore] 测试因 CounterModel 干扰而失败 | 极低 | 低 | CounterModel 仅在 test mod；现有测试用 inline FakeModel |

---

## 不做（明确边界）

- ❌ 不改 `crates/testing/src/mocks/mod.rs`（保持上游不动）
- ❌ 不改 `warmup_draft_kv` 内部逻辑（仅 pub(crate) 化）
- ❌ 不加 #[ignore] e2e 测试（已覆盖）
- ❌ 不改 scheduler.batch.rs
- ❌ 不动 self_spec.rs
- ❌ 不重写 CounterModel 为公开 struct（仅 inline test mod）

---

## 风险与决策记录

| ID | 决策 | 理由 | 日期 |
|----|------|------|------|
| D4-1 | 内联 CounterModel wrapper | 不改 crates/testing/；atomic 计数允许松散读 | 2026-06-26 |
| D4-2 | `warmup_draft_kv` 改 `pub(crate)` | 允许 test 直接调；不暴露给 crate 外 | 2026-06-26 |
| D4-3 | 直接构造 Batch 测内部行为 | 不依赖 step() 流程；fast 单测 | 2026-06-26 |
| D4-4 | 4 doc files 同步 | Wave 1/2/3 同模式 | 2026-06-26 |
| D4-5 | 1+4 = 5 commits | code + 4 docs | 2026-06-26 |

---

## 会话接续

Wave 4 完成后：
1. SESSION-HANDOFF（已更新）确认 Wave 4 状态
2. 下一 Wave 候选：Wave 5 (SPEC-BENCH-01/02 real-hardware benchmarks)

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-26 | 初版：基于对 SPEC-WARM-01 实现现状的代码探索 |
