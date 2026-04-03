# KV Cache Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect Scheduler's KV block information to Model's forward() method, enabling PagedAttention and PrefixCache to work correctly.

**Architecture:** Extend ModelBackend trait to receive KV cache block IDs, update Batch struct to carry block information, modify Qwen3Model to use forward_with_cache() for real PagedAttention.

**Tech Stack:** Rust, Candle (ML framework), vLLM-lite crates (traits, core, model)

---

## File Structure

```text
crates/traits/src/
├── types.rs          # MODIFY: Add Batch fields (kv_block_ids, num_computed_tokens, is_prefill)
├── model.rs          # MODIFY: Extend ModelBackend trait with new parameters

crates/core/src/
├── types.rs          # CHECK: Re-export changes from traits
├── scheduler.rs      # MODIFY: Populate new Batch fields in build_batch()
├── engine.rs         # MODIFY: Change Arc<M> to RefCell<Box<M>>
├── engine/batch.rs   # MODIFY: Pass new parameters to model.forward()

crates/model/src/
├── qwen3/model.rs    # MODIFY: Implement KV cache usage in forward()
├── fake.rs           # MODIFY: Update FakeModel to match new trait signature

tests/
# Add integration tests for KV cache end-to-end
```

---

## Task 1: Extend Batch Structure in traits

**Files:**

- Modify: `crates/traits/src/types.rs`
- Test: Run existing tests to verify no breakage

- [ ] **Step 1: Read current types.rs**

Run: `cat crates/traits/src/types.rs`

- [ ] **Step 2: Add new fields to Batch struct**

Edit `crates/traits/src/types.rs` - add after existing fields:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    pub seq_ids: Vec<SeqId>,
    pub input_tokens: Vec<Vec<TokenId>>,
    pub positions: Vec<Vec<usize>>,
    // NEW: KV cache information
    pub kv_block_ids: Vec<Vec<usize>>,     // Block IDs per sequence
    pub num_computed_tokens: Vec<usize>,   // Already computed tokens
    pub is_prefill: Vec<bool>,             // Prefill vs decode
}
```

- [ ] **Step 3: Add helper method**

Add to impl Batch:

```rust
impl Batch {
    pub fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }

    // NEW: Helper to check if batch has any prefill sequences
    pub fn has_prefill(&self) -> bool {
        self.is_prefill.iter().any(|&p| p)
    }

    // NEW: Helper to check if batch has any decode sequences
    pub fn has_decode(&self) -> bool {
        self.is_prefill.iter().any(|&p| !p)
    }
}
```

- [ ] **Step 4: Run tests to verify no breakage**

Run: `cargo test -p vllm-traits`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/traits/src/types.rs
git commit -m "feat(traits): add kv_block_ids, num_computed_tokens, is_prefill to Batch"
```

---

## Task 2: Extend ModelBackend Trait

**Files:**

- Modify: `crates/traits/src/model.rs`

- [ ] **Step 1: Read current model.rs**

Run: `cat crates/traits/src/model.rs`

- [ ] **Step 2: Update trait signature**

Replace the entire trait:

```rust
pub trait ModelBackend: Send + Sync {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> Result<BatchOutput>;

    fn forward_logits(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>>;
}
```

- [ ] **Step 3: Run cargo check to see affected files**

Run: `cargo check --workspace 2>&1 | head -50`
Expected: Errors showing files that need updating

- [ ] **Step 4: Commit**

```bash
git add crates/traits/src/model.rs
git commit -m "feat(traits): extend ModelBackend with KV cache parameters"
```

---

## Task 3: Update Qwen3Model to Implement New Trait

**Files:**

- Modify: `crates/model/src/qwen3/model.rs` (lines 316-389)
- Test: Run model tests

- [ ] **Step 1: Read the current forward implementation**

Run: `sed -n '316,389p' crates/model/src/qwen3/model.rs`

- [ ] **Step 2: Replace forward() implementation**

Replace the entire `impl ModelBackend for Qwen3Model` block:

```rust
impl ModelBackend for Qwen3Model {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> EngineResult<BatchOutput> {
        if seq_ids.is_empty() {
            return Ok(BatchOutput {
                seq_ids: vec![],
                next_tokens: vec![],
            });
        }

        // Group indices by prefill/decode status
        let mut prefill_indices: Vec<usize> = vec![];
        let mut decode_indices: Vec<usize> = vec![];

        for (i, &is_pf) in is_prefill.iter().enumerate() {
            if is_pf {
                prefill_indices.push(i);
            } else {
                decode_indices.push(i);
            }
        }

        let mut next_tokens = vec![0u32; seq_ids.len()];

        // Process prefill sequences
        if !prefill_indices.is_empty() {
            for &idx in &prefill_indices {
                let tokens = &input_tokens[idx];
                let pos = &positions[idx];
                let blocks = &kv_block_ids[idx];
                let computed = num_computed_tokens[idx];

                let (logits, _) = self.forward_with_cache(
                    tokens,
                    computed,
                    blocks,
                    pos,
                    true, // is_prefill
                )?;

                use candle_core::D;
                let next = logits.argmax(D::Minus1)?.to_vec1::<u32>()?[0];
                next_tokens[idx] = next;
            }
        }

        // Process decode sequences
        if !decode_indices.is_empty() {
            for &idx in &decode_indices {
                let tokens = &input_tokens[idx];
                let pos = &positions[idx];
                let blocks = &kv_block_ids[idx];
                let computed = num_computed_tokens[idx];

                let (logits, _) = self.forward_with_cache(
                    tokens,
                    computed,
                    blocks,
                    pos,
                    false, // is_decode
                )?;

                use candle_core::D;
                let next = logits.argmax(D::Minus1)?.to_vec1::<u32>()?[0];
                next_tokens[idx] = next;
            }
        }

        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }

    fn forward_logits(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> EngineResult<Vec<Vec<f32>>> {
        // For now, process each sequence individually
        let mut all_logits = Vec::new();
        let vocab_size = self.config.vocab_size();

        for (i, tokens) in input_tokens.iter().enumerate() {
            if tokens.is_empty() {
                all_logits.push(vec![0.0; vocab_size]);
                continue;
            }

            // Use last token for position
            let pos = if i < positions.len() {
                positions[i].last().copied().unwrap_or(0)
            } else {
                0
            };

            let blocks = kv_block_ids.get(i).map(|b| b.as_slice()).unwrap_or(&[]);
            let computed = num_computed_tokens.get(i).copied().unwrap_or(0);
            let is_pf = is_prefill.get(i).copied().unwrap_or(true);

            let (logits, _) = self.forward_with_cache(
                tokens,
                computed,
                blocks,
                &[pos],
                is_pf,
            )?;

            // Get logits for last token
            let last_logits: Vec<f32> = logits
                .squeeze(0)?
                .to_vec1::<f32>()?;

            all_logits.push(last_logits);
        }

        Ok(all_logits)
    }
}
```

- [ ] **Step 3: Run cargo check**

Run: `cargo check -p vllm-model 2>&1`
Expected: Should compile (may have warnings)

- [ ] **Step 4: Run tests**

Run: `cargo test -p vllm-model 2>&1`
Expected: Tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/qwen3/model.rs
git commit -m "feat(model): implement KV cache usage in Qwen3Model::forward"
```

---

## Task 4: Update FakeModel

**Files:**

- Modify: `crates/model/src/fake.rs`
- Test: Run tests

- [ ] **Step 1: Read fake.rs**

Run: `cat crates/model/src/fake.rs`

- [ ] **Step 2: Update FakeModel to implement new trait**

Replace with:

```rust
use vllm_traits::{BatchOutput, ModelBackend, Result, SeqId, TokenId};

pub struct FakeModel;

impl ModelBackend for FakeModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        let next_tokens: Vec<TokenId> = seq_ids
            .iter()
            .map(|&id| (id % 10000) as TokenId)
            .collect();

        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }

    fn forward_logits(
        &self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| tokens.iter().map(|_| 0.0).collect())
            .collect())
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p vllm-model 2>&1 | head -30`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/fake.rs
git commit -m "feat(model): update FakeModel to match new ModelBackend trait"
```

---

## Task 5: Update Scheduler to Populate Batch Fields

**Files:**

- Modify: `crates/core/src/scheduler.rs` (specifically build_batch method ~line 331)
- Test: Run scheduler tests

- [ ] **Step 1: Find build_batch method**

Run: `grep -n "pub fn build_batch" crates/core/src/scheduler.rs`
Result shows line number (around 331)

- [ ] **Step 2: Read current build_batch implementation**

Run: `sed -n '331,360p' crates/core/src/scheduler.rs`

- [ ] **Step 3: Add HashMap import if not present**

Run: `head -10 crates/core/src/scheduler.rs | grep "use std::collections"`
If not found, add: `use std::collections::HashMap;`

- [ ] **Step 4: Update build_batch to populate new fields**

Find the return statement in build_batch and replace the Batch construction. The key is after seq_ids, input_tokens, positions are built, add:

```rust
// Build index mapping to maintain order
let batch_len = seq_ids.len();
let seq_id_to_idx: HashMap<SeqId, usize> = seq_ids
    .iter()
    .enumerate()
    .map(|(i, &id)| (id, i))
    .collect();

// Pre-allocate vectors
let mut kv_block_ids: Vec<Vec<usize>> = vec![vec![]; batch_len];
let mut num_computed_tokens: Vec<usize> = vec![0; batch_len];
let mut is_prefill: Vec<bool> = vec![false; batch_len];

// Populate using index mapping
for seq in &self.running {
    if let Some(&idx) = seq_id_to_idx.get(&seq.id) {
        kv_block_ids[idx] = seq.kv_blocks.as_ref().clone();
        num_computed_tokens[idx] = seq.num_computed_tokens;
        is_prefill[idx] = seq.status == Status::Prefilling;
    }
}

// Update the Batch construction return
Batch {
    seq_ids,
    input_tokens,
    positions,
    kv_block_ids,
    num_computed_tokens,
    is_prefill,
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p vllm-core -- scheduler 2>&1 | tail -30`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "feat(core): populate KV cache fields in Scheduler::build_batch"
```

---

## Task 6: Update Engine to Use RefCell<Box<M>>

**Files:**

- Modify: `crates/core/src/engine.rs`
- Test: Run engine tests

- [ ] **Step 1: Read Engine struct definition**

Run: `sed -n '14,25p' crates/core/src/engine.rs`

- [ ] **Step 2: Add RefCell import**

Add at top of file: `use std::cell::RefCell;`

- [ ] **Step 3: Update Engine struct**

Replace target_model and draft_model:

```rust
pub struct Engine<M: ModelBackend> {
    pub scheduler: Scheduler,
    pub target_model: RefCell<Box<M>>,  // Changed from Arc<M>
    pub draft_model: RefCell<Box<M>>,   // Changed from Arc<M>
    pub max_draft_tokens: usize,
    pub speculative_mode: bool,
    pub error_count: usize,
    pub last_error: Option<String>,
    pub metrics: MetricsCollector,
    response_txs: HashMap<SeqId, mpsc::UnboundedSender<TokenId>>,
}
```

- [ ] **Step 4: Update with_config to use Box::new**

Find and update:

```rust
Self {
    scheduler: Scheduler::with_config(config, num_kv_blocks),
    target_model: RefCell::new(Box::new(target_model)),  // Changed
    draft_model: RefCell::new(Box::new(draft_model)),    // Changed
    max_draft_tokens,
    speculative_mode: false,
    error_count: 0,
    last_error: None,
    metrics: MetricsCollector::new(),
    response_txs: HashMap::with_capacity(max_seqs),
}
```

- [ ] **Step 5: Run cargo check**

Run: `cargo check -p vllm-core 2>&1 | head -30`

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/engine.rs
git commit -m "refactor(core): change Engine to use RefCell<Box<M>> for interior mutability"
```

---

## Task 7: Update Engine Batch Step to Pass New Parameters

**Files:**

- Modify: `crates/core/src/engine/batch.rs`
- Test: Run engine tests

- [ ] **Step 1: Read current batch.rs**

Run: `cat crates/core/src/engine/batch.rs`

- [ ] **Step 2: Update step() method**

Replace the forward call:

```rust
pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
    let start = std::time::Instant::now();
    let batch = self.scheduler.build_batch();
    if batch.is_empty() {
        return Ok(vec![]);
    }

    let output = self.target_model.borrow_mut().forward(
        &batch.seq_ids,
        &batch.input_tokens,
        &batch.positions,
        &batch.kv_block_ids,
        &batch.num_computed_tokens,
        &batch.is_prefill,
    )?;

    // ... rest of method unchanged
}
```

- [ ] **Step 3: Also update step_speculative if it exists**

Run: `grep -n "fn step_speculative" crates/core/src/engine/speculative.rs`
If exists, update similarly.

- [ ] **Step 4: Run tests**

Run: `cargo test -p vllm-core 2>&1 | tail -30`
Expected: Tests should pass

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/engine/batch.rs
git commit -m "feat(core): pass KV cache parameters in Engine::step"
```

---

## Task 8: Run Full Integration Test

**Files:**

- Test: Run full workspace tests

- [ ] **Step 1: Run full test suite**

Run: `cargo test --workspace 2>&1 | tail -50`
Expected: Most tests pass (some may need updates)

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings 2>&1 | tail -30`

- [ ] **Step 3: Fix any compilation errors**

If errors occur, fix them one by one.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: integrate KV cache into Model::forward for PagedAttention"
```

---

## Verification

After completing all tasks, verify:

1. **KV Cache Integration Works**:
    - PagedAttention reads/writes blocks correctly
    - Prefix cache hits work

2. **Tests Pass**:
    - `cargo test --workspace` passes
    - `cargo clippy --workspace -- -D warnings` passes

3. **No Regression**:
    - Existing functionality still works
    - OpenAI API endpoints work correctly

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-02-kv-cache-integration-plan.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
