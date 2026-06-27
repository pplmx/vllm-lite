# v24.0 Phase B — Unwrap Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the 6 production `unwrap()`/`expect()` calls that mask real errors into typed error variants with `?` propagation, and add `// invariant:` comments to the 51 production sites that represent legitimate invariants.

**Architecture:** Two prongs: (a) Bug fixes — 6 sites where the unwrap can panic in production get converted to typed errors with new variants on existing error enums; (b) Documentation — 51 sites get a `// invariant:` comment immediately above the call so a future reader understands why the panic is acceptable. The codebase already uses `?` propagation correctly for the vast majority of fallible operations; Phase B is surgical cleanup.

**Tech Stack:** Rust 2024 edition, thiserror for error enums, tracing for diagnostic logging.

**Spec:** [`docs/superpowers/specs/2026-06-28-v24-code-quality-hardening-design.md`](../specs/2026-06-28-v24-code-quality-hardening-design.md) §5 (revised 2026-06-28)

**Audit source:** `/tmp/phase_b_audit/SUMMARY.md` and 15 per-file audit reports under `/tmp/phase_b_audit/`.

**Sister plans:**
- Phase A — Lint Baseline: [`2026-06-28-v24-phase-a-lint-baseline.md`](2026-06-28-v24-phase-a-lint-baseline.md) ✅ shipped
- Phase C — API Ergonomics: not yet written
- Phase D — Module Boundaries: not yet written

---

## File Structure

| File | Change | Task |
|------|--------|------|
| `crates/model/src/kernels/cuda_graph/executor.rs` | Add typed error variant, replace unwrap | B-1 |
| `crates/core/src/engine.rs` | Add error variant, replace unwrap at :565 | B-2 |
| `crates/server/src/main.rs` | Convert 3 unwraps at :242, :324, :330 | B-2 |
| `crates/server/src/openai/batch/handler.rs` | Convert unwrap at :42 | B-2 |
| `crates/core/src/speculative/registry/lifecycle.rs` | Add `// invariant:` to 13 sites | B-3a |
| `crates/core/src/speculative/registry/loader.rs` | Add `// invariant:` to 4 sites | B-3a |
| `crates/model/src/arch/registry.rs` | Add `// invariant:` to 1 site | B-3a |
| `crates/core/src/engine/spec_dispatch/drafts.rs` | Add `// invariant:` to 1 site | B-3a |
| `crates/dist/src/grpc.rs` | Add `// invariant:` to 1 site | B-3b |
| `crates/dist/src/distributed_kv/cache.rs` | Add `// invariant:` to 1 site | B-3b |
| `crates/dist/src/distributed_kv/protocol.rs` | Add `// invariant:` to 1 site | B-3b |
| `crates/server/src/openai/batch/types.rs` | Add `// invariant:` to 1 site | B-3b |
| `crates/server/src/openai/batch/handler.rs` | Add `// invariant:` to 1 site | B-3b |
| `crates/server/src/openai/batch/manager.rs` | Add `// invariant:` to 1 site | B-3b |
| `crates/server/src/security/correlation.rs` | Add `// invariant:` to 1 site | B-3b |
| `crates/server/src/main.rs` | Add `// invariant:` to 1 (non-CONVERT) site | B-3b |
| `crates/model/src/gemma4/attention.rs` | Comments already adequate; verify only | B-3c |
| `crates/model/src/kv_cache.rs` | Add `// invariant:` to 1 site | B-3c |
| `crates/model/src/qwen3_5/block/linear.rs` | Add `// invariant:` to 1 site | B-3c |
| `crates/model/src/components/ssm.rs` | Add `// invariant:` to 1 site | B-3c |
| `crates/server/src/main.rs` | Add `// invariant:` to 2 (signal handler) sites | B-3c |
| `crates/core/src/engine.rs` | Add `// invariant:` to 2 (`duplicate draft id`) sites | B-3c |
| `crates/core/src/engine/spec_dispatch/drafts.rs` | Add `// invariant:` to 1 (caller contract) site | B-3c |
| `crates/server/src/main.rs` | Add `// invariant:` to 1 (`vram_budget_bytes validated`) site | B-3c |
| `crates/server/src/openai/chat.rs` | Add `// invariant:` to 2 (serialize) sites | B-3c |
| `crates/model/src/causal_lm/hybrid_lm.rs` | Add `// invariant:` to 1 site | B-3c |
| `crates/dist/build.rs` | Add `// invariant:` to 2 sites | B-3c |
| `crates/core/src/speculative/memory_budget.rs` | Add `// invariant:` to 1 site | B-3c |
| `crates/server/src/auth.rs` | Add `// invariant:` to 1 site | B-3c |
| `crates/server/src/api.rs` | Add `// invariant:` to 1 site | B-3c |
| `crates/core/src/circuit_breaker/strategy.rs` | Add `// invariant:` to 1 site | B-3c |
| `CHANGELOG.md` | Add Phase B entry | Final |
| `AGENTS.md` | Add "Invariant Comments" subsection to Lint Policy | Final |

---

## Task 1 (B-1): Fix `cuda_graph/executor.rs:222` Real Bug

This is the only CONVERT site with a real stability risk. `find_graph_key(batch_size)` may return a remapped key that isn't in `self.graphs` (race / cache eviction / edge case), causing a production panic.

**Files:**
- Modify: `/workspace/vllm-lite/crates/model/src/kernels/cuda_graph/executor.rs`

- [ ] **Step 1: Read the executor and its existing error type**

```bash
cat /workspace/vllm-lite/crates/model/src/kernels/cuda_graph/executor.rs
```

Identify:
- What error enum already exists (likely `CudaGraphError` or similar)
- The exact code at line 222
- The function signature (does it already return `Result`?)

- [ ] **Step 2: Add a `GraphNotFound` variant if missing**

If the existing error enum does not have a variant like `GraphNotFound { key: GraphKey }`, add one:

```rust
#[error("CUDA graph not found for key {key:?}")]
GraphNotFound { key: GraphKey },
```

If the function returns `Result<_, ExistingError>` already, no enum changes needed.

- [ ] **Step 3: Replace the unwrap at line 222**

Before:
```rust
let graph = self.graphs.get(&graph_key).unwrap();
```

After:
```rust
let graph = self.graphs
    .get(&graph_key)
    .ok_or_else(|| CudaGraphError::GraphNotFound { key: graph_key.clone() })?;
```

(Adjust error variant name and `GraphKey` type to match the actual types in the file.)

- [ ] **Step 4: Add a tracing log when the error is constructed**

If the surrounding function has a tracing context, add:

```rust
tracing::warn!(?graph_key, "CUDA graph cache miss; key remapped but not found");
```

This aids debugging without changing semantics.

- [ ] **Step 5: Write a negative unit test**

Find or create a `#[cfg(test)] mod tests` block in the executor. Add a test that:
- Constructs an executor with no graphs cached
- Calls the function with a key that won't be in the cache (or forces the `find_graph_key` mapping to return a non-existent key)
- Asserts the error is `GraphNotFound`

```rust
#[test]
fn execute_returns_graph_not_found_when_key_missing() {
    let executor = CudaGraphExecutor::new_for_test();
    let result = executor.execute_with_key(GraphKey::sentinel());
    assert!(matches!(result, Err(CudaGraphError::GraphNotFound { .. })));
}
```

(Adjust test scaffolding to match existing test patterns in the file.)

- [ ] **Step 6: Verify the change compiles and the test passes**

```bash
cargo build -p vllm-model --features cuda 2>&1 | tail -10
cargo test -p vllm-model --features cuda --lib cuda_graph 2>&1 | tail -20
```

Expected: builds clean, the new test passes (existing tests pass too).

- [ ] **Step 7: Commit**

```bash
git add crates/model/src/kernels/cuda_graph/executor.rs
git commit -m "fix(model): convert cuda_graph executor unwrap to typed error (race condition)"
```

---

## Task 2 (B-2): Convert Remaining 5 CONVERT Sites

Five production unwraps that mask real errors but aren't stability bugs (they fail at server startup or HTTP request boundaries, so the process exits either way, but the error reporting improves).

**Files:**
- Modify: `/workspace/vllm-lite/crates/core/src/engine.rs` (1 site at :565)
- Modify: `/workspace/vllm-lite/crates/server/src/main.rs` (3 sites at :242, :324, :330)
- Modify: `/workspace/vllm-lite/crates/server/src/openai/batch/handler.rs` (1 site at :42)

### 2a: `engine.rs:565` — Empty beam list

- [ ] **Step 1: Read the surrounding code**

```bash
sed -n '550,580p' /workspace/vllm-lite/crates/core/src/engine.rs
```

- [ ] **Step 2: Add or use an `EmptyBeamList` variant on `EngineError`**

If `EngineError` doesn't have such a variant, add:

```rust
#[error("beam search produced no candidate beams")]
EmptyBeamList,
```

- [ ] **Step 3: Replace the unwrap**

Before:
```rust
let best = beams.into_iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
```

After (adjust based on actual code shape):
```rust
let best = beams
    .into_iter()
    .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    .ok_or(EngineError::EmptyBeamList)?;
```

If the function signature returns `Result`, this works directly. If not, propagate up via `?`.

- [ ] **Step 4: Add a negative test**

In `engine.rs`'s existing `#[cfg(test)] mod tests`, add a test that calls the relevant function with an empty beam list and asserts `EngineError::EmptyBeamList`.

### 2b: `server/src/main.rs` — 3 sites

- [ ] **Step 1: Read main.rs around the 3 sites**

```bash
sed -n '235,250p; 318,335p' /workspace/vllm-lite/crates/server/src/main.rs
```

- [ ] **Step 2: Convert `:242` (tokenizer path)**

Before:
```rust
let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
```

After:
```rust
let tokenizer = Tokenizer::from_file(tokenizer_path)
    .map_err(|e| format!("failed to load tokenizer from {tokenizer_path:?}: {e}"))?;
```

Or if `main()` can return `Result<(), Box<dyn Error>>`:

```rust
let tokenizer = Tokenizer::from_file(tokenizer_path)?;
```

(Adjust to match main.rs's existing error handling pattern.)

- [ ] **Step 3: Convert `:324` and `:330` (TCP listener / axum serve)**

Before:
```rust
let listener = TcpListener::bind(addr).await.unwrap();
axum::serve(listener, app).await.unwrap();
```

After:
```rust
let listener = TcpListener::bind(&addr).await
    .map_err(|e| format!("failed to bind to {addr}: {e}"))?;
axum::serve(listener, app).await
    .map_err(|e| format!("server crashed: {e}"))?;
```

If main returns `Result`, use `?` propagation instead of `.map_err`. Match main.rs's existing style.

### 2c: `server/src/openai/batch/handler.rs:42` — Job not found

- [ ] **Step 1: Read the handler**

```bash
sed -n '35,60p' /workspace/vllm-lite/crates/server/src/openai/batch/handler.rs
```

- [ ] **Step 2: Convert to map to 404 error**

Before:
```rust
let job = self.manager.get_job(&job_id).await.unwrap();
```

After:
```rust
let job = self.manager.get_job(&job_id).await
    .ok_or_else(|| BatchError::NotFound { job_id: job_id.clone() })?;
```

(Adjust `BatchError` variant to match existing enum.)

- [ ] **Step 3: Add a negative test**

In `handler.rs`'s existing `#[cfg(test)] mod tests`, add a test that calls the handler with a non-existent job_id and asserts a 404 / `BatchError::NotFound`.

### Final steps for Task 2

- [ ] **Verify compilation**

```bash
cargo build --workspace --all-features 2>&1 | tail -10
```

Expected: clean build, no errors.

- [ ] **Run affected tests**

```bash
cargo test -p vllm-core --lib engine 2>&1 | tail -10
cargo test -p vllm-server --lib main openai::batch 2>&1 | tail -10
```

Expected: existing tests pass, new negative tests pass.

- [ ] **Commit (single commit covering all 3 files)**

```bash
git add crates/core/src/engine.rs crates/server/src/main.rs crates/server/src/openai/batch/handler.rs
git commit -m "fix(core,server): convert 5 production unwraps to typed errors (B-2)"
```

---

## Task 3 (B-3a): Add `// invariant:` Comments to RwLock/Mutex Cluster (21 sites)

21 `.expect("...poisoned")` calls on RwLock/Mutex across 5 files. The pattern is uniform; a single batch commit is appropriate.

**Files:**
- Modify: `/workspace/vllm-lite/crates/core/src/speculative/registry/lifecycle.rs` (13 sites)
- Modify: `/workspace/vllm-lite/crates/core/src/speculative/registry/loader.rs` (4 sites)
- Modify: `/workspace/vllm-lite/crates/model/src/arch/registry.rs` (1 site at :32)
- Modify: `/workspace/vllm-lite/crates/core/src/engine/spec_dispatch/drafts.rs` (1 site at :86)
- Modify: `/workspace/vllm-lite/crates/server/src/main.rs` (2 sites at :26, :40)

- [ ] **Step 1: Read each file and find the sites**

Use `rg` to locate exact lines:

```bash
rg "\.expect\(\"[^)]*poisoned" /workspace/vllm-lite/crates/ --type rust -n
```

This should list ~21 sites.

- [ ] **Step 2: Apply the standard comment pattern**

For each site, add the comment line immediately above the call:

```rust
// invariant: lock poisoning requires a panic while holding the lock; in this code path
// the lock is only held for synchronous field access (no await), so poisoning is impossible.
let guard = self.foo.write().expect("foo lock poisoned");
```

Or shorter version:
```rust
// invariant: lock is only held for sync field access; no panic possible while holding.
let guard = self.foo.write().expect("foo lock poisoned");
```

For `RwLock` (registry/lifecycle.rs, registry/loader.rs, arch/registry.rs): the current messages say `"mutex poisoned"` which is wrong (they're RwLock). Update the message too:

Before:
```rust
let guard = self.foo.write().expect("mutex poisoned");
```

After:
```rust
// invariant: lock is only held for sync field access; no panic possible while holding.
let guard = self.foo.write().expect("rwlock poisoned");
```

- [ ] **Step 3: Verify formatting**

```bash
cargo fmt --all
```

- [ ] **Step 4: Verify compilation and tests**

```bash
cargo build --workspace --all-features 2>&1 | tail -5
cargo test --workspace --lib 2>&1 | tail -5
```

Expected: clean build, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/speculative/registry/lifecycle.rs \
        crates/core/src/speculative/registry/loader.rs \
        crates/model/src/arch/registry.rs \
        crates/core/src/engine/spec_dispatch/drafts.rs \
        crates/server/src/main.rs
git commit -m "docs: add // invariant: comments to RwLock/Mutex .expect(\"poisoned\") cluster (B-3a)"
```

---

## Task 4 (B-3b): Add `// invariant:` Comments to SystemTime Cluster (8 sites)

8 `SystemTime::now().duration_since(UNIX_EPOCH).unwrap()` calls across 7 files. Uniform pattern.

**Files:**
- Modify: `/workspace/vllm-lite/crates/dist/src/grpc.rs` (1 site at :68)
- Modify: `/workspace/vllm-lite/crates/dist/src/distributed_kv/cache.rs` (1 site at :180)
- Modify: `/workspace/vllm-lite/crates/dist/src/distributed_kv/protocol.rs` (1 site at :63)
- Modify: `/workspace/vllm-lite/crates/server/src/openai/batch/types.rs` (1 site at :86)
- Modify: `/workspace/vllm-lite/crates/server/src/openai/batch/handler.rs` (1 site at :45)
- Modify: `/workspace/vllm-lite/crates/server/src/openai/batch/manager.rs` (1 site at :67)
- Modify: `/workspace/vllm-lite/crates/server/src/security/correlation.rs` (1 site at :33)
- Modify: `/workspace/vllm-lite/crates/server/src/main.rs` (1 site at :40 — note: different from B-3a sites)

- [ ] **Step 1: Locate all sites**

```bash
rg "SystemTime::now\(\).duration_since\(UNIX_EPOCH\)" /workspace/vllm-lite/crates/ --type rust -n
```

Expected: ~8 sites (possibly more if there are similar patterns with different variables).

- [ ] **Step 2: Apply the standard comment**

For each site:

```rust
// invariant: SystemTime::now() is always >= UNIX_EPOCH on any platform that has a
// working clock; duration_since(UNIX_EPOCH) cannot underflow.
let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
```

Or shorter:
```rust
// invariant: monotonic clock is always >= UNIX_EPOCH.
let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
```

- [ ] **Step 3: Verify**

```bash
cargo fmt --all
cargo build --workspace --all-features 2>&1 | tail -5
cargo test --workspace --lib 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add crates/dist/src/grpc.rs \
        crates/dist/src/distributed_kv/cache.rs \
        crates/dist/src/distributed_kv/protocol.rs \
        crates/server/src/openai/batch/types.rs \
        crates/server/src/openai/batch/handler.rs \
        crates/server/src/openai/batch/manager.rs \
        crates/server/src/security/correlation.rs \
        crates/server/src/main.rs
git commit -m "docs: add // invariant: comments to SystemTime cluster (B-3b)"
```

---

## Task 5 (B-3c): Add `// invariant:` Comments to Remaining Cluster (22 sites)

The remaining 22 INVARIANT sites span 14 files and represent diverse patterns: Tensor allocations, signal handlers, duplicate-key checks, caller contracts, serialization of known-good structs, HashMap-after-insert, Cargo env vars, and misc invariants.

**Files (14 total):**
- `crates/model/src/gemma4/attention.rs` (verify existing comments adequate; may need only polish)
- `crates/model/src/kv_cache.rs` (1 site at :29)
- `crates/model/src/qwen3_5/block/linear.rs` (1 site at :163)
- `crates/model/src/components/ssm.rs` (1 site at :301)
- `crates/server/src/main.rs` (2 signal handler sites at :340, :346 + 1 vram site at :143)
- `crates/core/src/engine.rs` (2 sites at :181, :232 — duplicate draft id)
- `crates/core/src/engine/spec_dispatch/drafts.rs` (1 site at :39 — caller contract)
- `crates/server/src/openai/chat.rs` (2 sites at :223, :243 — serialize)
- `crates/model/src/causal_lm/hybrid_lm.rs` (1 site at :182 — HashMap after insert)
- `crates/dist/build.rs` (2 sites at :5, :9 — Cargo env)
- `crates/core/src/speculative/memory_budget.rs` (1 site at :65 — u64::MAX)
- `crates/server/src/auth.rs` (1 site at :85 — Response builder)
- `crates/server/src/api.rs` (1 site at :53 — engine channel)
- `crates/core/src/circuit_breaker/strategy.rs` (1 site at :138 — last_error after loop)

- [ ] **Step 1: Locate all sites**

```bash
rg "\.expect\(|\.unwrap\(\)" /workspace/vllm-lite/crates/{model,core,server,dist}/src/ --type rust -n -g '!**/tests/**' -g '!**/target/**' > /tmp/b3c_sites.txt
```

Filter out sites already covered by B-1/B-2/B-3a/B-3b (the 35 sites) to get the remaining ~22.

- [ ] **Step 2: For each site, apply a context-appropriate comment**

Pattern templates by site type:

| Pattern | Comment template |
|---------|------------------|
| Tensor alloc (`Tensor::zeros(...).expect(...)`) | `// invariant: tensor shape is statically known; allocation cannot fail for fixed-size 1×1. ` (or larger shape with reasoning) |
| Signal handler install (`signal::ctrl_c().expect(...)`) | `// invariant: signal handler installation only fails if the OS is in an unrecoverable state; not recoverable anyway.` |
| Duplicate id check (`expect("duplicate draft id")`) | `// invariant: insert path checks for duplicates before calling this method; the duplicate case is a programmer error.` |
| Caller contract (`expect("called without draft_resolver")`) | `// invariant: caller contract — this method is only invoked after draft_resolver is set (see initialization in foo).` |
| Serialize known-good (`to_string().expect(...)`) | `// invariant: serializing a known-good struct (no exotic types); to_string cannot fail.` |
| HashMap after insert (`map.get(k).unwrap()`) | `// invariant: key was just inserted above; cannot be missing.` |
| Cargo env var (`env::var("CARGO_PKG_NAME").unwrap()`) | `// invariant: CARGO_* env vars are always set by Cargo during build.` |
| u64::MAX arithmetic (`u64::MAX - x`) | `// invariant: x is bounded by caller-side validation; cannot approach u64::MAX.` |
| Response builder | `// invariant: builder pattern with all required fields set above.` |
| Engine channel send | `// invariant: engine is shutdown only after all senders are dropped; sender outlives receiver here.` |
| last_error after loop | `// invariant: loop populates last_error on every iteration where condition was hit; if loop exits without setting, it's the success path.` |

Use the template that best matches each site's actual context. Customize the wording to be accurate for the specific code.

- [ ] **Step 3: Verify**

```bash
cargo fmt --all
cargo build --workspace --all-features 2>&1 | tail -5
cargo test --workspace --lib 2>&1 | tail -5
```

- [ ] **Step 4: Commit (single batch commit for all 14 files)**

```bash
git add crates/model/src/gemma4/attention.rs \
        crates/model/src/kv_cache.rs \
        crates/model/src/qwen3_5/block/linear.rs \
        crates/model/src/components/ssm.rs \
        crates/server/src/main.rs \
        crates/core/src/engine.rs \
        crates/core/src/engine/spec_dispatch/drafts.rs \
        crates/server/src/openai/chat.rs \
        crates/model/src/causal_lm/hybrid_lm.rs \
        crates/dist/build.rs \
        crates/core/src/speculative/memory_budget.rs \
        crates/server/src/auth.rs \
        crates/server/src/api.rs \
        crates/core/src/circuit_breaker/strategy.rs
git commit -m "docs: add // invariant: comments to remaining INVARIANT cluster (B-3c)"
```

---

## Task 6: Verify Production Unwrap Count

**Files:** none modified

- [ ] **Step 1: Run final unwrap count**

```bash
rg "\.unwrap\(\)|\.expect\(" /workspace/vllm-lite/crates/ --type rust -g '!**/tests/**' -g '!**/target/**' -g '!**/benches/**' | wc -l
```

Expected: ≤ 60 (audit baseline). After B-1/B-2 the count should drop slightly (CONVERT sites use `?` instead). The 51 INVARIANT sites stay but now have comments.

If count is > 60, investigate the new sites.

- [ ] **Step 2: Verify `// invariant:` comment coverage**

```bash
rg "// invariant:" /workspace/vllm-lite/crates/ --type rust | wc -l
```

Expected: ≥ 51 (the 51 INVARIANT sites from the audit). Higher is fine if some sites got multiple comments.

---

## Task 7: Verify Full CI

- [ ] **Step 1: `just fmt-check`**

```bash
just fmt-check
```

Expected: clean (formatter may auto-fix minor issues — if it does, run `cargo fmt --all` and commit the format fix as a separate `style: fmt` commit).

- [ ] **Step 2: `just clippy`**

```bash
just clippy
```

Expected: exit 0, no deny-tier errors.

- [ ] **Step 3: `just doc-check`**

```bash
just doc-check
```

Expected: exit 0.

- [ ] **Step 4: `just nextest`**

```bash
just nextest
```

Expected: all tests pass, no regressions.

- [ ] **Step 5: `just ci`**

```bash
just ci
```

Expected: all four steps pass.

---

## Task 8: Update AGENTS.md with Invariant Comments Convention

**Files:**
- Modify: `/workspace/vllm-lite/AGENTS.md`

Add a brief subsection under "Lint Policy" documenting the `// invariant:` convention.

- [ ] **Step 1: Find the Lint Policy section**

```bash
rg -n "## Lint Policy" /workspace/vllm-lite/AGENTS.md
```

- [ ] **Step 2: Insert new subsection after "Adding a new lint"**

Add (immediately after the "Adding a new lint" numbered list, before "Rationales for current allow list"):

```markdown
### Invariant comments

Production `.unwrap()` / `.expect()` calls that legitimately cannot fail must be
preceded by a `// invariant:` comment explaining why. This applies to:

- `RwLock` / `Mutex` `.expect("...poisoned")` — lock is only held for sync field access
- `SystemTime::now().duration_since(UNIX_EPOCH).unwrap()` — monotonic clock cannot underflow
- `.expect("duplicate <key>")` after a pre-check — programmer error path
- Tensor allocations with statically-known shapes
- Cargo env vars (always set by Cargo during build)
- Signal handler installation
- Serialize of known-good structs

If a `.unwrap()` / `.expect()` call is in production code and has no `// invariant:`
comment, treat it as a bug and convert it to typed error propagation. See Phase B
audit at `/tmp/phase_b_audit/SUMMARY.md` for the full inventory.
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs(agents): document // invariant: comment convention"
```

---

## Task 9: Phase B Completion Report (CHANGELOG)

**Files:**
- Modify: `/workspace/vllm-lite/CHANGELOG.md`

- [ ] **Step 1: Add Phase B entry under `[Unreleased]`**

Under the `### Changed` subsection (added in Phase A), append:

```markdown
- **Unwrap Cleanup (v24.0 Phase B)** — fixed real bug risk and improved error reporting:
  - **B-1**: `cuda_graph/executor.rs:222` race condition unwrap → typed `GraphNotFound` error
  - **B-2**: 5 production unwraps in `engine.rs`, `main.rs`, `handler.rs` → typed error variants
  - **B-3**: 51 production `// invariant:` comments added across 22 files documenting legitimate invariants
  - Baseline audit: spec originally claimed 787 production unwraps; actual was 60 (the 787 figure included inline `#[cfg(test)] mod tests` blocks). Spec target `≤160` was already met.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record v24.0 Phase B unwrap cleanup completion"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Spec §5 (revised 2026-06-28) → B-1/B-2 (CONVERT), B-3a/B-3b/B-3c (INVARIANT), AGENTS.md docs, CHANGELOG. All accounted for.
- [x] **Placeholder scan:** No "TBD"/"TODO"/"fill in". Each step has explicit file:line and code template.
- [x] **Type consistency:** Error enum names (`CudaGraphError`, `EngineError`, `BatchError`) are placeholders; implementer must verify against actual code. Tensors / GraphKey / etc. likewise.
- [x] **Dependency order:** Tasks 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9. Each builds on previous.

---

## Handoff

After Task 9 commit, Phase B is complete. Diff stats expected:
- ~50-100 lines of code change (B-1: 5 lines, B-2: ~30-50 lines)
- ~51 lines of comment additions (B-3)
- ~20 lines of doc updates (AGENTS.md + CHANGELOG.md)

Total: ~5 atomic commits covering 25+ files. Push to origin/main when user approves.

Next: Phase C plan (API Ergonomics) — to be written after Phase B ships.
