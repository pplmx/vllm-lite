---
phase: 17.4-speculative-warmup-metrics
reviewed: 2026-05-13T00:00:00Z
depth: deep
files_reviewed: 17
files_reviewed_list:
  - crates/core/src/engine.rs
  - crates/core/src/engine/speculative.rs
  - crates/core/src/speculative/adaptive.rs
  - crates/core/src/speculative/self_spec.rs
  - crates/core/src/speculative/verifier.rs
  - crates/core/src/speculative/model.rs
  - crates/core/src/scheduler/engine.rs
  - crates/core/src/scheduler/memory/mod.rs
  - crates/core/src/scheduler/batch.rs
  - crates/core/src/metrics/collector.rs
  - crates/core/src/metrics/enhanced.rs
  - crates/core/src/metrics/exporter.rs
  - crates/core/src/types.rs
  - crates/model/src/qwen3/model.rs
  - crates/model/src/qwen3_5/model.rs
  - crates/model/src/qwen3_5/hybrid.rs
  - crates/traits/src/model.rs
findings:
  critical: 1
  warning: 4
  info: 2
  total: 7
status: issues_found
---

# Phase 17.4: Code Review Report

**Reviewed:** 2026-05-13
**Depth:** deep (cross-file analysis: call chains, KV cache lifetime, position-to-RoPE consistency)
**Files Reviewed:** 17
**Status:** issues_found

## Summary

Reviewed the full v17.0 speculative decoding milestone: unified step dispatch, batched draft generation, logit-based verification, KV cache rollback, EWMA+deadband adaptive drafting, draft warmup, speculative metrics, per-request acceptance tracking, and `forward_to_layer` for Qwen3/Qwen3.5 models.

One BLOCKER found: a position off-by-one in `SelfSpeculativeModel::generate_draft` that causes RoPE position to lag one behind the KV cache write position when using prefill batches — producing incorrect positional encodings for all draft tokens.

Several warnings: missing cleanup of per-request acceptance DashMap entries on cancel/shutdown, potential inconsistency in `generate_batched_drafts` `is_prefill` propagation, unused `VerifierError::Verification` variant, and incompletely wired `DraftVerifier` trait with `SelfSpeculativeModel::verify` stub.

## Critical Issues

### CR-01: Position off-by-one in SelfSpeculativeModel::generate_draft

**File:** `crates/core/src/speculative/self_spec.rs:174-176`

**Issue:** The decode position for draft tokens is computed as `positions[last] + step` (line 175-176), but the KV cache write uses `current_num_computed` which starts at `batch.num_computed_tokens[batch_idx]`. For a prefill batch with `num_computed_tokens = N` and positions `[0,1,...,N-1]`, the first decode RoPE position is `N-1` while KV writes at position `N`. Every draft step is off by one — RoPE sees position `(N-1)+step` but KV is written at `N+step`. This produces incorrect positional embeddings for all draft tokens.

**Fix:**
```rust
// Before (line 172-177):
let step_position = if positions.is_empty() {
    vec![current_num_computed]
} else {
    let base = positions[positions.len().saturating_sub(1)];
    vec![base + step]
};

// After:
let step_position = vec![num_computed + step];
```
The `positions.is_empty()` branch already does this correctly. Remove the `else` branch entirely — position should always start at `num_computed` (the number of already-processed tokens).

---

## Warnings

### WR-01: Per-request acceptance DashMap entries leaked on cancel/shutdown

**Files:**
- `crates/core/src/engine.rs:341-347` (cancel_request)
- `crates/core/src/engine.rs:400` (Shutdown handler)
- `crates/core/src/metrics/collector.rs:108-120` (record_per_request_acceptance)

**Issue:** `cancel_request` and the `Shutdown` message handler free `response_txs` entries but never call `self.scheduler.metrics.remove_per_request(seq_id)`. The `per_request_acceptance` DashMap accumulates entries for cancelled and shutdown sequences indefinitely (no other code path removes them). The `remove_per_request` call exists in `step_speculative_inner:146` for normally-finished sequences only.

**Fix:** Add `self.scheduler.metrics.remove_per_request(seq_id)` in `cancel_request` after confirming the sequence exists and before returning. Similarly, in the `Shutdown` handler, iterate over all running sequences and call `remove_per_request` on each.

### WR-02: generate_batched_drafts propagates original is_prefill to all draft positions

**File:** `crates/core/src/engine/speculative.rs:206`

**Issue:** `pos_is_prefill.push(batch.is_prefill[i])` copies the original batch's prefill/decode status. When the initial batch is prefill (`is_prefill[i] = true`), all draft generation steps run in prefill mode. This causes the model to re-embed and re-process all accumulated tokens every step (progressively: N, N+1, N+2 tokens). While functionally correct, it O(n^2) recomputes embeddings and attention for tokens already in KV cache, and may produce unexpected interactions with models that branch on `is_prefill` for KV cache write logic (e.g., Qwen3 writes to fresh vs. existing blocks differently).

**Fix:** After the first step (or after warmup), set `pos_is_prefill` to `false` for subsequent draft positions. The first step may need prefill mode to populate the draft model's KV cache from the prompt, but subsequent steps are pure decode.

### WR-03: DraftVerifier::verify is a stub in SelfSpeculativeModel

**File:** `crates/core/src/speculative/self_spec.rs:204-211`

**Issue:** `SelfSpeculativeModel::verify` unconditionally returns `VerificationResult::new(_seq_id, _draft_tokens.to_vec())` — accepting ALL draft tokens without comparison against target logits. The `VerifierError::Verification` variant (defined in verifier.rs:13) is never constructed anywhere in the codebase. The actual verification logic lives in `Engine::verify_draft_tokens_logits` (speculative.rs:246-330), bypassing the `DraftVerifier` trait entirely. This creates a split-brain where the trait interface says one thing but the engine implements a different verification path.

**Fix:** Either: (a) implement `verify` on `SelfSpeculativeModel` to mirror the engine's logit-argmax logic and call it from the engine, or (b) remove `verify` and `accept` from `DraftVerifier` if they're part of a different verification architecture, or (c) add `#[allow(dead_code)]` with a comment explaining the design decision.

### WR-04: KV cache rollback does not invalidate physical tensor store entries

**File:** `crates/core/src/scheduler/memory/mod.rs:161-179`

**Issue:** `MemoryManager::rollback` frees logical block IDs from the scheduler's pool and truncates `seq.kv_blocks`, but never zeros or clears the physical KV cache entries in the model's tensor store (paged_tensor). The model implementations (Qwen3, Qwen3.5) maintain their own KV caches internally; freed block slots retain stale K/V data. While causal attention masking and `num_computed_tokens` bounds checking prevent reading stale data from reallocated blocks under *normal* operation, any attention implementation that reads block contents without position-based bounds (e.g., tile-based flash attention reading full tile rows) would silently consume stale data. No clearing function exists in `crates/model/src/paged_tensor/`.

**Fix:** Document the invariant that physical KV cache storage must only be read within the valid range `[0, num_computed_tokens)`. If feasible, add a `fn clear_block(&mut self, block_id: BlockId)` to the tensor store and call it from `MemoryManager::free()`.

---

## Info

### IN-01: speculative_efficiency metric computation is unclear

**File:** `crates/core/src/metrics/enhanced.rs:187-196`

**Issue:** `speculative_efficiency()` returns `draft / (draft + accepted)`, measuring the fraction of all generated tokens that were drafts (i.e., "overhead ratio"). The name "efficiency" is ambiguous — some readers may expect it to mean acceptance rate (`accepted / draft`). The deadband hysteresis and EWMA tests use `record_verification(total_draft, total_accepted)` which feeds into the sliding window as individual boolean samples (accepted or not), not as rates. These are two distinct concepts tracked in two different metric systems (AdaptiveDraftConfig/DraftAccuracyTracker vs EnhancedMetrics/EnhancedMetricsCollector).

**Fix:** Rename to `speculative_overhead_ratio` or add a doc comment clarifying: "Returns the fraction of generated tokens that were draft tokens (lower is better; 0.0 = no drafts, 1.0 = all computation was speculative)."

### IN-02: Test coverage gaps on rollback edge cases

**File:** `crates/core/src/scheduler/memory/mod.rs:161-179`

**Issue:** The `MemoryManager::rollback` method has no unit tests. Edge cases like `num_tokens == 0` (early return), `num_tokens > num_computed_tokens` (saturating_sub clamps to 0), partial block rollback (tokens spanning block boundary), and Arc re-creation are untested. The integration test `test_kv_rollback_rejected_drafts` in speculative.rs is `#[ignore]`d.

**Fix:** Add unit tests for rollback in `memory/mod.rs` covering: zero tokens, tokens within same block, tokens crossing block boundary, all tokens (full rollback), and sequence with already-minimal computed tokens.

---

_Reviewed: 2026-05-13_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
