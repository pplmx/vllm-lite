# Phase 33: Naming Consistency (v21.3) — SUMMARY

**Status:** Complete
**Milestone:** v21.0 P2/P3 Backlog Cleanup
**Requirements covered:** NAM-01, NAM-03, NAM-06, NAM-07

## What Was Delivered

### NAM-01: Renamed `flash_v3.rs` → `flash_attention_v3.rs`
- File renamed via `git mv` (preserves rename history)
- Updated `crates/model/src/components/attention/mod.rs`:
  - Module declaration: `pub mod flash_attention_v3;`
  - Re-export: `pub use flash_attention_v3::{...};`
  - Updated module doc comment
- Updated file header to reflect new name
- 1-line code change in body (`"FlashAttentionV3 forward"` log message unchanged — already correct)

### NAM-03: `NodeInfo` rename evaluation — kept with documented rationale
- Decision: **NOT to rename** — current name is concise and clear
- Rationale already documented in `AGENTS.md` Suffix Conventions table:
  - `*Info | Type is metadata-only (no behavior); bare name would be ambiguous | NodeInfo (vs graph Node)`
- NodeInfo specifically disambiguates from graph `Node` (used elsewhere)
- Alternative names (`NodeSummary`, `NodeMetadata`) would be equally descriptive but longer
- Audit allows either rename or documented rationale — chose documented rationale (already exists)

### NAM-06, NAM-07: Non-tensor single-letter variables in sampling code
Renamed for clarity per audit guidance:

| File | Before | After | Reason |
|------|--------|-------|--------|
| `crates/core/src/sampling.rs:45` | `let r = random_f32();` | `let random_threshold = random_f32();` | `r` was opaque |
| `crates/core/src/sampling.rs:88` | `let r = random_f32();` | `let random_threshold = random_f32();` | (same) |
| `crates/core/src/sampling.rs:105` | `let k = k.min(logits.len());` | `let top_k_limit = k.min(logits.len());` | top-k clamp |
| `crates/core/src/sampling.rs:149` | `let k = top_k.min(logits.len());` | `let top_k_limit = top_k.min(logits.len());` | (same) |
| `crates/core/src/engine.rs:584` | `let k = k.min(logits.len());` | `let top_k_limit = k.min(logits.len());` | (same) |

All renamed variables updated at use sites within each function. The `top_k` parameter name (function arguments) retained — function param names are not single-letter renames.

### NAM-02, NAM-04, NAM-05, NAM-08: AGENTS.md documentation
Already covered by existing AGENTS.md sections (Phase 30 set this up):
- **NAM-02** (verb policy): covered in Verb Policy table
- **NAM-04** (`create_*` vs `build_*`): covered in Verb Policy table (`create_*` = one-shot construction, `build_*` = builder finalization)
- **NAM-05** (async/sync split rationale): covered in API Conventions section ("Sync vs Async Trait Splits")
- **NAM-08** (test file location): covered in Test file location table

No documentation updates needed — existing AGENTS.md covers all NAM-02/04/05/08 requirements.

## Verification

| Check | Result |
|-------|--------|
| `cargo build --workspace --all-features` | Clean |
| `cargo test --workspace --all-features` | 1157 passed (no regression) |
| `cargo clippy --workspace --all-targets -- -D warnings` | Clean |
| `cargo fmt --all --check` | Clean |

## Deferred

- **NAM-04 (create_* vs build_* policy)**: Already documented in AGENTS.md Verb Policy table (Phase 30 outcome).
- **NAM-05 (async/sync split rationale)**: Already documented in API Conventions section (Phase 32 outcome).
- **NAM-08 (test file location convention)**: Already documented in Test file location table (Phase 30 outcome).
- **Additional `*Info` suffix renames**: No other `*Info` types exist in the codebase; convention is well-applied.

## Backward Compatibility

- File rename `flash_v3.rs` → `flash_attention_v3.rs` is internal — module path was internal to `crates/model/src/components/attention/`. The public types (`FlashAttentionV3`, `FlashAttentionV3Config`, `GqaFlashAttention`, `MqaFlashAttention`) retain their names and are re-exported from `mod.rs`.
- Variable renames are behavior-preserving and don't affect public API.
