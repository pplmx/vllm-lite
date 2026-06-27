# ADR-009: FP8 Quantizer Orphan Module Decision

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v20.2 outcome

## Context

The Phase 26 audit (MT-01 in `.planning/PROJECT.md`) found that `crates/model/src/components/kv_cache_fp8.rs` (289 LOC of FP8 E4M3 quantization code, see ADR-004) was an **orphan module**:

- The file existed in the source tree.
- The file compiled successfully (it had no syntax errors, no broken imports).
- But the file was **unreachable** from any other module — nothing in the crate imported from it, and no module tree entry referenced it.

The cause was an incomplete refactor from an earlier phase: when the `kv_cache_fp8.rs` module was relocated from one path to another (likely during the components-layer consolidation), the `pub mod` declaration in the parent module was never added back. The file survived in the new path but became invisible to the rest of the crate.

This produced several concrete problems:

1. **Dead code that compiles** — clippy and the compiler don't flag unreferenced modules as warnings; the 289 LOC just sat there.
2. **Test coverage gap** — the file's `#[cfg(test)]` module ran (it was reachable from itself via `mod tests` inside the file), but those tests verified only the quantizer's internal logic, not its integration with the rest of the cache pipeline.
3. **Onboarding confusion** — new contributors would find the file, try to use `Fp8Quantizer` or `KvCacheDtype`, and discover that `use crate::components::kv_cache_fp8::*` failed because the module wasn't declared in `components/mod.rs`.
4. **Audit backlog noise** — the v19.0 audit explicitly flagged this as MT-01; resolving it restored first-class status.

## Decision

Wire `kv_cache_fp8.rs` into the components module tree by adding a `pub mod` declaration and a `pub use` re-export in `crates/model/src/components/mod.rs`:

```rust
// crates/model/src/components/mod.rs
//! mod: module.

/// attention: attention module.
pub mod attention;
/// block: block module.
pub mod block;
/// decoder_block: decoder block module.
pub mod decoder_block;
/// gated_delta: gated delta module.
pub mod gated_delta;
/// kv_cache_fp8: kv cache fp8 module.
pub mod kv_cache_fp8;       // ← added (Phase 26 MT-01)
/// mlp: mlp module.
pub mod mlp;
...

pub use kv_cache_fp8::{Fp8Quantizer, KvCacheDtype};  // ← added re-export
```

After this change, `kv_cache_fp8` becomes reachable as `crate::components::kv_cache_fp8::*` and the two key types (`Fp8Quantizer`, `KvCacheDtype`) become reachable as `crate::components::{Fp8Quantizer, KvCacheDtype}` via the re-export. Downstream code can now actually use the FP8 quantizer without writing a direct path through `kv_cache_fp8`.

The wiring commit (Phase 26 MT-01, commit `f940639`) was a two-line change: add `pub mod kv_cache_fp8;` and the corresponding `pub use` line. No code in `kv_cache_fp8.rs` itself needed to change — the file was already correct, just invisible.

## Rationale

1. **Restores first-class status** — the module is now reachable through the public components API, matching its conceptual position in the cache precision layer (ADR-005).
2. **Zero risk** — the file was already compiling; the wiring change only adds visibility.
3. **Re-export enables ergonomic use** — `use crate::components::Fp8Quantizer` is shorter and more idiomatic than the deep path.
4. **Resolves the MT-01 audit finding** — the audit backlog item is closed.
5. **Matches sibling modules** — every other module in `components/mod.rs` follows the same `pub mod` + `pub use` pattern; this brings `kv_cache_fp8` into line.

Alternatives considered:

- **Delete the file** — rejected; the quantizer logic is correct and tested. Deletion would force a re-implementation later.
- **Move it back to its old path** — rejected; the new location under `components/` is the right architectural home (it's a shared component, not a model-specific concern).
- **Leave it orphaned** — rejected; the audit explicitly flagged this and the file is valuable code.
- **Wire it through a different module path** — rejected; `components/` is where all shared cache primitives live.

## Consequences

**Positive:**

- The 289 LOC of FP8 logic is now actually reachable — callers can use `Fp8Quantizer` and `KvCacheDtype` via the standard components import path.
- Future FP8-related work (e.g. wiring it into the paged tensor store, adding per-channel scaling) has a clear foundation.
- The MT-01 audit finding is closed; the audit backlog no longer flags it.
- Tests inside `kv_cache_fp8.rs` still run (they were always running) but now their integration with the rest of the crate is plausible.

**Negative:**

- The wiring was a fix for an *organisational* bug, not a logic bug. It does not add new functionality — anyone wanting FP8 quantisation still has to wire `Fp8Quantizer` into the paged tensor store themselves.
- The root cause (incomplete refactor leaving orphan files) is a process issue, not a code issue. Future refactors could produce the same bug; the only mitigation is review discipline.
- The file existed for some time (v15.0 → v20.2) in an orphan state; that was wasted review surface — anyone auditing the code may have assumed it was in use when it wasn't.

**Mitigations / migration paths:**

- Phase 26 also wired `debug.rs` (MT-02) using the same pattern; together these two fixes eliminated the orphan-file class of bugs.
- Future refactors that relocate modules should treat the `pub mod` declaration in the parent as a required step in the same commit, not a follow-up.
- A `cargo metadata` or clippy lint for "file in src/ but not reachable from lib.rs" would catch this class of bug automatically; worth adding to the verification checklist post-v20.0.
