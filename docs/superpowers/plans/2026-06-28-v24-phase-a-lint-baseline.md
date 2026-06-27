# v24.0 Phase A — Lint Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish workspace-wide clippy lint policy with severity tiers (deny correctness/suspicious/perf, warn pedantic/nursery, explicit allows), wire per-crate inheritance, and update CI so deny-level violations break the build while pedantic warnings are visible but non-blocking.

**Architecture:** Workspace-root `[workspace.lints.clippy]` table defines a single source of truth. Each crate inherits via `[lints] workspace = true`. The `just clippy` command switches from `-D warnings` to explicit per-group `-D` flags so pedantic warnings stay visible without breaking CI.

**Tech Stack:** Rust 2024 edition, Cargo workspace lints (stable since 1.74), clippy lint groups.

**Spec:** [`docs/superpowers/specs/2026-06-28-v24-code-quality-hardening-design.md`](../specs/2026-06-28-v24-code-quality-hardening-design.md) §4

**Sister plans (not yet written):**
- Phase B — Unwrap Cleanup (will be `2026-06-28-v24-phase-b-unwrap-cleanup.md`)
- Phase C — API Ergonomics
- Phase D — Module Boundaries

---

## File Structure

This phase modifies existing files only; no new source files.

| File | Change |
|------|--------|
| `Cargo.toml` | Add `[workspace.lints.clippy]` table |
| `crates/core/Cargo.toml` | Add `[lints] workspace = true` |
| `crates/dist/Cargo.toml` | Add `[lints] workspace = true` |
| `crates/model/Cargo.toml` | Add `[lints] workspace = true` |
| `crates/server/Cargo.toml` | Add `[lints] workspace = true` |
| `crates/testing/Cargo.toml` | Add `[lints] workspace = true` |
| `crates/traits/Cargo.toml` | Add `[lints] workspace = true` |
| `justfile` | Update `clippy` recipe to use explicit per-group `-D` flags |
| `AGENTS.md` | Add "Lint Policy" section documenting the tier system |

Total: 9 modified files, no new files.

---

## Task 1: Capture Pedantic Baseline

**Files:**
- Read-only: none
- Create: `docs/superpowers/specs/2026-06-28-v24-code-quality-hardening-design.md` already contains baseline (2005 warnings, 787 unwraps)

- [ ] **Step 1: Verify baseline counts are still current**

Run from `/workspace/vllm-lite`:

```bash
cargo clippy --workspace --all-targets --all-features -- -W clippy::pedantic 2>&1 | rg "warning:" | wc -l
```

Expected: ~2005 (within ±100 of spec baseline).

```bash
rg "\.unwrap\(\)|\.expect\(" /workspace/vllm-lite/crates/ --type rust -g '!**/tests/**' -g '!**/target/**' | wc -l
```

Expected: ~824 (Phase B target = ≤160, so this is the unwrap baseline).

- [ ] **Step 2: Record actual numbers**

Note the two numbers you see. They'll be the "before" values cited in Phase B and C completion reports.

No commit (no file changes yet).

---

## Task 2: Verify Deny-Level Is Currently Clean

Before configuring workspace lints, confirm that the lints we plan to deny (`correctness`, `suspicious`, `perf`) currently produce zero warnings. This prevents surprise build breaks when we add the config.

**Files:** none modified

- [ ] **Step 1: Run deny-level clippy check**

Run from `/workspace/vllm-lite`:

```bash
cargo clippy --workspace --all-targets --all-features -- \
  -D clippy::correctness \
  -D clippy::suspicious \
  -D clippy::perf \
  2>&1 | rg "error:" | head -20
```

Expected: empty output (no errors). The build may still produce non-clippy cargo output (like "Compiling...").

- [ ] **Step 2: If errors appear, record them**

If you see any `error:` lines, copy them into a scratch file. We'll need to fix them as part of Task 6. Do NOT proceed to Task 3 if there are errors — fix them first or escalate.

No commit.

---

## Task 3: Add Workspace Lint Table

**Files:**
- Modify: `/workspace/vllm-lite/Cargo.toml`

- [ ] **Step 1: Read current root Cargo.toml**

Already read above. Structure: `[workspace]` → `[workspace.package]` → `[workspace.metadata]` → `[workspace.dependencies]` → `[profile.dev]` → `[profile.release]`.

- [ ] **Step 2: Insert `[workspace.lints.clippy]` table after `[workspace.dependencies]`**

In `/workspace/vllm-lite/Cargo.toml`, insert the following block **after line 40 (the `[workspace.dependencies]` block ends with `tower-http = { ... }`) and before line 41 (the empty line preceding `[profile.dev]`)**:

```toml
# === Workspace lint policy ===
# Tiers:
#   deny  - breaks CI (`just clippy`)
#   warn  - visible in `cargo clippy`, not blocking
#   allow - explicitly silenced with rationale
#
# Phase A scope: deny correctness/suspicious/perf. Pedantic/nursery stay warn
# so they're visible during local dev but don't break CI. Phase B/C/D fix
# pedantic warnings opportunistically as they touch files.

[workspace.lints.clippy]
# deny tier: correctness / obvious bugs / perf footguns
correctness = { level = "deny", priority = -1 }
suspicious = { level = "deny", priority = -1 }
perf = { level = "deny", priority = -1 }

# warn tier: documentation, style, ergonomics
pedantic = { level = "warn", priority = -1 }
nursery = { level = "warn", priority = -1 }
missing_errors_doc = "warn"
missing_panics_doc = "warn"
module_name_repetitions = "warn"
must_use_candidate = "warn"
return_self_not_must_use = "warn"
missing_const_for_fn = "warn"
uninlined_format_args = "warn"

# allow tier: project-specific rationale
cast_precision_loss = "allow"          # model dim casts are intentional
cast_possible_truncation = "allow"     # same as above
cast_possible_wrap = "allow"           # same as above
cast_sign_loss = "allow"               # same as above
similar_names = "allow"                # tensor-math `q/k/v` shadows permitted
too_many_lines = "allow"               # addressed in Phase D
too_many_arguments = "allow"           # addressed in Phase C/D
multiple_crate_versions = "allow"      # tracked separately

[workspace.lints.rust]
unsafe_code = "warn"
missing_debug_implementations = "warn"
rust_2018_idioms = { level = "warn", priority = -1 }
```

The final file should look like (lines 40-end):

```toml
[workspace.dependencies]
thiserror = "2"
# ... (rest of existing deps) ...
tower-http = { version = "0.5", features = ["cors", "trace"] }

# === Workspace lint policy ===
# Tiers:
#   deny  - breaks CI (`just clippy`)
#   warn  - visible in `cargo clippy`, not blocking
#   allow - explicitly silenced with rationale
#
# Phase A scope: deny correctness/suspicious/perf. Pedantic/nursery stay warn
# so they're visible during local dev but don't break CI. Phase B/C/D fix
# pedantic warnings opportunistically as they touch files.

[workspace.lints.clippy]
# deny tier: correctness / obvious bugs / perf footguns
correctness = { level = "deny", priority = -1 }
suspicious = { level = "deny", priority = -1 }
perf = { level = "deny", priority = -1 }

# warn tier: documentation, style, ergonomics
pedantic = { level = "warn", priority = -1 }
nursery = { level = "warn", priority = -1 }
missing_errors_doc = "warn"
missing_panics_doc = "warn"
module_name_repetitions = "warn"
must_use_candidate = "warn"
return_self_not_must_use = "warn"
missing_const_for_fn = "warn"
uninlined_format_args = "warn"

# allow tier: project-specific rationale
cast_precision_loss = "allow"          # model dim casts are intentional
cast_possible_truncation = "allow"     # same as above
cast_possible_wrap = "allow"           # same as above
cast_sign_loss = "allow"               # same as above
similar_names = "allow"                # tensor-math `q/k/v` shadows permitted
too_many_lines = "allow"               # addressed in Phase D
too_many_arguments = "allow"           # addressed in Phase C/D
multiple_crate_versions = "allow"      # tracked separately

[workspace.lints.rust]
unsafe_code = "warn"
missing_debug_implementations = "warn"
rust_2018_idioms = { level = "warn", priority = -1 }

[profile.dev]
opt-level = 0
debug = true

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
```

- [ ] **Step 3: Verify workspace config parses**

Run from `/workspace/vllm-lite`:

```bash
cargo metadata --format-version 1 --no-deps > /dev/null
```

Expected: exit code 0, no error output. This validates the lint table syntax.

- [ ] **Step 4: Run clippy with the new config**

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings 2>&1 | rg "error:" | wc -l
```

Expected: 0 (deny-tier lints should all pass since Task 2 verified they're clean; pedantic warn-tier will become errors with `-D warnings` — see Task 5 where we update just clippy).

Wait — pedantic is configured as "warn". `-D warnings` promotes ALL warnings to errors, including the ~2005 pedantic warnings. This is expected and not yet fixed. **Do not be alarmed**; Task 5 changes `just clippy` to use specific `-D clippy::<group>` flags which will leave pedantic as warnings.

For verification, just confirm the number is around 2000 (the pedantic warnings are now being treated as errors, not a regression).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml
git commit -m "build(workspace): add workspace lint policy (deny correctness/suspicious/perf + warn pedantic)"
```

---

## Task 4: Add `[lints]` Inheritance to Each Crate

**Files:**
- Modify: `crates/core/Cargo.toml`
- Modify: `crates/dist/Cargo.toml`
- Modify: `crates/model/Cargo.toml`
- Modify: `crates/server/Cargo.toml`
- Modify: `crates/testing/Cargo.toml`
- Modify: `crates/traits/Cargo.toml`

Each crate needs a `[lints]` section that points back to the workspace table.

- [ ] **Step 1: Add `[lints]` section to `crates/traits/Cargo.toml`**

The file currently ends at line 13 (after `[features]`). Read the file first:

```bash
cat /workspace/vllm-lite/crates/traits/Cargo.toml
```

Append after the last existing section (e.g., after `[features]`):

```toml

[lints]
workspace = true
```

The blank line before `[lints]` is required by Cargo's TOML formatting conventions.

- [ ] **Step 2: Add `[lints]` section to `crates/core/Cargo.toml`**

Append after the last section (likely `[[bench]]` blocks):

```toml

[lints]
workspace = true
```

- [ ] **Step 3: Add `[lints]` section to `crates/model/Cargo.toml`**

Append after the last section (likely `[features]`):

```toml

[lints]
workspace = true
```

- [ ] **Step 4: Add `[lints]` section to `crates/server/Cargo.toml`**

Append after the last section (likely `[dev-dependencies]`):

```toml

[lints]
workspace = true
```

- [ ] **Step 5: Add `[lints]` section to `crates/testing/Cargo.toml`**

Append after the last section (likely `[features]`):

```toml

[lints]
workspace = true
```

- [ ] **Step 6: Add `[lints]` section to `crates/dist/Cargo.toml`**

Append after the last section (likely `[build-dependencies]`):

```toml

[lints]
workspace = true
```

- [ ] **Step 7: Verify all crates compile with inherited lints**

Run from `/workspace/vllm-lite`:

```bash
cargo check --workspace --all-targets --all-features 2>&1 | tail -20
```

Expected: `Finished` line, no errors. Warnings are OK (they're pedantic warns).

- [ ] **Step 8: Verify lints are actually inherited per crate**

Pick any crate (e.g., core) and check that pedantic warnings show up:

```bash
cargo clippy -p vllm-core --all-targets --all-features 2>&1 | rg "warning:" | wc -l
```

Expected: a number > 0 (proves pedantic warnings are showing in this crate).

Then verify the same crate has NO deny-tier errors when run with explicit denies:

```bash
cargo clippy -p vllm-core --all-targets --all-features -- \
  -D clippy::correctness \
  -D clippy::suspicious \
  -D clippy::perf \
  2>&1 | rg "error:" | wc -l
```

Expected: 0.

- [ ] **Step 9: Commit**

```bash
git add crates/*/Cargo.toml
git commit -m "build(crates): inherit workspace lint policy via [lints] workspace = true"
```

---

## Task 5: Update `just clippy` Command

**Files:**
- Modify: `/workspace/vllm-lite/justfile`

The current `just clippy` uses `-D warnings`, which would treat all ~2005 pedantic warnings as errors. Replace with explicit per-group `-D` flags so pedantic stays as warnings.

- [ ] **Step 1: Read current justfile clippy recipe**

Already shown above:

```just
# Run clippy (CI style)
clippy:
    cargo clippy --all-targets --workspace --all-features -- -D warnings
```

- [ ] **Step 2: Replace the recipe**

Change to:

```just
# Run clippy (CI style)
# Denies correctness/suspicious/perf. Pedantic/nursery are visible as warnings
# but not blocking. See `just clippy-pedantic` for pedantic-only view.
clippy:
    cargo clippy --all-targets --workspace --all-features -- \
        -D clippy::correctness \
        -D clippy::suspicious \
        -D clippy::perf

# Run clippy with pedantic+nursery warnings visible (local use, not CI)
clippy-pedantic:
    cargo clippy --all-targets --workspace --all-features -- \
        -W clippy::pedantic \
        -W clippy::nursery \
        -D clippy::correctness \
        -D clippy::suspicious \
        -D clippy::perf
```

- [ ] **Step 3: Verify new `just clippy` passes**

Run from `/workspace/vllm-lite`:

```bash
just clippy 2>&1 | tail -20
```

Expected: exit code 0, ends with `Finished` line. Warnings may appear (pedantic), no errors.

- [ ] **Step 4: Verify `just clippy-pedantic` shows pedantic warnings but no errors**

Run:

```bash
just clippy-pedantic 2>&1 | rg "error:" | wc -l
```

Expected: 0 errors. Warnings should appear (pedantic) but no deny-tier violations.

- [ ] **Step 5: Commit**

```bash
git add justfile
git commit -m "build(ci): update just clippy to use explicit per-group denies (pedantic stays as warn)"
```

---

## Task 6: Document Lint Policy in AGENTS.md

**Files:**
- Modify: `/workspace/vllm-lite/AGENTS.md`

Add a "Lint Policy" section so contributors know the tier system.

- [ ] **Step 1: Find a good insertion point in AGENTS.md**

Look for the `## Code Style Guidelines` section (line 160). We'll add the new section right after it (or before it — choose whichever keeps related content together).

- [ ] **Step 2: Insert the new section**

Insert the following block immediately after the `## Code Style Guidelines` section ends and before `## API Conventions`:

```markdown

---

## Lint Policy

Workspace-wide clippy configuration lives in the root `Cargo.toml` under
`[workspace.lints.clippy]`. Every crate inherits via `[lints] workspace = true`.

### Tiers

| Tier   | Lints                                                          | Effect                              |
| ------ | -------------------------------------------------------------- | ----------------------------------- |
| deny   | `correctness`, `suspicious`, `perf`                            | Breaks `just clippy` (CI blocks)    |
| warn   | `pedantic`, `nursery`, `missing_errors_doc`, `must_use_candidate`, etc. | Visible in `cargo clippy`, not blocking |
| allow  | `cast_precision_loss`, `too_many_lines`, `too_many_arguments`, etc. | Explicitly silenced with rationale |

### Local commands

```bash
# Standard CI check (deny-tier only)
just clippy

# Show pedantic+nursery warnings without breaking
just clippy-pedantic
```

### Adding a new lint

1. Identify which tier it belongs to (correctness/suspicious/perf → deny; otherwise → warn first, promote to deny later)
2. Add to `[workspace.lints.clippy]` in root `Cargo.toml`
3. Run `just clippy` to verify
4. If deny-tier and existing code violates it, fix the violations in the same PR

### Rationales for current allow list

| Lint                      | Why allowed                                                                |
| ------------------------- | -------------------------------------------------------------------------- |
| `cast_precision_loss`     | Model dim casts (`usize as f32`) are intentional for tensor math          |
| `cast_possible_truncation`| Same as above                                                              |
| `cast_possible_wrap`      | Same as above                                                              |
| `cast_sign_loss`          | Same as above                                                              |
| `similar_names`           | Tensor-math conventions (`q`, `k`, `v`, `b`, `c`, `h`, `d`)               |
| `too_many_lines`          | Phase D will refactor oversized files; lint enforced after that           |
| `too_many_arguments`      | Phase C builder API work will reduce; lint enforced after that            |
| `multiple_crate_versions` | Dependency cleanup tracked separately                                      |

---
```

- [ ] **Step 3: Verify AGENTS.md renders correctly**

Read the relevant section back:

```bash
rg -n "## Lint Policy" /workspace/vllm-lite/AGENTS.md
```

Expected: line number where the section header was inserted.

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md
git commit -m "docs(agents): document workspace lint policy (tiers + local commands)"
```

---

## Task 7: Verify Full CI

**Files:** none modified

End-to-end smoke test to confirm Phase A didn't break anything.

- [ ] **Step 1: Run `just fmt-check`**

```bash
just fmt-check 2>&1 | tail -5
```

Expected: no errors. If it fails, run `cargo fmt --all` to fix and commit the format-only change.

- [ ] **Step 2: Run `just clippy`**

```bash
just clippy 2>&1 | tail -10
```

Expected: exit code 0, ends with `Finished`. Warnings may appear (pedantic).

- [ ] **Step 3: Run `just doc-check`**

```bash
just doc-check 2>&1 | tail -10
```

Expected: exit code 0, no warnings (rustdoc is strict).

- [ ] **Step 4: Run `just nextest`**

```bash
just nextest 2>&1 | tail -20
```

Expected: all tests pass, 0 failures. Slow/#[ignore] tests are skipped (correct for this phase).

- [ ] **Step 5: Run `just ci` (full pipeline)**

```bash
just ci 2>&1 | tail -20
```

Expected: all four steps pass. This is the final acceptance gate for Phase A.

- [ ] **Step 6: Verify no unintended commits**

```bash
git log --oneline -10
```

Expected: 4 new commits on top of `main`:
1. `build(workspace): add workspace lint policy ...`
2. `build(crates): inherit workspace lint policy ...`
3. `build(ci): update just clippy to use explicit per-group denies ...`
4. `docs(agents): document workspace lint policy ...`

If you see anything else (e.g., a fmt fix), that's fine — note it in the phase completion report.

---

## Task 8: Phase A Completion Report

**Files:**
- Modify: `/workspace/vllm-lite/CHANGELOG.md`

Add a brief entry marking Phase A complete.

- [ ] **Step 1: Read current CHANGELOG.md head**

```bash
head -30 /workspace/vllm-lite/CHANGELOG.md
```

- [ ] **Step 2: Add Phase A entry**

If the changelog follows `## [Unreleased]` or `## [Next]` convention, add under that. Otherwise add at the top under a new `## v24.0 Phase A — Lint Baseline` heading:

```markdown
## v24.0 Phase A — Lint Baseline

- Added workspace lint policy (`[workspace.lints.clippy]`) with three tiers:
  deny correctness/suspicious/perf, warn pedantic/nursery, explicit allow list
  for project-specific cases.
- All 6 crates now inherit via `[lints] workspace = true`.
- `just clippy` switched from `-D warnings` to explicit per-group denies so
  pedantic warnings stay visible without breaking CI.
- New `just clippy-pedantic` recipe for local pedantic-only view.
- AGENTS.md updated with Lint Policy section documenting tiers and rationale.
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record v24.0 Phase A lint baseline completion"
```

---

## Self-Review Checklist (run after writing plan, before handoff)

- [x] **Spec coverage:** Spec §4 (Lint Baseline) → all four phases covered (workspace config in Task 3, per-crate in Task 4, CI in Task 5, docs in Task 6, verification in Task 7).
- [x] **Placeholder scan:** No "TBD"/"TODO"/"fill in". Exact paths, exact code, exact commands throughout.
- [x] **Type consistency:** Lint names match Cargo docs; crate names match workspace members; justfile recipe names match recipe definitions.
- [x] **Dependency order:** Tasks 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Each builds on previous.

---

## Handoff

After Task 8 commit, the user reviews the diff:

```bash
git log --oneline main..HEAD
git diff main..HEAD --stat
```

Phase A is complete when:
- 5 new commits on `main`
- `just ci` passes locally
- AGENTS.md has Lint Policy section
- CHANGELOG.md has Phase A entry

Next: Phase B plan (unwrap cleanup) will be written as a separate file
`docs/superpowers/plans/2026-06-28-v24-phase-b-unwrap-cleanup.md` once Phase A ships.
