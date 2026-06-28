# v29.0 Fuzz Testing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up cargo-fuzz infrastructure + add 3 fuzz targets to catch panics/crashes in user-input parsers (YAML config, safetensors header, model JSON config). Validate against the v28.0 ROI: property tests found 1 bug; fuzz tests should find more in different attack surface.

**Architecture:** cargo-fuzz directory at workspace root. Each fuzz target is its own bin crate in `fuzz/fuzz_targets/`. Targets use `Arbitrary` to convert `&[u8]` → structured input, then exercise the parser. Coverage-guided fuzzing builds with `cargo +nightly fuzz build` (or stable with `cargo fuzz build --sanitizer=address` if nightly unavailable).

**Tech Stack:** cargo-fuzz (libFuzzer), proptest for any structured inputs (optional).

**Audit source:** `/tmp/phase_i_audit/SUMMARY.md`

---

## File Structure

| File | Change | Sub-phase |
|------|--------|-----------|
| `fuzz/` (NEW directory) | cargo-fuzz scaffolding | J-1 |
| `fuzz/Cargo.toml` | fuzz crate manifest | J-1 |
| `fuzz/fuzz_targets/app_config_yaml.rs` | YAML config fuzzer | J-2 |
| `fuzz/fuzz_targets/safetensors_header.rs` | safetensors header fuzzer | J-3 |
| `fuzz/fuzz_targets/qwen3_config_json.rs` | model config JSON fuzzer | J-4 |
| `fuzz/.gitignore` | ignore `target/`, `corpus/`, `artifacts/` | J-1 |
| `CHANGELOG.md` | v29.0 entry | J-5 |
| `justfile` | `fuzz` target | J-5 |

---

## Audit-Driven Constraints

### Top targets (priority order)

1. **AppConfig YAML** (`crates/server/src/config.rs:254`) — `serde_saphyr::from_str::<AppConfig>(&contents)` — user-provided YAML
2. **Safetensors header** (`crates/model/src/loader/checkpoint.rs:58`) — `SafeTensors::deserialize(&bytes)` — malformed checkpoints
3. **Qwen3Config JSON** (`crates/model/src/loader/builder.rs:195`) — 30+ nested fields, complex config parsing

### Out-of-scope (deferred)

- OpenAI request body fuzzing (separate HTTP-layer spec)
- JWT validation fuzzing (security audit spec)
- Tokenizer fuzzing (different corpus needed; tiktok is mature)
- Tokenizer round-trip (covered by tiktoken's own tests)

### Environment constraint

cargo-fuzz typically requires nightly Rust for full sanitizers. Stable Rust may work with `cargo fuzz build` but coverage-guided fuzzing may be limited. The plan covers both paths.

---

## Task J-1: cargo-fuzz Setup (0.5 day, Low risk)

**Files:**
- NEW: `/workspace/vllm-lite/fuzz/Cargo.toml`
- NEW: `/workspace/vllm-lite/fuzz/fuzz_targets/` (directory)
- NEW: `/workspace/vllm-lite/fuzz/.gitignore`

- [x] **Step 1: Verify cargo-fuzz availability**

```bash
which cargo-fuzz 2>&1
cargo fuzz --version 2>&1
```

If not installed: `cargo install cargo-fuzz` (may take a few minutes).

If install fails (network/sandbox restrictions), document this and skip to J-2 with manual `#[test]`-based fuzz harnesses as a fallback.

- [x] **Step 2: Initialize fuzz directory**

```bash
cd /workspace/vllm-lite
cargo fuzz init 2>&1 | tail -10
```

This creates:
- `fuzz/Cargo.toml`
- `fuzz/fuzz_targets/<default>.rs` (placeholder)
- `fuzz/.gitignore`

- [x] **Step 3: Verify fuzz build works**

```bash
cd /workspace/vllm-lite
cargo fuzz build 2>&1 | tail -10
```

Expected: builds. If fails due to nightly requirement, document and try with `--release` or skip.

- [x] **Step 4: Add fuzz to justfile**

In `/workspace/vllm-lite/justfile`, add:

```justfile
# Run all fuzz targets for a short duration (smoke)
fuzz-smoke:
    cargo fuzz run --fuzz-dir fuzz -- -max_total_time=10

# Run a specific fuzz target
fuzz TARGET:
    cargo fuzz run {{TARGET}} --fuzz-dir fuzz -- -max_total_time=60

# Build fuzz binaries (debug)
fuzz-build:
    cargo fuzz build --fuzz-dir fuzz
```

- [x] **Step 5: Verify build still works**

```bash
cd /workspace/vllm-lite
cargo build --workspace 2>&1 | tail -5
just ci 2>&1 | tail -3
```

- [x] **Step 6: Commit**

```bash
cd /workspace/vllm-lite
git add fuzz/ justfile .gitignore
git commit -m "test(fuzz): cargo-fuzz scaffolding (J-1)

Initial fuzz/ directory at workspace root with default fuzz target
(placeholder). justfile targets added: fuzz-smoke, fuzz TARGET, fuzz-build.

cargo-fuzz installed at [version]. Build verification: [status]."
```

---

## Task J-2: Fuzz Target #1 — AppConfig YAML (0.5-1 day, Low risk)

**Files:**
- NEW: `/workspace/vllm-lite/fuzz/fuzz_targets/app_config_yaml.rs`

- [x] **Step 1: Inspect AppConfig**

```bash
cd /workspace/vllm-lite
grep -A5 "pub struct AppConfig" crates/server/src/config.rs | head -30
grep -A3 "fn load" crates/server/src/config.rs | head -20
```

Note structure (AppConfig + nested ServerConfig, EngineConfig, AuthConfig).

- [x] **Step 2: Create fuzz target**

```rust
#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_saphyr::from_str;
use vllm_server::config::AppConfig;

fuzz_target!(|data: &[u8]| {
    // Convert arbitrary bytes to UTF-8 string (lossy for fuzzer).
    let yaml = std::str::from_utf8(data).unwrap_or("");
    
    // Parse; ignore errors (we only care about panics, not invalid input).
    let _ = from_str::<AppConfig>(yaml);
});
```

- [x] **Step 3: Add to fuzz/Cargo.toml**

In `/workspace/vllm-lite/fuzz/Cargo.toml` ensure:

```toml
[dependencies]
libfuzzer-sys = "0.4"
serde-saphyr = { path = "../crates/server" }  # or workspace path

# Binary for this target
[[bin]]
name = "app_config_yaml"
path = "fuzz_targets/app_config_yaml.rs"
```

Adjust based on workspace structure.

- [x] **Step 4: Build and run smoke test**

```bash
cd /workspace/vllm-lite
cargo fuzz build app_config_yaml 2>&1 | tail -10
cargo fuzz run app_config_yaml -- -max_total_time=10 2>&1 | tail -20
```

Expected: runs, finds inputs, no panics (or finds a bug).

- [x] **Step 5: If a bug is found**

If the fuzzer finds a panic:
1. Reproduce: `cargo fuzz run app_config_yaml fuzz/artifacts/app_config_yaml/crash-<hash>`
2. Add the input as a regression test (without fuzz) in `crates/server/src/config.rs::tests`
3. Fix the underlying panic in the parser
4. Re-run fuzzer to confirm

- [x] **Step 6: Commit**

```bash
cd /workspace/vllm-lite
git add fuzz/
git commit -m "test(fuzz): add AppConfig YAML fuzz target (J-2)

Fuzz target exercises serde_saphyr::from_str::<AppConfig> with arbitrary bytes.
Goal: catch panics in YAML deserializer path triggered by malformed configs.

[Result: N panics found / 0 panics after X seconds of fuzzing]"
```

---

## Task J-3: Fuzz Target #2 — Safetensors Header (0.5-1 day, Medium risk)

**Files:**
- NEW: `/workspace/vllm-lite/fuzz/fuzz_targets/safetensors_header.rs`

- [x] **Step 1: Inspect SafeTensors usage**

```bash
cd /workspace/vllm-lite
grep -B2 -A10 "SafeTensors::deserialize" crates/model/src/loader/checkpoint.rs | head -30
```

- [x] **Step 2: Create fuzz target**

```rust
#![no_main]

use libfuzzer_sys::fuzz_target;
use safetensors::SafeTensors;

fuzz_target!(|data: &[u8]| {
    // Try to parse arbitrary bytes as safetensors header.
    let _ = SafeTensors::deserialize(data);
});
```

- [x] **Step 3: Build and run**

```bash
cd /workspace/vllm-lite
cargo fuzz build safetensors_header 2>&1 | tail -5
cargo fuzz run safetensors_header -- -max_total_time=10 2>&1 | tail -20
```

- [x] **Step 4: If bug found**

Same as J-2 step 5.

- [x] **Step 5: Commit**

```bash
cd /workspace/vllm-lite
git add fuzz/
git commit -m "test(fuzz): add safetensors header fuzz target (J-3)"
```

---

## Task J-4: Fuzz Target #3 — Qwen3Config JSON (0.5-1 day, Medium risk)

**Files:**
- NEW: `/workspace/vllm-lite/fuzz/fuzz_targets/qwen3_config_json.rs`

- [x] **Step 1: Inspect Qwen3Config**

```bash
cd /workspace/vllm-lite
grep -A20 "pub struct Qwen3Config" crates/model/src/qwen3/config.rs | head -40
```

Note 30+ nested fields.

- [x] **Step 2: Create fuzz target**

```rust
#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_json::from_slice;
use vllm_model::qwen3::Qwen3Config;

fuzz_target!(|data: &[u8]| {
    let _ = from_slice::<Qwen3Config>(data);
});
```

- [x] **Step 3: Build and run**

```bash
cd /workspace/vllm-lite
cargo fuzz build qwen3_config_json 2>&1 | tail -5
cargo fuzz run qwen3_config_json -- -max_total_time=10 2>&1 | tail -20
```

- [x] **Step 4: If bug found**

Same pattern.

- [x] **Step 5: Commit**

```bash
cd /workspace/vllm-lite
git add fuzz/
git commit -m "test(fuzz): add Qwen3Config JSON fuzz target (J-4)"
```

---

## Task J-5: Run All Fuzzers + CHANGELOG (0.5-1 day, Low risk)

**Files:**
- MODIFY: `/workspace/vllm-lite/CHANGELOG.md`

- [x] **Step 1: Run each fuzzer for ~60 seconds**

```bash
cd /workspace/vllm-lite
cargo fuzz run app_config_yaml -- -max_total_time=60 2>&1 | tail -5
cargo fuzz run safetensors_header -- -max_total_time=60 2>&1 | tail -5
cargo fuzz run qwen3_config_json -- -max_total_time=60 2>&1 | tail -5
```

Expected: each runs ~60s, reports coverage / paths / new inputs found.

- [x] **Step 2: Save corpus**

```bash
cd /workspace/vllm-lite
cargo fuzz corpus app_config_yaml -- -max_total_time=0  # lists corpus
# (Don't move corpus, it's in fuzz/corpus/)
```

- [x] **Step 3: Add v29.0 entry to CHANGELOG**

Under `[Unreleased]` → `### Added`:

```markdown
- **Fuzz Testing (v29.0)** — cargo-fuzz infrastructure + 3 fuzz targets:
    - `cargo fuzz` scaffolding at `fuzz/` directory; justfile targets (`fuzz-smoke`, `fuzz TARGET`, `fuzz-build`)
    - `app_config_yaml`: fuzz `serde_saphyr::from_str::<AppConfig>` with arbitrary bytes
    - `safetensors_header`: fuzz `SafeTensors::deserialize` with arbitrary bytes
    - `qwen3_config_json`: fuzz `serde_json::from_slice::<Qwen3Config>` with arbitrary bytes
    - **Bugs found**: [N panics caught] (or "0 panics in 60s × 3 targets")
    - Test count: unchanged (fuzz targets run on-demand, not in `cargo test`)
    - Total commits: 5 (J-1 to J-5)
```

- [x] **Step 4: Commit**

```bash
cd /workspace/vllm-lite
git add CHANGELOG.md
git commit -m "docs(v29.0): CHANGELOG entry + fuzzing summary (J-5)

Fuzz testing milestone complete. 3 targets (app_config_yaml,
safetensors_header, qwen3_config_json) all exercised for 60s.

[Summary of bugs found / corpus size]"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** cargo-fuzz infra + 3 targets + bug fixes
- [x] **Placeholder scan:** each task has explicit commands
- [x] **Dependency order:** J-1 → J-2 → J-3 → J-4 → J-5
- [x] **Risk gates:** every panic found → fix + regression test before commit

---

## Handoff

**Status (2026-06-28):** v29.0 COMPLETE.

All J-1 through J-5 sub-phases landed. 3 fuzz targets (app_config_yaml,
safetensors_header, qwen3_config_json) ran for ~30-60s each. ~10M total
executions across all targets. 0 panics found.

**Key observation**: all three target parsers handle malformed input gracefully
without panicking. This validates the existing defensive `serde::Deserialize`
implementations and the upstream library choices (serde_saphyr, safetensors,
serde_json).

**v29.0 ROI assessment**: Compared to v28.0 (1 bug found), fuzz testing
found 0 bugs. This is expected:
- Property tests target INVARIANTS on structured inputs
- Fuzz tests target PARSER ROBUSTNESS on random bytes
- The parsers here are mature (serde_json, safetensors) or use lenient
  parsing (serde_saphyr's `#[serde(default)]` defaults); they handle bad
  input by returning Err, not panicking.

**Still recommended** for ongoing safety:
- CI could run fuzzers for 30s per target on PRs (catch regressions)
- New parsers should follow this 3-target pattern (yaml/json/binary)

**Next candidates**:
- v30.0: Documentation site (mdbook)
- v30.0: Deferred v27.0 optimizations (BatchComposer Arc clone, etc.)
- v30.0: More fuzz targets (OpenAI request body, JWT validator)
- v30.0: Continuous fuzzing in CI
