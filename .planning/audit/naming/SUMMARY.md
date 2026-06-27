# Naming Audit Summary

**Generated:** 2026-06-27
**Phase:** 21 — Naming Audit (v19.0 milestone)
**Detailed findings:** [`REPORT.md`](./REPORT.md)
**Source files scanned:** 286 (excludes `target/`)
**Public types surveyed:** 345
**Public functions surveyed:** 842

## Total findings: 26

- **P0:** 0
- **P1:** 7
- **P2:** 19

### Distribution by dimension

| Dimension | Findings | P0 | P1 | P2 |
|-----------|----------|-----|-----|-----|
| NAME-01 File naming   | 4  | 0 | 2 | 2 |
| NAME-02 Type naming   | 9  | 0 | 1 | 8 |
| NAME-03 Function naming | 4 | 0 | 1 | 3 |
| NAME-04 Variable naming | 5 | 0 | 1 | 4 |
| NAME-05 Module naming  | 4 | 0 | 2 | 2 |

## Prioritized Findings

| ID | Dimension | Description | Severity | Source | Effort |
|----|-----------|-------------|----------|--------|--------|
| **NAME-F-01** | NAME-05 | **Orphan module**: `kv_cache_fp8.rs` (289 lines, defines `KvCacheDtype` enum + `Fp8Quantizer`) not declared in `components/mod.rs` — module is unreachable from crate root | P1 | `crates/model/src/components/kv_cache_fp8.rs` | 1h (add `pub mod kv_cache_fp8;`) |
| **NAME-F-02** | NAME-05 | **Orphan module**: `server/src/debug.rs` (175 lines, debug HTTP endpoints) not declared in `server/src/lib.rs` — module is unreachable | P1 | `crates/server/src/debug.rs` | 0.5h (add `pub mod debug;`) |
| **NAME-F-03** | NAME-01 | Stage-info-named file: `engine_v18_wiring.rs` (version `v18` in name, user-reported pain point) | P1 | `crates/core/tests/engine_v18_wiring.rs` | 0.5h (rename to `engine_wiring.rs`) |
| **NAME-F-04** | NAME-05 | Test files in `src/` not registered in `mod.rs`: `qwen3/model_tests.rs`, `qwen3_5/model_tests.rs`, `qwen3_5/speculative_tests.rs` (dead test code, may not be compiled) | P1 | `crates/model/src/qwen3/model_tests.rs:1`, `crates/model/src/qwen3_5/model_tests.rs:1`, `crates/model/src/qwen3_5/speculative_tests.rs:1` | 1h (move to `tests/` or convert to `#[cfg(test)] mod tests {}`) |
| **NAME-F-05** | NAME-04 | `data` variable name used 31 times — violates AGENTS.md "descriptive" guideline; appears in production code (`loader/checkpoint.rs`, `loader/io.rs`, `server/openai/chat.rs`) | P1 | `crates/model/src/loader/checkpoint.rs:44`, `crates/model/src/loader/io.rs:151-171`, `crates/server/src/openai/chat.rs:222-242` | 3h (rename; mostly test code + ~6 prod sites) |
| **NAME-F-06** | NAME-02 | `EmbeddingData` struct has potentially redundant `Data` suffix (output type already implies data) | P1 | `crates/server/src/openai/types.rs` | 0.5h (consider `Embedding` or `EmbeddingItem`) |
| **NAME-F-07** | NAME-03 | Inconsistent verb prefixes for read operations: `get_*` (14), `load_*` (8), `read_*` (4) — split is semantic but undocumented | P1 | scattered (e.g., `core/src/kv_cache/mod.rs`, `loader/checkpoint.rs`) | 1h (formalize verb policy in AGENTS.md) |
| NAME-F-08 | NAME-01 | Versioned filename `flash_v3.rs` carries algorithm version; could disambiguate as `flash_attention_v3.rs` | P2 | `crates/model/src/components/attention/flash_v3.rs` | 0.5h (rename) |
| NAME-F-09 | NAME-01 | File naming inconsistent across crates: `crates/model/src/qwen3_config.rs` (top-level) vs `crates/model/src/qwen3/` (submodule) — split concept | P2 | `crates/model/src/qwen3_config.rs` | 1h (move into `qwen3/` submodule) |
| NAME-F-10 | NAME-02 | `*Manager` suffix used 6 times — semantically OK but no formal guidance in AGENTS.md on when suffix is required | P2 | `crates/core/src/scheduler/{preemption,memory,...}` | 0.5h (doc update only) |
| NAME-F-11 | NAME-02 | `NodeInfo` struct uses `Info` suffix — marginal readability gain | P2 | `crates/dist/src/` | 0.5h (consider `NodeSummary` or `NodeMetadata`) |
| NAME-F-12 | NAME-02 | `RequestFactory` uses `Factory` suffix — only one such occurrence, OK | P2 | `crates/testing/src/request_factory.rs` | (no action) |
| NAME-F-13 | NAME-02 | `FlashAttentionV2`/`V3` carry algorithm versions in type names — conventional but version-bearing naming | P2 | `crates/model/src/components/attention/flash_v3.rs` | (no action; conventional) |
| NAME-F-14 | NAME-03 | Mixed `create_*` (7) and `build_*` (5) usage — semantically distinct but undocumented | P2 | scattered | 1h (AGENTS.md verb policy update) |
| NAME-F-15 | NAME-03 | No single-letter `fn` violations detected — fully snake_case | P2 | (negative finding) | (no action) |
| NAME-F-16 | NAME-03 | Async/sync split tracks resource nature (e.g., async for I/O, sync for in-memory) — consistent but undocumented | P2 | scattered | 0.5h (AGENTS.md update) |
| NAME-F-17 | NAME-04 | Single-letter tensor variables (`q`, `k`, `v`, `o`, `b`, `c`, `h`, `z`, `d`, `x`) used 472 times in non-test source — ML convention, technically violates AGENTS.md | P2 | `crates/model/src/components/attention/{gqa,mla,flash_v3}.rs`, `crates/model/src/components/{ssm,gated_delta}.rs`, etc. | 1h (AGENTS.md exemption for tensor-math) |
| NAME-F-18 | NAME-04 | Non-tensor single-letter variables in scheduler/sampling code (`r`, `k`, `pa`, `pb`, `id`) — could be renamed | P2 | `crates/core/src/sampling.rs:41-142`, `crates/core/src/scheduler/engine.rs:244-245` | 1h (rename) |
| NAME-F-19 | NAME-04 | Module naming test files mix `src/` and `tests/` locations — convention unclear | P2 | `crates/model/src/qwen3/model_tests.rs` (in src/) vs `crates/model/tests/qwen3_rope.rs` (in tests/) | 0.5h (AGENTS.md doc) |
| NAME-F-20 | NAME-05 | Module depth max=7 in `vllm-core` and `vllm-model` — deepest path is `scheduler/policy/trait_def.rs` (5 levels from crate root) | P2 | `crates/core/src/scheduler/policy/trait_def.rs` | (no action; justified by per-arch subdirs) |
| NAME-F-21 | NAME-05 | KV cache concept split across 3 locations: `core/src/kv_cache/`, `model/src/kv_cache.rs`, `model/src/components/kv_cache_fp8.rs` (orphan) | P2 | `crates/model/src/components/kv_cache_fp8.rs` (orphan, see NAME-F-01) | (covered by NAME-F-01) |
| NAME-F-22 | NAME-05 | Hand-authored files have 0 basename/mod mismatch — only generated file `vllm.distributed.rs` has mismatch (acceptable) | P2 | `crates/dist/src/generated/vllm.distributed.rs` | (no action; generated) |
| NAME-F-23 | NAME-01 | Hyphenated filenames: (none) — clean | P2 (negative finding) | — | (no action) |
| NAME-F-24 | NAME-01 | Uppercase in filenames: only generated `vllm.distributed.rs` (acceptable) | P2 (negative finding) | — | (no action) |
| NAME-F-25 | NAME-02 | No lowercase type names — fully PascalCase | P2 (negative finding) | — | (no action) |
| NAME-F-26 | NAME-04 | No `tmp`/`foo`/`bar` usage; only `data` (covered by NAME-F-05) | P2 (negative finding) | — | (no action) |

---

## Top 3 Action Items

1. **Fix orphan modules (NAME-F-01, NAME-F-02)** — Both `kv_cache_fp8.rs` (289 lines) and `debug.rs` (175 lines) are unreachable from their crate roots because `mod kv_cache_fp8;` and `mod debug;` declarations are missing. This is **dead code** in the current build; either register the modules or remove the files. **Effort: 1.5h total.**

2. **Rename `engine_v18_wiring.rs` (NAME-F-03)** — User-reported pain point. The file name encodes a version (`v18`) that will be meaningless once v18 ships. Rename to `engine_wiring.rs` (or similar semantic name). This is the canonical example of "stage-info named file" anti-pattern. **Effort: 0.5h.**

3. **Resolve test files in `src/` (NAME-F-04)** — Three test files (`qwen3/model_tests.rs`, `qwen3_5/model_tests.rs`, `qwen3_5/speculative_tests.rs`) live in `src/` but are not registered in `mod.rs`. They are **dead test code** that does not run in `cargo test`. Either move to `crates/model/tests/` directory or convert to `#[cfg(test)] mod tests {}` blocks within `qwen3/mod.rs` / `qwen3_5/mod.rs`. **Effort: 1h.**

---

## Suggested v20.0+ Phase

**Phase 25 — Naming Cleanup** (proposed, advisory)

Combine these in a single dedicated rename/registration phase:

1. Fix orphan modules: register `kv_cache_fp8`, `debug` (or delete if unused).
2. Rename `engine_v18_wiring.rs` → `engine_wiring.rs`.
3. Move or convert `model_tests.rs` / `speculative_tests.rs` test files.
4. Rename `data` → contextual names (`quantized`, `weights`, `raw_tensor`, etc.) — primarily test code, low risk.
5. Rename `EmbeddingData` → `Embedding` (or document the `*Data` suffix convention).
6. Update AGENTS.md:
   - Document verb policy: `get_*` (accessor), `load_*` (resource acquisition), `read_*` (stream I/O), `create_*` (resource), `build_*` (builder pattern).
   - Add explicit exemption for tensor-math single-letter variables (`q`, `k`, `v`, `o`, `b`, `c`, `h`, `z`, `d`, `x`) in attention/SSM/MLP modules.
   - Clarify test file location: prefer `crates/{crate}/tests/` for integration tests; `#[cfg(test)] mod tests {}` for unit tests; avoid test files in `src/` outside of `mod tests` blocks.
   - Add guidance on `*Manager`/`*Info`/`*Data` suffixes: suffix is required when the bare name is ambiguous (e.g., `Node` could be a graph node; `NodeInfo` clarifies it is metadata).

**Effort estimate:** 8-12 hours of focused renaming + documentation work.

**Risk:** Low — most changes are mechanical. The orphan module fix (NAME-F-01/02) is highest-impact because it could expose latent compilation errors that the current dead-code state hides.

---

## Cross-references

- **Phase 20 (Architecture Audit)** — likely flags `kv_cache_fp8.rs` as well (orphan module = unreachable code = ARCH concern).
- **Phase 22 (Docs Audit)** — should verify `AGENTS.md` accuracy on the conventions actually enforced (this audit found ~472 single-letter variables, contradicting the stated "no single-letter except indices" rule).
- **Phase 24 (Synthesis)** — should correlate this audit's findings with architecture/test/docs audits to produce a unified remediation backlog.

---

*End of SUMMARY.md*
