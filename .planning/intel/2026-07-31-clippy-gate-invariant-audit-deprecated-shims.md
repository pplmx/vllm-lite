# RIL: clippy gate + invariant audit + deprecated shim removal — 2026-07-31

## Context

Autonomous engineering loop, v31.0 "Perfection & Elegance". Two rounds completed,
3 commits on `main` (all green; 19 test failures are environmental only — no GPU,
sandbox blocks socket binding for gRPC/OTLP tests).

## Round 1 — AGENTS.md quality-gate compliance (commits 33c9285d, a58ae6f9, ec9f5a58)

### Problem
- `cargo clippy --all-targets --workspace --all-features -- -D warnings` (the
  AGENTS.md pre-commit gate) failed with 13 errors across 6 files.
- Production `unwrap()`/`expect()` sites without `// invariant:` comments
  (AGENTS.md invariant policy — "treat it as a bug").

### Changes
- Clippy fixes: `float_cmp` allows with rationale on exact-literal f32 test
  assertions (vllm-traits/src/sampling.rs, core/src/types/request.rs);
  `unsafe_code` allow on the env-var test module (sequence_packing.rs) with
  SAFETY rationale; moved `use` before statements (multi_gpu_scheduler.rs);
  `u64::from(i)` + collection-kept-alive allow (gpu_engine_e2e.rs);
  `future_not_send` allow with rationale (multi-node gRPC bootstrap — only call
  site awaits directly on the main task); significant-drop-tightening resolved
  by restructuring the lock guard to per-iteration acquisition in
  paged_kv_cache_wrapper.rs (read_layer_block returns owned Vecs).
- `// invariant:` comments added at all remaining production unwrap/expect
  sites (16 sites across 7 files): rope.rs dims4() x5, auth.rs response
  builders x2, health_handlers.rs RwLock read, openai/chat.rs serialize x4 +
  seq_id_rx, openai/completions.rs seq_id_rx x2, paged_kv_cache_wrapper.rs
  Arc::try_unwrap, testing/device.rs.

### Verification
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: 0 errors.
- `cargo fmt --all --check` clean; `RUSTDOCFLAGS="-D warnings" cargo doc --workspace` clean.
- Tests: 2079 passed / 19 env-fail / 67 skipped — identical to baseline.

## Round 2 — Deprecated public shims removed (commit 5929c4b9, task-003)

### Problem
31-C "Migrate deprecated import paths" was the last open master-plan checkbox.
Three deprecated public surfaces with zero internal callers and elapsed
deprecation windows: `vllm_core::speculative::draft_registry` module (orphan
file — not compiled since the `registry/` reorganization), `vllm_dist::NcclAllReduce`
alias, `vllm_server::openai::types::EmbeddingData` alias.

### Changes
- Deleted `crates/core/src/speculative/draft_registry.rs` (orphan).
- Removed `NcclAllReduce` type alias + re-exports (dist lib.rs, tensor_parallel/mod.rs);
  removed the compile-only alias test; fixed the two doc comments referencing it.
- Removed `EmbeddingData` alias (server openai types.rs).
- Refreshed public-api baselines via `just public-api-baseline` (API shrink allowed;
  refresh also caught up stale entries so baselines match current toolchain output).
- `bash .planning/phase-12e/check-public-api.sh` → OK.
- Marked 31-C master-plan item done; CHANGELOG entry added.

### Verification
- Build (default + all-features) green; tests: 2104 run (6 new tests from a
  concurrent agent's middleware work), 2085 passed / 19 env-fail / 67 skipped.

## Notes
- Sandbox environment: rustup crashes under sandbox (SIGCHLD handler blocked),
  so `cargo public-api` / `just ci` pieces requiring rustup need escalated
  permissions. gRPC/OTLP/CUDA tests fail in-sandbox (no GPU; socket bind blocked).
- A concurrent agent instance worked in the same workspace (middleware stack
  ordering fix in crates/server + its own RIL graph updates); coordinate by
  avoiding in-progress files/components.
