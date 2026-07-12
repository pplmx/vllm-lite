---
gsd_state_version: 1.0
milestone: v31.0
milestone_name: Perfection & Elegance
status: in_progress
last_updated: "2026-07-12T13:30:00.000Z"
last_activity: 2026-07-12 — Technical due diligence P0 batch: REL-01, OBS-01, DEP-01, SEC-01 committed
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 6
  completed_plans: 3
  percent: 50
---

# Project State

> **Doc navigation:** [`.planning/DOC-MAP.md`](./DOC-MAP.md) — authority matrix for README / docs / planning.

## Current Position

Phase: v31.0 Phase D (Multi-Node)
Plan: `.planning/v31.0-MASTER-PLAN.md`
Status: v30 shipped (CHANGELOG authoritative); v31 in progress
Last activity: 2026-07-12 — Technical due diligence P0 batch committed
(`0f3f9db` REL-01, `32b1f71` OBS-01, `3dcec01` DEP-01, `3b97440` SEC-01)

## v30 Outcomes (shipped per CHANGELOG)

- Phases 8–19: file splits, YaRN, architecture unification, OPS-05 seams
- Phases K–P: mutation testing, fuzz CI, proptest, docs, ADR, tutorial
- 1235 tests; doc coverage **70.8% raw / 67.4% real** (target 65% real ✅)

## v31 Active Work

- [x] Chunked prefill continuation — all architectures + unit tests
- [x] `docs/architecture.md`, README honesty, OPERATIONS.md, ADR-019
- [x] `.planning/DOC-MAP.md` — doc authority matrix
- [x] Doc coverage 55% → **67.4% real** / 70.8% raw (target 65% real)
- [x] Phase 12e `public-api-check` in CI + baselines refreshed
- [x] **Phase 31-D KV block transfer (OPS-31d)** — `TransferKVBlock` gRPC RPC, `BlockDataSource` trait, `DistributedKVCache::fetch_block` fan-out, 64 MiB symmetric message limit. Test count: vllm-dist 64 → 87 (+23); workspace 1307 → 1338 (+31). Phase plan: `.planning/phase-19/ops-31d-kv-block-transfer.md`.

## Technical Due Diligence P0 (started 2026-07-12)

Closed four of the six P0 items called out in
`docs/technical-due-diligence/`. Each PR is self-contained and
carries the issue ID + doc reference in its commit message.

- [x] **REL-01** (`0f3f9db`) — engine mailbox now bounded
  (`engine_mailbox_capacity`, default 256). Saturated mailbox
  surfaces as `503 engine_overloaded` (distinct from
  `engine_unavailable`). All chat/completions/embeddings handlers
  updated. New `crates/server/tests/overload_integration.rs`
  covers the three paths (Full / Closed / under-capacity).
- [x] **OBS-01** (`32b1f71`) — `/metrics` now reads from
  `engine.scheduler.metrics` instead of a freshly constructed
  duplicate collector. New `crates/server/tests/metrics_wiring.rs`
  guards the Arc-sharing invariant.
- [x] **DEP-01** (`3dcec01`) — Dockerfile bumped to MSRV 1.88 +
  `--locked` + named `runtime` stage + curl-based HEALTHCHECK;
  docker-compose points at real paths and emits `VLLM_*` env vars;
  Helm chart emits `VLLM_MODEL` / `VLLM_MAX_BATCH_SIZE` /
  `VLLM_KV_BLOCKS` / `VLLM_TENSOR_PARALLEL_SIZE`. New
  `scripts/smoke-deployment.sh` re-checks every change in CI.
  Added `rust-toolchain.toml`.
- [x] **SEC-01** (`3b97440`) — `/debug/*` and `/shutdown` now
  require admin auth (valid Bearer when keys configured; 503
  `admin_disabled` when none configured). Startup logs a loud
  warning when binding to a non-loopback address without auth.
  New `--insecure-allow-public-no-auth` escape hatch for
  intentional internal deployments. 9 new integration tests in
  `crates/server/tests/admin_gating.rs`.

## Deferred Items (v32+)

- **ARCH-01** (prefix cache refcount) — still open; root cause is
  the missing block-ownership contract between `RadixTree` and
  `MemoryManager`. The minimal-viable fix is to plumb a single
  `BlockManager`; longer-term this needs the broader refactor
  called out in `docs/technical-due-diligence/architecture-performance.md#arch-01`.
- **ARCH-02** (sampling params) — still open; requires adding
  `SamplingParams` to the `Batch` value type and switching
  `Engine::step_regular` from `model.forward` (which does greedy
  internally) to `model.forward_logits` + `core::sampling::sample_batch`.
  Multi-file refactor; deferred to a dedicated phase.
- **GOV-01**, **PERF-01**, **API-01**, **CI-01** — see
  `docs/technical-due-diligence/governance-release.md` and
  `architecture-performance.md`.

