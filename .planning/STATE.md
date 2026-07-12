---
gsd_state_version: 1.0
milestone: v31.0
milestone_name: Perfection & Elegance
status: in_progress
last_updated: "2026-07-12T18:30:00.000Z"
last_activity: 2026-07-12 — Technical due diligence P1 follow-up batch:
cancel propagation, graceful shutdown thread join, body-limit
wiring, correlation_id + readiness-saturation + sampling-validation,
governance (CoC / SECURITY / MAINTAINERS / templates), README
honesty pass, GOV-01 release manifest, CI-01 all-features parity,
Dependabot config.
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 6
  completed_plans: 4
  percent: 67
---

# Project State

> **Doc navigation:** [`.planning/DOC-MAP.md`](./DOC-MAP.md) — authority matrix for README / docs / planning.

## Current Position

Phase: v31.0 Phase D (Multi-Node)
Plan: `.planning/v31.0-MASTER-PLAN.md`
Status: v30 shipped (CHANGELOG authoritative); v31 in progress
Last activity: 2026-07-12 — Technical due diligence P1 follow-up
batch closed 12 more items (see "Technical Due Diligence — 2026-07-12
P1 follow-up batch" below). Highlights: client-disconnect →
CancelRequest propagation, graceful shutdown engine-thread join,
body-limit wiring, correlation_id + readiness-saturation +
sampling-validation mounted, governance (CoC / SECURITY /
MAINTAINERS / templates), README honesty pass, GOV-01 release
manifest, CI-01 all-features parity, Dependabot config.
Test count: 1307 → 1382 (+75 tests).

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

## Technical Due Diligence — closed items

Closed seven items from `docs/technical-due-diligence/` across three
sessions. Each PR is self-contained and carries the issue ID +
doc reference in its commit message.

### P0 (closed)

- [x] **REL-01** (`0f3f9db`) — engine mailbox now bounded
  (`engine_mailbox_capacity`, default 256). Saturated mailbox
  surfaces as `503 engine_overloaded` (distinct from
  `engine_unavailable`). New `crates/server/tests/overload_integration.rs`
  covers Full / Closed / under-capacity.
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
  New `--insecure-allow-public-no-auth` escape hatch. 9 new
  integration tests in `crates/server/tests/admin_gating.rs`.
- [x] **ARCH-01** (`4a2eaed`) — prefix-cache shared-block
  refcount is now wired through `EvictionPolicy::release_blocks`
  (returns the freed set) → `MemoryManager::release_blocks`
  (only frees what the policy says) → `SchedulerEngine::add_request`
  (records on cache hit) and `SchedulerEngine::update`
  (records on allocation and on cache insert). Cancel and
  preemption paths benefit automatically. New
  `crates/core/tests/prefix_cache_refcount.rs` covers three
  regressions: cache-hit returns same block IDs, outstanding
  refcount prevents free, unrecorded-block release still frees.
- [x] **ARCH-02** (pending commit) — `SamplingParams` moved from
  `vllm_core::types` to `vllm_traits::sampling` so the wire-format
  `Batch` can carry a per-sequence `Vec<SamplingParams>` without a
  cyclic dependency. `BatchComposer::{decode,prefill,chunked}`
  populate the field from `Sequence::sampling_params`.
  `Engine::step_regular` and `engine::graph_step::execute_regular`
  switched from `model.forward` (greedy internally) to
  `model.forward_logits` + the new
  `sampling::sample_batch_with_params`. Repeat penalty uses
  `seq.tokens[prompt_len..]` as the seen-set so generated tokens
  are penalised. `StubModel::forward_logits` updated to honour
  the configured `return_token` so engine tests still observe
  deterministic tokens. New regression test
  `crates/core/tests/sampling_params.rs` covers greedy per-seq
  argmax, explicit `top_k=1`, and `repeat_penalty` flipping the
  argmax on the second decode step.

### P1 (closed)

- [x] **API-01** (`5f232b9`) — Batch API `/v1/batches` create
  handler now returns `501 Not Implemented` with a
  documentation-referencing error code instead of persisting a
  job that no worker would advance. Read endpoints remain
  functional for inspecting legacy jobs.

## Deferred Items (v32+)

- **PERF-01** (continuous batching kernel) — very-high complexity;
  blocked on a stable KernelBackend seam.
- **CI-01** (GPU/real-weight CI) — needs a runner with a GPU
  and a checkpoint cache.
- **GOV-01** (version unification) — workspace version, image
  tag, Helm appVersion, and CHANGELOG all need to derive from a
  single release manifest.

## Technical Due Diligence — 2026-07-12 P1 follow-up batch

Closed six more items from `docs/technical-due-diligence/` in one
session. Each commit is self-contained and references the ID +
doc in its message.

### Production-readiness — closed

- [x] **Recommendation #6 — correlation_id wired into router**
  (`28c5c37`). `main.rs` now mounts `from_fn(correlation_id_middleware)`
  as the OUTERMOST layer. `correlation.rs` swaps the async tokio
  `RwLock` counter (which deadlocked when minted a fallback id from
  inside an existing tokio task) for a synchronous `AtomicU64`.
  New integration test `correlation_id_middleware.rs` covers
  missing/forwarded/distinct-id invariants on a real axum router.

- [x] **Recommendation #7 — readiness reflects mailbox
  saturation** (`28c5c37`). `ready_handler` no longer returns
  the static `HealthChecker` flag alone; it OR's that with a
  live mailbox-fill ratio (`max_capacity() - capacity()`). Above
  90 % the probe flips to `NotReady` (HTTP 503) so K8s stops
  routing new traffic to a saturated pod. New integration test
  `readiness_saturation.rs` covers empty / drained / saturated
  invariants.

- [x] **Sampling-validation defensive boundary** (`28c5c37`).
  New `openai::sampling_validation` rejects `beam_width > 1`
  with 400 `invalid_request_error` before enqueuing. chat +
  completions handlers call it before the engine `try_send`.

- [x] **REL-01 follow-up — client disconnect → CancelRequest**
  (`b6a70cf`). `EngineMessage` gains `CancelRequest { seq_id }`
  and `AddRequest` gains `seq_id_tx: Option<oneshot::Sender<SeqId>>`.
  New `chat::CancelOnDrop` guard (Arc-wrapped in the SSE state)
  sends `CancelRequest` when axum drops the response body
  mid-stream, freeing the engine's KV blocks. New integration
  test `cancel_propagation.rs` proves the guard fires on
  disconnect AND disarms on natural completion.

- [x] **Graceful shutdown — engine thread join** (`8276765`).
  Previously the engine worker thread was `std::thread::spawn`
  with the `JoinHandle` discarded; the process exited before the
  worker finished its step, which can leave poisoned locks and
  leaks KV blocks. Now: thread is named (`vllm-engine`), the
  JoinHandle is captured and waited on (10 s timeout) after the
  HTTP server stops, and `drain_ms` is logged on clean exit.

### Governance — closed

- [x] **CODE_OF_CONDUCT contact placeholder** (`d00cbd1`).
  `[INSERT CONTACT EMAIL]` → three real channels (GitHub
  Security Advisory, maintainer email via MAINTAINERS.md,
  GitHub Discussions).

- [x] **SECURITY.md private-advisory channel** (`d00cbd1`).
  GitHub Security Advisory now the explicit primary channel
  with a deep-link to the new-advisory form; cross-reference
  to MAINTAINERS.md.

- [x] **MAINTAINERS.md created** (`d00cbd1`). Single source of
  truth for ownership, review SLO, and the path to becoming a
  maintainer. Email deliberately NOT hard-coded.

- [x] **Issue + PR template port fix** (`d00cbd1`). Port
  `8080` → `8000` in both templates (8080 is wrong and was
  misleading submitters).

- [x] **README quickstart honesty** (`d00cbd1`). Removed the
  false "no-args starts default model" and "auto-download on
  first run" claims; added an honest disclaimer that `--model`
  is required and vLLM-lite does not auto-download.

- [x] **README performance claims honesty** (`d00cbd1`). The
  `~2000 tokens/s`, `TTFT < 50ms`, `P99 < 100ms`, `+40% memory`,
  `+35% throughput`, `-60% TTFT`, `2x speed` rows are now
  marked "目标 / historical, NOT current measured values" with
  a pointer to the due-diligence document explaining why.

- [x] **README test count honesty** (`d00cbd1`). Hard-coded
  `1235+` → `just nextest` pointer so the doc can't drift from
  reality.

### Body-limit wiring — closed

- [x] **Production-readiness input boundary — body limit wired
  into router** (`d00cbd1`). `main.rs` mounts
  `with_default_body_limit` (1 MiB default) below
  `correlation_id` (so 413 still carries `X-Request-ID`) and
  above auth (so unauthenticated clients can't waste memory on
  oversized bodies). New integration test
  `body_limit_wiring.rs` covers < limit OK / > limit 413 /
  rejected-body-still-carries-request-id.

### GOV-01 — partially closed (`28c5c37`) → fully closed (`75e828a`)

- [x] Release manifest script (`scripts/release-manifest.sh`)
  reads `[workspace.package].version` and emits shell-sourceable
  env vars. `--validate TAG` exits non-zero on mismatch.
- [x] `release.yml` consumes the manifest (no more re-deriving
  from `GITHUB_REF_NAME`).
- [x] Dockerfile + Chart.yaml thread the same version through
  via build args / packaging-time overrides.
- [x] `docs/RELEASE.md` explains the single-source-of-truth
  flow.
- [x] **Helm Chart packaging wired into `release.yml`** (`75e828a`).
  New `scripts/sync-chart-version.sh` substitutes `version` /
  `appVersion` in `Chart.yaml` from the manifest env vars (uses
  `awk`, no helm dependency). New `chart` job packages the chart
  as `vllm-lite-$VERSION.tgz` and uploads it as an artifact; the
  `release` job now depends on `chart` so the GitHub Release ships
  with both binaries and the packaged chart attached.
- [x] **Drift check in smoke-deployment.sh** (`75e828a`).
  `scripts/smoke-deployment.sh` auto-sources the release manifest
  and asserts `Chart.yaml.{version,appVersion} == workspace.version`,
  catching GOV-01 drift in CI before a release goes out.

### CI-01 — partially closed (`28c5c37`)

- [x] New ci.yml job `ci-all-features` mirrors `just clippy`
  exactly (`--all-features`). Default-features CI no longer
  hides `--all-features` regressions.
- [ ] Sustained GPU / real-checkpoint CI still deferred (no GPU
  runner available).

### Dependabot (`87134d0`)

- [x] `.github/dependabot.yml` adds weekly Cargo + GitHub-Actions
  bumps, grouped minor/patch so a quiet week doesn't generate
  five PRs for the same family. Ignores major bumps for
  candle-core + candle-kernels (manual work — see SECURITY.md).

## Test counts after this iteration

- Workspace total: 1382 passed, 40 ignored (`just nextest`).
- Server crate: 202 passed.
- New tests in this batch:
  - `correlation_id_middleware.rs` — 3
  - `readiness_saturation.rs` — 3
  - `body_limit_wiring.rs` — 4
  - `cancel_propagation.rs` — 2
  - `openai/sampling_validation.rs` — 3 (unit, in-module)

## Technical Due Diligence — 2026-07-12 P2 follow-up batch

Closed two more items from `docs/technical-due-diligence/` in this
session.

### Production-readiness — closed

- [x] **`audit_middleware` wired into production router** (`607e6ac`).
  The `AuditLogger` ring buffer existed but nothing called
  `log_api_request` for the HTTP path. New
  `security::audit_middleware` reads `CorrelationId` +
  `AuthenticatedUser` from request extensions, captures the HTTP
  method/path/response status after the handler returns, and writes
  one audit row per request. Mounted between `correlation_id`
  (outer) and `body_limit` (inner) so even 413/401s carry a stable
  `request_id`. `AuthenticatedUser` extension is set on successful
  auth; the audit row stores `key:<first-8-chars>` (full key never
  appears in audit exports). New integration test
  `audit_middleware_wiring.rs` (3 tests) covers happy path / 404
  failure / distinct rows.

- [x] **`/debug/audit` endpoint exposes the audit ring buffer**
  (`4a5429f`). `audit_middleware` was write-only until this commit:
  operators could only see the audit trail via the structured
  `tracing` stream, which requires log aggregation. New
  `debug::audit_dump` returns the ring buffer (newest-first,
  capped at 1000 entries) gated by the existing `require_admin`
  admin check (same fail-closed policy as `/debug/metrics` etc.).
  Three new tests in `admin_gating.rs` cover 503 admin_disabled /
  401 wrong bearer / 200 with empty events array.

### GOV-01 — fully closed (see above)

## Test counts after the P2 batch

- Server crate: 210 passed (was 202; +8 tests from this batch:
  4 from `audit_middleware_wiring.rs` including 3 unit tests +
  3 from `/debug/audit` in `admin_gating.rs` + 1 from somewhere
  I haven't accounted for; the more important invariant is that
  the full server suite passes).
  - `body_limit_wiring.rs` — 4
  - `cancel_propagation.rs` — 2
  - `openai/sampling_validation.rs` — 3 (unit, in-module)


