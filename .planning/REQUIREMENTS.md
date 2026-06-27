# Requirements: vllm-lite

**Defined:** 2026-06-27 (v20.0); 2026-06-27 (v21.0 added); 2026-06-27 (v22.0 added)
**Milestone:** v22.0 Production Hardening
**Core Value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## v22.0 Requirements (CURRENT)

Closure of remaining tech debt from v18.0-v21.0 — production hardening. Each requirement maps to one sub-phase (v22.1-v22.4). Source: PROJECT.md Active section, CONCERNS.md (2026-05-13 baseline, items still relevant post-v21.0).

### Phase 36 (v22.1): Critical Bug Fixes (~25h)

- [ ] **OPS-02**: Fix `Engine::step()` speculative-mode hang (pre-existing bug; unblocks 2 Phase 19 e2e tests currently `#[ignore]`d)
- [ ] **OPS-03**: Resolve 5 pre-existing cargo doc broken-link warnings in `engine.rs` + `components/mod.rs`
- [ ] **GGUF-01**: Resolve actionable TODO in `crates/model/src/quantize/gguf.rs` (gguf parser placeholder from v20.6 CMT-02)
- [ ] **FINAL-01**: All 1146+ tests remain green post-fix

### Phase 37 (v22.2): Security Hardening (~25h)

- [ ] **SEC-01**: Implement JWT cryptographic signature verification — HMAC-SHA256 for `secret`-based JWTs; RSA/ECDSA for `public_key_pem`-based JWTs
- [ ] **SEC-02**: Wire `RbacMiddleware` into request pipeline (currently no-op pass-through at `rbac.rs:82-84`)
- [ ] **SEC-03**: Add request size limits via `tower_http::limit::RequestBodyLimitLayer`
- [ ] **SEC-04**: Audit log integration test (verify `security/audit.rs` emits events for authenticated requests)
- [ ] **SEC-05**: Move hardcoded Grafana credentials from `docker-compose.yml` to `.env` file (already gitignored)
- [ ] **SEC-06**: TLS hardening — replace `unwrap()` with proper error in `security/tls.rs:63`
- [ ] **FINAL-01**: Auth/middleware integration tests pass; no new security regressions

### Phase 38 (v22.3): Production Polish (~15h)

- [ ] **RFU-05**: Migrate from `std::sync::Mutex` → `parking_lot::Mutex` in scheduler/engine paths (eliminates poison check; covers 24 mutex `.lock().unwrap()` sites from CONCERNS.md)
- [ ] **OPS-01**: Decide fate of `speculative.rs` mock usage in production paths (real draft loading or document mock-only status)
- [ ] **PERF-01**: Replace `MlaKvCache::write_compressed` full-cache materialization with `Tensor::slice_assign` or equivalent
- [ ] **PERF-02**: Replace `model_type.to_lowercase()` in arch detection with `eq_ignore_ascii_case` (avoid per-load String allocation)
- [ ] **PERF-03**: Replace `once_cell::sync::Lazy` with `std::sync::LazyLock` in `arch/registry.rs` (Rust 1.80+)
- [ ] **DOC-01**: Verify cargo doc warnings resolved (carry-over from OPS-03 if any remain)
- [ ] **FINAL-01**: All 1146+ tests remain green post-polish

### Phase 39 (v22.4): Engine Refactor + Final Verification (~10h)

- [ ] **ARF-06**: Split `engine.rs` God module (1,038 LOC) into focused sub-modules (deferred from v20.0 Phase 27)
- [ ] **ARF-07**: Unify `engine/spec_dispatch` tree post-ML-02 (collapse duplicate abstractions)
- [ ] **FINAL-01**: All 1146+ tests remain green post-refactor
- [ ] **FINAL-02**: `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [ ] **FINAL-03**: `cargo fmt --all --check` clean
- [ ] **FINAL-04**: `cargo test --workspace --all-features` ≥ 1146 tests pass
- [ ] **FINAL-05**: `.planning/PROJECT.md` and `.planning/STATE.md` updated with v22.0 outcomes

## v21.0 Requirements (HISTORICAL — Shipped 2026-06-27)

All 38 of 42 v21.0 requirements satisfied (4 deferred: API-04 already in Phase 27; API-10 traits intentionally not object-safe). See `v21.0-MILESTONE-AUDIT.md` for full verification.

## v20.0 Requirements (HISTORICAL — Shipped 2026-06-27)

All 48 requirements satisfied. See `v20.0-MILESTONE-AUDIT.md` for full verification.

## v2 Requirements (deferred)

Tracked but not in v22.0 scope:

### New model capabilities

- **NMC-01**: Long context support (>32K tokens; modern LLMs support 128K-200K)
- **NMC-02**: Multimodal/Vision encoder (replace placeholder `components/vision.rs` with real implementation)
- **NMC-03**: Tool calling / Function calling (OpenAI-compatible `tools` array partially supported; full implementation pending)

### Operational

- **OPS-04**: Real-model benchmark (vs stub backend) — deferred from v18.0; requires GPU environment
- **OPS-05**: Multi-node / vllm-dist resurrection (feature-gated; vllm-dist investment decision in ADR-015)

### Architecture follow-ups

- **ARF-08**: Dynamic KV cache block allocation with memory pressure signals
- **ARF-09**: Chunked prefill production rollout (deferred from v15.0; chunks interleaved with decode for better batching)

### Refactor follow-ups

- **RFU-06**: Doc coverage push 97.8% → 99%+ (if v22.x doesn't push higher incidentally)

## Out of Scope (v22.0)

Explicitly excluded from v22.0:

| Feature | Reason |
|---------|--------|
| Long context (>32K) | New capability, deferred to v23.0+ candidate |
| Multimodal/Vision | New capability, deferred to v23.0+ candidate |
| Multi-node / vllm-dist resurrection | Feature-gated; multi-node work separate cycle |
| Real-model benchmarks | Requires GPU environment (no GPU currently) |
| Performance optimization beyond audit findings | Orthogonal to hardening scope |
| New architectures (Falcon, DeepSeek, etc.) | Out of scope for hardening |
| Tree-based speculation | Too complex (carried from v18.0) |
| Online fine-tuning | 长期愿景 |
| WebAssembly support | 长期愿景 |

## Traceability

### v22.0 Requirements (CURRENT)

| Requirement | Phase | Status |
|-------------|-------|--------|
| OPS-02      | Phase 36 | Pending |
| OPS-03      | Phase 36 | Pending |
| GGUF-01     | Phase 36 | Pending |
| SEC-01      | Phase 37 | Pending |
| SEC-02      | Phase 37 | Pending |
| SEC-03      | Phase 37 | Pending |
| SEC-04      | Phase 37 | Pending |
| SEC-05      | Phase 37 | Pending |
| SEC-06      | Phase 37 | Pending |
| RFU-05      | Phase 38 | Pending |
| OPS-01      | Phase 38 | Pending |
| PERF-01     | Phase 38 | Pending |
| PERF-02     | Phase 38 | Pending |
| PERF-03     | Phase 38 | Pending |
| DOC-01      | Phase 38 | Pending |
| ARF-06      | Phase 39 | Pending |
| ARF-07      | Phase 39 | Pending |
| FINAL-01 (per phase) | Phase 36-39 | Pending |

### v21.0 Requirements (HISTORICAL — Shipped 2026-06-27)

38/42 satisfied (4 deferred with rationale). See v21.0 milestone audit for details.

### v20.0 Requirements (HISTORICAL — Shipped 2026-06-27)

48/48 Complete ✅

**Coverage:**
- v22.0 requirements: 21 total (4 bug + 6 sec + 6 polish + 5 engine+FINAL across 4 phases)
- Mapped to phases: 21
- Unmapped: 0 ✓
- v21.0 historical: 38/42 Complete ✅
- v20.0 historical: 48/48 Complete ✅

---

*Requirements defined: 2026-06-27 (v20.0)*
*Last updated: 2026-06-27 after v22.0 Production Hardening scope definition (21 requirements across Phase 36-39)*
