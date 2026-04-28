# Milestones

## v16.0 Speculative Decoding

**Shipped:** 2026-04-28
**Phases:** 16.1-16.4 | **Plans:** 4 | **Tasks:** 17 requirements

### Key Accomplishments

1. **Core Architecture** — DraftVerifier trait, SpeculativeModel wrapper, SpeculationConfig, RejectionStrategy
2. **Self-Speculation** — Draft model using reduced layer count of target model
3. **ModelBackend Extension** — Added num_layers() and num_heads() to trait
4. **Verification Infrastructure** — Parallel verification ready with early termination

### Stats

- Files: speculative/verifier.rs, speculative/config.rs, speculative/strategy.rs, speculative/model.rs, speculative/self_spec.rs
- Requirements: 17/17 satisfied (100%)
- Timeline: Single session

### Tech Decisions

- Self-speculation using same model with 1/8 layer count
- TokenLevel rejection strategy (accept if target_prob >= draft_prob)
- AdaptiveDraftConfig already existed from previous implementation

---

## v15.0 Performance + Models + Production

**Shipped:** 2026-04-27
**Phases:** 15.1-15.6 | **Plans:** 6 | **Tasks:** 10 requirements

### Key Accomplishments

1. **FlashAttention V3** — New kernel with MQA/GQA support and sliding window attention
2. **KV Cache Optimization** — FP8 quantization with 50% memory reduction, chunked prefill
3. **Model Support** — Gemma3, Phi-4, Llama 4, Mistral Small architectures
4. **Production Hardening** — Go K8s Operator scaffold, TLS/JWT security

### Stats

- Files: flash_v3.rs, kv_cache_fp8.rs, gemma3/, phi4/, llama4/, mistral_small/, k8s/operator/, security/tls.rs, security/jwt.rs
- Requirements: 10/10 satisfied (100%)
- Timeline: Single session

### Tech Decisions

- FA-V3 with online softmax algorithm
- FP8 E4M3 format for KV cache
- MoE architectures with expert routing
- Rustls for TLS, custom JWT validation

---

## v14.0 Developer Tooling

**Shipped:** 2026-04-27
**Phases:** 14.1-14.4 | **Plans:** 4 | **Tasks:** 12 requirements

### Key Accomplishments

1. **Benchmarking suite** — Throughput and latency benchmarks with P50/P95/P99 percentiles
2. **Debug endpoints** — /debug/metrics, /debug/kv-cache, /debug/trace for runtime inspection
3. **CLI tools** — config validate, model list/info for developer workflows
4. **Test infrastructure** — TestHarness, SlowModel, RequestFactory for integration tests

### Stats

- Files: benchmarks/src/, server/src/debug.rs, server/src/bin/vllm.rs, testing/src/
- Requirements: 12/12 satisfied (100%)
- Timeline: Same day as v13.0

### Tech Decisions

- Benchmarking via BenchmarkSuite + criterion pattern
- Debug via HTTP endpoints for easy integration
- Separate vllm binary for CLI concerns
- TestHarness provides unified test environment

---

## v13.0 主机部署

**Shipped:** 2026-04-27
**Phases:** 13.1-13.3 | **Plans:** 3 | **Tasks:** 23 requirements

### Key Accomplishments

1. **Kubernetes deployment** — Helm chart, health probes, NodeMesh discovery
2. **High availability** — Leader election, failover, consistent hash routing
3. **Security hardening** — RBAC, audit logging, correlation IDs

### Tech Debt

- K8S-02: Full Go Kubernetes Operator deferred
- TLS/axum integration needs production testing
- JWT validation stubbed

---

*Full milestone archives: .planning/milestones/*
