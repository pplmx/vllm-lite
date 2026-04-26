# Milestones

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
