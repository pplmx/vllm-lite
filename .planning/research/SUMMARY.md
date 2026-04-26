# Research Summary: vllm-lite v13.0 Host Deployment

**Domain:** LLM Inference Engine Production Deployment
**Researched:** 2026-04-27
**Overall confidence:** MEDIUM-HIGH

## Executive Summary

Host deployment transforms vllm-lite from a single-node inference engine into a production-grade serving platform. The existing codebase already has foundational building blocks—distributed KV cache scaffolding in `crates/dist/`, health check infrastructure in `crates/server/src/health.rs`, and auth/rate limiting—requiring orchestration layer additions rather than ground-up implementation.

The research identifies 4 deployment dimensions: multi-node cluster orchestration, Kubernetes-native deployment, high availability/disaster recovery, and security controls. Each dimension contains table stakes features (required for production) and differentiators (competitive advantage).

**Key insight:** vllm-lite is already 40-60% ready for K8s deployment. The gap is primarily orchestration layer (manifests, probes, metrics) not engine modifications.

## Key Findings

### Stack: What's Already There
- **Distributed KV cache** infrastructure in `crates/dist/src/distributed_kv/` (scaffolding only, needs network transport)
- **Health checker** framework in `crates/server/src/health.rs` (needs HTTP endpoints)
- **Config loading** in `crates/server/src/config.rs` (needs ConfigMap path support)
- **Auth/rate limiting** in `crates/server/src/auth.rs` and `crates/server/src/backpressure.rs`
- **Tensor parallelism** in `crates/dist/src/tensor_parallel/` (single-node multi-GPU)

### Architecture: Deployment Pattern
- Stateless inference pods with StatefulSet for ordinal identity
- Headless Service for peer discovery
- ConfigMap/Secret for configuration (12-factor compliant)
- Lease-based leader election for HA

### Critical Pitfall: Over-engineering
The biggest risk is attempting multi-node distributed KV cache coherence in v13.0. This requires:
- Network transport implementation
- Consistency protocol
- Request routing across nodes

**Recommendation:** Single-node per pod for v13.0, multi-node in v13.1+.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Kubernetes Foundation (v13.0)
- **Addresses:** K8s deployment, health probes, ConfigMap support, graceful shutdown
- **Avoids:** Multi-node coordination, distributed KV cache, custom operators

### Phase 2: Observability & HA (v13.1)
- **Addresses:** Prometheus metrics, HPA support, leadership election, warm standby
- **Avoids:** Custom operators, predictive scaling

### Phase 3: Multi-Node (v13.2)
- **Addresses:** Node discovery, inter-node transport, request routing
- **Note:** Requires distributed KV cache network implementation

### Phase 4: Day-2 Operations (v14.0)
- **Addresses:** Kubernetes Operator, advanced scaling policies, multi-region

**Phase ordering rationale:**
1. Foundation first (K8s basics) — lowest risk, immediate value
2. Observability next — enables operational visibility
3. HA after observability — metrics needed for failover decisions
4. Multi-node last — highest complexity, depends on all prior phases

**Research flags for phases:**
- Phase 1: Likely needs deeper research on GPU operator compatibility
- Phase 3: Likely needs deeper research on inter-node transport (TCP vs RDMA)

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Existing codebase well-understood |
| Features | MEDIUM-HIGH | Based on K8s docs + LLM serving patterns |
| Architecture | MEDIUM | Deployment patterns are standard, vllm-lite specifics need validation |
| Pitfalls | MEDIUM | Common patterns identified, edge cases may exist |

## Gaps to Address

- [ ] GPU operator compatibility testing (NVIDIA Device Plugin version matrix)
- [ ] Memory requirements per model size (needed for resource limits)
- [ ] Inter-node transport latency budget (TCP vs RDMA decision)
- [ ] Multi-tenant isolation requirements (RBAC scope)
