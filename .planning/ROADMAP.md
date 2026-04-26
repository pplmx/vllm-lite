# Phase 13 Roadmap: 主机部署

## Overview

**Milestone:** Phase 13 — 主机部署
**Core Value:** Production-ready host deployment with cluster, K8s, HA, ops, and security
**Phases:** 3 | **Requirements:** 23 | **Started:** 2026-04-27

---

## Phase Summary

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 13.1 | K8s 基础 | Kubernetes 部署基础设施 | K8S-01~05, OPS-01~02, CLUSTER-01~03 | Complete |
| 13.2 | 高可用 | HA 和可观测性 | HA-01~04, OPS-03~04, CLUSTER-04 | Complete |
| 13.3 | 安全加固 | 安全强化和运维 | SEC-01~05, OPS-05 | Complete |

---

## Phase 13.1: K8s 基础

**Goal:** Kubernetes deployment infrastructure — Helm chart, Operator, health probes, node discovery

**Requirements:**
- K8S-01: Helm chart for declarative deployment
- K8S-02: Kubernetes Operator (CRD)
- K8S-03: Liveness and Readiness probes
- K8S-04: ConfigMap integration
- K8S-05: Graceful shutdown
- OPS-01: Deployment scripts
- OPS-02: Health check endpoints
- CLUSTER-01: Node discovery via DNS
- CLUSTER-02: NodeMesh extension
- CLUSTER-03: gRPC transport layer

**Success Criteria:**
1. `helm install vllm-lite ./charts/vllm-lite` deploys 3 replicas
2. `kubectl get pods` shows 3 running pods
3. `/health/live` returns 200 when process alive
4. `/health/ready` returns 200 when model loaded
5. Headless service returns peer pod IPs
6. gRPC ping/pong between pods succeeds
7. Rolling update completes without downtime

**Implementation Notes:**
- Reference `crates/server/src/health.rs` for health endpoints
- Reference `crates/dist/src/tensor_parallel/` for NodeMesh
- Create `k8s/charts/vllm-lite/` for Helm chart
- Create `k8s/operator/` for Operator code (Go)

---

## Phase 13.2: 高可用

**Goal:** High availability with leader election, automatic failover, and observability

**Requirements:**
- HA-01: Leader election (K8s Lease API)
- HA-02: Automatic failover
- HA-03: PodDisruptionBudget
- HA-04: In-flight request preservation
- OPS-03: Prometheus metrics endpoint
- OPS-04: HPA support
- CLUSTER-04: Consistent hash routing

**Success Criteria:**
1. Kill leader pod, follower takes over within 30s
2. In-flight requests complete or are retried
3. HPA scales replicas based on queue depth
4. `/metrics` exposes request_count, queue_depth, gpu_memory_used
5. Consistent hash routes same prompt to same node
6. PodDisruptionBudget allows max 1 unavailable during upgrade

**Implementation Notes:**
- Reference `crates/core/src/engine.rs` for scheduler HA
- Use K8s Lease API for leader election
- Implement request tracking for failover preservation

---

## Phase 13.3: 安全加固

**Goal:** Security hardening — TLS, mTLS, authentication, RBAC, audit logging

**Requirements:**
- SEC-01: TLS termination
- SEC-02: mTLS for inter-node
- SEC-03: API authentication
- SEC-04: RBAC
- SEC-05: Audit logging
- OPS-05: Structured logging

**Success Criteria:**
1. External HTTPS endpoint with valid certificate
2. Node-to-node communication encrypted
3. Unauthenticated requests rejected
4. RBAC restricts model access by role
5. Audit log captures all API calls with user identity
6. Structured logs include request_id correlation

**Implementation Notes:**
- Reference `crates/server/src/auth.rs` for auth middleware
- Use `rustls` for TLS implementation
- Reference `crates/server/src/backpressure.rs` for rate limiting

---

## Phase Transition Triggers

After each phase, run verification and update ROADMAP.md progress.

---

## Long-term Vision

Phase 14: Cross-region replication

---
*Roadmap created: 2026-04-27*
*Last updated: 2026-04-27 after initial creation*
