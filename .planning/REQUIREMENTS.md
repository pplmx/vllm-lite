# vllm-lite v13.0 Requirements

**Milestone:** v13.0 — 主机部署
**Created:** 2026-04-27
**Status:** Active

## Milestone Requirements

### 1. 多节点集群支持 (Multi-Node Cluster)

- [ ] **CLUSTER-01**: Nodes can discover peer nodes via Kubernetes headless service DNS
- [ ] **CLUSTER-02**: DeviceMesh extends to NodeMesh for multi-node tensor parallelism
- [ ] **CLUSTER-03**: gRPC transport layer for inter-node KV cache and all-reduce operations
- [ ] **CLUSTER-04**: Consistent hash routing for request distribution across nodes

### 2. Kubernetes 集成 (Kubernetes Integration)

- [ ] **K8S-01**: Helm chart for declarative deployment with configurable values
- [ ] **K8S-02**: Kubernetes Operator (CRD) for custom resource management
- [ ] **K8S-03**: Liveness and Readiness probes exposed via HTTP endpoints
- [ ] **K8S-04**: ConfigMap integration for engine configuration
- [ ] **K8S-05**: Graceful shutdown with request draining

### 3. 高可用/故障恢复 (High Availability)

- [ ] **HA-01**: Leader election using K8s Lease API
- [ ] **HA-02**: Automatic failover when leader node becomes unavailable
- [ ] **HA-03**: PodDisruptionBudget for safe evictions during cluster upgrades
- [ ] **HA-04**: In-flight request preservation during failover

### 4. 运维工具 (Operations Tools)

- [ ] **OPS-01**: Deployment script for one-click cluster setup
- [ ] **OPS-02**: Health check endpoints (/health/live, /health/ready)
- [ ] **OPS-03**: Prometheus metrics endpoint (/metrics)
- [ ] **OPS-04**: HPA (Horizontal Pod Autoscaler) support based on request queue depth
- [ ] **OPS-05**: Structured logging with request correlation IDs

### 5. 安全强化 (Security Hardening)

- [ ] **SEC-01**: TLS termination support via cert-manager integration
- [ ] **SEC-02**: mTLS for inter-node communication
- [ ] **SEC-03**: API authentication (JWT/API key validation)
- [ ] **SEC-04**: RBAC for multi-tenant access control
- [ ] **SEC-05**: Audit logging for API requests

---

## Out of Scope

- **Edge/mobile deployment** — Focus on host/datacenter
- **Full service mesh (Istio/Linkerd)** — Defer to future release
- **Multi-tenant KV cache isolation** — Requires coherence protocol v2
- **Dynamic model loading** — Static model per deployment

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLUSTER-01 | 13.1 | — |
| CLUSTER-02 | 13.1 | — |
| CLUSTER-03 | 13.1 | — |
| CLUSTER-04 | 13.2 | — |
| K8S-01 | 13.1 | — |
| K8S-02 | 13.1 | — |
| K8S-03 | 13.1 | — |
| K8S-04 | 13.1 | — |
| K8S-05 | 13.1 | — |
| HA-01 | 13.2 | — |
| HA-02 | 13.2 | — |
| HA-03 | 13.2 | — |
| HA-04 | 13.2 | — |
| OPS-01 | 13.1 | — |
| OPS-02 | 13.1 | — |
| OPS-03 | 13.2 | — |
| OPS-04 | 13.2 | — |
| OPS-05 | 13.3 | — |
| SEC-01 | 13.3 | — |
| SEC-02 | 13.3 | — |
| SEC-03 | 13.3 | — |
| SEC-04 | 13.3 | — |
| SEC-05 | 13.3 | — |

---

*Requirements defined: 2026-04-27*
