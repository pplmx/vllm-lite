# Feature Landscape: vllm-lite v13.0 Host Deployment

**Domain:** LLM Inference Engine — Host Deployment
**Researched:** 2026-04-27

## Table Stakes

Features users expect from a production inference deployment. Missing = not production-ready.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **K8s Deployment** | Standard for production infra | Low | CRD + Operator |
| **Multi-replica** | HA, throughput | Med | Pod scaling, request routing |
| **Health checks** | K8s probe integration | Low | Extend existing HealthChecker |
| **Metrics** | Observability | Low | Extend existing metrics |
| **Config management** | Declarative deployment | Med | CRD + ConfigMap |
| **Leader election** | Scheduler HA | High | K8s Lease API |
| **Graceful shutdown** | Zero-downtime deploys | Med | Drain requests, complete KV |

## Differentiators

Features that set vllm-lite apart from competitors. Not expected, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Cross-node KV cache** | Efficient long-context sharing | High | Coherence protocol needed |
| **Lightweight operator** | Less resource overhead | Low | vs vLLM Ray-based |
| **Rust-native coordination** | Type-safe cluster management | Med | vs Python/Golang alternatives |
| **Hybrid attention+SSM** | Qwen3.5 hybrid models | Med | Existing, extend to cluster |

## Anti-Features

Features to explicitly NOT build in v13.0.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Full service mesh (Istio)** | Complexity, resource cost | Optional, opt-in later |
| **Multi-tenant isolation** | Significant complexity | Phase 2 or 3 |
| **Dynamic model loading** | KV cache invalidation complexity | Static model per deployment |
| **Cross-cluster federation** | Not needed yet | Single cluster focus |

## Feature Dependencies

```
Phase 1: K8s Deployment
├── K8s Operator (Go)
│   └── CRD definitions
├── StatefulSet controller
│   └── ConfigMap integration
└── Health check extension
    └── Readying checking model loaded

Phase 2: Multi-Node Coordination
├── NodeMesh
│   └── DeviceMesh extension
├── ClusterManager
│   └── Service discovery
└── Cross-node KV cache
    └── DistributedKVCache extension

Phase 3: High Availability
├── LeaderElection
│   └── K8s Lease API
├── ClusterSchedulerEngine
│   └── SchedulerEngine extension
└── Request routing
    └── Ingress to leader

Phase 4: Security Hardening
├── TLS/mTLS
│   └── rustls + cert-manager
├── JWT auth
│   └── AuthMiddleware extension
└── Cluster rate limiting
    └── Distributed rate limiter
```

## MVP Recommendation

**v13.0 scope (4 phases as planned):**

Prioritize:
1. K8s Deployment (Phase 1) — operational foundation
2. Multi-Node Coordination (Phase 2) — core differentiation
3. Leader Election (Phase 3) — production reliability
4. Security (Phase 4) — hardening

Defer:
- **Multi-tenant isolation**: Requires separate auth namespace, not v13.0
- **Dynamic model loading**: KV cache coherence too complex for v13.0
- **Full service mesh**: Add as optional integration later

## Phase Deliverables

| Phase | Deliverable | Acceptance Criteria |
|-------|-------------|---------------------|
| 1 | K8s Operator + CRDs | `kubectl apply -f` deploys N replicas |
| 2 | Multi-node TP | TP=2 across 2 pods, correct all-reduce |
| 3 | HA failover | Kill leader, follower takes over, no dropped requests |
| 4 | mTLS + JWT | All node-to-node encrypted, JWT validation works |

## Sources

- vllm-lite existing features
- Kubernetes production deployment patterns
- vLLM Ray vs K8s comparison (community wisdom)
- Leader election patterns
