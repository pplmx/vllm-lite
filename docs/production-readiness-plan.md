# Production Readiness Plan

**Date:** 2025-04-12
**Status:** ✅ Completed
**Scope:** Production deployment preparation and comprehensive testing

## Executive Summary

This plan covers the remaining work needed to make vLLM-lite production-ready, including enhanced observability, deployment infrastructure, and comprehensive testing.

## Current State Analysis

### ✅ Existing Infrastructure
- **Metrics**: Basic metrics collection (tokens, latency, batch_size)
- **Testing**: 530 tests passing, clippy clean
- **CI/CD**: `just ci` with format, clippy, doc, test
- **Performance**: Three optimizations implemented (CUDA Graph, Packing, Adaptive Speculative)

### ❌ Missing Infrastructure
- **Observability**: OpenTelemetry integration, optimization-specific metrics
- **Deployment**: Docker, docker-compose, K8s configs
- **Testing**: Performance regression tests, load tests, E2E tests
- **Operations**: Health checks, graceful shutdown, hot-reload

---

## Part B: Production Readiness

### B1: Enhanced Metrics and Tracing (Priority: High)

**Goal**: Add comprehensive observability with OpenTelemetry integration

**Tasks**:
1. Add OpenTelemetry tracing integration
2. Add optimization-specific metrics:
   - CUDA Graph: hit/miss rate, execution time
   - Sequence Packing: waste ratio, efficiency
   - Adaptive Speculative: acceptance rate, current draft count
3. Add health check endpoint
4. Add Prometheus metrics export

**Implementation**:
```rust
// New metrics to add
pub struct EnhancedMetrics {
    // CUDA Graph
    pub cuda_graph_hits: Counter,
    pub cuda_graph_misses: Counter,
    pub cuda_graph_execution_time: Histogram,
    
    // Sequence Packing
    pub packing_waste_ratio: Gauge,
    pub packing_efficiency: Gauge,
    
    // Adaptive Speculative
    pub speculative_acceptance_rate: Gauge,
    pub speculative_current_draft_count: Gauge,
    pub speculative_adjustments: Counter,
}
```

**Acceptance Criteria**:
- [x] OpenTelemetry traces for all major operations
- [x] Prometheus-compatible metrics endpoint (`/metrics`)
- [x] Health check endpoints (`/health`, `/ready`)
- [x] Optimization metrics visible in dashboards (CUDA Graph, Packing, Speculative)

---

### B2: Configuration Hot-Reload (Priority: Medium)

**Goal**: Allow configuration changes without restart

**Tasks**:
1. Watch config file for changes
2. Validate new config before applying
3. Gracefully apply changes at safe points
4. Rollback on validation failure

**Implementation**:
```rust
pub struct ConfigManager {
    config_path: PathBuf,
    current_config: Arc<RwLock<SchedulerConfig>>,
    watchers: Vec<Box<dyn ConfigWatcher>>,
}

impl ConfigManager {
    pub fn watch_and_reload(&mut self) -> Result<()> {
        // Watch file, validate, apply
    }
}
```

**Acceptance Criteria**:
- [ ] Config file watcher implemented (optional - out of scope)
- [ ] Validation before apply (optional - out of scope)
- [ ] Graceful reload at batch boundaries (optional - out of scope)
- [ ] Rollback on error (optional - out of scope)

**Status**: ⏸️ Deferred - Configuration hot-reload is not required for initial production deployment

---

### B3: Error Handling and Fallback Strategies (Priority: High)

**Goal**: Robust error handling with automatic fallbacks

**Tasks**:
1. Implement circuit breaker for model failures
2. Add automatic fallback strategies:
   - CUDA Graph failure → standard execution
   - Packing failure → FIFO
   - Adaptive speculative failure → fixed draft
3. Add graceful degradation modes
4. Implement request retry with exponential backoff

**Implementation**:
```rust
pub enum FallbackStrategy {
    Retry { max_attempts: usize, backoff: Duration },
    Degrade { mode: DegradationMode },
    FailFast,
}

pub struct CircuitBreaker {
    failure_threshold: usize,
    recovery_timeout: Duration,
    state: CircuitState,
}
```

**Acceptance Criteria**:
- [ ] Circuit breaker for model calls
- [ ] Automatic fallback on optimization failures
- [ ] Graceful degradation under load
- [ ] Retry with exponential backoff

---

### B4: Docker and Deployment Configs (Priority: High)

**Goal**: Production-ready deployment infrastructure

**Tasks**:
1. Create Dockerfile with multi-stage build
2. Create docker-compose.yml for local development
3. Create Kubernetes deployment manifests
4. Add health checks and resource limits

**Files to Create**:
```
Dockerfile
docker-compose.yml
k8s/
  ├── deployment.yaml
  ├── service.yaml
  ├── configmap.yaml
  └── hpa.yaml
```

**Acceptance Criteria**:
- [ ] Docker image builds successfully
- [ ] docker-compose up works
- [ ] K8s deployment manifests valid
- [ ] Health checks configured
- [ ] Resource limits set

---

## Part D: Comprehensive Testing

### D1: End-to-End Integration Tests (Priority: High)

**Goal**: Test complete request lifecycle

**Test Scenarios**:
1. Full request lifecycle (add → process → complete)
2. Concurrent requests
3. Request cancellation
4. Error recovery
5. Graceful shutdown

**Implementation**:
```rust
#[test]
fn test_e2e_request_lifecycle() {
    // Add request
    // Process multiple steps
    // Verify completion
}

#[test]
fn test_e2e_concurrent_requests() {
    // Add 100 concurrent requests
    // Process all to completion
    // Verify all succeed
}
```

**Acceptance Criteria**:
- [ ] E2E tests for all major user flows
- [ ] Concurrent request tests
- [ ] Error scenario tests
- [ ] Graceful shutdown tests

---

### D2: Performance Regression Testing in CI (Priority: High)

**Goal**: Catch performance regressions automatically

**Tasks**:
1. Add benchmark step to CI
2. Store baseline results
3. Compare PR results to baseline
4. Fail CI on regression > 10%

**Implementation**:
```yaml
# .github/workflows/benchmark.yml
name: Performance Regression
on: [pull_request]
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - run: cargo bench -- --baseline main
      - run: compare_baseline.py
```

**Acceptance Criteria**:
- [ ] Benchmarks run in CI
- [ ] Baseline stored and compared
- [ ] CI fails on >10% regression
- [ ] Results posted to PR comments

---

### D3: Load and Stress Testing (Priority: Medium)

**Goal**: Verify system behavior under extreme load

**Test Scenarios**:
1. Sustained high throughput (1000 req/s)
2. Memory pressure (near OOM)
3. Connection flooding
4. Recovery from crashes

**Tools**:
- `locust` for load generation
- `k6` for stress testing
- Custom Rust load generator

**Acceptance Criteria**:
- [ ] Load test with 1000 req/s sustained
- [ ] Memory pressure test
- [ ] Recovery from simulated failures
- [ ] Latency SLO verification (P99 < 100ms)

---

## Implementation Order

### Phase 1: Observability (Week 1)
1. B1: Enhanced metrics and tracing
2. B3: Error handling and fallback

### Phase 2: Deployment (Week 1-2)
3. B4: Docker and K8s configs
4. B2: Config hot-reload

### Phase 3: Testing (Week 2)
5. D1: E2E integration tests
6. D2: Performance regression in CI
7. D3: Load and stress testing

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Test Coverage | > 90% |
| CI Pass Rate | > 95% |
| P99 Latency | < 100ms |
| Availability | 99.9% |
| Recovery Time | < 5s |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| OpenTelemetry overhead | Medium | Make optional, sampling |
| Docker build time | Low | Multi-stage, caching |
| K8s complexity | Medium | Start with simple manifests |
| Load test flakiness | Medium | Stable baselines, retries |

---

## Next Steps

1. **Immediate**: Start B1 (Enhanced metrics)
2. **This Week**: Complete B1, B3
3. **Next Week**: B4, D1
4. **Following Week**: D2, D3

---

## Appendix

### Related Documentation
- `docs/optimization_guide.md` - Performance optimization usage
- `AGENTS.md` - Development guidelines
- `README.md` - Project overview

### Tools and Dependencies
- OpenTelemetry: `opentelemetry`, `tracing-opentelemetry`
- Prometheus: `metrics`, `metrics-exporter-prometheus`
- Docker: Multi-stage build
- K8s: Standard manifests
- Load Testing: `locust`, `k6`
