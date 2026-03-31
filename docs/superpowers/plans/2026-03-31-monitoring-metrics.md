# Monitoring Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 添加完整监控指标，支持吞吐量、延迟、GPU 资源监控

**Tech Stack:** Rust, axum, tokio

---

## Task 1: MetricsCollector 结构

**Files:**
- Create: `crates/core/src/metrics.rs`

- [ ] **Step 1: 定义指标结构**

```rust
#[derive(Clone)]
pub struct MetricsCollector {
    tokens_total: Arc<AtomicU64>,
    requests_total: Arc<AtomicU64>,
    latencies: Arc<Mutex<Vec<f64>>>,
    batch_sizes: Arc<Mutex<Vec<usize>>>,
    resource_stats: Arc<Mutex<ResourceStats>>,
}
```

- [ ] **Step 2: 实现收集方法**

```rust
impl MetricsCollector {
    pub fn new() -> Self { ... }
    pub fn record_tokens(&self, count: u64) { ... }
    pub fn record_request(&self) { ... }
    pub fn record_latency(&self, ms: f64) { ... }
    pub fn record_batch_size(&self, size: usize) { ... }
    pub fn snapshot(&self) -> MetricsSnapshot { ... }
}
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(core): add metrics collector"
```

---

## Task 2: Engine 集成

**Files:**
- Modify: `crates/core/src/engine.rs`

- [ ] **Step 1: 添加 metrics 字段**

```rust
pub struct Engine<M: ModelBackend> {
    // ... existing
    pub metrics: MetricsCollector,
}
```

- [ ] **Step 2: 在 step 中收集指标**

```rust
pub fn step(&mut self) -> Result<()> {
    let batch = self.scheduler.build_batch();
    let start = Instant::now();
    
    // ... existing
    
    self.metrics.record_batch_size(batch.seq_ids.len());
    self.metrics.record_tokens(total_tokens);
    self.metrics.record_latency(start.elapsed().as_millis() as f64);
    
    Ok(())
}
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(core): integrate metrics collection in engine"
```

---

## Task 3: API 端点

**Files:**
- Modify: `crates/server/src/api.rs`

- [ ] **Step 1: 添加 stats 端点**

```rust
async fn get_stats(State(state): State<EngineState>) -> Json<MetricsSnapshot> {
    let snapshot = state.engine.metrics.snapshot();
    Json(snapshot)
}
```

- [ ] **Step 2: 添加 Prometheus 端点**

```rust
async fn get_prometheus(State(state): State<EngineState>) -> String {
    format_prometheus(&state.engine.metrics.snapshot())
}
```

- [ ] **Step 3: 注册路由**

```rust
.get("/v1/stats", get_stats)
.get("/metrics", get_prometheus)
```

- [ ] **Step 4: 提交**

```bash
git commit -m "feat(server): add metrics API endpoints"
```

---

## Task 4: 测试

**Files:**
- Add: `crates/server/tests/metrics.rs`

- [ ] **Step 1: 添加测试**

```rust
#[test]
fn test_metrics_api() {
    // 发送请求
    // 检查 /v1/stats 返回正确数据
}
```

- [ ] **Step 2: 提交**

```bash
git commit -m "test(server): add metrics API tests"
```

---

## Verification Checklist

- [ ] 吞吐量正确计算
- [ ] 延迟分位数正确
- [ ] API 返回正确数据
- [ ] 测试通过