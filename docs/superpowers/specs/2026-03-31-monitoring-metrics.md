# vLLM-lite Monitoring Metrics Design

## 1. Overview

添加完整的监控指标，为生产环境提供系统状态可视化。

**目标：**
- 吞吐量 (tokens/sec)
- 延迟分布 (P50, P90, P99)
- GPU 利用率
- 显存使用
- Batch 大小分布
- 请求队列状态

## 2. 指标定义

### 2.1 吞吐量指标

```rust
pub struct ThroughputMetrics {
    pub tokens_per_second: f64,      // 总吞吐量
    pub requests_per_second: f64,    // 请求吞吐量
    pub prefill_tokens_per_second:   // Prefill 吞吐量
    pub decode_tokens_per_second:    // Decode 吞吐量
}
```

### 2.2 延迟指标

```rust
pub struct LatencyMetrics {
    pub prefill_latency_ms: Histogram,   // Prefill 延迟
    pub decode_latency_ms: Histogram,    // 单次 decode 延迟
    pub total_latency_ms: Histogram,     // 端到端延迟
    pub time_to_first_token_ms: f64,     // 首个 token 时间
}
```

### 2.3 资源指标

```rust
pub struct ResourceMetrics {
    pub gpu_utilization_percent: f32,    // GPU 利用率
    pub gpu_memory_used_mb: u64,         // 显存使用
    pub gpu_memory_total_mb: u64,        // 显存总量
    pub kv_cache_usage_percent: f64,     // KV Cache 使用率
}
```

### 2.4 Batch 指标

```rust
pub struct BatchMetrics {
    pub current_batch_size: usize,
    pub avg_batch_size: f64,
    pub max_batch_size: usize,
    pub batch_size_distribution: HashMap<usize, u64>,  // 各大小出现次数
    pub prefill_decode_ratio: f64,  // prefill:decode 比例
}
```

### 2.5 请求指标

```rust
pub struct RequestMetrics {
    pub active_requests: usize,
    pub waiting_requests: usize,
    pub finished_requests: u64,
    pub failed_requests: u64,
    pub avg_prompt_length: f64,
    pub avg_completion_length: f64,
}
```

## 3. 数据收集

### 3.1 Engine 集成

```rust
pub struct Engine<M: ModelBackend> {
    // ... existing fields
    metrics_collector: MetricsCollector,
}

impl<M: ModelBackend> Engine<M> {
    pub fn step(&mut self) -> Result<()> {
        let start = Instant::now();
        
        // ... existing logic
        
        self.metrics_collector.record_batch(
            batch_size,
            prefill_tokens,
            decode_tokens,
            elapsed,
        );
        
        Ok(())
    }
}
```

### 3.2 指标收集器

```rust
pub struct MetricsCollector {
    throughput: Arc<Mutex<ThroughputCounter>>,
    latency: Arc<Mutex<LatencyHistogram>>,
    resources: Arc<Mutex<ResourceMonitor>>,
    batch: Arc<Mutex<BatchTracker>>,
}

impl MetricsCollector {
    pub fn new() -> Self { ... }
    
    pub fn record_batch(&self, size: usize, prefill: usize, decode: usize, elapsed: Duration) { ... }
    pub fn record_latency(&self, phase: &str, ms: f64) { ... }
    pub fn get_metrics(&self) -> MetricsSnapshot { ... }
}
```

## 4. 指标导出

### 4.1 HTTP 端点

```rust
// GET /metrics
// 返回 Prometheus 格式

// GET /stats
// 返回 JSON 格式
```

```json
{
  "throughput": {
    "tokens_per_second": 150.5,
    "requests_per_second": 2.3
  },
  "latency": {
    "p50_ms": 45.2,
    "p90_ms": 120.5,
    "p99_ms": 250.0
  },
  "resources": {
    "gpu_utilization_percent": 85.0,
    "gpu_memory_used_mb": 8192,
    "kv_cache_usage_percent": 65.0
  },
  "batch": {
    "current_batch_size": 8,
    "avg_batch_size": 5.2
  }
}
```

### 4.2 Prometheus 格式

```
# HELP vllm_tokens_per_second Total tokens processed per second
# TYPE vllm_tokens_per_second gauge
vllm_tokens_per_second 150.5

# HELP vllm_latency_p50 P50 latency in milliseconds
# TYPE vllm_latency_p50 gauge
vllm_latency_p50 45.2

# HELP vllm_gpu_memory_used GPU memory used in MB
# TYPE vllm_gpu_memory_used gauge
vllm_gpu_memory_used 8192
```

## 5. 实现方案

### 5.1 指标收集器

```rust
pub struct MetricsCollector {
    // 滑动窗口计数器
    tokens_total: AtomicU64,
    requests_total: AtomicU64,
    
    // 延迟直方图 (简化的)
    latencies: Mutex<Vec<f64>>,
    
    // 资源监控
    resource_stats: Mutex<ResourceStats>,
}
```

### 5.2 直方图实现

由于 Rust 标准库没有直方图，用分桶实现：

```rust
impl LatencyHistogram {
    pub fn new() -> Self {
        let mut buckets = HashMap::new();
        // P50 需要 50 个 bucket，简化处理用固定分桶
        for &ms in &[10, 25, 50, 100, 200, 500, 1000, 2000, 5000] {
            buckets.insert(ms, 0);
        }
        Self { buckets, total: 0 }
    }
    
    pub fn record(&mut self, ms: f64) {
        // 找到合适的 bucket
        for (&threshold, count) in self.buckets.iter_mut() {
            if ms <= threshold as f64 {
                *count += 1;
                break;
            }
        }
        self.total += 1;
    }
    
    pub fn percentile(&self, p: f64) -> f64 {
        let target = (self.total as f64 * p).ceil() as u64;
        let mut cumsum = 0;
        for (&threshold, count) in self.buckets.iter() {
            cumsum += count;
            if cumsum >= target {
                return threshold as f64;
            }
        }
        // 超过最大 bucket
        *self.buckets.keys().max().unwrap() as f64
    }
}
```

## 6. API 集成

### 6.1 Stats 端点

```rust
// GET /v1/stats
async fn stats() -> Json<MetricsSnapshot> {
    let metrics = engine.get_metrics();
    Json(metrics)
}
```

### 6.2 Prometheus 端点

```rust
// GET /metrics
async fn metrics() -> String {
    let metrics = engine.get_metrics();
    format_prometheus(&metrics)
}
```

## 7. 实现计划

- [ ] 添加 MetricsCollector 结构
- [ ] 实现指标收集
- [ ] 添加 HTTP 端点
- [ ] Prometheus 格式支持
- [ ] 测试验证

## 8. 测试场景

### Test 1: 吞吐量计算

```
发送 100 个请求
期望: tokens_per_second 正确计算
```

### Test 2: 延迟分位数

```
发送 1000 个请求，延迟随机
期望: P50/P90/P99 正确
```

### Test 3: 资源监控

```
运行推理
期望: GPU 显存使用正确反映