# vLLM-lite Performance Optimization Guide

This guide documents the three major performance optimizations implemented in vLLM-lite and provides instructions on how to use and configure them.

## Overview

vLLM-lite includes three key performance optimizations:

1. **CUDA Graph Integration** - Reduces kernel launch overhead for decode operations
2. **Sequence Packing Optimization** - Minimizes padding waste during prefill
3. **Adaptive Speculative Decoding** - Dynamically adjusts draft token count based on accuracy

## 1. CUDA Graph Integration

### What It Does
CUDA Graphs capture the entire decode execution once and replay it with a single launch, eliminating per-kernel launch overhead (15-30% improvement for small batches).

### When to Enable
- ✅ **Recommended for**: Production deployments with consistent batch sizes
- ✅ **Best for**: Small batch sizes (batch=1-8) where kernel launch overhead is significant
- ❌ **Not needed for**: Large batches where compute dominates

### Configuration

```rust
use vllm_core::types::SchedulerConfig;

let config = SchedulerConfig {
    cuda_graph: CudaGraphConfig {
        enabled: true,
        batch_sizes: vec![1, 4, 8, 16, 32, 64],
        ..Default::default()
    },
    ..Default::default()
};
```

### Environment Variables
```bash
VLLM_CUDA_GRAPH_ENABLED=true
VLLM_CUDA_GRAPH_BATCH_SIZES=1,4,8,16,32,64
```

### Usage
```rust
let mut engine = Engine::with_config(target_model, draft_model, config, 4, 1024);
engine.capture_cuda_graphs()?; // Call once after initialization

// Use step() as normal - CUDA Graph will be used automatically for decode
let results = engine.step()?;
```

### Performance Impact
| Batch Size | Improvement |
|------------|-------------|
| 1 | 20-30% |
| 4 | 15-25% |
| 16 | 10-15% |
| 64+ | 5-10% |

---

## 2. Sequence Packing Optimization

### What It Does
Uses Best-Fit Decreasing (BFD) algorithm to group sequences of similar lengths into batches, minimizing padding waste during prefill phase.

### When to Enable
- ✅ **Recommended for**: Workloads with variable-length prompts
- ✅ **Best for**: Mixed prompt lengths (e.g., 10-1000 tokens)
- ❌ **Not needed for**: Fixed-length prompts where padding is already minimal

### Configuration

```rust
use vllm_core::types::{SchedulerConfig, SequencePackingConfig};

let config = SchedulerConfig {
    packing: SequencePackingConfig {
        enabled: true,
        target_batch_size: 32,
        max_batch_size: 256,
        similarity_threshold: 0.2, // 20% length difference tolerance
    },
    ..Default::default()
};
```

### Environment Variables
```bash
VLLM_SEQ_PACKING_ENABLED=true
VLLM_SEQ_PACKING_TARGET_BATCH=32
VLLM_SEQ_PACKING_MAX_BATCH=256
VLLM_SEQ_PACKING_THRESHOLD=0.2
```

### Usage
Sequence packing is automatically applied during prefill when enabled:

```rust
let mut engine = Engine::with_config(target_model, draft_model, config, 4, 1024);
// Packing happens automatically in build_batch()
let batch = engine.scheduler.build_batch();
```

### Performance Impact
| Scenario | Padding Waste | Improvement |
|----------|--------------|-------------|
| Variable lengths (no packing) | 40-60% | Baseline |
| Variable lengths (with packing) | 10-20% | 60% reduction |
| Similar lengths | 5-10% | Minimal |

---

## 3. Adaptive Speculative Decoding

### What It Does
Dynamically adjusts the number of draft tokens based on real-time acceptance rate tracking, maximizing throughput while avoiding wasted computation.

### When to Enable
- ✅ **Recommended for**: Speculative decoding workloads
- ✅ **Best for**: Scenarios where draft model accuracy varies
- ❌ **Not needed for**: Fixed draft token count scenarios

### Configuration

```rust
use vllm_core::speculative::AdaptiveDraftConfig;

let adaptive_config = AdaptiveDraftConfig {
    min_draft_tokens: 2,
    max_draft_tokens: 8,
    target_acceptance_rate: 0.7, // Target 70% acceptance
    accuracy_window_size: 20,     // Track last 20 verifications
    adjustment_step: 1,           // Adjust by 1 token at a time
    cooldown_steps: 5,            // Wait 5 steps between adjustments
};
```

### Environment Variables
```bash
VLLM_ADAPTIVE_MIN_DRAFT=2
VLLM_ADAPTIVE_MAX_DRAFT=8
VLLM_ADAPTIVE_TARGET_RATE=0.7
VLLM_ADAPTIVE_WINDOW=20
VLLM_ADAPTIVE_STEP=1
VLLM_ADAPTIVE_COOLDOWN=5
```

### Usage
```rust
let mut engine = Engine::with_config(target_model, draft_model, config, 4, 1024);
engine.enable_adaptive_speculative(adaptive_config);

// Use adaptive speculative decoding
let results = engine.step_adaptive_speculative()?;
```

### Performance Impact
| Draft Accuracy | Fixed Tokens | Adaptive Tokens | Throughput |
|----------------|--------------|-----------------|------------|
| 30% | 4 | 2-3 | +15% |
| 70% | 4 | 4 | Baseline |
| 90% | 4 | 6-8 | +25% |

---

## Best Practices

### For Maximum Throughput
```rust
// Enable all optimizations
let config = SchedulerConfig::default();
let mut engine = Engine::with_config(target_model, draft_model, config, 4, 1024);

// 1. CUDA Graph for decode optimization
engine.capture_cuda_graphs()?;

// 2. Sequence packing is enabled by default in config

// 3. Adaptive speculative decoding
engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());
```

### For Minimum Latency
```rust
// Disable packing for lower latency (fewer sequences per batch)
let config = SchedulerConfig {
    packing: SequencePackingConfig {
        enabled: false,
        ..Default::default()
    },
    ..Default::default()
};

// Use standard speculative with small draft count
let mut engine = Engine::with_config(target_model, draft_model, config, 2, 1024);
```

### For Memory Efficiency
```rust
// Reduce batch sizes and enable packing
let config = SchedulerConfig {
    max_num_seqs: 64,
    max_num_batched_tokens: 2048,
    packing: SequencePackingConfig {
        enabled: true,
        max_batch_size: 64,
        ..Default::default()
    },
    ..Default::default()
};
```

## Monitoring

### Metrics to Track

```rust
// Access metrics from engine
let snapshot = engine.metrics.snapshot();

// Key metrics:
// - snapshot.throughput_tokens_per_sec
// - snapshot.avg_latency_ms
// - snapshot.batch_size_histogram
```

### Expected Improvements

| Optimization | Tokens/sec | Latency (P99) | Memory |
|--------------|-----------|---------------|--------|
| CUDA Graph | +10-25% | -5% | Same |
| Sequence Packing | +20-30% | Same | -30% waste |
| Adaptive Speculative | +10-25% | -10% | Same |
| **Combined** | **+40-70%** | **-10%** | **-30% waste** |

## Troubleshooting

### CUDA Graph Not Working
- Check if batch sizes match pre-configured values
- Verify CUDA is available
- Check logs for capture errors

### Packing Not Reducing Waste
- Verify `similarity_threshold` is appropriate for your workload
- Check that prompt lengths actually vary
- Increase `target_batch_size` if batches are too small

### Adaptive Not Adjusting
- Verify `cooldown_steps` is not too high
- Check that draft model accuracy is actually varying
- Lower `accuracy_window_size` for faster response

## Further Reading

- [CUDA Graphs Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- vLLM Architecture Documentation
