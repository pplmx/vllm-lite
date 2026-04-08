# Scheduler Optimization Design

## Overview

Optimize the vLLM-lite scheduler for better performance, smarter scheduling decisions, and improved memory efficiency while maintaining backward compatibility.

## Problem Statement

Current scheduler has three main issues:

1. **Performance**: Multiple iterations over running sequences (4x in build_batch)
2. **Scheduling**: Fixed P/D separation ratio, naive preemption selection
3. **Memory**: Simple block allocation, basic eviction policy

## Architecture

### Core Changes

```text
Scheduler
├── Queue (unchanged)
├── BlockAllocator (enhanced allocation strategy)
├── PrefixCache (enhanced eviction)
├── PreemptionManager (smarter victim selection)
└── BatchBuilder (new - single-pass batch construction)
```

### Single-Pass Batch Construction

Replace 4 separate iterations with 1 pass that collects:

- Decode sequences
- Prefill sequences  
- Stats updates

```rust
struct BatchBuilder {
    decode_candidates: Vec<SequenceRef>,
    prefill_candidates: Vec<SequenceRef>,
}

impl BatchBuilder {
    fn scan(&mut self, running: &[Sequence]) {
        for seq in running {
            match seq.status {
                Status::Decoding => self.decode_candidates.push(seq),
                Status::Prefilling => self.prefill_candidates.push(seq),
                _ => {}
            }
        }
    }
}
```

## Components

### 1. BatchBuilder (New)

**Responsibility**: Single-pass batch construction

```rust
pub struct BatchBuilder {
    config: SchedulerConfig,
    decode_preference_ratio: f32,  // Adaptive
}

impl BatchBuilder {
    fn build(&mut self, running: &[Sequence], budget: usize) -> Batch {
        // Single pass: collect candidates
        // Second pass: select based on budget and policy
        // Third pass: finalize (can be merged)
    }

    fn compute_adaptive_ratio(&self, running: &[Sequence]) -> f32 {
        // Count prefill vs decode
        // If many prefills pending -> favor prefill
        // If high decode throughput -> favor decode
    }
}
```

### 2. Adaptive P/D Separation

**Current**: Fixed ratio (e.g., 70% decode, 30% prefill)

**Proposed**: Dynamic ratio based on:

- Queue pressure (waiting count)
- Running state (prefill vs decode ratio)
- Memory pressure

```rust
fn compute_adaptive_ratio(&self, scheduler: &Scheduler) -> f32 {
    let waiting = scheduler.waiting_count();
    let running = scheduler.running_count();

    let prefill_waiting = scheduler.queue.waiting_iter()
        .filter(|s| s.status == Status::Prefilling)
        .count();

    let decode_running = running.saturating_sub(
        scheduler.running().iter()
            .filter(|s| s.status == Status::Prefilling)
            .count()
    );

    // If lots of prefill waiting, increase prefill budget
    if prefill_waiting > decode_running.saturating_mul(2) {
        return 0.4; // 40% decode, 60% prefill
    }

    // Default to decode-heavy
    0.7
}
```

### 3. Smart Preemption

**Current**: Select victim with minimum `consecutive_decode_rounds`

**Proposed**: Multi-factor scoring

```rust
fn select_victim(&self, running: &[Sequence], memory_pressure: f32) -> Option<Sequence> {
    running.iter()
        .filter(|s| s.status == Status::Decoding)
        .min_by_key(|s| {
            let decode_score = s.consecutive_decode_rounds;
            let progress_score = s.tokens.len() * 100 / s.max_tokens;
            let length_score = s.tokens.len();

            // Memory pressure affects weighting
            let weight = if memory_pressure > 0.8 { 2 } else { 1 };

            (decode_score * weight) - (progress_score / 10) - (length_score / 20)
        })
        .cloned()
}
```

Victim selection factors:

1. `consecutive_decode_rounds` - longer decode = better preempt candidate
2. `progress` - how close to completion (prefer early-stage)
3. `prompt_length` - shorter prompts cheaper to re-compute
4. `memory_pressure` - high pressure = more aggressive

### 4. Enhanced Dynamic Batching

**Current**: Simple memory-based adjustment

**Proposed**: Consider prompt length distribution

```rust
fn compute_optimal_batch_size(&self, scheduler: &Scheduler) -> usize {
    let base = self.config.max_num_seqs;
    let memory_factor = self.memory_utilization_factor(scheduler);
    let queue_factor = self.queue_pressure_factor(scheduler);
    let diversity_factor = self.prompt_diversity_factor(scheduler);

    // If prompts vary greatly in length, smaller batches reduce waste
    // If all similar, larger batches more efficient

    (base as f32 * memory_factor * queue_factor * diversity_factor) as usize
}
```

### 5. Block Allocation Strategy

**Current**: First-fit allocation

**Proposed**: Segmented/free-list hybrid

```rust
impl BlockAllocator {
    fn allocate_segmented(&mut self, num_blocks: usize) -> Vec<usize> {
        // Keep track of allocation size histogram
        // Prefer sizes that are "common" for better reuse
    }
}
```

### 6. Prefix Cache Eviction

**Current**: Simple LRU

**Proposed**: Weight-aware LRU

```rust
fn compute_eviction_score(&self, entry: &CacheEntry) -> f32 {
    let recency = self.now - entry.last_access;
    let size = entry.blocks.len();
    let hits = entry.hit_count;

    // Prefer evicting: large, old, low-hit entries
    recency as f32 * 0.5 + size as f32 * 0.3 - hits as f32 * 0.2
}
```

## Data Flow

```text
Request Arrival
    │
    ▼
┌─────────────────┐
│  add_request    │──▶ Check prefix cache
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  build_batch    │──▶ Single-pass scan (new)
└─────────────────┘
    │
    ├─▶ Adaptive P/D ratio
    │
    ├─▶ BatchBuilder.select()
    │
    └─▶ finalize_batch (optimized)
    │
    ▼
┌─────────────────┐
│  model.forward  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  update         │──▶ Process results
└─────────────────┘
    │
    ├─▶ Check preemption need
    │
    ├─▶ Smart victim selection
    │
    └─▶ Update prefix cache
```

## Error Handling

- **Allocation failure**: Trigger preemption before returning error
- **Empty batch**: Return early, no processing needed
- **Preemption failure**: Log warning, continue with reduced batch

## Testing

### Unit Tests

1. `test_single_pass_batch_equivalence` - Verify new impl matches old output
2. `test_adaptive_ratio_convergence` - Ratio stabilizes under load
3. `test_smart_preemption_factors` - Each factor affects selection
4. `test_dynamic_batching_bounds` - Always within min/max

### Integration Tests

1. `test_high_concurrency_batching` - 100+ concurrent requests
2. `test_preemption_under_memory_pressure` - Simulate OOM
3. `test_prefix_cache_eviction_patterns` - Various access patterns

### Performance Benchmarks

1. `benchmark_batch_construction` - Single-pass vs multi-pass
2. `benchmark_preemption_selection` - With/without smart selection
3. `benchmark_full_pipeline` - End-to-end throughput

## Backward Compatibility

- All config fields remain
- Default values produce identical behavior to current implementation
- New features disabled by default, opt-in via config

## Config Changes

```rust
pub struct SchedulerConfig {
    // Existing fields...

    // New fields (default to current behavior)
    pub enable_adaptive_pd_ratio: bool = false,
    pub enable_smart_preemption: bool = false,
    pub preemption_memory_threshold: f32 = 0.8,
    pub enable_enhanced_dynamic_batching: bool = false,
}
```

## Implementation Order

1. BatchBuilder single-pass (performance - highest impact)
2. Adaptive P/D ratio (scheduling - medium impact)
3. Smart preemption (reliability - prevents OOM)
4. Enhanced dynamic batching (throughput - variable impact)
5. Block allocation improvements (memory - incremental)
6. Prefix cache enhancements (memory - incremental)

## Success Criteria

- Batch construction time: **>30% reduction**
- Preemption accuracy: **>20% improvement** (fewer unnecessary preemptions)
- Memory efficiency: **>10% improvement** in cache hit rate
- Overall throughput: **>20% improvement** on typical workloads
