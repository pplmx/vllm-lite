# Sequence Packing Optimization Design

**Date:** 2025-04-11  
**Status:** Approved  
**Author:** vLLM-lite Team

## Summary

Implement Sequence Packing Optimization for the prefill phase to reduce padding waste and improve memory utilization by grouping sequences of similar lengths into the same batch using the Best-Fit Decreasing (BFD) bin packing algorithm.

## Background

### Problem Statement

Current `BatchComposer` builds batches by taking sequences in FIFO order from the request queue. This leads to significant padding waste when sequences of varying lengths are batched together:

```
Current behavior:
Batch: [seq_len=10, seq_len=100, seq_len=1000]
Max sequence length = 1000
Padding waste = (1000-10) + (1000-100) + (1000-1000) = 990 tokens (99% waste!)

With Sequence Packing:
Batch 1: [seq_len=10, seq_len=95, seq_len=100, seq_len=105]
Batch 2: [seq_len=1000]
Padding waste = (105-10)+(105-95)+(105-100)+(105-105) + 0 = 15 tokens (3.5% waste)
```

### Solution

Use the **Best-Fit Decreasing (BFD)** bin packing algorithm to group sequences of similar lengths together, minimizing the maximum sequence length in each batch and thus reducing padding overhead.

## Goals

| Goal | Metric | Target |
|------|--------|--------|
| Reduce padding waste | Padding tokens / Total tokens | < 20% (from ~50%) |
| Improve memory utilization | Effective token ratio | > 80% (from ~50%) |
| Maintain latency | P50/P99 latency | No regression |
| Zero correctness impact | Output parity | 100% |

## Non-Goals

- Decode phase optimization (token count is always 1)
- Dynamic batch size adjustment based on GPU utilization
- Mixed prefill/decode batches
- Priority-based preemption within batches

## Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SchedulerEngine                          │
│                         │                                   │
│                         ▼                                   │
│               PhaseScheduler.select_phase()                 │
│                         │                                   │
│                         ▼                                   │
│            BatchComposer.compose_optimized()                │
│                         │                                   │
│            ┌────────────┴────────────┐                     │
│            ▼                          ▼                     │
│    Phase::Prefill              Phase::Decode                │
│            │                          │                     │
│            ▼                          ▼                     │
│    SequencePacker              Standard compose             │
│    .pack_sequences()              (no change)               │
│            │                                                │
│            ▼                                                │
│    Best-Fit Decreasing                                      │
│    Algorithm                                                │
│            │                                                │
│            ▼                                                │
│    Optimized Batch                                          │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. SequencePackingConfig

Configuration for sequence packing optimization.

```rust
#[derive(Clone, Debug)]
pub struct SequencePackingConfig {
    /// Enable sequence packing optimization
    pub enabled: bool,
    /// Target batch size for packing
    pub target_batch_size: usize,
    /// Maximum batch size (hard limit)
    pub max_batch_size: usize,
    /// Length similarity threshold (0.0-1.0)
    pub similarity_threshold: f32,
}

impl Default for SequencePackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_batch_size: 32,
            max_batch_size: 256,
            similarity_threshold: 0.2, // 20% length difference tolerance
        }
    }
}
```

#### 2. SequencePacker

Core packing algorithm using Best-Fit Decreasing.

```rust
pub struct SequencePacker {
    config: SequencePackingConfig,
}

/// Result of sequence packing
pub struct PackedBatch {
    pub sequences: Vec<Sequence>,
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub padding_waste: usize,
}

impl SequencePacker {
    pub fn new(config: SequencePackingConfig) -> Self {
        Self { config }
    }

    /// Pack sequences using Best-Fit Decreasing algorithm
    pub fn pack_sequences(&self, sequences: Vec<Sequence>) -> Vec<PackedBatch> {
        if sequences.is_empty() {
            return vec![];
        }

        if !self.config.enabled {
            return vec![self.create_single_batch(sequences)];
        }

        // 1. Sort by length descending (Decreasing)
        let mut sorted: Vec<Sequence> = sequences;
        sorted.sort_by_key(|s| std::cmp::Reverse(s.tokens.len()));

        let mut batches: Vec<PackedBatch> = Vec::new();

        for seq in sorted {
            // 2. Find best fit batch
            let best_fit = self.find_best_fit(&batches, &seq);

            if let Some(idx) = best_fit {
                batches[idx].sequences.push(seq);
                self.update_batch_stats(&mut batches[idx]);
            } else {
                // 3. Create new batch
                batches.push(self.create_batch_with_sequence(seq));
            }
        }

        batches
    }

    /// Find the batch that best fits the sequence (minimizes waste)
    fn find_best_fit(&self, batches: &[PackedBatch], seq: &Sequence) -> Option<usize> {
        let seq_len = seq.tokens.len();

        batches
            .iter()
            .enumerate()
            .filter(|(_, b)| b.sequences.len() < self.config.max_batch_size)
            .filter(|(_, b)| {
                // Check length similarity
                let max_len = b.max_seq_len.max(seq_len);
                let min_len = b.sequences.iter().map(|s| s.tokens.len()).min().unwrap_or(seq_len);
                let diff = (max_len - min_len) as f32 / max_len as f32;
                diff <= self.config.similarity_threshold
            })
            .min_by_key(|(_, b)| {
                // Best fit = minimum additional padding required
                let new_max = b.max_seq_len.max(seq_len);
                new_max - b.max_seq_len
            })
            .map(|(idx, _)| idx)
    }

    fn create_batch_with_sequence(&self, seq: Sequence) -> PackedBatch {
        let len = seq.tokens.len();
        PackedBatch {
            batch_size: 1,
            max_seq_len: len,
            padding_waste: 0,
            sequences: vec![seq],
        }
    }

    fn update_batch_stats(&mut self, batch: &mut PackedBatch) {
        batch.batch_size = batch.sequences.len();
        batch.max_seq_len = batch.sequences.iter()
            .map(|s| s.tokens.len())
            .max()
            .unwrap_or(0);
        batch.padding_waste = batch.sequences.iter()
            .map(|s| batch.max_seq_len - s.tokens.len())
            .sum();
    }

    fn create_single_batch(&self, sequences: Vec<Sequence>) -> PackedBatch {
        let max_len = sequences.iter()
            .map(|s| s.tokens.len())
            .max()
            .unwrap_or(0);
        let waste: usize = sequences.iter()
            .map(|s| max_len - s.tokens.len())
            .sum();

        PackedBatch {
            batch_size: sequences.len(),
            max_seq_len: max_len,
            padding_waste: waste,
            sequences,
        }
    }
}
```

#### 3. BatchComposer Integration

Extend `BatchComposer` to use sequence packing for prefill batches.

```rust
pub struct BatchComposer {
    config: BatchCompositionConfig,
    packing_config: SequencePackingConfig,
}

impl BatchComposer {
    pub fn with_packing(
        config: BatchCompositionConfig,
        packing_config: SequencePackingConfig,
    ) -> Self {
        Self {
            config,
            packing_config,
        }
    }

    /// Compose batch with optional sequence packing optimization
    pub fn compose(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        match phase {
            Phase::Prefill if self.packing_config.enabled => {
                self.compose_prefill_with_packing(sequences)
            }
            _ => self.compose_standard(sequences, phase),
        }
    }

    fn compose_prefill_with_packing(&self, sequences: Vec<Sequence>) -> Batch {
        let packer = SequencePacker::new(self.packing_config.clone());
        let packed_batches = packer.pack_sequences(sequences);

        // Select the batch with the best packing (lowest waste per sequence)
        let best_batch = packed_batches
            .into_iter()
            .min_by(|a, b| {
                let waste_a = a.padding_waste as f32 / a.sequences.len() as f32;
                let waste_b = b.padding_waste as f32 / b.sequences.len() as f32;
                waste_a.partial_cmp(&waste_b).unwrap()
            });

        match best_batch {
            Some(batch) => self.build_batch_from_sequences(batch.sequences, Phase::Prefill),
            None => Batch::empty(),
        }
    }

    fn compose_standard(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        // Original implementation
        self.build_batch_from_sequences(sequences, phase)
    }

    fn build_batch_from_sequences(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        // Build Batch from sequences (original logic)
        // ... existing implementation ...
    }
}
```

#### 4. SchedulerEngine Configuration

Add sequence packing config to `SchedulerConfig`.

```rust
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    // ... existing fields ...
    /// Sequence packing configuration
    pub packing: SequencePackingConfig,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            packing: SequencePackingConfig::default(),
        }
    }
}
```

Update `SchedulerEngine::new()`:

```rust
impl SchedulerEngine {
    pub fn new(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
        let batch_config = BatchCompositionConfig {
            max_batch_size: config.max_num_seqs,
            max_token_budget: config.max_num_batched_tokens,
            enable_similarity_grouping: false,
        };

        // Use packing-aware batch composer
        let batch_composer = BatchComposer::with_packing(
            batch_config,
            config.packing.clone(),
        );

        // ... rest of initialization ...
    }
}
```

### Algorithm: Best-Fit Decreasing (BFD)

**Why BFD?**
- Time complexity: O(n log n) - efficient for large n
- Approximation ratio: 1.22 - good quality results
- Simple to implement and understand

**Algorithm Steps:**

```
1. Sort sequences by length (descending) - O(n log n)
2. Initialize empty batch list
3. For each sequence:
   a. Find the batch that would have the minimum additional padding
   b. Check if sequence length is within similarity threshold
   c. If found, add to that batch
   d. Else, create new batch
4. Return list of packed batches
```

**Example:**

```
Input sequences (lengths): [1000, 100, 95, 10, 200]

Step 1 - Sort: [1000, 200, 100, 95, 10]

Step 2-3 - Pack:
- 1000 → Batch 1: [1000]
- 200 → Batch 2: [200]
- 100 → Batch 2: [200, 100] (max_len=200, waste=100)
- 95 → Batch 2: [200, 100, 95] (max_len=200, waste=105)
- 10 → Batch 2: [200, 100, 95, 10] (max_len=200, waste=195)

Result: 2 batches
- Batch 1: [1000] (waste: 0)
- Batch 2: [200, 100, 95, 10] (waste: 195)

vs FIFO (no packing):
- Batch 1: [1000, 100, 95, 10, 200] (waste: 1800)

Improvement: 90% reduction in padding waste!
```

### Configuration Options

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_SEQ_PACKING_ENABLED` | Enable sequence packing | `true` |
| `VLLM_SEQ_PACKING_TARGET_BATCH` | Target batch size | `32` |
| `VLLM_SEQ_PACKING_MAX_BATCH` | Maximum batch size | `256` |
| `VLLM_SEQ_PACKING_THRESHOLD` | Similarity threshold (0.0-1.0) | `0.2` |

**YAML Configuration:**

```yaml
scheduler:
  # ... existing config ...
  packing:
    enabled: true
    target_batch_size: 32
    max_batch_size: 256
    similarity_threshold: 0.2
```

### Fallback Mechanism

If sequence packing fails or produces suboptimal results:

```rust
fn compose_prefill_with_packing(&self, sequences: Vec<Sequence>) -> Batch {
    let packer = SequencePacker::new(self.packing_config.clone());
    let packed_batches = packer.pack_sequences(sequences.clone());

    // Check if packing actually helped
    let best_packed = packed_batches.iter()
        .min_by_key(|b| b.padding_waste);
    
    let fifo_waste = calculate_fifo_waste(&sequences);
    
    match best_packed {
        Some(batch) if batch.padding_waste < fifo_waste => {
            // Packing helped, use it
            self.build_batch_from_sequences(batch.sequences.clone(), Phase::Prefill)
        }
        _ => {
            // Packing didn't help, use FIFO
            tracing::debug!("Sequence packing did not improve waste, using FIFO");
            self.build_batch_from_sequences(sequences, Phase::Prefill)
        }
    }
}
```

## Data Flow

### Prefill Batch Construction

```
1. SchedulerEngine.build_batch()
   └── PhaseScheduler.select_phase() -> Phase::Prefill
       └── Select candidate sequences
           └── BatchComposer.compose(sequences, Phase::Prefill)
               └── compose_prefill_with_packing()
                   ├── SequencePacker.pack_sequences()
                   │   ├── Sort by length
                   │   ├── Best-fit bin packing
                   │   └── Return PackedBatches
                   ├── Select best batch (min waste)
                   └── build_batch_from_sequences()
```

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_packer_reduces_waste() {
        let sequences = vec![
            create_seq(1, 1000),
            create_seq(2, 100),
            create_seq(3, 95),
            create_seq(4, 10),
            create_seq(5, 200),
        ];

        let config = SequencePackingConfig::default();
        let packer = SequencePacker::new(config);
        let batches = packer.pack_sequences(sequences);

        // Calculate total waste
        let total_waste: usize = batches.iter()
            .map(|b| b.padding_waste)
            .sum();

        // FIFO would have waste = 1800
        // BFD should have waste < 300 (improvement > 80%)
        assert!(total_waste < 300, 
            "Expected waste < 300, got {}", total_waste);
    }

    #[test]
    fn test_packer_respects_max_batch_size() {
        let sequences: Vec<_> = (0..100)
            .map(|i| create_seq(i, 100 + i))
            .collect();

        let config = SequencePackingConfig {
            max_batch_size: 10,
            ..Default::default()
        };
        let packer = SequencePacker::new(config);
        let batches = packer.pack_sequences(sequences);

        for batch in &batches {
            assert!(batch.sequences.len() <= 10,
                "Batch size {} exceeds max {}", 
                batch.sequences.len(), 10);
        }
    }

    #[test]
    fn test_packer_disabled_returns_single_batch() {
        let sequences = vec![
            create_seq(1, 100),
            create_seq(2, 200),
        ];

        let config = SequencePackingConfig {
            enabled: false,
            ..Default::default()
        };
        let packer = SequencePacker::new(config);
        let batches = packer.pack_sequences(sequences);

        assert_eq!(batches.len(), 1);
    }
}
```

### Integration Tests

```rust
#[test]
fn test_end_to_end_prefill_with_packing() {
    let config = SchedulerConfig {
        packing: SequencePackingConfig {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = SchedulerEngine::new(config, 1024);

    // Add requests with varying lengths
    engine.add_request(Request::new(0, vec![1; 1000], 10));  // len=1000
    engine.add_request(Request::new(0, vec![1; 100], 10));   // len=100
    engine.add_request(Request::new(0, vec![1; 95], 10));     // len=95
    engine.add_request(Request::new(0, vec![1; 10], 10));    // len=10

    let batch = engine.build_batch();

    // Should pack shorter sequences together
    assert!(batch.seq_ids.len() >= 1);
}
```

## Performance Targets

| Metric | Baseline (FIFO) | Target (With Packing) | Improvement |
|--------|-----------------|----------------------|-------------|
| Prefill padding waste | ~50% | < 20% | 60% reduction |
| Effective token ratio | ~50% | > 80% | 60% increase |
| Memory utilization | Baseline | +30-50% | Significant |
| P50 latency | X | ~X | No regression |
| P99 latency | X | ~X | No regression |
| Throughput | Baseline | +20-30% | Measurable |

## Migration Path

### Phase 1: Basic Implementation
- Implement `SequencePacker` with BFD algorithm
- Integrate into `BatchComposer` for prefill only
- Add configuration options
- Add comprehensive tests

### Phase 2: Refinement
- Add runtime metrics (waste tracking)
- Add automatic fallback when packing doesn't help
- Optimize algorithm for edge cases

### Phase 3: Advanced Features (Future)
- Dynamic threshold adjustment based on workload
- Consider KV cache block alignment
- Multi-objective optimization (latency + throughput)

## Success Criteria

- [ ] All existing tests pass
- [ ] Padding waste reduced by > 50% in benchmarks
- [ ] No latency regression in integration tests
- [ ] Configuration toggles work as expected
- [ ] Documentation updated
- [ ] Code review approved

## Open Questions

1. **Threshold tuning**: Is 20% the optimal similarity threshold for all workloads?
2. **Batch size**: Should target_batch_size be dynamic based on GPU utilization?
3. **Decode phase**: Should we apply length-based reordering to decode batches (though padding isn't an issue)?

## Appendix

### Related Code

- `crates/core/src/scheduler/batch_composer.rs` - Batch composition
- `crates/core/src/scheduler/phase_scheduler.rs` - Phase management
- `crates/core/src/types.rs` - Configuration types

### References

- [Bin Packing Problem](https://en.wikipedia.org/wiki/Bin_packing_problem)
- [Best-Fit Decreasing Algorithm](https://en.wikipedia.org/wiki/Bin_packing_problem#First-fit_algorithm)
- vLLM's continuous batching paper
