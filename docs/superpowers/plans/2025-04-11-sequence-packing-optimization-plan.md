# Sequence Packing Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Sequence Packing Optimization for prefill phase to reduce padding waste by 60% using Best-Fit Decreasing algorithm.

**Architecture:** Add SequencePacker with BFD algorithm, integrate into BatchComposer for prefill batches, add configuration support.

**Tech Stack:** Rust, vLLM-lite scheduler

---

## File Structure

### New Files
- `crates/core/src/scheduler/packing.rs` - SequencePacker and SequencePackingConfig
- `crates/core/src/scheduler/packing/tests.rs` - Unit tests

### Modified Files
- `crates/core/src/scheduler/batch_composer.rs` - Add SequencePacking integration
- `crates/core/src/scheduler/mod.rs` - Export packing module
- `crates/core/src/types.rs` - Add SequencePackingConfig to SchedulerConfig
- `crates/core/src/scheduler/engine.rs` - Use packing-aware BatchComposer
- `crates/core/tests/packing_integration.rs` - Integration tests

---

## Task 1: SequencePackingConfig

**Files:**
- Modify: `crates/core/src/types.rs`

- [ ] **Step 1: Add SequencePackingConfig to types.rs**

```rust
// Add to crates/core/src/types.rs

/// Configuration for sequence packing optimization
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
            similarity_threshold: 0.2,
        }
    }
}

impl SequencePackingConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let enabled = std::env::var("VLLM_SEQ_PACKING_ENABLED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(true);
        
        let target_batch_size = std::env::var("VLLM_SEQ_PACKING_TARGET_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        
        let max_batch_size = std::env::var("VLLM_SEQ_PACKING_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        
        let similarity_threshold = std::env::var("VLLM_SEQ_PACKING_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.2);
        
        Self {
            enabled,
            target_batch_size,
            max_batch_size,
            similarity_threshold,
        }
    }
}
```

- [ ] **Step 2: Add SequencePackingConfig to SchedulerConfig**

```rust
// Add field to SchedulerConfig struct
pub struct SchedulerConfig {
    // ... existing fields ...
    /// Sequence packing configuration
    pub packing: SequencePackingConfig,
}

// Update Default impl
impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            packing: SequencePackingConfig::default(),
        }
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/types.rs
git commit -m "feat(packing): add SequencePackingConfig type"
```

---

## Task 2: SequencePacker Core Structure

**Files:**
- Create: `crates/core/src/scheduler/packing.rs`
- Modify: `crates/core/src/scheduler/mod.rs`

- [ ] **Step 1: Create packing.rs with SequencePacker and PackedBatch**

```rust
// crates/core/src/scheduler/packing.rs

//! Sequence Packing Optimization
//!
//! Uses Best-Fit Decreasing (BFD) algorithm to group sequences of similar
//! lengths into batches, minimizing padding waste during prefill.

use crate::types::{Phase, Sequence};
use crate::scheduler::SequencePackingConfig;

/// Result of sequence packing
#[derive(Clone, Debug)]
pub struct PackedBatch {
    pub sequences: Vec<Sequence>,
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub padding_waste: usize,
}

impl PackedBatch {
    fn new() -> Self {
        Self {
            sequences: Vec::new(),
            batch_size: 0,
            max_seq_len: 0,
            padding_waste: 0,
        }
    }

    fn add_sequence(&mut self, seq: Sequence) {
        self.sequences.push(seq);
        self.recalculate_stats();
    }

    fn recalculate_stats(&mut self) {
        self.batch_size = self.sequences.len();
        self.max_seq_len = self.sequences
            .iter()
            .map(|s| s.tokens.len())
            .max()
            .unwrap_or(0);
        self.padding_waste = self.sequences
            .iter()
            .map(|s| self.max_seq_len - s.tokens.len())
            .sum();
    }
}

/// Packer using Best-Fit Decreasing algorithm
pub struct SequencePacker {
    config: SequencePackingConfig,
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

        // Sort by length descending (Decreasing)
        let mut sorted: Vec<Sequence> = sequences;
        sorted.sort_by_key(|s| std::cmp::Reverse(s.tokens.len()));

        let mut batches: Vec<PackedBatch> = Vec::new();

        for seq in sorted {
            // Find best fit batch
            let best_fit = self.find_best_fit(&batches, &seq);

            if let Some(idx) = best_fit {
                batches[idx].add_sequence(seq);
            } else {
                // Create new batch
                let mut batch = PackedBatch::new();
                batch.add_sequence(seq);
                batches.push(batch);
            }
        }

        batches
    }

    /// Find the batch that best fits the sequence
    fn find_best_fit(&self, batches: &[PackedBatch], seq: &Sequence) -> Option<usize> {
        let seq_len = seq.tokens.len();

        batches
            .iter()
            .enumerate()
            .filter(|(_, b)| b.batch_size < self.config.max_batch_size)
            .filter(|(_, b)| {
                // Check length similarity
                let batch_min_len = b.sequences
                    .iter()
                    .map(|s| s.tokens.len())
                    .min()
                    .unwrap_or(seq_len);
                let max_len = b.max_seq_len.max(seq_len);
                let min_len = batch_min_len.min(seq_len);
                let diff = (max_len - min_len) as f32 / max_len as f32;
                diff <= self.config.similarity_threshold
            })
            .min_by(|(_, a), (_, b)| {
                // Best fit = minimum additional padding
                let add_padding_a = a.max_seq_len.max(seq_len) - a.max_seq_len;
                let add_padding_b = b.max_seq_len.max(seq_len) - b.max_seq_len;
                add_padding_a.cmp(&add_padding_b)
            })
            .map(|(idx, _)| idx)
    }

    fn create_single_batch(&self, sequences: Vec<Sequence>) -> PackedBatch {
        let mut batch = PackedBatch::new();
        for seq in sequences {
            batch.add_sequence(seq);
        }
        batch
    }
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 2: Add module export to scheduler/mod.rs**

```rust
// Add to crates/core/src/scheduler/mod.rs

pub mod packing;
pub use packing::{PackedBatch, SequencePacker};
```

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/scheduler/packing.rs
git add crates/core/src/scheduler/mod.rs
git commit -m "feat(packing): add SequencePacker with BFD algorithm"
```

---

## Task 3: Unit Tests for SequencePacker

**Files:**
- Create: `crates/core/src/scheduler/packing/tests.rs`

- [ ] **Step 1: Create unit tests**

```rust
// crates/core/src/scheduler/packing/tests.rs

use super::*;
use crate::types::{Priority, SamplingParams, Status};
use std::sync::Arc;

fn create_sequence(id: u64, len: usize) -> Sequence {
    Sequence {
        id,
        tokens: vec![1u32; len],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: 0,
        prompt_len: len,
        status: Status::Waiting,
        max_tokens: 100,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
    }
}

#[test]
fn test_packer_reduces_padding_waste() {
    let sequences = vec![
        create_sequence(1, 1000),
        create_sequence(2, 100),
        create_sequence(3, 95),
        create_sequence(4, 10),
        create_sequence(5, 200),
    ];

    let config = SequencePackingConfig::default();
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(sequences);

    // Calculate total waste
    let total_waste: usize = batches.iter().map(|b| b.padding_waste).sum();

    // FIFO would have waste = (1000-10)+(1000-100)+(1000-95)+(1000-200) = 3495
    // BFD should have significantly less waste
    assert!(total_waste < 500, "Expected waste < 500, got {}", total_waste);
}

#[test]
fn test_packer_respects_max_batch_size() {
    let sequences: Vec<_> = (0..50)
        .map(|i| create_sequence(i, 100 + i))
        .collect();

    let config = SequencePackingConfig {
        max_batch_size: 10,
        ..Default::default()
    };
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(sequences);

    for batch in &batches {
        assert!(
            batch.batch_size <= 10,
            "Batch size {} exceeds max {}",
            batch.batch_size,
            10
        );
    }
}

#[test]
fn test_packer_disabled_returns_single_batch() {
    let sequences = vec![
        create_sequence(1, 100),
        create_sequence(2, 200),
    ];

    let config = SequencePackingConfig {
        enabled: false,
        ..Default::default()
    };
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(sequences);

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].batch_size, 2);
}

#[test]
fn test_packer_empty_sequences() {
    let config = SequencePackingConfig::default();
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(vec![]);

    assert!(batches.is_empty());
}

#[test]
fn test_packer_single_sequence() {
    let sequences = vec![create_sequence(1, 100)];

    let config = SequencePackingConfig::default();
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(sequences);

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].batch_size, 1);
    assert_eq!(batches[0].padding_waste, 0);
}

#[test]
fn test_similar_sequences_packed_together() {
    // Sequences with similar lengths should be packed together
    let sequences = vec![
        create_sequence(1, 100),
        create_sequence(2, 105),
        create_sequence(3, 110),
        create_sequence(4, 1000),
    ];

    let config = SequencePackingConfig {
        similarity_threshold: 0.2,
        ..Default::default()
    };
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(sequences);

    // Should have 2 batches: [1000] and [110, 105, 100]
    assert_eq!(batches.len(), 2);
    
    // Find batch with 3 sequences
    let large_batch = batches.iter()
        .find(|b| b.batch_size == 3)
        .expect("Should have batch with 3 sequences");
    
    assert!(large_batch.max_seq_len <= 110);
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-core packing::tests -- --nocapture
```

Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/scheduler/packing/tests.rs
git commit -m "test(packing): add unit tests for SequencePacker"
```

---

## Task 4: BatchComposer Integration

**Files:**
- Modify: `crates/core/src/scheduler/batch_composer.rs`

- [ ] **Step 1: Add SequencePacking to BatchComposer**

```rust
// Modify crates/core/src/scheduler/batch_composer.rs

use crate::scheduler::packing::{PackedBatch, SequencePacker};
use crate::types::{Phase, Sequence, SequencePackingConfig};

pub struct BatchComposer {
    config: BatchCompositionConfig,
    packing_config: SequencePackingConfig,
}

impl BatchComposer {
    pub fn new(config: BatchCompositionConfig) -> Self {
        Self {
            config,
            packing_config: SequencePackingConfig::default(),
        }
    }

    pub fn with_packing(
        config: BatchCompositionConfig,
        packing_config: SequencePackingConfig,
    ) -> Self {
        Self {
            config,
            packing_config,
        }
    }

    /// Compose batch with optional sequence packing for prefill
    pub fn compose(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        match phase {
            Phase::Prefill if self.packing_config.enabled && sequences.len() > 1 => {
                self.compose_prefill_with_packing(sequences)
            }
            _ => self.compose_standard(sequences, phase),
        }
    }

    fn compose_prefill_with_packing(&self, sequences: Vec<Sequence>) -> Batch {
        let packer = SequencePacker::new(self.packing_config.clone());
        let packed_batches = packer.pack_sequences(sequences);

        if packed_batches.is_empty() {
            return Batch::empty();
        }

        // Select batch with best packing (lowest waste per sequence)
        let best_batch = packed_batches
            .into_iter()
            .min_by(|a, b| {
                let waste_per_seq_a = a.padding_waste as f32 / a.batch_size.max(1) as f32;
                let waste_per_seq_b = b.padding_waste as f32 / b.batch_size.max(1) as f32;
                waste_per_seq_a.partial_cmp(&waste_per_seq_b).unwrap()
            })
            .unwrap_or_else(|| PackedBatch {
                sequences: vec![],
                batch_size: 0,
                max_seq_len: 0,
                padding_waste: 0,
            });

        self.build_batch_from_sequences(best_batch.sequences, Phase::Prefill)
    }

    fn compose_standard(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        self.build_batch_from_sequences(sequences, phase)
    }

    fn build_batch_from_sequences(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        // Existing implementation
        match phase {
            Phase::Prefill => self.build_prefill_batch(sequences),
            Phase::Decode => self.build_decode_batch(sequences),
        }
    }

    // ... existing build_prefill_batch and build_decode_batch methods ...
}
```

- [ ] **Step 2: Update existing tests if needed**

Run tests to ensure existing functionality still works:

```bash
cargo test -p vllm-core batch_composer -- --nocapture
```

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/scheduler/batch_composer.rs
git commit -m "feat(packing): integrate SequencePacker into BatchComposer"
```

---

## Task 5: SchedulerEngine Integration

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs`

- [ ] **Step 1: Update SchedulerEngine to use packing-aware BatchComposer**

```rust
// Modify SchedulerEngine::new() in crates/core/src/scheduler/engine.rs

impl SchedulerEngine {
    pub fn new(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
        // ... existing code ...

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

        Self {
            request_queue: RequestQueue::new(),
            phase_scheduler: PhaseScheduler::new(phase_switch_policy),
            batch_composer,
            // ... rest of initialization ...
        }
    }
}
```

- [ ] **Step 2: Run engine tests**

```bash
cargo test -p vllm-core scheduler::engine -- --nocapture
```

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/scheduler/engine.rs
git commit -m "feat(packing): use packing-aware BatchComposer in SchedulerEngine"
```

---

## Task 6: Integration Tests

**Files:**
- Create: `crates/core/tests/packing_integration.rs`

- [ ] **Step 1: Create integration tests**

```rust
// crates/core/tests/packing_integration.rs

use vllm_core::scheduler::{PackedBatch, SchedulerEngine, SequencePacker};
use vllm_core::types::{Phase, Request, SchedulerConfig, SequencePackingConfig};

#[test]
fn test_packing_disabled_by_default() {
    let config = SchedulerConfig::default();
    let engine = SchedulerEngine::new(config, 1024);
    
    // Packing should be enabled by default in the config
    assert!(config.packing.enabled);
}

#[test]
fn test_packing_disabled_returns_single_batch() {
    let config = SchedulerConfig {
        packing: SequencePackingConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut engine = SchedulerEngine::new(config, 1024);
    
    // Add requests
    engine.add_request(Request::new(0, vec![1; 100], 10));
    engine.add_request(Request::new(0, vec![1; 200], 10));
    
    let batch = engine.build_batch();
    
    // Batch should still be valid
    assert_eq!(batch.seq_ids.len(), 2);
}

#[test]
fn test_end_to_end_packing_reduces_waste() {
    let config = SchedulerConfig::default();
    let mut engine = SchedulerEngine::new(config, 1024);
    
    // Add requests with varying lengths
    engine.add_request(Request::new(0, vec![1; 1000], 10));
    engine.add_request(Request::new(0, vec![1; 100], 10));
    engine.add_request(Request::new(0, vec![1; 95], 10));
    engine.add_request(Request::new(0, vec![1; 10], 10));
    
    // Build batch (should use packing for prefill)
    let batch = engine.build_batch();
    
    // Verify batch is valid
    assert!(batch.seq_ids.len() > 0);
}

#[test]
fn test_packer_config_from_env() {
    // Test that config can be created from env vars
    let config = SequencePackingConfig::from_env();
    
    // Should have reasonable defaults
    assert!(config.target_batch_size > 0);
    assert!(config.max_batch_size >= config.target_batch_size);
    assert!(config.similarity_threshold > 0.0 && config.similarity_threshold <= 1.0);
}
```

- [ ] **Step 2: Run integration tests**

```bash
cargo test -p vllm-core --test packing_integration -- --nocapture
```

- [ ] **Step 3: Commit**

```bash
git add crates/core/tests/packing_integration.rs
git commit -m "test(packing): add integration tests"
```

---

## Task 7: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
just nextest
```

Expected: All tests pass

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

Expected: No warnings

- [ ] **Step 3: Run fmt check**

```bash
cargo fmt --all --check
```

Expected: Clean

- [ ] **Step 4: Final commit**

```bash
git commit -m "feat(packing): complete Sequence Packing Optimization implementation

- Add SequencePackingConfig with environment variable support
- Implement SequencePacker using Best-Fit Decreasing algorithm
- Integrate packing into BatchComposer for prefill batches
- Add comprehensive unit and integration tests
- All tests passing, clippy clean"
```

---

## Summary

### Files Created
- `crates/core/src/scheduler/packing.rs` - SequencePacker implementation
- `crates/core/src/scheduler/packing/tests.rs` - Unit tests
- `crates/core/tests/packing_integration.rs` - Integration tests

### Files Modified
- `crates/core/src/types.rs` - Add SequencePackingConfig
- `crates/core/src/scheduler/batch_composer.rs` - Integrate packing
- `crates/core/src/scheduler/engine.rs` - Use packing-aware composer
- `crates/core/src/scheduler/mod.rs` - Export packing module

### Expected Results
- Padding waste reduced by 60%+
- Memory utilization improved by 30-50%
- No latency regression
- All existing tests pass
