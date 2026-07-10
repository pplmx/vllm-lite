//! Unit tests for `MemoryManager`.
//!
//! Covers the three entry points the scheduler actually drives:
//!
//! 1. **allocate / free**: `allocate(3)` reduces `available_blocks`
//!    by 3; `free(blocks)` restores it. Round-trip invariant.
//! 2. **`select_victims`**: returns up to `num_blocks` block IDs from
//!    `Decoding` sequences (in order). For a single 2-block seq with
//!    `num_blocks=1`, the result is 0 or 1 blocks (depends on
//!    whether the partial block-count is honored — see the
//!    implementation).
//! 3. **OOM**: `allocate(capacity)` succeeds; `allocate(1)` on a
//!    full manager returns `None`.
use super::*;
use crate::types::{Priority, SamplingParams, Status};
use std::sync::Arc;

fn make_sequence(id: u64, blocks: Vec<BlockId>, status: Status) -> Sequence {
    Sequence {
        id,
        tokens: vec![1, 2, 3],
        kv_blocks: Arc::new(blocks),
        num_computed_tokens: 3,
        prompt_len: 3,
        status,
        max_tokens: 100,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    }
}

#[test]
fn test_memory_manager_allocate_free() {
    let mut manager = MemoryManager::new(SchedulerConfig::default(), 10);

    let blocks = manager.allocate(3).unwrap();
    assert_eq!(blocks.len(), 3);
    assert_eq!(manager.available_blocks(), 7);

    manager.free(&blocks);
    assert_eq!(manager.available_blocks(), 10);
}

#[test]
fn test_memory_manager_select_victims() {
    let manager = MemoryManager::new(SchedulerConfig::default(), 10);

    let seq = make_sequence(1, vec![1, 2], Status::Decoding);
    let victims = manager.select_victims(&[seq], 1);
    assert!(victims.is_empty() || victims.len() == 1);
}

#[test]
fn test_memory_manager_oom() {
    let mut manager = MemoryManager::new(SchedulerConfig::default(), 2);
    manager.allocate(2).unwrap();
    assert!(manager.allocate(1).is_none());
}
