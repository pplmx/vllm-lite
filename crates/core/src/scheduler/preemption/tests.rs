//! Unit tests for `PreemptionManager`.
//!
//! Covers two surfaces:
//!
//! 1. **Preemption decision** (`should_preempt(running, waiting,
//!    needed, available)`): true only when *all four* conditions
//!    hold — at least one running seq, at least one waiting seq,
//!    not enough available blocks, more blocks needed than a
//!    single running seq owns. The 5 tests walk each combination
//!    of hold/not-hold.
//! 2. **Victim selection** (`select_victim`): when a single
//!    running seq exists, no victim is selected (cannot preempt
//!    the only running work); with multiple running seqs, the one
//!    with the **fewest** consecutive decode rounds is chosen
//!    (least-progress-first heuristic).
//! 3. **Stats**: `preempted_count` / `rejected_count` increment on
//!    the matching record call, and `reset_stats` clears both.
use super::*;
use crate::types::{Priority, SamplingParams, Status};

fn make_sequence(id: u64, decode_rounds: u32) -> Sequence {
    Sequence {
        id,
        tokens: vec![1, 2, 3],
        kv_blocks: std::sync::Arc::new(vec![]),
        num_computed_tokens: 3,
        prompt_len: 3,
        status: Status::Decoding,
        max_tokens: 100,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: decode_rounds,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    }
}

#[test]
fn test_should_preempt_no_waiting() {
    let manager = PreemptionManager::new(SchedulerConfig::default());
    assert!(!manager.should_preempt(5, 0, 10, 1));
}

#[test]
fn test_should_preempt_no_running() {
    let manager = PreemptionManager::new(SchedulerConfig::default());
    assert!(!manager.should_preempt(0, 5, 10, 1));
}

#[test]
fn test_should_preempt_enough_blocks() {
    let manager = PreemptionManager::new(SchedulerConfig::default());
    assert!(!manager.should_preempt(5, 3, 10, 15));
}

#[test]
fn test_should_preempt_single_running() {
    let manager = PreemptionManager::new(SchedulerConfig::default());
    assert!(!manager.should_preempt(1, 5, 10, 1));
}

#[test]
fn test_should_preempt_all_conditions_met() {
    let manager = PreemptionManager::new(SchedulerConfig::default());
    assert!(manager.should_preempt(3, 5, 10, 5));
}

#[test]
fn test_select_victim_single() {
    let manager = PreemptionManager::new(SchedulerConfig::default());
    let running = vec![make_sequence(1, 5)];
    assert!(manager.select_victim(&running).is_none());
}

#[test]
fn test_select_victim_multiple() {
    let manager = PreemptionManager::new(SchedulerConfig::default());
    let running = vec![
        make_sequence(1, 10),
        make_sequence(2, 5),
        make_sequence(3, 15),
    ];
    let result = manager.select_victim(&running);
    assert!(result.is_some());
    let (idx, seq) = result.unwrap();
    assert_eq!(idx, 1);
    assert_eq!(seq.id, 2);
}

#[test]
fn test_stats() {
    let mut manager = PreemptionManager::new(SchedulerConfig::default());
    assert_eq!(manager.preempted_count(), 0);
    assert_eq!(manager.rejected_count(), 0);

    manager.record_preemption();
    assert_eq!(manager.preempted_count(), 1);

    manager.record_rejection();
    assert_eq!(manager.rejected_count(), 1);

    manager.reset_stats();
    assert_eq!(manager.preempted_count(), 0);
    assert_eq!(manager.rejected_count(), 0);
}
