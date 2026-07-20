//! Compile-time and behaviour tests for the `CudaGraphExecutor` trait surface.
//!
//! The trait is consumed by `vllm-core` via `Box<dyn CudaGraphExecutor + Send>`
//! and implemented by `vllm_model::kernels::cuda_graph::BatchCudaGraphExecutor`.
//! This file verifies the contract in isolation from either crate, so the
//! abstraction stays honest even if the concrete type evolves.

use std::sync::atomic::{AtomicUsize, Ordering};
use vllm_traits::{Batch, BatchOutput, CudaGraphExecutor, GraphExecutionError};

/// Counting fake executor used to assert call order, capture/execute counts,
/// and that the trait is object-safe (`Box<dyn CudaGraphExecutor + Send>`).
#[derive(Debug, Default)]
struct FakeExecutor {
    enabled: bool,
    captures: AtomicUsize,
    executes: AtomicUsize,
}

impl FakeExecutor {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::default()
        }
    }

    fn captures(&self) -> usize {
        self.captures.load(Ordering::Relaxed)
    }

    fn executes(&self) -> usize {
        self.executes.load(Ordering::Relaxed)
    }
}

impl CudaGraphExecutor for FakeExecutor {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn execute(&self, _batch: &Batch) -> Result<BatchOutput, GraphExecutionError> {
        self.executes.fetch_add(1, Ordering::Relaxed);
        Ok(BatchOutput {
            seq_ids: vec![],
            next_tokens: vec![],
        })
    }

    fn capture_all_graphs(&mut self) -> Result<(), GraphExecutionError> {
        // Mirrors `BatchCudaGraphExecutor::capture_all_graphs`: the call
        // always bumps the counter, but only does real work when enabled.
        // It does not flip `enabled` — capture is a no-op when disabled.
        self.captures.fetch_add(1, Ordering::Relaxed);
        if !self.enabled {
            return Ok(());
        }
        Ok(())
    }
}

#[test]
fn trait_object_compiles_and_is_send() {
    // The `let` binding forces the compiler to confirm the trait object type
    // is well-formed. If `CudaGraphExecutor` ever stops being object-safe,
    // this line will fail to type-check.
    let exec: Box<dyn CudaGraphExecutor + Send> = Box::new(FakeExecutor::new(true));
    assert!(exec.is_enabled());

    // A `Send` executor must remain Send across `move`. We move it into a
    // thread to make the compiler prove `Send` is satisfied; if `Send`
    // were ever dropped from the trait bound, this line would fail.
    let moved = exec;
    let handle = std::thread::spawn(move || {
        let _ = moved.is_enabled();
    });
    handle.join().expect("spawned thread should not panic");
}

#[test]
fn is_enabled_reflects_underlying_state() {
    let enabled = Box::new(FakeExecutor::new(true));
    let disabled = Box::new(FakeExecutor::new(false));
    assert!(enabled.is_enabled());
    assert!(!disabled.is_enabled());
}

#[test]
fn execute_returns_batch_output_and_increments_counter() {
    let exec = FakeExecutor::new(true);
    let batch = Batch {
        seq_ids: vec![1, 2],
        input_tokens: vec![vec![10], vec![20]],
        positions: vec![vec![0], vec![0]],
        kv_block_ids: vec![vec![0], vec![1]],
        num_computed_tokens: vec![0, 0],
        is_prefill: vec![false, false],
        sampling_params: vec![vllm_traits::SamplingParams::default(); 2],
        phase: vllm_traits::BatchPhase::Decode,
        total_tokens: 2,
        max_seq_len: 1,
    };
    let out = exec.execute(&batch).expect("execute should succeed");
    let empty_seq_ids: Vec<u64> = vec![];
    // P36: BatchOutput::next_tokens is `Vec<SampledToken>` — the
    // empty-batch case yields an empty `Vec<SampledToken>`.
    let empty_next_tokens: Vec<vllm_traits::SampledToken> = vec![];
    assert_eq!(out.seq_ids, empty_seq_ids);
    assert_eq!(out.next_tokens, empty_next_tokens);
    assert_eq!(exec.executes(), 1);

    exec.execute(&batch).unwrap();
    assert_eq!(exec.executes(), 2);
}

#[test]
fn capture_all_graphs_increments_counter_without_changing_enabled() {
    let mut exec = FakeExecutor::new(true);
    assert!(exec.is_enabled());
    exec.capture_all_graphs().unwrap();
    assert_eq!(exec.captures(), 1);
    assert!(exec.is_enabled(), "capture must not flip the enabled flag");

    // Re-capture is idempotent — counter still moves but no error.
    exec.capture_all_graphs().unwrap();
    assert_eq!(exec.captures(), 2);
    assert!(exec.is_enabled());
}

#[test]
fn disabled_executor_capture_is_noop_but_succeeds() {
    // Mirrors the real `BatchCudaGraphExecutor::capture_all_graphs` contract:
    // a disabled executor returns `Ok(())` without doing real work and does
    // not transition to the enabled state.
    let mut exec = FakeExecutor::new(false);
    exec.capture_all_graphs().unwrap();
    assert_eq!(exec.captures(), 1);
    assert!(
        !exec.is_enabled(),
        "capture on a disabled executor must stay disabled"
    );
}

#[test]
fn execute_then_capture_preserves_order() {
    // Verifies the trait methods are independently callable and that mut vs
    // immut borrows are respected (the compiler enforces this; we just
    // observe the resulting state).
    let mut exec = FakeExecutor::new(true);
    let batch = Batch {
        seq_ids: vec![],
        input_tokens: vec![],
        positions: vec![],
        kv_block_ids: vec![],
        num_computed_tokens: vec![],
        is_prefill: vec![],
        sampling_params: vec![],
        phase: vllm_traits::BatchPhase::Prefill,
        total_tokens: 0,
        max_seq_len: 0,
    };
    exec.execute(&batch).unwrap();
    exec.execute(&batch).unwrap();
    exec.capture_all_graphs().unwrap();
    assert_eq!(exec.executes(), 2);
    assert_eq!(exec.captures(), 1);
}
