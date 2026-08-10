//! Regression (RIL ISS-031): steady-state decode must not waste every other
//! step on an empty prefill batch.
//!
//! `PhaseScheduler::should_switch_to_prefill` used `decode_queue_len <
//! min_decode_batch_size` to decide when to visit prefill. In steady-state
//! decode all sequences are RUNNING, not queued, so `decode_queue_len` reads
//! 0 — the scheduler oscillated Decode -> Prefill(empty) -> Decode, idling
//! every other step (~50% throughput loss). The fix: never switch to prefill
//! when there is no prefill work at all.

use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::Request;
use vllm_traits::ModelBackend;

#[test]
fn steady_state_decode_does_not_idle_steps() {
    let target = vllm_testing::StubModel::returning(7);
    let mut engine = Engine::new_boxed(Box::new(target), None::<Box<dyn ModelBackend>>);
    let (tx, _rx) = mpsc::channel(64);
    // 3-token prompt, 12 generated tokens: 1 prefill step + 12 decode steps.
    engine.add_request(Request::new(1, vec![1, 2, 3], 12), tx);

    // Run exactly the number of steps the sequence needs: 1 prefill step +
    // 11 decode steps (12 generated tokens reach max_tokens). Pre-fix,
    // oscillation made every other step empty, so 12 steps yielded only 6-7
    // tokens and the sequence was still running.
    let mut total_emitted = 0usize;
    let mut empty_steps = 0usize;
    for _ in 0..12 {
        let result = engine.step().unwrap();
        if result.is_empty() {
            empty_steps += 1;
        } else {
            total_emitted += result.len();
        }
    }

    assert_eq!(
        empty_steps, 0,
        "steady-state decode must not idle steps on empty prefill batches \
         (got {empty_steps} empty steps) (RIL ISS-031)"
    );
    assert!(
        total_emitted >= 12,
        "12 steps must complete the 12-token generation (emitted {total_emitted})"
    );
}
