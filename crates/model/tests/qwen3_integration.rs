//! Slow Qwen3 integration tests against on-disk checkpoints.
//!
//! Default `just nextest` skips these (`#[ignore]`). Run with:
//! `just nextest-checkpoint` or `cargo nextest run -p vllm-model --test qwen3_integration --run-ignored all`
//!
//! Requires weights at [`support::qwen3::model_dir()`] (override via `VLLM_TEST_MODEL_DIR`).
//! Fast smoke without this file: `arch_checkpoint_smoke::test_qwen3_checkpoint_forward_smoke`.

mod support;

use candle_core::D;
use support::qwen3::{HIDDEN_SIZE, VOCAB_SIZE};
use vllm_traits::ModelBackend;

fn require_model() -> support::on_disk::CachedModel {
    let fixture = support::qwen3::Qwen3Fixture::cpu();
    if !fixture.weights_available() {
        panic!(
            "Qwen3 checkpoint missing at {} (set {} or run on a machine with weights)",
            fixture.model_dir().display(),
            support::qwen3::ENV_VAR
        );
    }
    fixture.load_model().expect("load Qwen3 checkpoint")
}

fn argmax_token(logits: &candle_core::Tensor) -> u32 {
    logits
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(D::Minus1)
        .unwrap()
        .to_vec0::<u32>()
        .unwrap()
}

/// Single load, multiple assertions: prefill/decode shapes, embeddings, determinism.
#[test]
#[ignore = "slow: on-disk Qwen3 checkpoint (run: just nextest-checkpoint)"]
fn test_qwen3_checkpoint_e2e() {
    let mut model = require_model();

    // --- prefill + decode on block 0 (sequential KV growth) ---
    let tokens = vec![1u32, 2, 3, 4, 5];
    let positions: Vec<usize> = (0..tokens.len()).collect();
    let (logits, _) = model
        .forward_with_cache(&tokens, 0, &[0], &positions, true)
        .expect("prefill");
    assert_eq!(logits.dims(), [1, 5, VOCAB_SIZE]);
    let prefill_next = logits
        .narrow(1, 4, 1)
        .unwrap()
        .squeeze(1)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(D::Minus1)
        .unwrap()
        .to_vec0::<u32>()
        .unwrap();
    assert!(prefill_next < VOCAB_SIZE as u32);

    let (decode_logits, _) = model
        .forward_with_cache(&[42], 5, &[0], &[5], false)
        .expect("decode after prefill");
    assert_eq!(decode_logits.dims(), [1, 1, VOCAB_SIZE]);
    let decode_next = argmax_token(&decode_logits);
    assert!(decode_next < VOCAB_SIZE as u32);

    // --- embedding sanity (no KV) ---
    let hi = vec![6023u32];
    let embeddings = model.embed(&[hi], &[vec![0]]).expect("embed");
    assert_eq!(embeddings[0].len(), HIDDEN_SIZE);
    let non_zero = embeddings[0].iter().filter(|&&x| x != 0.0).count();
    assert!(non_zero as f32 / HIDDEN_SIZE as f32 > 0.5);

    // --- determinism on a fresh KV block (avoid pollution from block 0 above) ---
    const FRESH_BLOCK: usize = 1;
    let probe = vec![6023u32];
    let (l1, _) = model
        .forward_with_cache(&probe, 0, &[FRESH_BLOCK], &[0], true)
        .unwrap();
    let (l2, _) = model
        .forward_with_cache(&probe, 0, &[FRESH_BLOCK], &[0], true)
        .unwrap();
    assert_eq!(argmax_token(&l1), argmax_token(&l2));

    // --- different prompts differ (separate KV blocks, each from empty cache) ---
    let (la, _) = model
        .forward_with_cache(&[6023], 0, &[2], &[0], true)
        .unwrap();
    let (lb, _) = model
        .forward_with_cache(&[14947], 0, &[3], &[0], true)
        .unwrap();
    let a = la.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = lb.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let cosine = dot / (norm_a * norm_b);
    assert!(cosine < 0.99, "different tokens should differ, cosine={cosine}");
}
