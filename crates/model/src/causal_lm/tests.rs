//! Unit tests for the `causal_lm` module.
//!
//! Covers three high-level entry points plus regression tests for the
//! `squeeze` no-op edge case:
//!
//! 1. **`forward_with_paged_kv` end-to-end**: builds a minimal
//!    one-layer model (one `new_block`, one `norm`, one `lm_head`,
//!    one `PagedKvCache`), runs a 3-token prefill and a single-step
//!    decode through the same cache, and verifies both outputs
//!    preserve the `[batch, seq, vocab]` shape contract.
//! 2. **`greedy_sample_token`**: with prefill=true and a known
//!    logit matrix, the function picks the last-position argmax
//!    (position 1, value 0.9 → token 3).
//! 3. **`mean_pool_embeddings`**: with an empty token list, returns
//!    a zero vector of the requested embedding dimension.
//! 4. **`squeeze` no-op regression**: Candle's `squeeze(dim)` is a no-op
//!    on dimensions with size > 1, so chained squeezes can leave a
//!    residual rank-2 tensor that `to_vec1()` rejects. The
//!    `flatten_all()` fix in `greedy_sample_token` / `logits_to_vector`
//!    guarantees rank-1 before extraction regardless of how many
//!    squeezes actually reduced the rank. These tests feed rank-2
//!    inputs directly and assert the correct token/value is returned
//!    (i.e., the function doesn't panic with "unexpected rank").
//!
//! All tests run on `Device::Cpu` with `DType::F32`.
use super::*;
use crate::components::decoder_block::new_block;
use crate::config::ModelConfig;
use candle_nn::{Embedding, VarBuilder};

#[test]
fn test_forward_with_paged_kv_prefill_and_decode() {
    let config = ModelConfig::test_tiny();
    let device = Device::Cpu;
    let layer = new_block(&config, 0, &Device::Cpu).unwrap();
    let layers = vec![layer];

    let vocab = config.vocab_size;
    let hidden = config.hidden_size;
    let embeddings = Tensor::zeros((vocab, hidden), candle_core::DType::F32, &device).unwrap();
    let embed_tokens = Embedding::new(embeddings, hidden);

    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
    let norm = candle_nn::linear(hidden, hidden, vb.pp("norm")).unwrap();
    let lm_head = candle_nn::linear(hidden, vocab, vb.pp("lm_head")).unwrap();

    let mut kv_cache = PagedKvCache::new(
        1,
        config.num_heads,
        config.head_dim,
        16,
        device.clone(),
        false,
    )
    .unwrap();

    let tokens = vec![1u32, 2, 3];
    let positions: Vec<usize> = (0..tokens.len()).collect();
    let (prefill_logits, _) = forward_with_paged_kv(
        &embed_tokens,
        &layers,
        &norm,
        &lm_head,
        &device,
        vocab,
        &tokens,
        0,
        &[0],
        &positions,
        true,
        &mut kv_cache,
    )
    .unwrap();
    assert_eq!(prefill_logits.dims(), &[1, tokens.len(), vocab]);

    let (decode_logits, _) = forward_with_paged_kv(
        &embed_tokens,
        &layers,
        &norm,
        &lm_head,
        &device,
        vocab,
        &tokens,
        tokens.len(),
        &[0],
        &[tokens.len()],
        false,
        &mut kv_cache,
    )
    .unwrap();
    assert_eq!(decode_logits.dims(), &[1, 1, vocab]);
}

#[test]
fn test_greedy_sample_prefill_takes_last_position() {
    let device = Device::Cpu;
    let vocab = 4;
    let logits = Tensor::from_slice(
        &[0.1f32, 0.2, 0.9, 0.3, 0.4, 0.1, 0.2, 0.5],
        (1, 2, vocab),
        &device,
    )
    .unwrap();
    let token = greedy_sample_token(&logits, true).unwrap();
    assert_eq!(token, 3);
}

#[test]
fn test_mean_pool_empty_tokens() {
    let device = Device::Cpu;
    let emb = Embedding::new(
        Tensor::zeros((8, 16), candle_core::DType::F32, &device).unwrap(),
        16,
    );
    let pooled = mean_pool_embeddings(&emb, &[], &device, 16).unwrap();
    assert_eq!(pooled.len(), 16);
    assert!(pooled.iter().all(|v| *v == 0.0));
}

// === squeeze no-op regression tests ===
//
// Candle's `squeeze(dim)` on a dimension with size > 1 is a no-op
// (returns the same tensor). The `flatten_all()` fix in
// `greedy_sample_token` and `logits_to_vector` guarantees rank-1
// before `to_vec1()` / `argmax_logits` regardless of how many
// squeeze calls actually reduced the rank.
//
// These tests feed rank-2 [batch, vocab] tensors directly to the
// decode path (is_prefill=false). Without `flatten_all()`, the chained
// `squeeze(0).squeeze(0)` would be a no-op on dim 0 (size > 1) and
// `to_vec1()` would panic with "unexpected rank".

#[test]
fn test_greedy_sample_decode_squeeze_noop_rank2() {
    // [batch=2, vocab=4] — squeeze(0) is a no-op (dim 0 has size 2).
    // The highest logit is 0.9 at position (0, 3), so the flattened
    // argmax should be token 3.
    let device = Device::Cpu;
    let logits = Tensor::from_slice(
        &[0.1f32, 0.2, 0.9, 0.3, 0.4, 0.1, 0.2, 0.5],
        (2, 4),
        &device,
    )
    .unwrap();
    let token = greedy_sample_token(&logits, false).unwrap();
    assert_eq!(token, 2); // 0.9 is at index 2 in the flattened tensor
}

#[test]
fn test_logits_to_vector_decode_squeeze_noop_rank2() {
    // Same [2, 4] tensor — flatten_all() must produce a rank-1
    // vector of 8 elements, preserving element order.
    let device = Device::Cpu;
    let logits = Tensor::from_slice(
        &[0.1f32, 0.2, 0.9, 0.3, 0.4, 0.1, 0.2, 0.5],
        (2, 4),
        &device,
    )
    .unwrap();
    let vec = logits_to_vector(&logits, false).unwrap();
    assert_eq!(vec.len(), 8);
    assert!((vec[0] - 0.1).abs() < 1e-6);
    assert!((vec[2] - 0.9).abs() < 1e-6);
    assert!((vec[7] - 0.5).abs() < 1e-6);
}

#[test]
fn test_greedy_sample_prefill_squeeze_noop_rank3() {
    // [batch=1, seq=3, vocab=4] — narrow(1, 2, 1) gives [1, 1, 4],
    // squeeze(1) is a no-op on dim 1 if size != 1 (but here it IS
    // size 1 after narrow, so this tests the prefill path with
    // a multi-position tensor).
    let device = Device::Cpu;
    let logits = Tensor::from_slice(
        &[
            0.1f32, 0.2, 0.9, 0.3, // pos 0
            0.4, 0.1, 0.2, 0.5, // pos 1
            0.3, 0.3, 0.3, 0.9, // pos 2 — highest logit at token 3
        ],
        (1, 3, 4),
        &device,
    )
    .unwrap();
    let token = greedy_sample_token(&logits, true).unwrap();
    assert_eq!(token, 3);
}

#[test]
fn test_logits_to_vector_prefill_squeeze_noop_rank3() {
    // Same [1, 3, 4] tensor — narrow(1, 2, 1) gives [1, 1, 4],
    // flatten_all() produces a rank-1 vector of 4 elements.
    let device = Device::Cpu;
    let logits = Tensor::from_slice(
        &[
            0.1f32, 0.2, 0.9, 0.3, 0.4, 0.1, 0.2, 0.5, 0.3, 0.3, 0.3, 0.9,
        ],
        (1, 3, 4),
        &device,
    )
    .unwrap();
    let vec = logits_to_vector(&logits, true).unwrap();
    // RIL ISS-023: logits_to_vector returns ALL positions (3 positions x
    // 4 vocab = 12 elements) so the speculative verifier can check each
    // draft; callers wanting only the last position take the last vocab_size.
    assert_eq!(vec.len(), 12);
    // Last position (pos 2): [0.3, 0.3, 0.3, 0.9] at indices 8..12.
    assert!((vec[11] - 0.9).abs() < 1e-6);
}
