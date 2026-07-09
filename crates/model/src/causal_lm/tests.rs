//! Unit tests for the `causal_lm` module.
//!
//! Covers three high-level entry points:
//!
//! 1. **forward_with_paged_kv end-to-end**: builds a minimal
//!    one-layer model (one `new_block`, one `norm`, one `lm_head`,
//!    one `PagedKvCache`), runs a 3-token prefill and a single-step
//!    decode through the same cache, and verifies both outputs
//!    preserve the `[batch, seq, vocab]` shape contract.
//! 2. **`greedy_sample_token`**: with prefill=true and a known
//!    logit matrix, the function picks the last-position argmax
//!    (position 1, value 0.9 → token 3).
//! 3. **`mean_pool_embeddings`**: with an empty token list, returns
//!    a zero vector of the requested embedding dimension.
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
    let layer = new_block(&config, 0).unwrap();
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
