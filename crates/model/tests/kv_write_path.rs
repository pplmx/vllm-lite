//! Integration test: a REAL model (`LlamaModel` with KV-writing
//! `RopeGqaDecoderBlock`) writes its prefill KV into the blocks it is given —
//! the data path that `StubModel`'s passthrough block skips entirely. This is
//! a regression guard for the KV-write path (RIL ISS-019/ISS-022 class): if
//! `write_prefill_kv` or the block indexing regresses, the KV would land in
//! the wrong blocks (or stay zero) and this fails.

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use vllm_model::config::ModelConfig;
use vllm_model::llama::LlamaModel;
use vllm_traits::ModelBackend;

/// Deterministic non-zero, non-constant weights (so `LayerNorm` has non-zero
/// variance and the attention produces non-zero K/V).
fn fill_weight(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| 0.02 * ((i as f32).mul_add(0.13, seed).sin() + 1.5))
        .collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn tiny_weights(cfg: &ModelConfig) -> HashMap<String, Tensor> {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let inter = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let mut w = HashMap::new();
    for layer in 0..cfg.num_layers {
        let p = |name: &str| format!("model.layers.{layer}.{name}");
        w.insert(
            p("self_attn.q_proj.weight"),
            fill_weight(&[nh * hd, h], 1.0),
        );
        w.insert(
            p("self_attn.k_proj.weight"),
            fill_weight(&[nkv * hd, h], 2.0),
        );
        w.insert(
            p("self_attn.v_proj.weight"),
            fill_weight(&[nkv * hd, h], 3.0),
        );
        w.insert(
            p("self_attn.o_proj.weight"),
            fill_weight(&[h, nh * hd], 4.0),
        );
        w.insert(p("mlp.gate_proj.weight"), fill_weight(&[inter, h], 5.0));
        w.insert(p("mlp.up_proj.weight"), fill_weight(&[inter, h], 6.0));
        w.insert(p("mlp.down_proj.weight"), fill_weight(&[h, inter], 7.0));
        w.insert(p("input_layernorm.weight"), fill_weight(&[h], 8.0));
        w.insert(p("post_attention_layernorm.weight"), fill_weight(&[h], 9.0));
    }
    w.insert(
        "model.embed_tokens.weight".to_string(),
        fill_weight(&[vocab, h], 10.0),
    );
    w.insert("model.norm.weight".to_string(), fill_weight(&[h], 11.0));
    w.insert("lm_head.weight".to_string(), fill_weight(&[vocab, h], 12.0));
    w
}

#[test]
fn real_model_prefill_writes_kv_into_given_blocks() {
    let device = Device::Cpu;
    let cfg = ModelConfig::test_tiny();
    let weights = tiny_weights(&cfg);
    let mut model = LlamaModel::from_weights(cfg, &device, weights, 16, false).unwrap();

    // 20-token prompt => ceil(20 / 16) = 2 blocks. Prefill into blocks [0, 1].
    let seq_len = 20usize;
    let prompt: Vec<Vec<u32>> = vec![(0..seq_len as u32).collect()];
    let positions: Vec<Vec<usize>> = vec![(0..seq_len).collect()];
    let block_ids: Vec<Vec<usize>> = vec![vec![0, 1]];

    let logits = model
        .forward_logits(&[1], &prompt, &positions, &block_ids, &[0], &[true])
        .unwrap();
    assert_eq!(logits.len(), 1);
    drop(logits);

    // Read back the KV (scope the cache guard so it drops early).
    let (k01, v01, k2) = {
        let cache = model.paged_kv_cache();
        let cache = cache.lock();
        let (k01, v01) = cache.read_kv(0, &[0, 1], seq_len).unwrap();
        let (k2, _) = cache.read_kv(0, &[2], 16).unwrap();
        drop(cache);
        (k01, v01, k2)
    };

    // The prefill must have written NON-zero KV into blocks 0 and 1.
    let k01: Vec<f32> = k01.flatten_all().unwrap().to_vec1().unwrap();
    let v01: Vec<f32> = v01.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        k01.iter().any(|x| x.abs() > 1e-6),
        "prefill must write non-zero K into the given blocks (KV-write path regression)"
    );
    assert!(
        v01.iter().any(|x| x.abs() > 1e-6),
        "prefill must write non-zero V into the given blocks (KV-write path regression)"
    );

    // A block that was NOT given (block 2) must remain untouched (all zero).
    let k2: Vec<f32> = k2.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        k2.iter().all(|x| x.abs() < 1e-9),
        "blocks not given to the prefill must remain empty"
    );
}
