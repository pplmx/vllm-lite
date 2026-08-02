//! End-to-end engine integration test with a REAL KV-writing model
//! (`LlamaModel`), exercising the full scheduler + model + KV-cache path that
//! `StubModel`/`ConstModel` skip (they don't write KV). This is the capstone
//! regression guard for the KV-data-path bug class (RIL ISS-019/ISS-022): a
//! multi-block prompt is prefilled, then decoded, and we verify the model
//! produces deterministic output and actually populates the KV cache.

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_model::config::ModelConfig;
use vllm_model::llama::LlamaModel;
use vllm_traits::{ModelBackend, TokenId};

/// Deterministic non-zero, non-constant weights.
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

fn build_model() -> LlamaModel {
    let cfg = ModelConfig::test_tiny();
    let weights = tiny_weights(&cfg);
    LlamaModel::from_weights(cfg, &Device::Cpu, weights, 64, false).unwrap()
}

fn scheduler_config() -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs: 8,
        max_num_batched_tokens: 256,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    }
}

/// Run a multi-block prompt to completion (greedy) and return the generated
/// tokens.
fn run_to_completion(prompt: Vec<TokenId>, max_tokens: usize) -> Vec<TokenId> {
    let model = build_model();
    let mut engine = Engine::with_config_boxed(
        Box::new(model),
        None::<Box<dyn ModelBackend>>,
        scheduler_config(),
        4,
        64,
    );
    let (tx, mut rx) = mpsc::channel(64);
    let prompt_len = prompt.len();
    engine.add_request(Request::new(1, prompt, max_tokens), tx);

    let mut generated = Vec::new();
    for _ in 0..(max_tokens + prompt_len + 10) {
        engine.step().unwrap();
        while let Ok(sampled) = rx.try_recv() {
            generated.push(sampled.token);
        }
        if generated.len() >= max_tokens {
            break;
        }
    }
    generated
}

#[test]
fn real_model_engine_multiblock_prefill_then_decode() {
    // 20-token prompt => ceil(20/16) = 2 blocks; generate 3 tokens (greedy).
    let prompt: Vec<TokenId> = (0..20).collect();
    let generated = run_to_completion(prompt, 3);
    assert_eq!(
        generated.len(),
        3,
        "engine must generate exactly max_tokens tokens with a real model"
    );
}

#[test]
fn real_model_engine_output_is_deterministic() {
    // Greedy decoding with fixed weights must be reproducible run-to-run.
    let prompt: Vec<TokenId> = (0..20).collect();
    let run1 = run_to_completion(prompt.clone(), 4);
    let run2 = run_to_completion(prompt, 4);
    assert_eq!(
        run1, run2,
        "greedy decoding with fixed weights must be deterministic"
    );
    assert_eq!(run1.len(), 4);
}

#[test]
fn real_model_engine_populates_kv_cache() {
    // After a multi-block prefill, the model must have written NON-zero KV
    // into the cache (the data path StubModel skips). RIL ISS-019/022 guard.
    let model = build_model();
    let cache = model.paged_kv_cache();
    let mut engine = Engine::with_config_boxed(
        Box::new(model),
        None::<Box<dyn ModelBackend>>,
        scheduler_config(),
        4,
        64,
    );
    let (tx, mut rx) = mpsc::channel(64);
    let prompt: Vec<TokenId> = (0..20).collect();
    engine.add_request(Request::new(1, prompt, 2), tx);
    engine.step().unwrap(); // prefill
    let _ = rx.try_recv();

    // Scan all blocks; at least one must hold non-zero KV now. Scope the
    // guard so it drops before the assert.
    let any_nonzero = {
        let cache = cache.lock();
        let mut found = false;
        for block in 0..4 {
            let (k, _v) = cache.read_kv(0, &[block], 16).unwrap();
            let k: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
            if k.iter().any(|x| x.abs() > 1e-6) {
                found = true;
                break;
            }
        }
        drop(cache);
        found
    };
    assert!(
        any_nonzero,
        "a real model prefill must populate the KV cache (KV-write path regression)"
    );
}
