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

/// Regression (RIL ISS-026): a speculative prefill whose prompt + drafts
/// cross a block boundary must NOT silently fall back to writing the
/// overflow KV into block 0.
///
/// The verifier runs the target over `input_len + drafts` tokens, but the
/// sequence's block table is pre-allocated for `input_len` tokens only
/// (`write_prefill_kv` falls back to `block_ids.get(block_idx).unwrap_or(0)`
/// for a missing block). With a 16-token prompt (exactly one block) and
/// max_draft=4, the draft KV for positions 16..19 lands in block 0 offsets
/// 0..3, overwriting the prompt's KV. The first block must be byte-identical
/// to a non-speculative prefill of the same prompt.
#[test]
fn real_model_speculative_prefill_does_not_corrupt_first_block() {
    // `ModelConfig` is not `Clone`; two identical configurations are built
    // from the same deterministic constructor so target and draft weights
    // match exactly.
    let cfg = ModelConfig::test_tiny();
    let weights = tiny_weights(&cfg);
    let target = LlamaModel::from_weights(
        ModelConfig::test_tiny(),
        &Device::Cpu,
        weights.clone(),
        64,
        false,
    )
    .unwrap();
    let draft =
        LlamaModel::from_weights(ModelConfig::test_tiny(), &Device::Cpu, weights, 64, false)
            .unwrap();
    let target_cache = target.paged_kv_cache();

    let mut engine = Engine::with_config_boxed(
        Box::new(target),
        Some(Box::new(draft)),
        scheduler_config(),
        4, // max_draft
        64,
    );
    engine.enable_speculative();

    let (tx, _rx) = mpsc::channel(64);
    // Exactly one block of prompt: 16 tokens. Drafts (4) then overflow into
    // the missing second block.
    let prompt: Vec<TokenId> = (0..16).collect();
    engine.add_request(Request::new(1, prompt.clone(), 20), tx);
    engine.step().unwrap();

    // Reference: non-speculative prefill of the same prompt writes the
    // canonical KV into block 0.
    let ref_model = build_model();
    let ref_cache = ref_model.paged_kv_cache();
    let mut ref_engine = Engine::with_config_boxed(
        Box::new(ref_model),
        None::<Box<dyn ModelBackend>>,
        scheduler_config(),
        4,
        64,
    );
    let (tx2, _rx2) = mpsc::channel(64);
    ref_engine.add_request(Request::new(1, prompt, 20), tx2);
    ref_engine.step().unwrap();

    let (k_spec, v_spec) = {
        let cache = target_cache.lock();
        cache.read_kv(0, &[0], 16).unwrap()
    };
    let (k_ref, v_ref) = {
        let cache = ref_cache.lock();
        cache.read_kv(0, &[0], 16).unwrap()
    };

    let k_spec_v: Vec<f32> = k_spec.flatten_all().unwrap().to_vec1().unwrap();
    let k_ref_v: Vec<f32> = k_ref.flatten_all().unwrap().to_vec1().unwrap();
    let k_max_diff = k_spec_v
        .iter()
        .zip(k_ref_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let v_spec_v: Vec<f32> = v_spec.flatten_all().unwrap().to_vec1().unwrap();
    let v_ref_v: Vec<f32> = v_ref.flatten_all().unwrap().to_vec1().unwrap();
    let v_max_diff = v_spec_v
        .iter()
        .zip(v_ref_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        k_max_diff < 1e-5 && v_max_diff < 1e-5,
        "speculative prefill corrupted block 0 KV (k max diff {k_max_diff}, v max diff {v_max_diff}); \
         draft KV must not fall back into block 0 when the block table is full"
    );
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

/// End-to-end smoke test: a GQA model (`num_kv_heads < num_heads`) runs a
/// multi-block prefill through the real forward path and produces finite,
/// non-zero logits. Grouping CORRECTNESS is pinned at the unit level by
/// `test_gqa_attention_matches_naive_reference`; here we verify the GQA path
/// (`expand_kv` with grouping) integrates without error and computes
/// non-trivial logits. (A logits-equivalence check against a replicated-MHA
/// model is degenerate on a tiny model — the residual stream washes out the
/// attention's grouping-dependent contribution.)
#[test]
fn gqa_model_forward_produces_nontrivial_logits() {
    let mut cfg = ModelConfig::test_tiny();
    cfg.num_kv_heads = 2; // GQA: 4 query heads, 2 KV heads
    let weights = tiny_weights(&cfg);
    let mut model = LlamaModel::from_weights(cfg, &Device::Cpu, weights, 64, false).unwrap();

    let prompt: Vec<Vec<TokenId>> = vec![(0..20).collect()];
    let positions: Vec<Vec<usize>> = vec![(0..20).collect()];
    let block_ids: Vec<Vec<usize>> = vec![vec![0, 1]];
    let logits = model
        .forward_logits(&[1], &prompt, &positions, &block_ids, &[0], &[true])
        .unwrap();
    let last = &logits[0];
    assert!(
        last.iter().all(|x| x.is_finite()),
        "GQA logits must be finite"
    );
    assert!(
        last.iter().any(|x| x.abs() > 1e-6),
        "GQA logits must be non-zero"
    );
}
