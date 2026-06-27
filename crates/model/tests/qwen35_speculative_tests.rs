//! Speculative-decoding parity tests for Qwen3.5 hybrid models (Phase 5 Wave 4).

use candle_core::Device;
use vllm_model::qwen3_5::Qwen35HybridModel;
use vllm_model::qwen3_config::{Qwen3Config, TextConfig};
use vllm_traits::{ModelBackend, TokenId};

fn tiny_text_config(layer_types: Vec<&str>) -> TextConfig {
    TextConfig {
        hidden_size: Some(64),
        num_hidden_layers: Some(layer_types.len()),
        num_attention_heads: Some(2),
        num_key_value_heads: Some(2),
        intermediate_size: Some(128),
        layer_types: Some(layer_types.into_iter().map(str::to_string).collect()),
        ..Default::default()
    }
}

fn tiny_hybrid_config() -> Qwen3Config {
    Qwen3Config {
        vocab_size: Some(128),
        head_dim: Some(32),
        text_config: Some(TextConfig {
            num_hidden_layers: Some(2),
            num_attention_heads: Some(2),
            num_key_value_heads: Some(2),
            hidden_size: Some(64),
            intermediate_size: Some(128),
            layer_types: Some(vec![
                "linear_attention".to_string(),
                "full_attention".to_string(),
            ]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn tiny_config(layer_types: Vec<&str>) -> Qwen3Config {
    Qwen3Config {
        vocab_size: Some(32),
        head_dim: Some(32),
        text_config: Some(tiny_text_config(layer_types)),
        ..Default::default()
    }
}

fn prefill(model: &mut Qwen35HybridModel, seq_id: u64, tokens: &[TokenId]) {
    let positions: Vec<usize> = (0..tokens.len()).collect();
    let block_ids = vec![0usize; tokens.len()];
    model
        .forward(
            &[seq_id],
            &[tokens.to_vec()],
            &[positions],
            &[block_ids],
            &[0],
            &[true],
        )
        .expect("prefill");
}

fn decode_token(
    model: &mut Qwen35HybridModel,
    seq_id: u64,
    token: TokenId,
    num_computed: usize,
) -> TokenId {
    let out = model
        .forward(
            &[seq_id],
            &[vec![token]],
            &[vec![num_computed]],
            &[vec![0usize]],
            &[num_computed],
            &[false],
        )
        .expect("decode forward");
    out.next_tokens[0]
}

fn decode_token_to_layer(
    model: &mut Qwen35HybridModel,
    seq_id: u64,
    token: TokenId,
    num_computed: usize,
    upto_layer: usize,
) -> TokenId {
    let out = model
        .forward_to_layer(
            &[seq_id],
            &[vec![token]],
            &[vec![num_computed]],
            &[vec![0usize]],
            &[num_computed],
            &[false],
            upto_layer,
        )
        .expect("decode forward_to_layer");
    out.next_tokens[0]
}

/// Mirrors `SelfSpeculativeModel::generate_draft` single-sequence decode loop (Wave 4 / 5.4.4).
fn simulate_draft_tokens(
    model: &mut Qwen35HybridModel,
    seq_id: u64,
    start_token: TokenId,
    start_num_computed: usize,
    draft_layers: usize,
    num_tokens: usize,
) -> Vec<TokenId> {
    let mut current_token = start_token;
    let mut num_computed = start_num_computed;
    let mut drafts = Vec::with_capacity(num_tokens);

    for _ in 0..num_tokens {
        let next = decode_token_to_layer(model, seq_id, current_token, num_computed, draft_layers);
        drafts.push(next);
        current_token = next;
        num_computed += 1;
    }

    drafts
}

#[test]
fn test_forward_to_layer_matches_full_forward_full_attention_only() {
    let config = tiny_config(vec!["full_attention", "full_attention"]);
    let device = Device::Cpu;
    let tokens = vec![3u32, 7, 11];
    let seq_id = 42u64;

    let mut full_model = Qwen35HybridModel::new(config.clone(), device.clone(), 16, false).unwrap();
    prefill(&mut full_model, seq_id, &tokens);
    let full_token = decode_token(
        &mut full_model,
        seq_id,
        *tokens.last().unwrap(),
        tokens.len(),
    );

    let mut partial_model = Qwen35HybridModel::new(config, device, 16, false).unwrap();
    let num_layers = partial_model.num_layers();
    prefill(&mut partial_model, seq_id, &tokens);
    let partial_token = decode_token_to_layer(
        &mut partial_model,
        seq_id,
        *tokens.last().unwrap(),
        tokens.len(),
        num_layers,
    );

    assert_eq!(full_token, partial_token);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_gdn_prefill_decode_sequence_is_stable() {
    let config = tiny_config(vec!["linear_attention", "linear_attention"]);
    let device = Device::Cpu;
    let tokens = vec![1u32, 2, 3, 4];
    let prompt_len = tokens.len();
    let seq_id = 7u64;

    let mut model = Qwen35HybridModel::new(config.clone(), device.clone(), 16, false).unwrap();
    prefill(&mut model, seq_id, &tokens);
    let mut chain = tokens.clone();
    let mut decoded = Vec::new();
    for step in 0..3 {
        let num_computed = prompt_len + step;
        let next = decode_token(&mut model, seq_id, *chain.last().unwrap(), num_computed);
        decoded.push(next);
        chain.push(next);
    }

    let mut reference = Qwen35HybridModel::new(config, device, 16, false).unwrap();
    prefill(&mut reference, seq_id, &tokens);
    let mut ref_chain = tokens;
    for step in 0..3 {
        let num_computed = prompt_len + step;
        let next = decode_token(
            &mut reference,
            seq_id,
            *ref_chain.last().unwrap(),
            num_computed,
        );
        assert_eq!(next, decoded[step]);
        ref_chain.push(next);
    }
}

#[test]
fn test_multi_seq_gdn_state_isolation() {
    let config = tiny_hybrid_config();
    let device = Device::Cpu;
    let seq_a = 1u64;
    let seq_b = 2u64;
    let tokens_a = vec![1u32, 2, 3, 4];
    let tokens_b = vec![10u32, 20, 30, 40];

    let mut model = Qwen35HybridModel::new(config.clone(), device.clone(), 16, false).unwrap();
    prefill(&mut model, seq_a, &tokens_a);
    prefill(&mut model, seq_b, &tokens_b);

    let a1 = decode_token(&mut model, seq_a, *tokens_a.last().unwrap(), tokens_a.len());
    let _b1 = decode_token(&mut model, seq_b, *tokens_b.last().unwrap(), tokens_b.len());
    let a2 = decode_token(&mut model, seq_a, a1, tokens_a.len() + 1);

    let mut reference = Qwen35HybridModel::new(config, device, 16, false).unwrap();
    prefill(&mut reference, seq_a, &tokens_a);
    let r1 = decode_token(
        &mut reference,
        seq_a,
        *tokens_a.last().unwrap(),
        tokens_a.len(),
    );
    let r2 = decode_token(&mut reference, seq_a, r1, tokens_a.len() + 1);

    assert_eq!(a1, r1);
    assert_eq!(a2, r2);
}

#[test]
fn test_partial_draft_forward_smoke() {
    let config = tiny_hybrid_config();
    let device = Device::Cpu;
    let tokens = vec![5u32, 6, 7, 8];
    let seq_id = 99u64;

    let mut model = Qwen35HybridModel::new(config, device, 16, false).unwrap();
    prefill(&mut model, seq_id, &tokens);

    let drafts = simulate_draft_tokens(
        &mut model,
        seq_id,
        *tokens.last().unwrap(),
        tokens.len(),
        1,
        3,
    );
    assert_eq!(drafts.len(), 3);
}

#[test]
fn test_full_forward_unchanged_after_partial_draft() {
    // Partial draft must not corrupt per-seq GDN state used by a subsequent full decode.
    let config = tiny_hybrid_config();
    let device = Device::Cpu;
    let tokens = vec![2u32, 4, 6, 8];
    let seq_id = 11u64;
    let start_token = *tokens.last().unwrap();
    let start_num_computed = tokens.len();

    let mut baseline = Qwen35HybridModel::new(config.clone(), device.clone(), 16, false).unwrap();
    prefill(&mut baseline, seq_id, &tokens);
    let expected = decode_token(&mut baseline, seq_id, start_token, start_num_computed);

    let mut after_draft = Qwen35HybridModel::new(config, device, 16, false).unwrap();
    prefill(&mut after_draft, seq_id, &tokens);
    let _drafts = simulate_draft_tokens(
        &mut after_draft,
        seq_id,
        start_token,
        start_num_computed,
        1,
        2,
    );
    let actual = decode_token(&mut after_draft, seq_id, start_token, start_num_computed);

    assert_eq!(
        expected, actual,
        "partial draft steps must not alter full-decode logits for the same input"
    );
}
