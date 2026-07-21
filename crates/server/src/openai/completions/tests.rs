//! Unit tests for the `OpenAI` `/v1/completions` endpoint.
//!
//! Covers the validation path (empty prompt → 400) and the
//! engine-channel error mapping (closed channel → 503
//! `engine_unavailable`). The handler is exercised without a live
//! engine by relying on `test_fixtures::api_state`.
use super::*;
use crate::security::correlation::CorrelationId;
use axum::Extension;
use vllm_traits::TokenId;

fn create_test_state() -> crate::ApiState {
    crate::test_fixtures::api_state(vllm_model::config::Architecture::Qwen3)
}

#[tokio::test]
async fn test_completions_empty_prompt() {
    let state = create_test_state();
    let req = CompletionRequest {
        model: None,
        prompt: String::new(),
        temperature: None,
        top_p: None,
        max_tokens: Some(100),
        stream: None,
        n: None,
        stop: None,
        user: None,
        seed: None,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: None,
        echo: None,
        suffix: None,
        best_of: None,
    };

    let result = completions(
        State(state),
        Extension(CorrelationId("test-correlation-id".into())),
        Json(req),
    )
    .await;
    assert!(result.is_err());
    let (status, _) = result.unwrap_err();
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_completions_with_valid_max_tokens() {
    let state = create_test_state();
    let req = CompletionRequest {
        model: None,
        prompt: "Hello".to_string(),
        temperature: None,
        top_p: None,
        max_tokens: Some(10),
        stream: None,
        n: None,
        stop: None,
        user: None,
        seed: None,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: None,
        echo: None,
        suffix: None,
        best_of: None,
    };

    // With no engine running, this will fail to send to engine
    // but we can verify it doesn't fail on validation. The closed-channel
    // error surfaces as 503 SERVICE_UNAVAILABLE with `engine_unavailable`
    // code (see `completions` handler) — distinguishable from a real
    // server-side bug, and safe for clients to retry.
    let result = completions(
        State(state),
        Extension(CorrelationId("test-correlation-id".into())),
        Json(req),
    )
    .await;
    assert!(result.is_err());
    let (status, body) = result.unwrap_err();
    assert_eq!(status, axum::http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body.error.code.as_deref(), Some("engine_unavailable"));
}

// =============================================================================
// P37 v0.x wire-type follow-up — engine wire-through: unit tests for
// `rank_by_mean_logprob`. Pin the helper's contract end-to-end so future
// refactors (e.g. switching to a length-penalty variant) trip the suite.
// =============================================================================

fn sampled(token: TokenId, logprob: f32) -> vllm_traits::SampledToken {
    vllm_traits::SampledToken {
        token,
        logprob,
        top_logprobs: Vec::new(),
    }
}

#[test]
fn test_rank_by_mean_logprob_single_candidate_returns_zero() {
    // Degenerate case: only one candidate. The ranker must return
    // 0 (the only valid index).
    let candidates: Vec<Vec<vllm_traits::SampledToken>> = vec![vec![sampled(10, -1.0)]];
    assert_eq!(rank_by_mean_logprob(&candidates), 0);
}

#[test]
fn test_rank_by_mean_logprob_distinct_means_picks_highest() {
    // Three candidates with distinct mean logprobs — the ranker
    // must return the index of the candidate with the highest
    // mean logprob.
    let candidates: Vec<Vec<vllm_traits::SampledToken>> = vec![
        vec![sampled(10, -1.0), sampled(11, -1.0)], // mean = -1.0
        vec![sampled(20, -0.5), sampled(21, -0.5)], // mean = -0.5
        vec![sampled(30, 0.0), sampled(31, 0.0)],   // mean =  0.0
    ];
    assert_eq!(rank_by_mean_logprob(&candidates), 2);
}

#[test]
fn test_rank_by_mean_logprob_equal_means_tie_breaks_on_seq_id() {
    // Two candidates with identical mean logprob (-1.0). The
    // tie-break rule says the LOWER index (seq_id proxy) wins.
    // Index 0 must win over index 1.
    let candidates: Vec<Vec<vllm_traits::SampledToken>> = vec![
        vec![sampled(10, -1.0), sampled(11, -1.0)], // mean = -1.0, idx 0
        vec![sampled(20, -1.0), sampled(21, -1.0)], // mean = -1.0, idx 1
    ];
    assert_eq!(rank_by_mean_logprob(&candidates), 0);
}

#[test]
fn test_rank_by_mean_logprob_empty_candidates_returns_zero() {
    // Defensive default: an empty input must not panic. The ranker
    // returns 0 (the only valid index in an empty slice).
    let candidates: Vec<Vec<vllm_traits::SampledToken>> = Vec::new();
    assert_eq!(rank_by_mean_logprob(&candidates), 0);
}

#[test]
fn test_rank_by_mean_logprob_length_normalized() {
    // Two candidates with different LENGTHS but the same mean
    // logprob. The ranker uses the arithmetic mean (length-normalized
    // by design) so longer + shorter completions can be compared
    // directly per OpenAI's "mean log probability" wording.
    let candidates: Vec<Vec<vllm_traits::SampledToken>> = vec![
        vec![sampled(10, -1.0), sampled(11, -1.0), sampled(12, -1.0)], // 3 tokens, mean = -1.0
        vec![sampled(20, -1.0), sampled(21, -1.0)],                    // 2 tokens, mean = -1.0
    ];
    // Both have mean -1.0; tie-break on lower seq_id → idx 0 wins.
    assert_eq!(rank_by_mean_logprob(&candidates), 0);

    // Now flip the means: candidate 1 has higher mean → idx 1 wins.
    let candidates: Vec<Vec<vllm_traits::SampledToken>> = vec![
        vec![sampled(10, -2.0), sampled(11, -2.0), sampled(12, -2.0)], // mean = -2.0
        vec![sampled(20, -0.5), sampled(21, -0.5)],                    // mean = -0.5
    ];
    assert_eq!(rank_by_mean_logprob(&candidates), 1);
}
