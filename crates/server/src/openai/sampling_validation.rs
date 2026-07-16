//! Sampling parameter validation at the HTTP boundary.
//!
//! `vllm_core::sampling::sample_batch_with_params` documents that
//! `beam_width > 1` is not implemented and would otherwise silently
//! degrade to greedy decoding, which is a contract violation for any
//! caller that explicitly asked for beam search.
//!
//! This module exposes [`validate_sampling_params`], which every
//! handler must call before constructing an `EngineMessage::AddRequest`.
//! The OpenAI request types do not currently expose `beam_width` (the
//! upstream OpenAI API doesn't either), so this check is
//! defensive — it documents the contract and gates any future field
//! addition. When a misconfigured request does arrive, it gets a
//! `400 invalid_request_error` instead of a degraded response.
//!
//! [`validate_chat_request_fields`] and [`validate_completion_request_fields`]
//! reject OpenAI request fields that are declared on the wire types
//! but not yet honoured by the engine (architecture-performance §5.1
//! "API-01: OpenAI 兼容是部分兼容"). The accepted path is honest
//! rejection (400 invalid_request_error) — silent acceptance + ignored
//! field would be a contract violation for any caller that explicitly
//! asked for `n > 1` or `stop` sequences.
//!
//! `top_p` is HONOURED: the engine's `sample_batch_with_params`
//! reads `SamplingParams::top_p` and applies nucleus sampling
//! before selecting the next token. Validation rejects out-of-range
//! values (`top_p <= 0`, `top_p > 1`, `NaN`) with `400` so the
//! caller learns about the bad value before paying the cost of
//! enqueuing the request.

use axum::{Json, http::StatusCode};
use vllm_core::types::SamplingParams;

use super::types::{ChatRequest, CompletionRequest, ErrorResponse};

/// Validate a `top_p` value from an HTTP request.
///
/// Per the OpenAI API specification the valid range is `(0, 1]`:
/// `top_p = 0` would select zero tokens, `top_p > 1` is undefined
/// (the nucleus set would include more than the full vocabulary),
/// and `NaN` would make the math ill-defined.
///
/// `None` is accepted (use the engine default of `1.0`, i.e. no
/// nucleus filtering).
///
/// # Errors
///
/// Returns `Err((StatusCode::BAD_REQUEST, …))` when `top_p` is
/// outside `(0, 1]` or `NaN`. The error message names the field
/// and the bad value so callers can adapt without reading the
/// source.
pub fn validate_top_p(top_p: Option<f32>) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(p) = top_p {
        if p.is_nan() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    "top_p must be a finite number in the (0, 1] interval (got NaN)",
                    "invalid_request_error",
                )),
            ));
        }
        if p <= 0.0 {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    "top_p must be > 0; top_p = 0 would select zero tokens",
                    "invalid_request_error",
                )),
            ));
        }
        if p > 1.0 {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    "top_p must be <= 1 (per OpenAI spec); values > 1 are undefined",
                    "invalid_request_error",
                )),
            ));
        }
    }
    Ok(())
}

/// Reject sampling parameters the engine cannot honour.
///
/// # Errors
///
/// Returns `Err((StatusCode::BAD_REQUEST, …))` when
/// `params.beam_width > 1`. The OpenAI API does not expose beam
/// search, so this is currently a no-op for HTTP callers; if a
/// future API field surfaces `beam_width`, callers will get a
/// clear 400 instead of silent greedy fallback.
pub fn validate_sampling_params(
    params: &SamplingParams,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if params.beam_width > 1 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "beam search is not supported in this build; set beam_width = 1 (default)",
                "invalid_request_error",
            )),
        ));
    }
    Ok(())
}

/// Reject chat-request fields the engine does not yet honour, and
/// validate the ones it does.
///
/// Currently rejects:
/// - `n != 1` — the engine emits exactly one completion per request.
///   OpenAI's `n` would generate `n` independent completions in
///   parallel; we have no equivalent path. `None` and `Some(1)` are
///   both accepted (the latter is the default).
/// - non-empty `stop` — the engine stops at `max_tokens` or natural
///   EOS only. Accepting `stop` and ignoring it would silently
///   truncate at `max_tokens` even when a stop sequence was emitted,
///   which is a contract violation.
///
/// Validates:
/// - `top_p` — must be in `(0, 1]`. The engine honours `top_p` via
///   `sample_batch_with_params`; an out-of-range value would either
///   crash the sampler or silently produce garbage, so we reject
///   up front.
///
/// # Errors
///
/// Returns `Err((StatusCode::BAD_REQUEST, …))` when any check fires.
/// The error message names the rejected field so callers can adapt
/// without having to read the source.
pub fn validate_chat_request_fields(
    req: &ChatRequest,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    validate_top_p(req.top_p)?;
    if let Some(n) = req.n
        && n != 1
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "n > 1 is not supported in this build; the engine emits exactly one completion per request (omit n or set n = 1)",
                "invalid_request_error",
            )),
        ));
    }
    if let Some(stop) = &req.stop
        && !stop.is_empty()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "stop sequences are not yet honoured; the engine stops at max_tokens or natural EOS only (omit stop or send an empty array)",
                "invalid_request_error",
            )),
        ));
    }
    Ok(())
}

/// Reject completion-request fields the engine does not yet honour,
/// and validate the ones it does.
///
/// Mirror of [`validate_chat_request_fields`] for the legacy
/// `/v1/completions` endpoint. Same three checks (`n != 1`,
/// non-empty `stop`, out-of-range `top_p`).
///
/// # Errors
///
/// Returns `Err((StatusCode::BAD_REQUEST, …))` when any check fires.
pub fn validate_completion_request_fields(
    req: &CompletionRequest,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    validate_top_p(req.top_p)?;
    if let Some(n) = req.n
        && n != 1
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "n > 1 is not supported in this build; the engine emits exactly one completion per request (omit n or set n = 1)",
                "invalid_request_error",
            )),
        ));
    }
    if let Some(stop) = &req.stop
        && !stop.is_empty()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "stop sequences are not yet honoured; the engine stops at max_tokens or natural EOS only (omit stop or send an empty array)",
                "invalid_request_error",
            )),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_sampling_params_pass_validation() {
        // `SamplingParams::default()` is greedy (beam_width = 1);
        // validate() must accept it so the existing HTTP flow keeps
        // working unchanged.
        let params = SamplingParams::default();
        validate_sampling_params(&params).expect("default params must validate");
    }

    #[test]
    fn beam_width_one_passes() {
        let params = SamplingParams::builder().with_beam_width(1).build();
        validate_sampling_params(&params).expect("beam_width = 1 must pass");
    }

    #[test]
    fn beam_width_two_is_rejected_with_400() {
        let params = SamplingParams::builder().with_beam_width(2).build();
        let err = validate_sampling_params(&params).expect_err("beam_width = 2 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        let body = err.1.0;
        assert_eq!(body.error.error_type, "invalid_request_error");
        assert!(
            body.error.message.contains("beam search"),
            "error message must explain the rejection: got '{}'",
            body.error.message
        );
    }

    // ChatRequest field-validation tests

    fn chat_request_with_n(n: Option<i64>) -> ChatRequest {
        ChatRequest {
            model: "test-model".to_string(),
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n,
            stop: None,
            user: None,
        }
    }

    fn chat_request_with_stop(stop: Option<Vec<String>>) -> ChatRequest {
        ChatRequest {
            model: "test-model".to_string(),
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n: None,
            stop,
            user: None,
        }
    }

    fn chat_request_with_top_p(top_p: Option<f32>) -> ChatRequest {
        ChatRequest {
            model: "test-model".to_string(),
            messages: vec![],
            temperature: None,
            top_p,
            max_tokens: None,
            stream: None,
            n: None,
            stop: None,
            user: None,
        }
    }

    #[test]
    fn chat_request_default_passes_field_validation() {
        let req = chat_request_with_n(None);
        validate_chat_request_fields(&req).expect("n = None must pass");
    }

    #[test]
    fn chat_request_n_one_passes_field_validation() {
        let req = chat_request_with_n(Some(1));
        validate_chat_request_fields(&req).expect("n = 1 must pass");
    }

    #[test]
    fn chat_request_n_two_is_rejected_with_400() {
        let req = chat_request_with_n(Some(2));
        let err = validate_chat_request_fields(&req).expect_err("n = 2 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        let body = err.1.0;
        assert_eq!(body.error.error_type, "invalid_request_error");
        assert!(
            body.error.message.contains("n > 1"),
            "error message must mention n > 1: got '{}'",
            body.error.message
        );
    }

    #[test]
    fn chat_request_empty_stop_array_passes() {
        let req = chat_request_with_stop(Some(vec![]));
        validate_chat_request_fields(&req).expect("empty stop array must pass");
    }

    #[test]
    fn chat_request_non_empty_stop_is_rejected() {
        let req = chat_request_with_stop(Some(vec!["\n".to_string()]));
        let err = validate_chat_request_fields(&req).expect_err("non-empty stop must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        let body = err.1.0;
        assert_eq!(body.error.error_type, "invalid_request_error");
        assert!(
            body.error.message.contains("stop sequences"),
            "error message must mention stop sequences: got '{}'",
            body.error.message
        );
    }

    // CompletionRequest field-validation tests

    fn completion_request_with_n(n: Option<i64>) -> CompletionRequest {
        CompletionRequest {
            model: None,
            prompt: "hello".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n,
            stop: None,
            user: None,
        }
    }

    fn completion_request_with_stop(stop: Option<Vec<String>>) -> CompletionRequest {
        CompletionRequest {
            model: None,
            prompt: "hello".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n: None,
            stop,
            user: None,
        }
    }

    fn completion_request_with_top_p(top_p: Option<f32>) -> CompletionRequest {
        CompletionRequest {
            model: None,
            prompt: "hello".to_string(),
            temperature: None,
            top_p,
            max_tokens: None,
            stream: None,
            n: None,
            stop: None,
            user: None,
        }
    }

    #[test]
    fn completion_request_default_passes_field_validation() {
        let req = completion_request_with_n(None);
        validate_completion_request_fields(&req).expect("n = None must pass");
    }

    #[test]
    fn completion_request_n_two_is_rejected_with_400() {
        let req = completion_request_with_n(Some(2));
        let err = validate_completion_request_fields(&req).expect_err("n = 2 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        let body = err.1.0;
        assert_eq!(body.error.error_type, "invalid_request_error");
    }

    #[test]
    fn completion_request_non_empty_stop_is_rejected() {
        let req = completion_request_with_stop(Some(vec!["END".to_string()]));
        let err =
            validate_completion_request_fields(&req).expect_err("non-empty stop must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        let body = err.1.0;
        assert!(body.error.message.contains("stop sequences"));
    }

    // top_p validation tests — covers both the standalone
    // `validate_top_p` helper (called from each request-validator)
    // and the end-to-end chat / completion validators.

    #[test]
    fn top_p_none_passes() {
        // None means "use the engine default" (1.0 = no nucleus
        // filtering); both the standalone helper and the chat /
        // completion validators must accept it.
        validate_top_p(None).expect("top_p = None must pass");
        let chat = chat_request_with_top_p(None);
        validate_chat_request_fields(&chat).expect("chat with top_p = None must pass");
        let comp = completion_request_with_top_p(None);
        validate_completion_request_fields(&comp).expect("completion with top_p = None must pass");
    }

    #[test]
    fn top_p_one_passes() {
        // top_p = 1.0 is the inclusive upper bound; means "consider
        // all tokens" and matches the engine default.
        validate_top_p(Some(1.0)).expect("top_p = 1.0 must pass");
        let chat = chat_request_with_top_p(Some(1.0));
        validate_chat_request_fields(&chat).expect("chat with top_p = 1.0 must pass");
    }

    #[test]
    fn top_p_intermediate_passes() {
        // A typical value (0.9) must pass validation.
        validate_top_p(Some(0.9)).expect("top_p = 0.9 must pass");
        let comp = completion_request_with_top_p(Some(0.9));
        validate_completion_request_fields(&comp).expect("completion with top_p = 0.9 must pass");
    }

    #[test]
    fn top_p_zero_is_rejected() {
        // top_p = 0 would select zero tokens — undefined behaviour
        // for the sampler; reject with 400.
        let err = validate_top_p(Some(0.0)).expect_err("top_p = 0 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.0.error.message.contains("> 0"));

        let chat = chat_request_with_top_p(Some(0.0));
        let err = validate_chat_request_fields(&chat).expect_err("chat top_p = 0 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn top_p_negative_is_rejected() {
        // Negative values are nonsensical; reject up front rather
        // than letting them through to the sampler.
        let err = validate_top_p(Some(-0.1)).expect_err("top_p < 0 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn top_p_above_one_is_rejected() {
        // Per OpenAI spec the upper bound is inclusive 1; values
        // > 1 are undefined.
        let err = validate_top_p(Some(1.5)).expect_err("top_p > 1 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.0.error.message.contains("<= 1"));

        let comp = completion_request_with_top_p(Some(2.0));
        let err = validate_completion_request_fields(&comp)
            .expect_err("completion top_p = 2 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn top_p_nan_is_rejected() {
        // NaN would make the nucleus comparison ill-defined; the
        // sampler would either panic or produce undefined behaviour.
        let err = validate_top_p(Some(f32::NAN)).expect_err("top_p = NaN must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.0.error.message.contains("NaN"));
    }
}
