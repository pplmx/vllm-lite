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

use super::types::{ChatRequest, CompletionRequest, ErrorResponse, ResponseFormat};

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

/// Validate the `response_format` field on a chat request.
///
/// v0.2 accepts only [`ResponseFormat::Text`] and
/// [`ResponseFormat::JsonObject`]. The `{type: "json_schema"}`
/// variant (which would require a grammar-constrained decoder) is
/// rejected at the **serde layer** because it is not a declared
/// variant on [`ResponseFormat`] — the deserialization error surfaces
/// as a 400 with serde's standard "unknown variant" message.
///
/// This function is a no-op pass-through today but exists for three
/// reasons:
///
/// 1. **Documentation** — the function name + doc-comment make the
///    "v0.2 accepts text + json_object only" contract explicit at
///    the validator layer, not just at the type definition.
/// 2. **Forward-compatibility** — if v0.3 ever adds stricter checks
///    (e.g. format-specific syntax requirements), the hook is here.
/// 3. **Pattern parity** — mirrors [`validate_chat_request_fields`]
///    so handlers have a single call site for all request-level
///    checks, and tracing/audit can find the validator by name.
///
/// # Errors
///
/// Currently infallible. Preserves the [`Result`] return type for
/// forward-compatibility — if v0.3 adds strict checks (e.g.
/// `JsonObject` payload must already contain a JSON example in the
/// prompt), the validator would reject malformed usage here.
///
/// `#[allow(clippy::missing_const_for_fn)]` — clippy flags the
/// body as eligible for `const fn` because it contains no runtime
/// operations. We intentionally keep it non-`const` for
/// forward-compatibility: future validators will need runtime
/// operations (regex checks, format validation, etc.) and we don't
/// want the signature to change when that happens. The `Json<T>`
/// return type already precludes `const fn` on stable Rust.
#[allow(clippy::missing_const_for_fn)]
pub fn validate_chat_response_format(
    _format: Option<&ResponseFormat>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    Ok(())
}

/// Validate an OpenAI `frequency_penalty` or `presence_penalty` value.
///
/// Per the OpenAI API specification the valid range for both fields
/// is `[-2.0, 2.0]`. Values outside that range are rejected with
/// `400 invalid_request_error`; `NaN` is rejected (the sampler
/// math is ill-defined); `±infinity` is rejected (the engine would
/// either crash or silently saturate).
///
/// `None` is accepted (use the engine default of no penalty — for
/// `frequency_penalty` this maps to `repeat_penalty = 1.0`; for
/// `presence_penalty` this is the no-op default).
///
/// **Honoring note (v0.3 declaration + P29 sign-aware engine):**
/// the engine's `apply_repeat_penalty` implements frequency-style
/// semantics (penalty proportional to occurrence count). The chat
/// / completions handlers map `frequency_penalty` to
/// `repeat_penalty = (1.0 + value).max(1e-3)`. For positive
/// `frequency_penalty` the engine divides positive logits and
/// multiplies negative logits by `repeat_penalty` (P29 sign-aware
/// refactor), giving the correct "penalize repetition" semantic.
/// For negative `frequency_penalty` (boost) the engine uses the
/// same sign-aware multiply, giving the correct "boost repetition"
/// semantic. The 1e-3 floor prevents divide-by-zero for extreme
/// negative values (e.g. `frequency_penalty = -1.5` would otherwise
/// produce `repeat_penalty = -0.5`). `presence_penalty` is honored
/// end-to-end via the new `apply_presence_penalty` helper (P28).
/// See the
/// `frequency_penalty` / `presence_penalty` field doc-comments on
/// [`ChatRequest`] for the full per-field rationale.
///
/// # Errors
///
/// Returns `Err((StatusCode::BAD_REQUEST, …))` when `value` is
/// outside `[-2.0, 2.0]`, `NaN`, or `±infinity`. The error message
/// names the field so callers can adapt without reading the source.
pub fn validate_penalty(
    value: Option<f32>,
    field_name: &str,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    const PENALTY_MIN: f32 = -2.0;
    const PENALTY_MAX: f32 = 2.0;
    if let Some(v) = value {
        if v.is_nan() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    format!(
                        "{field_name} must be a finite number in the [-2.0, 2.0] interval (got NaN)"
                    )
                    .as_str(),
                    "invalid_request_error",
                )),
            ));
        }
        if !v.is_finite() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    format!(
                        "{field_name} must be a finite number in the [-2.0, 2.0] interval (got {v})"
                    )
                    .as_str(),
                    "invalid_request_error",
                )),
            ));
        }
        if v < PENALTY_MIN || v > PENALTY_MAX {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    format!(
                        "{field_name} must be in the [-2.0, 2.0] interval per OpenAI spec (got {v})"
                    )
                    .as_str(),
                    "invalid_request_error",
                )),
            ));
        }
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
/// - `frequency_penalty` / `presence_penalty` — must be in
///   `[-2.0, 2.0]` and finite (per OpenAI spec). Both fields are
///   honored end-to-end (`frequency_penalty` via the sign-aware
///   `apply_repeat_penalty` mapping `(1.0 + value).max(1e-3)`,
///   P29; `presence_penalty` via the new `apply_presence_penalty`
///   helper, P28).
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
    validate_chat_response_format(req.response_format.as_ref())?;
    validate_penalty(req.frequency_penalty, "frequency_penalty")?;
    validate_penalty(req.presence_penalty, "presence_penalty")?;
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
/// `/v1/completions` endpoint. Same set of checks (`n != 1`,
/// non-empty `stop`, out-of-range `top_p`, out-of-range penalties).
///
/// # Errors
///
/// Returns `Err((StatusCode::BAD_REQUEST, …))` when any check fires.
pub fn validate_completion_request_fields(
    req: &CompletionRequest,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    validate_top_p(req.top_p)?;
    validate_penalty(req.frequency_penalty, "frequency_penalty")?;
    validate_penalty(req.presence_penalty, "presence_penalty")?;
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
            response_format: None,
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
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
            response_format: None,
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
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
            response_format: None,
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
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

    // response_format validation tests (P22 v0.2 wire-type follow-up)

    #[test]
    fn chat_response_format_none_passes_validation() {
        // Omitted / `None` is the default path; must pass.
        validate_chat_response_format(None)
            .expect("response_format = None must pass validation (default path)");
    }

    #[test]
    fn chat_response_format_text_passes_validation() {
        // `{type: "text"}` is the OpenAI default; explicitly setting
        // it must pass with no validation errors.
        validate_chat_response_format(Some(&ResponseFormat::Text))
            .expect("response_format = Text must pass validation (OpenAI default)");
    }

    #[test]
    fn chat_response_format_json_object_passes_validation() {
        // `{type: "json_object"}` is accepted as a v0.2 declaration
        // pass-through. Honoring is a no-op today (deferred to v0.3
        // / v32+), but the wire-type contract accepts the value.
        validate_chat_response_format(Some(&ResponseFormat::JsonObject))
            .expect("response_format = JsonObject must pass validation (v0.2 pass-through)");
    }

    #[test]
    fn chat_request_with_response_format_text_passes_full_field_validation() {
        // Integration with `validate_chat_request_fields`: a request
        // that has both `response_format = Text` and other valid
        // fields must pass the full chat-request validator.
        let req = ChatRequest {
            model: "test-model".to_string(),
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n: None,
            stop: None,
            user: None,
            response_format: Some(ResponseFormat::Text),
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
        };
        validate_chat_request_fields(&req)
            .expect("chat request with response_format = Text must pass full field validation");
    }

    #[test]
    fn chat_request_with_response_format_json_object_passes_full_field_validation() {
        // Same as above but with `JsonObject` — verifies the
        // validator flow is wired correctly for both accepted
        // variants.
        let req = ChatRequest {
            model: "test-model".to_string(),
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n: None,
            stop: None,
            user: None,
            response_format: Some(ResponseFormat::JsonObject),
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
        };
        validate_chat_request_fields(&req).expect(
            "chat request with response_format = JsonObject must pass full field validation",
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
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
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
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
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
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
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

    // P23 v0.2 wire-type follow-up: `seed` field declaration.
    //
    // `seed` per the OpenAI spec accepts any integer (the contract is
    // "best effort to sample deterministically" — there is no range
    // validation, no NaN check, no length cap). Honoring is a no-op
    // today (the sampler is unseeded) so the only contract these tests
    // pin is "the field flows through the validator without rejection".
    // If v32+ adds engine-side RNG seeding and a tighter validation
    // contract, these tests grow the new shape; for v0.2 the validator
    // is intentionally permissive.

    #[test]
    fn chat_request_with_seed_none_passes_full_field_validation() {
        // Baseline: omitted seed (None) is the default path; must
        // pass `validate_chat_request_fields` unchanged.
        let req = chat_request_with_n(None);
        validate_chat_request_fields(&req).expect("seed = None must pass validation");
    }

    #[test]
    fn chat_request_with_seed_positive_passes_full_field_validation() {
        // OpenAI spec: any integer is accepted. A typical positive
        // seed value must pass.
        let mut req = chat_request_with_n(None);
        req.seed = Some(42);
        validate_chat_request_fields(&req).expect(
            "chat with seed = Some(42) must pass validation (OpenAI spec accepts any integer)",
        );
    }

    #[test]
    fn chat_request_with_seed_negative_passes_full_field_validation() {
        // Negative integers are valid i64 values per the OpenAI spec;
        // we must not silently reject them.
        let mut req = chat_request_with_n(None);
        req.seed = Some(-1);
        validate_chat_request_fields(&req).expect(
            "chat with seed = Some(-1) must pass validation (any i64 is valid per OpenAI spec)",
        );
    }

    #[test]
    fn chat_request_with_seed_zero_passes_full_field_validation() {
        // Zero is a valid i64 — many RNG implementations treat seed=0
        // as a valid input (e.g. `rand::SeedableRng::seed_from_u64(0)`).
        // Must not be silently rejected.
        let mut req = chat_request_with_n(None);
        req.seed = Some(0);
        validate_chat_request_fields(&req)
            .expect("chat with seed = Some(0) must pass validation (any i64 is valid)");
    }

    #[test]
    fn chat_request_with_seed_i64_min_passes_full_field_validation() {
        // Boundary: i64::MIN is the most-negative value; must not
        // overflow or trigger a range check.
        let mut req = chat_request_with_n(None);
        req.seed = Some(i64::MIN);
        validate_chat_request_fields(&req)
            .expect("chat with seed = Some(i64::MIN) must pass validation (any i64 is valid)");
    }

    #[test]
    fn chat_request_with_seed_i64_max_passes_full_field_validation() {
        // Boundary: i64::MAX is the most-positive value; must not
        // overflow or trigger a range check.
        let mut req = chat_request_with_n(None);
        req.seed = Some(i64::MAX);
        validate_chat_request_fields(&req)
            .expect("chat with seed = Some(i64::MAX) must pass validation (any i64 is valid)");
    }

    #[test]
    fn completion_request_with_seed_some_passes_full_field_validation() {
        // The completions endpoint accepts the same seed contract;
        // mirror the chat test on `CompletionRequest`.
        let req = CompletionRequest {
            model: None,
            prompt: "hello".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n: None,
            stop: None,
            user: None,
            seed: Some(42),
            frequency_penalty: None,
            presence_penalty: None,
        };
        validate_completion_request_fields(&req).expect(
            "completion with seed = Some(42) must pass validation (any i64 is valid per OpenAI spec)",
        );
    }

    #[test]
    fn completion_request_with_seed_none_passes_full_field_validation() {
        // Baseline: omitted seed (None) is the default path on
        // /v1/completions as well.
        let req = CompletionRequest {
            model: None,
            prompt: "hello".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n: None,
            stop: None,
            user: None,
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
        };
        validate_completion_request_fields(&req)
            .expect("completion with seed = None must pass validation");
    }

    // P27 v0.3 wire-type follow-up: `frequency_penalty` +
    // `presence_penalty` field declaration. Per OpenAI spec the
    // valid range for both fields is `[-2.0, 2.0]` with finite
    // values only (no NaN, no ±infinity). The validator rejects
    // out-of-range / non-finite values with `400
    // invalid_request_error`; honoring end-to-end is documented on
    // the field doc-comments in `types.rs`.

    // Standalone `validate_penalty` helper — covers the basic
    // acceptance + rejection paths the helper enforces.

    #[test]
    fn validate_penalty_none_passes() {
        // Omitted / `None` is the default path; must pass.
        validate_penalty(None, "frequency_penalty")
            .expect("penalty = None must pass (default path)");
    }

    #[test]
    fn validate_penalty_zero_passes() {
        // 0.0 is the OpenAI default and falls in the [-2.0, 2.0]
        // interval; must pass.
        validate_penalty(Some(0.0), "presence_penalty")
            .expect("penalty = 0.0 must pass (OpenAI default)");
    }

    #[test]
    fn validate_penalty_positive_in_range_passes() {
        // A typical positive value (1.0) must pass.
        validate_penalty(Some(1.0), "frequency_penalty")
            .expect("penalty = 1.0 must pass (in [-2.0, 2.0] interval)");
    }

    #[test]
    fn validate_penalty_negative_in_range_passes() {
        // A typical negative value (-1.0) must pass. The validator
        // accepts the full [-2.0, 2.0] interval per OpenAI spec;
        // clamping to no-penalty is the handler's job (see the
        // wire-through blocks in chat.rs / completions.rs).
        validate_penalty(Some(-1.0), "presence_penalty")
            .expect("penalty = -1.0 must pass (in [-2.0, 2.0] interval)");
    }

    #[test]
    fn validate_penalty_lower_boundary_passes() {
        // -2.0 is the inclusive lower bound per OpenAI spec.
        validate_penalty(Some(-2.0), "frequency_penalty")
            .expect("penalty = -2.0 must pass (inclusive lower bound)");
    }

    #[test]
    fn validate_penalty_upper_boundary_passes() {
        // 2.0 is the inclusive upper bound per OpenAI spec.
        validate_penalty(Some(2.0), "presence_penalty")
            .expect("penalty = 2.0 must pass (inclusive upper bound)");
    }

    #[test]
    fn validate_penalty_above_upper_bound_is_rejected() {
        // 2.5 is just outside the upper bound; must be rejected
        // with 400. The error message must name the field so the
        // caller can adapt.
        let err = validate_penalty(Some(2.5), "frequency_penalty")
            .expect_err("penalty = 2.5 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.0.error.message.contains("frequency_penalty"));
        assert!(err.1.0.error.message.contains("[-2.0, 2.0]"));
    }

    #[test]
    fn validate_penalty_below_lower_bound_is_rejected() {
        // -2.5 is just outside the lower bound; must be rejected
        // with 400.
        let err = validate_penalty(Some(-2.5), "presence_penalty")
            .expect_err("penalty = -2.5 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.0.error.message.contains("presence_penalty"));
    }

    #[test]
    fn validate_penalty_nan_is_rejected() {
        // NaN would make the sampler math ill-defined; the
        // validator must reject it.
        let err = validate_penalty(Some(f32::NAN), "frequency_penalty")
            .expect_err("penalty = NaN must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.0.error.message.contains("NaN"));
    }

    #[test]
    fn validate_penalty_positive_infinity_is_rejected() {
        // +infinity would saturate the engine's logit-divide math;
        // the validator must reject it.
        let err = validate_penalty(Some(f32::INFINITY), "presence_penalty")
            .expect_err("penalty = +inf must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_penalty_negative_infinity_is_rejected() {
        // -infinity would also saturate the engine's logit-divide
        // math; the validator must reject it.
        let err = validate_penalty(Some(f32::NEG_INFINITY), "frequency_penalty")
            .expect_err("penalty = -inf must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    // End-to-end: `validate_chat_request_fields` integration with
    // the penalty validators. Same shape as the P22 response_format
    // / P23 seed tests — verify the validator stack accepts valid
    // penalty values and rejects invalid ones with the right error
    // category.

    fn chat_request_with_frequency_penalty(fp: Option<f32>) -> ChatRequest {
        let mut req = chat_request_with_n(None);
        req.frequency_penalty = fp;
        req
    }

    fn chat_request_with_presence_penalty(pp: Option<f32>) -> ChatRequest {
        let mut req = chat_request_with_n(None);
        req.presence_penalty = pp;
        req
    }

    #[test]
    fn chat_request_with_frequency_penalty_none_passes_field_validation() {
        let req = chat_request_with_frequency_penalty(None);
        validate_chat_request_fields(&req).expect("chat with frequency_penalty = None must pass");
    }

    #[test]
    fn chat_request_with_frequency_penalty_zero_passes_field_validation() {
        let req = chat_request_with_frequency_penalty(Some(0.0));
        validate_chat_request_fields(&req)
            .expect("chat with frequency_penalty = 0.0 must pass (OpenAI default)");
    }

    #[test]
    fn chat_request_with_frequency_penalty_positive_passes_field_validation() {
        let req = chat_request_with_frequency_penalty(Some(1.0));
        validate_chat_request_fields(&req)
            .expect("chat with frequency_penalty = 1.0 must pass (in [-2.0, 2.0])");
    }

    #[test]
    fn chat_request_with_frequency_penalty_negative_passes_field_validation() {
        // Negative values are accepted by the validator (in the
        // [-2.0, 2.0] range); the handler clamps them to no-penalty
        // when forwarding to the engine, but the validator itself
        // doesn't clamp.
        let req = chat_request_with_frequency_penalty(Some(-1.5));
        validate_chat_request_fields(&req)
            .expect("chat with frequency_penalty = -1.5 must pass (in [-2.0, 2.0])");
    }

    #[test]
    fn chat_request_with_frequency_penalty_out_of_range_is_rejected() {
        let req = chat_request_with_frequency_penalty(Some(2.5));
        let err = validate_chat_request_fields(&req)
            .expect_err("chat with frequency_penalty = 2.5 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.0.error.message.contains("frequency_penalty"));
    }

    #[test]
    fn chat_request_with_frequency_penalty_nan_is_rejected() {
        let req = chat_request_with_frequency_penalty(Some(f32::NAN));
        let err = validate_chat_request_fields(&req)
            .expect_err("chat with frequency_penalty = NaN must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn chat_request_with_presence_penalty_none_passes_field_validation() {
        let req = chat_request_with_presence_penalty(None);
        validate_chat_request_fields(&req).expect("chat with presence_penalty = None must pass");
    }

    #[test]
    fn chat_request_with_presence_penalty_positive_passes_field_validation() {
        let req = chat_request_with_presence_penalty(Some(1.0));
        validate_chat_request_fields(&req)
            .expect("chat with presence_penalty = 1.0 must pass (in [-2.0, 2.0])");
    }

    #[test]
    fn chat_request_with_presence_penalty_out_of_range_is_rejected() {
        let req = chat_request_with_presence_penalty(Some(-2.5));
        let err = validate_chat_request_fields(&req)
            .expect_err("chat with presence_penalty = -2.5 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.0.error.message.contains("presence_penalty"));
    }

    #[test]
    fn chat_request_with_both_penalties_set_passes_field_validation() {
        // Both penalty fields set together must pass — the
        // validator must accept independent penalties on the same
        // request.
        let req = ChatRequest {
            model: "test-model".to_string(),
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n: None,
            stop: None,
            user: None,
            response_format: None,
            seed: None,
            frequency_penalty: Some(0.5),
            presence_penalty: Some(-0.5),
        };
        validate_chat_request_fields(&req)
            .expect("chat with both penalties set must pass full field validation");
    }

    #[test]
    fn completion_request_with_frequency_penalty_none_passes_field_validation() {
        // Mirror of the chat test on `CompletionRequest`. Same
        // contract: None is the default path; must pass.
        let req = CompletionRequest {
            model: None,
            prompt: "hello".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n: None,
            stop: None,
            user: None,
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
        };
        validate_completion_request_fields(&req)
            .expect("completion with frequency_penalty = None must pass");
    }

    #[test]
    fn completion_request_with_frequency_penalty_positive_passes_field_validation() {
        let req = CompletionRequest {
            model: None,
            prompt: "hello".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n: None,
            stop: None,
            user: None,
            seed: None,
            frequency_penalty: Some(1.0),
            presence_penalty: None,
        };
        validate_completion_request_fields(&req)
            .expect("completion with frequency_penalty = 1.0 must pass (in [-2.0, 2.0])");
    }

    #[test]
    fn completion_request_with_frequency_penalty_out_of_range_is_rejected() {
        let req = CompletionRequest {
            model: None,
            prompt: "hello".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n: None,
            stop: None,
            user: None,
            seed: None,
            frequency_penalty: Some(3.0),
            presence_penalty: None,
        };
        let err = validate_completion_request_fields(&req)
            .expect_err("completion with frequency_penalty = 3.0 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.0.error.message.contains("frequency_penalty"));
    }

    #[test]
    fn completion_request_with_presence_penalty_out_of_range_is_rejected() {
        let req = CompletionRequest {
            model: None,
            prompt: "hello".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            n: None,
            stop: None,
            user: None,
            seed: None,
            frequency_penalty: None,
            presence_penalty: Some(3.0),
        };
        let err = validate_completion_request_fields(&req)
            .expect_err("completion with presence_penalty = 3.0 must be rejected");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.0.error.message.contains("presence_penalty"));
    }
}
