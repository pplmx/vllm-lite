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

use axum::{Json, http::StatusCode};
use vllm_core::types::SamplingParams;

use super::types::ErrorResponse;

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
        let body = err.1 .0;
        assert_eq!(body.error.error_type, "invalid_request_error");
        assert!(
            body.error.message.contains("beam search"),
            "error message must explain the rejection: got '{}'",
            body.error.message
        );
    }
}
