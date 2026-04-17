use vllm_core::error::{EngineError, Result};
use vllm_core::types::Request;

#[test]
fn test_error_message_format() {
    let err = EngineError::ModelError("test error message".to_string());
    assert_eq!(err.to_string(), "model forward failed: test error message");
}

#[test]
fn test_seq_not_found_error() {
    let err = EngineError::SeqNotFound { id: 42 };
    let msg = err.to_string();
    assert!(msg.contains("42"));
    assert!(msg.contains("not found"));
}

#[test]
fn test_model_error() {
    let err = EngineError::ModelError("Forward pass failed".to_string());
    assert!(err.to_string().contains("Forward pass failed"));
}

#[test]
fn test_invalid_request_error() {
    let err = EngineError::InvalidRequest("empty prompt".to_string());
    let msg = err.to_string();
    assert!(msg.contains("invalid request"));
    assert!(msg.contains("empty prompt"));
}

#[test]
fn test_sampling_error() {
    let err = EngineError::SamplingError("invalid temperature".to_string());
    let msg = err.to_string();
    assert!(msg.contains("sampling failed"));
    assert!(msg.contains("invalid temperature"));
}

#[test]
fn test_error_from_trait() {
    let model_err = vllm_traits::ModelError::new("OOM error");
    let engine_err: EngineError = model_err.into();
    assert!(matches!(engine_err, EngineError::ModelError(_)));
    assert!(engine_err.to_string().contains("OOM error"));
}

#[test]
fn test_error_result_type() {
    fn fallible() -> Result<i32> {
        Err(EngineError::SeqNotFound { id: 99 })
    }
    let result = fallible();
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        EngineError::SeqNotFound { id: 99 }
    ));
}

#[test]
fn test_error_debug_format() {
    let err = EngineError::ModelError("test".to_string());
    let debug = format!("{:?}", err);
    assert!(debug.contains("ModelError"));
}

#[test]
fn test_request_zero_prompt() {
    let req = Request::new(1, vec![], 10);
    assert_eq!(req.prompt.len(), 0);
    assert_eq!(req.max_tokens, 10);
}

#[test]
fn test_request_normal() {
    let req = Request::new(1, vec![10, 20, 30], 50);
    assert_eq!(req.prompt.len(), 3);
    assert_eq!(req.max_tokens, 50);
}
