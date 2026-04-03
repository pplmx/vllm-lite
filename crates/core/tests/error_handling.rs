use vllm_core::error::EngineError;
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
