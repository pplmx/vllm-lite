use axum::http::StatusCode;
use vllm_model::config::Architecture;
use vllm_server::openai::models::models_handler;
use vllm_server::test_fixtures::api_state;

#[tokio::test]
async fn test_models_handler_returns_list() {
    let state = api_state(Architecture::Qwen3);

    let response = models_handler(axum::extract::State(state)).await;

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body();
    let bytes = axum::body::to_bytes(body, 1024 * 1024).await.unwrap();
    let json_str = String::from_utf8(bytes.to_vec()).unwrap();

    println!("Response: {json_str}");
    assert!(json_str.contains("\"object\":\"list\""));
    assert!(json_str.contains("\"model\""));
}
