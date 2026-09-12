use axum::http::StatusCode;
use vllm_model::config::Architecture;
use vllm_server::openai::models::model_by_id_handler;
use vllm_server::openai::models::models_handler;
use vllm_server::test_fixtures::api_state;

#[tokio::test]
async fn test_models_handler_returns_list() {
    let state = api_state(Architecture::Qwen3);

    let response = models_handler(axum::extract::State(state)).await;

    assert_eq!(response.status(), StatusCode::OK);
    // RIL ISS-118: /v1/models must be uncacheable (OpenAI sends
    // cache-control: no-cache) so proxies don't serve a stale model list.
    assert_eq!(
        response
            .headers()
            .get("cache-control")
            .and_then(|v| v.to_str().ok()),
        Some("no-cache"),
        "GET /v1/models must carry cache-control: no-cache"
    );

    let body = response.into_body();
    let bytes = axum::body::to_bytes(body, 1024 * 1024).await.unwrap();
    let json_str = String::from_utf8(bytes.to_vec()).unwrap();

    println!("Response: {json_str}");
    assert!(json_str.contains("\"object\":\"list\""));
    assert!(json_str.contains("\"model\""));
}

/// RIL ISS-106: `GET /v1/models/{id}` must resolve the id the list
/// endpoint advertises (same single-element payload) and return a clean
/// OpenAI `404 model_not_found` for unknown ids — the route was missing
/// entirely, so SDKs that follow a model list with a per-model lookup hit
/// axum's default empty 404.
#[tokio::test]
async fn test_model_by_id_returns_served_model() {
    let state = api_state(Architecture::Qwen3);
    // The served model name comes from the tokenizer (Qwen3 fixture has
    // no HF backend → model_name None, handler falls back to "unknown").
    let model_name = state
        .tokenizer
        .model_name()
        .unwrap_or_else(|| "unknown".to_string());

    let response = model_by_id_handler(
        axum::extract::Path(model_name.clone()),
        axum::extract::State(state),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    // RIL ISS-118: by-id lookup is also uncacheable (mirrors the list).
    assert_eq!(
        response
            .headers()
            .get("cache-control")
            .and_then(|v| v.to_str().ok()),
        Some("no-cache")
    );
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json_str = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        json_str.contains("\"object\":\"list\""),
        "by-id payload must mirror the list endpoint: {json_str}"
    );
    assert!(
        json_str.contains(&format!("\"id\":\"{model_name}\"")),
        "must return the requested model id: {json_str}"
    );
}

#[tokio::test]
async fn test_model_by_id_unknown_returns_404_openai_error() {
    let state = api_state(Architecture::Qwen3);

    let response = model_by_id_handler(
        axum::extract::Path("no-such-model".to_string()),
        axum::extract::State(state),
    )
    .await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json_str = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        json_str.contains("\"model_not_found\""),
        "unknown model must be a clean OpenAI model_not_found error: {json_str}"
    );
    assert!(
        json_str.contains("\"invalid_request_error\""),
        "unknown model type must be invalid_request_error: {json_str}"
    );
}
