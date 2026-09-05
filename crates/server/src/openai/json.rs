//! Shared `OpenaiJson<T>` request extractor for the OpenAI surface.
//!
//! axum's default `Json<T>` rejection behavior does not match the
//! OpenAI wire contract: a wrong-shaped-but-valid body returns
//! `422 Unprocessable Entity` with a plain-text body (`Failed to
//! deserialize the JSON body into the target type: ...`), a missing
//! `Content-Type` returns `415`, and the message leaks serde/type
//! internals. OpenAI clients parse error bodies as `{error:
//! {message, type, code}}` and mostly treat non-200 / non-400 as
//! opaque, so these defaults break client error handling (RIL ISS-104).
//!
//! This wrapper intercepts every [`JsonRejection`] and maps it to the
//! same OpenAI-format `ErrorResponse` the handlers' own validation
//! paths produce: `400 invalid_request_error` for syntax and
//! shape-mismatch failures (client-correctable input), `415` for a
//! missing/wrong `Content-Type`. The serde detail is kept (stripped of
//! the axum "into the target type" wrapper) because it names the
//! offending request field — genuinely useful for debugging — but the
//! envelope is now standard.
#![allow(clippy::module_name_repetitions)]

use axum::{
    Json,
    extract::{FromRequest, Request, rejection::JsonRejection},
    http::StatusCode,
};
use serde::de::DeserializeOwned;

use super::types::ErrorResponse;

/// Request extractor that yields the inner `T` on success and maps every
/// JSON-body failure to an OpenAI-format error response.
///
/// Mirror-usable in a handler pattern: `OpenaiJson(req):
/// OpenaiJson<ChatRequest>` binds `req` to the inner `ChatRequest` just
/// like `Json(req): Json<ChatRequest>` did.
#[derive(Debug)]
pub struct OpenaiJson<T>(pub T);

/// Strip axum's `Failed to deserialize the JSON body into the target
/// type:` wrapper from a `JsonDataError` so clients see only the
/// field-level serde detail (e.g. `messages: invalid type: string
/// "hi", expected a sequence`) rather than internal extractor phrasing.
fn strip_axum_prefix(message: &str) -> String {
    message
        .strip_prefix("Failed to deserialize the JSON body into the target type:")
        .map_or_else(|| message, str::trim_start)
        .to_string()
}

impl<S, T> FromRequest<S> for OpenaiJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = (StatusCode, Json<ErrorResponse>);

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            Err(rejection) => {
                let (status, message) = match rejection {
                    // Valid JSON that does not match the request shape —
                    // a client-correctable mistake. OpenAI reports this
                    // as 400 invalid_request_error, not 422.
                    JsonRejection::JsonDataError(err) => (
                        StatusCode::BAD_REQUEST,
                        format!(
                            "Invalid request body: {}",
                            strip_axum_prefix(&err.to_string())
                        ),
                    ),
                    // Malformed JSON.
                    JsonRejection::JsonSyntaxError(err) => {
                        (StatusCode::BAD_REQUEST, format!("Invalid JSON: {err}"))
                    }
                    // Missing / wrong Content-Type.
                    JsonRejection::MissingJsonContentType(_) => (
                        StatusCode::UNSUPPORTED_MEDIA_TYPE,
                        "Content-Type must be application/json".to_string(),
                    ),
                    // Body could not be read (oversize / closed stream).
                    JsonRejection::BytesRejection(err) => (
                        StatusCode::BAD_REQUEST,
                        format!("Could not read request body: {err}"),
                    ),
                    // Any future rejection kind — never leak a raw
                    // plain-text body.
                    other => (
                        StatusCode::BAD_REQUEST,
                        format!("Invalid request body: {other}"),
                    ),
                };
                Err((
                    status,
                    Json(ErrorResponse::new(&message, "invalid_request_error")),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{Request, header},
        routing::post,
    };
    use serde::{Deserialize, Serialize};
    use tower::ServiceExt;

    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    struct TestBody {
        #[allow(dead_code)]
        messages: Vec<String>,
    }

    #[axum::debug_handler]
    async fn echo_handler(OpenaiJson(body): OpenaiJson<TestBody>) -> Json<TestBody> {
        Json(body)
    }

    fn app() -> Router {
        Router::new().route("/", post(echo_handler))
    }

    #[tokio::test]
    async fn valid_body_passes_through() {
        let resp = app()
            .oneshot(
                Request::post("/")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"messages": ["hi"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(String::from_utf8_lossy(&body), r#"{"messages":["hi"]}"#);
    }

    #[tokio::test]
    async fn wrong_shape_returns_400_openai_error() {
        // RIL ISS-104: a wrong-shaped but valid JSON body was 422 plain
        // text (`Failed to deserialize the JSON body into the target
        // type: ...`) — now 400 with an OpenAI error envelope.
        let resp = app()
            .oneshot(
                Request::post("/")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"messages": "hi"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(
            body.contains("\"error\""),
            "must be an OpenAI error envelope: {body}"
        );
        assert!(
            body.contains("\"invalid_request_error\""),
            "must be invalid_request_error: {body}"
        );
        assert!(
            body.contains("messages"),
            "must keep the offending field name: {body}"
        );
        assert!(
            !body.contains("into the target type"),
            "must not leak the axum extractor wrapper: {body}"
        );
    }

    #[tokio::test]
    async fn malformed_json_returns_400_openai_error() {
        let resp = app()
            .oneshot(
                Request::post("/")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{ not json"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("\"invalid_request_error\""));
    }

    #[tokio::test]
    async fn missing_content_type_returns_415_openai_error() {
        let resp = app()
            .oneshot(
                Request::post("/")
                    .body(Body::from(r#"{"messages":["hi"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(
            body.contains("\"invalid_request_error\""),
            "415 body must still be OpenAI-format: {body}"
        );
        assert!(body.contains("Content-Type"));
    }

    #[tokio::test]
    async fn missing_fields_still_lists_them_cleanly() {
        // A body missing required fields entirely (no `messages` at all).
        let resp = app()
            .oneshot(
                Request::post("/")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("\"error\""));
        assert!(body.contains("messages"));
    }
}
