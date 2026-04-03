use axum::{
    body::Body,
    response::Response,
    Router,
};
use http::{Request, StatusCode};
use tower::ServiceExt;

pub async fn send_request(app: &Router, request: Request<Body>) -> Response {
    app.oneshot(request).await.expect("Failed to send request")
}

pub fn assert_response_status(response: &Response, expected: StatusCode) {
    assert_eq!(response.status(), expected);
}

pub fn get_body_bytes(response: &Response) -> Vec<u8> {
    let body = response.body();
    let bytes = hyper::body::to_bytes(body.clone())
        .now_or_never()
        .unwrap_or_default();
    bytes.to_vec()
}

pub fn get_body_string(response: &Response) -> String {
    String::from_utf8_lossy(&get_body_bytes(response)).to_string()
}
