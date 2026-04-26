use axum::{
    extract::Request,
    http::HeaderValue,
    middleware::Next,
    response::Response,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

pub const REQUEST_ID_HEADER: &str = "X-Request-ID";

#[derive(Debug, Clone)]
pub struct CorrelationId(pub String);

#[derive(Clone)]
pub struct CorrelationIdMiddleware {
    id_generator: Arc<RwLock<u64>>,
}

impl CorrelationIdMiddleware {
    pub fn new() -> Self {
        Self {
            id_generator: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn generate_id(&self) -> String {
        let mut counter = self.id_generator.write().await;
        *counter += 1;
        format!("{:x}-{:x}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64, *counter)
    }

    pub fn extract_id(headers: &axum::http::HeaderMap) -> Option<String> {
        headers
            .get(REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    }
}

impl Default for CorrelationIdMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

pub async fn correlation_id_middleware(
    request: Request,
    next: Next,
) -> Response {
    let middleware = CorrelationIdMiddleware::new();

    let request_id = CorrelationIdMiddleware::extract_id(request.headers())
        .unwrap_or_else(|| {
            tokio::runtime::Handle::current()
                .block_on(middleware.generate_id())
        });

    info!(
        request_id = %request_id,
        method = %request.method(),
        uri = %request.uri(),
        "Request started"
    );

    let mut request = request;
    request.headers_mut().insert(
        REQUEST_ID_HEADER,
        HeaderValue::from_str(&request_id).unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );

    let response = next.run(request).await;

    info!(request_id = %request_id, "Request completed");

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_id() {
        let middleware = CorrelationIdMiddleware::new();
        let id1 = middleware.generate_id().await;
        let id2 = middleware.generate_id().await;

        assert!(!id1.is_empty());
        assert!(!id2.is_empty());
        assert_ne!(id1, id2);
    }

    #[tokio::test]
    async fn test_extract_id() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(REQUEST_ID_HEADER, "test-id-123".parse().unwrap());

        let id = CorrelationIdMiddleware::extract_id(&headers);
        assert_eq!(id, Some("test-id-123".to_string()));
    }

    #[tokio::test]
    async fn test_extract_id_missing() {
        let headers = axum::http::HeaderMap::new();
        let id = CorrelationIdMiddleware::extract_id(&headers);
        assert_eq!(id, None);
    }
}
