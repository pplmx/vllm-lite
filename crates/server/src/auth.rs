use axum::{
    extract::Request,
    http::{header::AUTHORIZATION, HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct AuthMiddleware {
    api_keys: Arc<Vec<String>>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
}

pub struct RateLimiter {
    requests: HashMap<String, Vec<Instant>>,
    max_requests: usize,
    window_secs: u64,
}

impl RateLimiter {
    fn new(max_requests: usize, window_secs: u64) -> Self {
        Self {
            requests: HashMap::new(),
            max_requests,
            window_secs,
        }
    }
    
    async fn check_rate_limit(&mut self, key: &str) -> bool {
        let now = Instant::now();
        let window = Duration::from_secs(self.window_secs);
        
        let times = self.requests.entry(key.to_string()).or_default();
        times.retain(|t| now.duration_since(*t) < window);
        
        if times.len() >= self.max_requests {
            return false;
        }
        
        times.push(now);
        true
    }
}

impl AuthMiddleware {
    pub fn new(api_keys: Vec<String>, max_requests: usize, window_secs: u64) -> Self {
        Self {
            api_keys: Arc::new(api_keys),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new(max_requests, window_secs))),
        }
    }
    
    pub async fn verify(&self, headers: &HeaderMap) -> Result<String, StatusCode> {
        let auth_header = headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok());
        
        let api_key = auth_header
            .and_then(|h| h.strip_prefix("Bearer "))
            .ok_or(StatusCode::UNAUTHORIZED)?;
        
        if !self.api_keys.is_empty() && !self.api_keys.contains(&api_key.to_string()) {
            return Err(StatusCode::UNAUTHORIZED);
        }
        
        let mut limiter = self.rate_limiter.write().await;
        if !limiter.check_rate_limit(api_key).await {
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }
        
        Ok(api_key.to_string())
    }
}

pub async fn auth_middleware(
    auth: axum::extract::State<Arc<AuthMiddleware>>,
    request: Request,
    next: Next,
) -> Response {
    match auth.verify(request.headers()).await {
        Ok(_) => next.run(request).await,
        Err(status) => Response::builder().status(status).body("".into()).unwrap(),
    }
}