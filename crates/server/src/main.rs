#![allow(dead_code)]

mod api;
mod auth;
mod cli;
mod config;
mod health;
mod logging;
pub mod openai;

use crate::auth::AuthMiddleware;
use crate::openai::batch::manager::BatchManager;
use axum::{
    Router, extract::State, http::StatusCode, response::Response, routing::get, routing::post,
};
use candle_core::Device;
use clap::Parser;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::metrics::{EnhancedMetricsCollector, PrometheusExporter};
use vllm_core::types::EngineMessage;
use vllm_model::loader::ModelLoader;
use vllm_model::tokenizer::Tokenizer;
use vllm_server::health::HealthChecker;

/// Shared state for all API handlers
#[derive(Clone)]
pub struct ApiState {
    /// Channel to send messages to the inference engine
    pub engine_tx: api::EngineHandle,
    /// Tokenizer for encoding/decoding text
    pub tokenizer: Arc<Tokenizer>,
    /// Batch manager for handling batch API requests
    pub batch_manager: Arc<BatchManager>,
    /// Authentication middleware (None if disabled)
    pub auth: Option<Arc<AuthMiddleware>>,
    /// Health checker for liveness/readiness probes
    pub health: Arc<std::sync::RwLock<HealthChecker>>,
    /// Enhanced metrics collector
    pub metrics: Arc<EnhancedMetricsCollector>,
}

/// Health check endpoint - liveness probe
async fn health_handler(State(state): State<ApiState>) -> Response {
    let health = state.health.read().unwrap();
    let status = health.check_liveness();
    let http_status = StatusCode::from_u16(status.http_status()).unwrap_or(StatusCode::OK);

    let body = json!({ "status": status.as_str() });
    Response::builder()
        .status(http_status)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&body).unwrap_or_default().into())
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}

/// Readiness check endpoint
async fn ready_handler(State(state): State<ApiState>) -> Response {
    let health = state.health.read().unwrap();
    let status = health.check_readiness();
    let http_status =
        StatusCode::from_u16(status.http_status()).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);

    let body = json!({ "status": status.as_str() });
    Response::builder()
        .status(http_status)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&body).unwrap_or_default().into())
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}

/// Prometheus metrics endpoint
async fn metrics_handler(State(state): State<ApiState>) -> Response {
    let exporter = PrometheusExporter::new(state.metrics.clone(), 9090);
    let output = exporter.export_to_string().await;

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/plain; charset=utf-8")
        .body(output.into())
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}

#[tokio::main]
async fn main() {
    let cli = cli::CliArgs::parse();
    let app_config = cli.to_app_config();

    if let Err(errors) = app_config.validate() {
        for err in &errors {
            tracing::error!(error = %err, "Config validation failed");
        }
        eprintln!("Config validation failed:");
        for err in errors {
            eprintln!("  - {}", err);
        }
        std::process::exit(1);
    }

    let log_dir = app_config.server.log_dir.as_ref().map(PathBuf::from);
    logging::init_logging(log_dir, &app_config.server.log_level);

    tracing::info!(config = ?app_config, "Starting vllm-lite");

    let device = Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu);
    tracing::info!(device = ?device, "Using device");

    let model_path = cli.model_path().display().to_string();
    tracing::info!(model_path = %model_path, "Loading model from");

    let tensor_parallel_size = app_config.engine.tensor_parallel_size;
    tracing::info!(
        tensor_parallel_size = tensor_parallel_size,
        "Tensor parallel size"
    );

    let loader = ModelLoader::builder(device.clone())
        .with_model_dir(model_path.clone())
        .with_kv_blocks(app_config.engine.num_kv_blocks)
        .with_kv_quantization(app_config.engine.kv_quantization)
        .build()
        .unwrap_or_else(|e| panic!("Failed to create loader: {}", e));

    let model = loader
        .load_model()
        .unwrap_or_else(|e| panic!("Failed to load model: {}", e));

    // Only load draft model if speculative decoding is enabled
    let draft_model = if app_config.engine.max_draft_tokens > 0 {
        tracing::info!("Loading draft model (speculative decoding enabled)");
        Some(
            loader
                .load_model()
                .unwrap_or_else(|e| panic!("Failed to load draft model: {}", e)),
        )
    } else {
        tracing::info!("Skipping draft model (speculative decoding disabled)");
        None
    };

    let mut engine = Engine::new(model, draft_model);
    // Don't enable speculative mode - it causes hangs with mismatched draft model
    // engine.enable_speculative();

    let (msg_tx, msg_rx) = mpsc::unbounded_channel::<EngineMessage>();
    let engine_shutdown_tx = msg_tx.clone();

    std::thread::spawn(move || {
        engine.run(msg_rx);
    });

    let tokenizer_path = PathBuf::from(&model_path).join("tokenizer.json");
    let tokenizer: Arc<Tokenizer> = if tokenizer_path.exists() {
        match Tokenizer::from_file(tokenizer_path.to_str().unwrap()) {
            Ok(t) => {
                tracing::info!("Loaded tokenizer from {:?}", tokenizer_path);
                // Test encoding
                let test_tokens = t.encode("hi");
                let test_decode = t.decode(&test_tokens);
                tracing::info!(
                    "Tokenizer test: 'hi' -> {:?}, decode -> '{}'",
                    test_tokens,
                    test_decode
                );
                Arc::new(t)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to load tokenizer from file, using default");
                Arc::new(Tokenizer::new())
            }
        }
    } else {
        tracing::warn!("No tokenizer.json found in model directory, using default tokenizer");
        Arc::new(Tokenizer::new())
    };
    let batch_manager = Arc::new(BatchManager::new());

    let auth_middleware = if !app_config.auth.api_keys.is_empty() {
        Some(Arc::new(AuthMiddleware::new(
            app_config.auth.api_keys.clone(),
            app_config.auth.rate_limit_requests,
            app_config.auth.rate_limit_window_secs,
        )))
    } else {
        None
    };

    // Initialize health checker and metrics collector
    let health_checker = Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true)));
    let metrics_collector = Arc::new(EnhancedMetricsCollector::new());

    let state = ApiState {
        engine_tx: msg_tx.clone(),
        tokenizer,
        batch_manager,
        auth: auth_middleware.clone(),
        health: health_checker,
        metrics: metrics_collector,
    };

    use openai::batch::handler::{create_batch, get_batch, get_batch_results, list_batches};
    use openai::chat::chat_completions;
    use openai::completions::completions as openai_completions;
    use openai::embeddings::embeddings;

    let mut app = Router::new()
        // OpenAI API
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(openai_completions))
        .route("/v1/embeddings", post(embeddings))
        // Batch API
        .route("/v1/batches", post(create_batch))
        .route("/v1/batches", get(list_batches))
        .route("/v1/batches/{id}", get(get_batch))
        .route("/v1/batches/{id}/results", get(get_batch_results))
        // Health, readiness, and metrics endpoints
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/metrics", get(metrics_handler))
        .route("/health/details", get(api::health_details))
        .with_state(state);

    if let Some(auth) = auth_middleware {
        app = app.layer(axum::middleware::from_fn_with_state(
            auth,
            auth::auth_middleware,
        ));
    }

    let app = app.route("/shutdown", get(api::shutdown).with_state(msg_tx));

    let addr = format!("{}:{}", app_config.server.host, app_config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    tracing::info!(address = %addr, "Server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    tracing::info!("Shutting down gracefully");
    let _ = engine_shutdown_tx.send(EngineMessage::Shutdown);
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received");
}
