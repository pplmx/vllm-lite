#![allow(dead_code)]

mod api;
mod auth;
mod cli;
mod config;
mod logging;
pub mod openai;

use crate::auth::AuthMiddleware;
use crate::openai::batch::manager::BatchManager;
use axum::{Router, routing::get, routing::post};
use candle_core::Device;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::EngineMessage;
use vllm_model::loader::ModelLoader;
use vllm_model::tokenizer::Tokenizer;

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

    let loader = ModelLoader::new(device.clone());
    let model = loader
        .load_model(
            &model_path,
            app_config.engine.num_kv_blocks,
            app_config.engine.kv_quantization,
        )
        .unwrap_or_else(|e| panic!("Failed to load model: {}", e));

    // For speculative decoding, we need a proper draft model
    // Note: Loading twice doubles memory - consider optimizing for production
    let draft_model = loader
        .load_model(
            &model_path,
            app_config.engine.num_kv_blocks,
            app_config.engine.kv_quantization,
        )
        .unwrap_or_else(|e| panic!("Failed to load draft model: {}", e));

    let mut engine = Engine::new(model, draft_model);
    // Don't enable speculative mode - it causes hangs with mismatched draft model
    // engine.enable_speculative();

    let (msg_tx, msg_rx) = mpsc::unbounded_channel::<EngineMessage>();
    let engine_shutdown_tx = msg_tx.clone();

    std::thread::spawn(move || {
        engine.run(msg_rx);
    });

    let tokenizer = Arc::new(Tokenizer::new());
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

    let state = ApiState {
        engine_tx: msg_tx.clone(),
        tokenizer,
        batch_manager,
        auth: auth_middleware.clone(),
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
        .route("/v1/batches/:id", get(get_batch))
        .route("/v1/batches/:id/results", get(get_batch_results))
        // 运维 (保留 api.rs)
        .route("/metrics", get(api::get_prometheus))
        .route("/health", get(api::health))
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
