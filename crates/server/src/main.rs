mod api;
mod config;
mod logging;
pub mod openai;

use axum::{routing::get, routing::post, Router};
use candle_core::Device;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::EngineMessage;
use vllm_model::loader::ModelLoader;
use vllm_model::tokenizer::Tokenizer;
use crate::openai::batch::manager::BatchManager;

#[derive(Clone)]
pub struct ApiState {
    pub engine_tx: api::EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    pub batch_manager: Arc<BatchManager>,
}

fn load_config() -> config::AppConfig {
    let config_path = std::env::args()
        .skip(1)
        .find(|arg| arg.starts_with("--config="))
        .map(|arg| PathBuf::from(arg.trim_start_matches("--config=")));

    let config = config::AppConfig::load(config_path);

    if let Err(errors) = config.validate() {
        for err in &errors {
            tracing::error!(error = %err, "Config validation failed");
        }
        eprintln!("Config validation failed:");
        for err in errors {
            eprintln!("  - {}", err);
        }
        std::process::exit(1);
    }

    config
}

fn get_model_path() -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "/models/Qwen2.5-0.5B-Instruct".to_string())
}

#[tokio::main]
async fn main() {
    let app_config = load_config();

    let log_dir = app_config.server.log_dir.as_ref().map(PathBuf::from);
    logging::init_logging(log_dir, &app_config.server.log_level);

    tracing::info!(config = ?app_config, "Starting vllm-lite");

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    tracing::info!(device = ?device, "Using device");

    let model_path = get_model_path();
    tracing::info!(model_path = %model_path, "Loading model from");

    let loader = ModelLoader::new(device.clone());
    let model = loader
        .load_model(&model_path)
        .expect("Failed to load model");

    // For speculative decoding, we need a proper draft model
    // For now, disable speculative mode by using the same model as draft
    // (this doubles memory but ensures compatibility)
    let draft_model = loader
        .load_model(&model_path)
        .expect("Failed to load draft model");

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
    let state = ApiState {
        engine_tx: msg_tx.clone(),
        tokenizer,
        batch_manager,
    };

    use openai::chat::chat_completions;
    use openai::completions::completions as openai_completions;
    use openai::embeddings::embeddings;
    use openai::batch::handler::{
        create_batch, get_batch, get_batch_results, list_batches,
    };

    let app = Router::new()
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
        .with_state(state);

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
