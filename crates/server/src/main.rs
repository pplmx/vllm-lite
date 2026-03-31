mod api;

use axum::{routing::get, routing::post, Router};
use std::sync::Arc;
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::EngineMessage;
use vllm_model::config::Qwen3Config;
use vllm_model::qwen3::model::Qwen3Model;
use vllm_model::tokenizer::Tokenizer;
use candle_core::Device;

#[derive(Clone)]
pub struct ApiState {
    pub engine_tx: api::EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
}

#[tokio::main]
async fn main() {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    let config = Qwen3Config::default();
    println!("Using config: {:?}", config);

    let model = Qwen3Model::new(config.clone(), device.clone())
        .expect("Failed to create model");
    let draft_model = Qwen3Model::new(config, device)
        .expect("Failed to create draft model");
    
    let mut engine = Engine::new(
        model,
        draft_model,
    );
    engine.enable_speculative();

    let (msg_tx, msg_rx) = mpsc::unbounded_channel::<EngineMessage>();

    std::thread::spawn(move || {
        engine.run(msg_rx);
    });

    let tokenizer = Arc::new(Tokenizer::new());
    let state = ApiState {
        engine_tx: msg_tx.clone(),
        tokenizer,
    };

    let app = Router::new()
        .route("/v1/completions", post(api::completions))
        .route("/v1/stats", get(api::get_stats))
        .route("/metrics", get(api::get_prometheus))
        .route("/health", get(api::health))
        .route("/ready", get(api::ready))
        .with_state(state);

    let app = app.route("/shutdown", get(api::shutdown).with_state(msg_tx));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("vllm-lite (real weights) listening on http://0.0.0.0:8000");
    axum::serve(listener, app).await.unwrap();
}