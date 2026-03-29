mod api;

use axum::{routing::post, Router};
use std::env;
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{EngineMessage, SchedulerConfig};
use vllm_model::loader::ModelLoader;
use vllm_model::qwen3::model::Qwen3Model;
use candle_core::Device;

#[tokio::main]
async fn main() {
    let model_path = env::var("MODEL_PATH")
        .unwrap_or_else(|_| "./models/qwen3-7b".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu);
    let loader = ModelLoader::new(device.clone());
    
    println!("Loading model from: {}", model_path);
    let (config, weights) = loader.load(&model_path).expect("Failed to load model");
    println!("Loaded config: {:?}", config);
    println!("Loaded {} weights", weights.len());
    
    let model = Qwen3Model::new(config.clone(), device.clone()).expect("Failed to create model");
    let draft_model = Qwen3Model::new(config, device).expect("Failed to create draft model");
    
    let sched_config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
    };
    let mut engine = Engine::with_config(
        model,
        draft_model,
        sched_config,
        4,  // max_draft_tokens
        1024,
    );

    let (msg_tx, msg_rx) = mpsc::unbounded_channel::<EngineMessage>();

    std::thread::spawn(move || {
        engine.run(msg_rx);
    });

    let app = Router::new()
        .route("/v1/completions", post(api::completions))
        .with_state(msg_tx);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("vllm-lite (real weights) listening on http://0.0.0.0:8000");
    axum::serve(listener, app).await.unwrap();
}