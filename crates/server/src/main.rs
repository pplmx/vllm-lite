mod api;

use axum::{routing::post, Router};
use std::sync::Arc;
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{EngineMessage, SchedulerConfig};
use vllm_model::qwen3::model::Qwen3Model;
use vllm_model::config::Qwen3Config;
use candle_core::Device;

#[tokio::main]
async fn main() {
    let device = Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu);

    let config = Qwen3Config {
        vocab_size: 151936,
        hidden_size: 3584,
        num_hidden_layers: 28,
        num_attention_heads: 28,
        num_key_value_heads: 8,
        intermediate_size: 18944,
        sliding_window: Some(32768),
        rope_theta: 10000.0,
        max_position_embeddings: 32768,
        rms_norm_eps: 1e-6,
    };

    let model = Qwen3Model::new(config, device).unwrap();
    let model = Arc::new(model);

    let sched_config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
    };

    let mut engine = Engine::with_config_arc(model.clone(), model, sched_config, 4, 1024);

    let (msg_tx, msg_rx) = mpsc::unbounded_channel::<EngineMessage>();

    std::thread::spawn(move || {
        engine.run(msg_rx);
    });

    let app = Router::new()
        .route("/v1/completions", post(api::completions))
        .with_state(msg_tx);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("vllm-lite listening on http://0.0.0.0:8000");
    axum::serve(listener, app).await.unwrap();
}