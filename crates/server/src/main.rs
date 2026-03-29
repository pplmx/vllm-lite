mod api;

use axum::{routing::post, Router};
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{EngineMessage, SchedulerConfig};
use vllm_model::fake::FakeModel;

#[tokio::main]
async fn main() {
    let model = FakeModel::new(32000);
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
    };
    let mut engine = Engine::with_config(model, config);

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