mod api;

use axum::{routing::post, Router};
use std::sync::{Arc, Mutex};
use vllm_core::engine::Engine;
use vllm_model::fake::FakeModel;

#[tokio::main]
async fn main() {
    let model = FakeModel::new(32000);
    let engine = Engine::new(model);
    let state = Arc::new(Mutex::new(engine));

    let app = Router::new()
        .route("/v1/completions", post(api::completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("vllm-lite listening on http://0.0.0.0:8000");
    axum::serve(listener, app).await.unwrap();
}
