//! `vllm-server` binary entry point: parse CLI args, load config, construct the engine, bind the axum router, and run until SIGINT/SIGTERM.
//!
//! Used by the `vllm-server` package; the library form is `vllm_server` for embedding tests + integration.
//!
//! Bootstrap helpers (engine construction, speculative configuration,
//! tokenizer loading, HTTP health/readiness/metrics handlers) live in
//! the `bootstrap` submodule to keep this entry file focused on wiring.

mod bootstrap;
mod debug;

use anyhow::{Context, Result};
use axum::{
    Router,
    routing::{get, post},
};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::mpsc;
use vllm_core::types::EngineMessage;
use vllm_server::auth::AuthMiddleware;
use vllm_server::openai::batch::BatchManager;
use vllm_server::openai::batch::handler::{
    create_batch, get_batch, get_batch_results, list_batches,
};
use vllm_server::openai::chat::chat_completions;
use vllm_server::openai::completions::completions as openai_completions;
use vllm_server::openai::embeddings::embeddings;
use vllm_server::openai::models::models_handler;
use vllm_server::{ApiState, api, auth, cli, health::HealthChecker, logging};

#[tokio::main]
#[allow(clippy::too_many_lines)] // server bootstrap: linear startup sequence with no natural decomposition
async fn main() -> Result<()> {
    let cli = cli::CliArgs::parse();
    let app_config = cli.to_app_config();

    if let Err(errors) = app_config.validate() {
        for err in &errors.0 {
            tracing::error!(error = %err, "Config validation failed");
        }
        // Distinct exit code (78) for config errors — distinguishable from
        // generic startup failures in supervisor restart policies.
        std::process::exit(78);
    }

    let log_dir = app_config.server.log_dir.as_ref().map(PathBuf::from);
    logging::init_logging(log_dir, &app_config.server.log_level);

    tracing::info!("Starting vllm-lite");

    let (mut engine, loader, device) = bootstrap::engine::build_engine(&app_config, &cli)
        .context("failed to construct inference engine")?;

    // v18.0: wire a real DraftLoader so the resolver can actually load draft
    // weights from disk. The engine installs a NoopLoader by default in the
    // v18.0 constructors, which would silently fall back to self-spec for any
    // declared spec. Replace it with ServerDraftLoader when a resolver is
    // installed (i.e. the v18.0 path was taken).
    if engine.draft_resolver.is_some() {
        let draft_loader = vllm_server::draft_loader::ServerDraftLoader::new(
            device,
            &app_config.engine.draft_specs,
        )
        .with_kv_blocks(app_config.engine.num_kv_blocks)
        .with_kv_quantization(app_config.engine.kv_quantization)
        .with_allow_stub(cli.model.allow_stub);
        tracing::info!(
            registered_drafts = draft_loader.len(),
            "Installed ServerDraftLoader (replaces NoopLoader)"
        );
        if !engine.set_draft_loader(Arc::new(draft_loader)) {
            tracing::warn!(
                "Engine.set_draft_loader was a no-op: resolver missing. \
                 Drafts will fall back to self-spec."
            );
        }
    }

    tracing::debug!(
        draft_enabled = app_config.engine.max_draft_tokens > 0,
        kv_blocks = app_config.engine.num_kv_blocks,
        has_resolver = engine.draft_resolver.is_some(),
        "Engine configured"
    );

    bootstrap::engine::configure_speculative(&app_config, &mut engine);

    // REL-01 (technical due diligence): use a bounded mailbox so a
    // flood of HTTP requests fails fast with `503 engine_overloaded`
    // instead of building an unbounded backlog. The default capacity
    // (256) absorbs short bursts while bounding memory; tunable via
    // `app_config.engine.engine_mailbox_capacity`.
    let mailbox_capacity = app_config.engine.engine_mailbox_capacity.max(1);
    let (msg_tx, msg_rx) = mpsc::channel::<EngineMessage>(mailbox_capacity);
    let engine_shutdown_tx = msg_tx.clone();

    std::thread::spawn(move || {
        engine.run(msg_rx);
    });

    let tokenizer = bootstrap::tokenizer::load_tokenizer(cli.model_path());
    let batch_manager = Arc::new(BatchManager::new());

    let auth_middleware = if app_config.auth.api_keys.is_empty() {
        None
    } else {
        Some(Arc::new(AuthMiddleware::new(
            app_config.auth.api_keys.clone(),
            app_config.auth.rate_limit_requests,
            app_config.auth.rate_limit_window_secs,
        )))
    };

    // Initialize health checker and metrics collector
    let health_checker = Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true)));
    let metrics_collector = Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new());

    let architecture = loader.architecture();

    let state = ApiState {
        engine_tx: msg_tx.clone(),
        tokenizer,
        architecture,
        batch_manager,
        auth: auth_middleware.clone(),
        health: health_checker,
        metrics: metrics_collector,
    };

    let mut app: axum::Router<()> = Router::new()
        // OpenAI API
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(openai_completions))
        .route("/v1/embeddings", post(embeddings))
        // Batch API
        .route("/v1/batches", post(create_batch))
        .route("/v1/batches", get(list_batches))
        .route("/v1/batches/{id}", get(get_batch))
        .route("/v1/batches/{id}/results", get(get_batch_results))
        // Health, readiness, and metrics endpoints (K8s-compatible paths)
        .route("/health/live", get(bootstrap::handlers::health_handler))
        .route("/health/ready", get(bootstrap::handlers::ready_handler))
        .route("/health", get(bootstrap::handlers::health_handler))
        .route("/ready", get(bootstrap::handlers::ready_handler))
        .route("/metrics", get(bootstrap::handlers::metrics_handler))
        .route("/health/details", get(api::health_details))
        // Debug endpoints
        .route("/debug/metrics", get(debug::metrics_snapshot))
        .route("/debug/kv-cache", get(debug::kv_cache_dump))
        .route("/debug/trace", get(debug::trace_status))
        // Shutdown
        .route("/shutdown", get(api::shutdown))
        .with_state(state);

    if let Some(auth) = auth_middleware {
        app = app.layer(axum::middleware::from_fn_with_state(
            auth,
            auth::auth_middleware,
        ));
    }

    let addr = format!("{}:{}", app_config.server.host, app_config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind server socket to {addr}"))?;
    tracing::info!(address = %addr, "Server listening");

    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .with_context(|| format!("server crashed while serving on {addr}"))?;

    tracing::info!("Shutting down gracefully");
    let _ = engine_shutdown_tx.send(EngineMessage::Shutdown);
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        // invariant: signal handler installation only fails if the OS is in an unrecoverable
        // state; not recoverable from this process anyway.
        signal::ctrl_c()
            .await
            // invariant: pre-conditions make this infallible at this call site.
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        // invariant: signal handler installation only fails if the OS is in an unrecoverable
        // state; not recoverable from this process anyway.
        signal::unix::signal(signal::unix::SignalKind::terminate())
            // invariant: pre-conditions make this infallible at this call site.
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }

    tracing::info!("Shutdown signal received");
}
