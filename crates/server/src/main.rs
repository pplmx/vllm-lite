//! `vllm-server` binary entry point: parse CLI args, load config, construct the engine, bind the axum router, and run until SIGINT/SIGTERM.
//!
//! Used by the `vllm-server` package; the library form is `vllm_server` for embedding tests + integration.
mod debug;

use anyhow::{Context, Result};
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
use vllm_core::types::AdaptiveDraftConfig;
use vllm_core::types::EngineMessage;
use vllm_core::types::SchedulerConfig;
use vllm_model::loader::ModelLoader;
use vllm_model::tokenizer::Tokenizer;
use vllm_server::auth::AuthMiddleware;
use vllm_server::openai::batch::BatchManager;
use vllm_server::openai::batch::handler::{
    create_batch, get_batch, get_batch_results, list_batches,
};
use vllm_server::openai::chat::chat_completions;
use vllm_server::openai::completions::completions as openai_completions;
use vllm_server::openai::embeddings::embeddings;
use vllm_server::openai::models::models_handler;
use vllm_server::{ApiState, api, auth, cli, config::AppConfig, health::HealthChecker, logging};

/// Health check endpoint - liveness probe
async fn health_handler(State(state): State<ApiState>) -> Response {
    // invariant: lock is only held for synchronous field access; no panic possible while holding.
    let status = state.health.read().unwrap().check_liveness();
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
    // invariant: lock is only held for synchronous field access; no panic possible while holding.
    let status = state.health.read().unwrap().check_readiness();
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

/// Build the loader, model, optional draft model, and engine from CLI + config.
///
/// Returns the constructed engine and the model loader (the latter is retained
/// because the engine stores a reference to its architecture for routing).
#[allow(clippy::too_many_lines)]
fn build_engine(
    app_config: &AppConfig,
    cli: &cli::CliArgs,
) -> Result<(Engine, ModelLoader, Device)> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    tracing::info!(device = ?device, "Device initialized");

    let model_path = cli.model_path().display().to_string();
    tracing::debug!(model_path = %model_path, "Model path configured");

    let loader = ModelLoader::builder(device.clone())
        .with_model_dir(model_path.clone())
        .with_kv_blocks(app_config.engine.num_kv_blocks)
        .with_kv_quantization(app_config.engine.kv_quantization)
        .with_allow_stub(cli.model.allow_stub)
        .build()
        .context("failed to create model loader")?;

    let model = loader
        .load_model()
        .context("failed to load model weights")?;

    tracing::info!(
        model_path = %model_path,
        device = ?device,
        "Model loaded"
    );

    // Only load draft model if speculative decoding is enabled.
    let draft_model = if app_config.engine.max_draft_tokens > 0 {
        tracing::info!("Loading draft model (speculative decoding enabled)");
        Some(loader.load_model().context("failed to load draft model")?)
    } else {
        tracing::info!("Skipping draft model (speculative decoding disabled)");
        None
    };

    // v18.0: build the engine using with_budget_boxed / with_drafts_boxed when
    // the server config declares a VRAM budget or external draft specs. The
    // legacy new_boxed path is preserved for backward compatibility.
    let draft_specs: Vec<vllm_core::speculative::DraftSpec> = app_config
        .engine
        .draft_specs
        .iter()
        .map(|c| {
            let mut spec =
                vllm_core::speculative::DraftSpec::new(c.id.clone(), c.path.clone(), c.num_layers);
            if c.weight_size_bytes > 0 {
                spec = spec.with_weight_size(c.weight_size_bytes);
            }
            if let Some(arch) = &c.architecture {
                spec = spec.with_arch_hint(arch.clone());
            }
            spec
        })
        .collect();

    let engine = if let Some(budget_bytes) = app_config.engine.vram_budget_bytes {
        let budget = Arc::new(
            vllm_core::speculative::MemoryBudget::new(budget_bytes)
                .context("server config: invalid vram_budget_bytes")?,
        );
        tracing::info!(
            budget_bytes,
            draft_specs = draft_specs.len(),
            "Constructing Engine with VRAM budget (v18.0 path)"
        );
        Engine::with_budget_boxed(
            model,
            draft_model,
            draft_specs,
            budget,
            SchedulerConfig::default(),
            app_config.engine.max_draft_tokens,
            app_config.engine.num_kv_blocks,
        )
    } else if !draft_specs.is_empty() {
        tracing::info!(
            draft_specs = draft_specs.len(),
            "Constructing Engine with draft specs (v18.0 path, no budget)"
        );
        Engine::with_drafts_boxed(
            model,
            draft_model,
            draft_specs,
            SchedulerConfig::default(),
            app_config.engine.max_draft_tokens,
            app_config.engine.num_kv_blocks,
        )
    } else {
        Engine::new_boxed(model, draft_model)
    };

    Ok((engine, loader, device))
}

/// Wire optional speculative-decoding knobs onto a freshly constructed engine.
fn configure_speculative(app_config: &AppConfig, engine: &mut Engine) {
    if app_config.engine.max_draft_tokens > 0 {
        if app_config.engine.enable_adaptive_speculative {
            tracing::info!(
                "Enabling adaptive speculative decoding (max_draft_tokens={})",
                app_config.engine.max_draft_tokens
            );
            engine.enable_adaptive_speculative(AdaptiveDraftConfig {
                min_draft_tokens: 1,
                max_draft_tokens: app_config.engine.max_draft_tokens,
                target_acceptance_rate: 0.5,
                accuracy_window_size: 10,
                adjustment_step: 1,
                cooldown_steps: 5,
                ewma_alpha: 0.1,
                deadband_threshold: 0.05,
            });
        } else {
            tracing::info!(
                "Enabling speculative decoding (max_draft_tokens={})",
                app_config.engine.max_draft_tokens
            );
            engine.enable_speculative();
        }
    }
}

/// Load the tokenizer from `<model_dir>/tokenizer.json`, or fall back to a
/// default-constructed tokenizer. Returns the `Arc<Tokenizer>` ready for use.
fn load_tokenizer(model_dir: &std::path::Path) -> Arc<Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        tracing::warn!("No tokenizer.json found in model directory, using default tokenizer");
        return Arc::new(Tokenizer::new());
    }
    let Some(path_str) = tokenizer_path.to_str() else {
        tracing::error!(
            path = ?tokenizer_path,
            "Tokenizer path is not valid UTF-8; falling back to default tokenizer"
        );
        return Arc::new(Tokenizer::new());
    };
    match Tokenizer::from_file(path_str) {
        Ok(t) => {
            tracing::info!("Tokenizer loaded");
            Arc::new(t)
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to load tokenizer from file, using default");
            Arc::new(Tokenizer::new())
        }
    }
}

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

    let (mut engine, loader, device) =
        build_engine(&app_config, &cli).context("failed to construct inference engine")?;

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

    configure_speculative(&app_config, &mut engine);

    let (msg_tx, msg_rx) = mpsc::unbounded_channel::<EngineMessage>();
    let engine_shutdown_tx = msg_tx.clone();

    std::thread::spawn(move || {
        engine.run(msg_rx);
    });

    let tokenizer = load_tokenizer(cli.model_path());
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
    let metrics_collector = Arc::new(EnhancedMetricsCollector::new());

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
        .route("/health/live", get(health_handler))
        .route("/health/ready", get(ready_handler))
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/metrics", get(metrics_handler))
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
