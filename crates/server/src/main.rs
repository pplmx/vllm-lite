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
use vllm_server::security::correlation::correlation_id_middleware;
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

    // SEC-01 (technical due diligence): refuse to be quiet about an
    // unauthenticated bind to a non-loopback address. The previous
    // behavior was to start silently with no auth, which let any
    // network-reachable client hit `/v1/chat/completions`,
    // `/debug/*`, and `/shutdown`.
    //
    // We deliberately *warn* rather than *refuse*: failing closed
    // would break local dev where `127.0.0.1` is the implicit bind
    // but operators may also legitimately smoke-test on a LAN. The
    // warning is structured so log-aggregation rules can alert on
    // it. Pass `--insecure-allow-public-no-auth` (or
    // `VLLM_INSECURE_ALLOW_PUBLIC_NO_AUTH=true`) to silence it for
    // intentional internal deployments.
    {
        let resolved_keys = app_config.auth.resolve_api_keys();
        let auth_configured = !resolved_keys.is_empty();
        let bind_exposes_public = !vllm_server::config::is_loopback_address(&app_config.server.host);
        if bind_exposes_public && !auth_configured && !cli.security.insecure_allow_public_no_auth {
            tracing::warn!(
                host = %app_config.server.host,
                "SEC-01: server is bound to a non-loopback address with no API \
                 keys configured. Anyone reachable on the network can invoke \
                 inference, read /debug/*, and trigger /shutdown. Set \
                 --api-key / VLLM_API_KEY or pass --insecure-allow-public-no-auth \
                 for intentional internal deployments."
            );
        }
    }

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

    // OBS-01 (technical due diligence): wire the HTTP `/metrics` exporter to
    // the *engine's* metrics collector, not a freshly constructed duplicate.
    // Previously we created a separate `EnhancedMetricsCollector` here which
    // meant the Prometheus exporter read zero values while the engine kept
    // its real counters internally — `/metrics` showed an idle system while
    // `/health/details` showed the live counters. Cloning the engine's Arc
    // (before the engine is moved into the worker thread below) gives both
    // surfaces the same source of truth.
    let metrics_collector = engine.scheduler.metrics.clone();

    // Production-readiness recommendation (graceful shutdown): keep
    // the engine worker's JoinHandle so we can wait for it to exit
    // AFTER sending `EngineMessage::Shutdown`. Without this, the
    // process exits immediately after the HTTP server returns and
    // the worker thread is dropped mid-step, which (a) loses any
    // KV blocks it was holding and (b) can panic on `unwrap()` of
    // a poisoned lock if we abort mid-write. The thread is
    // detached-by-Arc-join: we call `JoinHandle::join()` with a
    // timeout below (see `engine_thread.join()`).
    let engine_thread = std::thread::Builder::new()
        .name("vllm-engine".to_string())
        .spawn(move || {
            engine.run(msg_rx);
        })
        .with_context(|| "failed to spawn vllm-engine worker thread")?;

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

    // Initialize health checker
    let health_checker = Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true)));

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
        .route("/health/live", get(vllm_server::health_handlers::health_handler))
        .route("/health/ready", get(vllm_server::health_handlers::ready_handler))
        .route("/health", get(vllm_server::health_handlers::health_handler))
        .route("/ready", get(vllm_server::health_handlers::ready_handler))
        .route("/metrics", get(vllm_server::health_handlers::metrics_handler))
        .route("/health/details", get(api::health_details))
        // Debug endpoints
        .route("/debug/metrics", get(debug::metrics_snapshot))
        .route("/debug/kv-cache", get(debug::kv_cache_dump))
        .route("/debug/trace", get(debug::trace_status))
        // Shutdown
        .route("/shutdown", get(api::shutdown))
        .with_state(state);

    // Mount correlation_id_middleware as the OUTERMOST layer: it must
    // see every request (auth-gated or not) and stamp an `X-Request-ID`
    // header before auth/size-limit/audit run, so even rejected
    // requests get a stable ID in the response and logs.
    //
    // Production-readiness recommendation 6: thread a single
    // correlation ID through HTTP → scheduler → token stream so
    // operators can trace a request across the whole pipeline.
    app = app.layer(axum::middleware::from_fn(correlation_id_middleware));

    // Production-readiness recommendation (input boundary protection):
    // cap the inbound JSON body at `DEFAULT_BODY_LIMIT_BYTES` (1 MiB)
    // before auth/handlers can be reached. This blocks trivial JSON
    // allocation attacks where a multi-MiB prompt exhausts memory
    // before any application-level validation runs. Larger limits can
    // be configured via the `with_body_size_limit` helper if a
    // deployment legitimately needs them; the default keeps a known
    // upper bound on memory pressure per request.
    //
    // Body limit lives BELOW correlation_id so a 413 response still
    // carries an `X-Request-ID` header (operators need it to trace
    // the rejected request), and ABOVE auth so unauthenticated
    // clients can't waste server memory on oversized bodies either.
    app = vllm_server::security::size_limit::with_default_body_limit(app);

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

    tracing::info!("HTTP server stopped; draining engine");

    // Production-readiness recommendation (graceful shutdown):
    // 1. Tell the engine to exit its run loop. The `try_send`
    //    keeps working because the receiver is still alive (the
    //    worker thread holds it).
    // 2. Wait for the worker to acknowledge shutdown by joining
    //    the thread. Cap at 10s so a stuck engine can't pin the
    //    process forever; operators can `SIGKILL` if that's
    //    needed.
    let _ = engine_shutdown_tx.send(EngineMessage::Shutdown);
    const ENGINE_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
    let join_start = std::time::Instant::now();
    // `JoinHandle::join()` is blocking; we wrap it with a
    // background timeout thread so the main thread can still log
    // progress and exit the process cleanly if the join takes
    // too long. We don't abort the engine — that's the
    // operator's call — we just stop blocking the process.
    let join_handle = std::thread::spawn(move || {
        let _ = engine_thread.join();
        join_start.elapsed()
    });
    let deadline = std::time::Instant::now() + ENGINE_DRAIN_TIMEOUT;
    while !join_handle.is_finished() {
        if std::time::Instant::now() >= deadline {
            tracing::warn!(
                timeout_secs = ENGINE_DRAIN_TIMEOUT.as_secs(),
                "engine thread did not exit within drain timeout; exiting anyway"
            );
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    if join_handle.is_finished() {
        let elapsed = join_handle.join().unwrap_or(ENGINE_DRAIN_TIMEOUT);
        tracing::info!(
            drain_ms = %u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            "engine thread joined cleanly"
        );
    }
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
