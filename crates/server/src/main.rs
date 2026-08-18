//! `vllm-server` binary entry point: parse CLI args, load config, construct the engine, bind the axum router, and run until SIGINT/SIGTERM.
//!
//! Used by the `vllm-server` package; the library form is `vllm_server` for embedding tests + integration.
//!
//! Bootstrap helpers (engine construction, speculative configuration,
//! tokenizer loading, HTTP health/readiness/metrics handlers) live in
//! the `bootstrap` submodule to keep this entry file focused on wiring.

mod bootstrap;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::mpsc;
use vllm_core::types::EngineMessage;
use vllm_server::openai::batch::BatchManager;
use vllm_server::{ApiState, cli, health::HealthChecker, logging};

#[tokio::main]
#[allow(clippy::too_many_lines)] // server bootstrap: linear startup sequence with no natural decomposition
async fn main() -> Result<()> {
    const ENGINE_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

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

    // P43 T5: when the `opentelemetry` feature is enabled *and* OTLP is
    // enabled in config, initialise tracing with the OTLP bridge instead of
    // the default `logging::init_logging`. The `OtlpGuard` returned here
    // flushes pending spans on drop (graceful shutdown).
    //
    // When OTLP is disabled (or the feature is off), fall through to the
    // existing `logging::init_logging` path — console + optional JSON file.
    #[cfg(feature = "opentelemetry")]
    let _otlp_guard: Option<vllm_core::tracing_init::OtlpGuard> = if app_config
        .observability
        .otlp
        .enabled
    {
        use tracing_subscriber::EnvFilter;
        let env_filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(&app_config.server.log_level));
        match vllm_core::tracing_init::init_tracing_with_otlp(
            env_filter,
            &app_config.observability.otlp,
        ) {
            Ok(guard) => Some(guard),
            Err(e) => {
                // Subscriber may already be initialised (e.g. a prior call
                // raced). Warn and continue without the OTLP span bridge;
                // the metrics exporter still runs independently.
                logging::init_logging(log_dir, &app_config.server.log_level);
                tracing::warn!(error = %e, "OTLP tracing init failed; continuing without OTLP span bridge");
                None
            }
        }
    } else {
        logging::init_logging(log_dir, &app_config.server.log_level);
        None
    };

    #[cfg(not(feature = "opentelemetry"))]
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
        let bind_exposes_public =
            !vllm_server::config::is_loopback_address(&app_config.server.host);
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

    // Phase 41 OPS-32a second-half: when multi-node is enabled, spawn
    // the gRPC server that answers `TransferKVBlock` calls with real
    // K/V bytes from the engine's wired `PagedKvCache`. The bootstrap
    // is a no-op on single-node builds (the function returns Ok("")
    // and the gRPC submodule compiles to nothing without the
    // `multi-node` Cargo feature).
    if app_config.server.multi_node.enabled {
        let node_id =
            bootstrap::grpc::spawn_multi_node_grpc_server(&engine, &app_config.server.multi_node)
                .await
                .context("failed to spawn multi-node gRPC server")?;
        tracing::info!(
            node_id,
            bind_addr = %app_config.server.multi_node.bind_addr,
            "Multi-node KV block transfer enabled"
        );
    }

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

    // P43 T5: spawn the OTLP metrics background task if enabled. The
    // `OtlpGuard` from `init_tracing_with_otlp` (above) handles span flushing
    // on shutdown; this handle aborts the metrics polling task. Both are
    // dropped at the end of `main` via the `#[cfg]` guards below.
    #[cfg(feature = "opentelemetry")]
    let _otlp_handle: Option<bootstrap::observability::OtlpHandle> =
        bootstrap::observability::spawn_otlp_exporter(
            metrics_collector.clone(),
            app_config.observability.otlp.clone(),
        )
        .map_err(|e| {
            tracing::warn!(error = %e, "OTLP exporter spawn failed; continuing without OTLP metrics");
            e
        })
        .ok()
        .flatten();

    // Production-readliness recommendation (graceful shutdown): keep
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

    // RIL ISS-080: enforce auth from ALL resolved key sources (inline +
    // env + file), matching the SEC-01 posture computed above. Pre-fix this
    // gated on the inline `api_keys` list only, so `--api-key-file` /
    // `VLLM_API_KEYS_FILE` deployments saw no startup warning while the
    // middleware stayed None and the inference API ran unauthenticated.
    let auth_middleware = vllm_server::app::build_auth_middleware(&app_config.auth);

    // Initialize health checker
    let health_checker = Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true)));

    // Production-readiness recommendation: shared in-memory
    // audit ring buffer (10 000 events) backed by a structured
    // `tracing` event per row. Mounted as a layer below; the
    // `Arc` here lets us share the same instance between
    // `ApiState` (for direct handler access if needed) and the
    // middleware `State` extractor.
    let audit_logger = Arc::new(vllm_server::security::audit::AuditLogger::new(10_000));

    let architecture = loader.architecture();

    // Production-readiness §4: extract the model's max context
    // length so the chat/completions handlers can return a
    // `400 context_length_exceeded` instead of wasting KV
    // blocks on an over-budget prompt. The field name follows
    // the HuggingFace convention (`max_position_embeddings`),
    // which the Qwen3 / Llama / Mistral checkpoints all carry.
    // An explicit `--max-model-len` (RIL ISS-044) overrides the
    // checkpoint value. For architectures that declare neither
    // (stub models, GGUF variants, etc.) the value stays `None`
    // and request validation falls back to a hard `max_tokens`
    // ceiling (see `openai::chat::check_context_length`) so an
    // unknown-context model cannot drive an unbounded generation.
    let max_model_len = app_config.engine.max_model_len.or_else(|| {
        loader
            .config_json()
            .get("max_position_embeddings")
            .and_then(serde_json::value::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
    });

    // Production-readiness §10: read capability flags so
    // capability-gated endpoints (e.g. `/v1/embeddings`) can
    // refuse on stubs / unknown architectures rather than
    // silently returning meaningless data.
    let arch_capabilities = loader.capabilities();

    let state = ApiState {
        engine_tx: msg_tx.clone(),
        tokenizer,
        architecture,
        batch_manager,
        auth: auth_middleware.clone(),
        audit: audit_logger.clone(),
        health: health_checker.clone(),
        metrics: metrics_collector,
        max_model_len,
        arch_capabilities,
    };

    // Production router: routes + middleware stack. The ordering
    // invariants (correlation outermost of the security stack, body
    // limit above auth, audit above both) live in `app::build_app` and
    // are covered by `crates/server/tests/middleware_wiring.rs`.
    let cors_config = app_config.cors.into_runtime();
    let app =
        vllm_server::app::build_app(state, auth_middleware, audit_logger.clone(), &cors_config);

    let addr = format!("{}:{}", app_config.server.host, app_config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind server socket to {addr}"))?;
    tracing::info!(address = %addr, "Server listening");

    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(shutdown_signal(
            health_checker,
            app_config.server.shutdown_drain_grace_secs,
        ))
        .await
        .with_context(|| format!("server crashed while serving on {addr}"))?;

    tracing::info!("HTTP server stopped; draining engine");

    // Production-readiness recommendation (graceful shutdown):
    // 1. Tell the engine to exit its run loop. `blocking_send` is the
    //    synchronous mpsc send — it always returns immediately (with
    //    Err only when the channel is closed) and never produces a
    //    future that needs awaiting. The receiver stays alive (the
    //    worker thread holds it) until we join below, so the send
    //    succeeds.
    // 2. Wait for the worker to acknowledge shutdown by joining
    //    the thread. Cap at 10s so a stuck engine can't pin the
    //    process forever; operators can `SIGKILL` if that's
    //    needed.
    if let Err(e) = engine_shutdown_tx.blocking_send(EngineMessage::Shutdown) {
        tracing::warn!(error = %e, "engine shutdown send failed");
    }
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

/// Wait for a shutdown signal (Ctrl+C on all platforms, SIGTERM on Unix).
async fn wait_for_shutdown_signal() {
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
}

/// Mark the health checker as not-ready and wait for the drain grace period.
async fn drain_gracefully(
    health_checker: &std::sync::RwLock<HealthChecker>,
    drain_grace_secs: u64,
) {
    // Production-readiness §7 step 1: flip readiness=false BEFORE
    // returning from this future. axum's `with_graceful_shutdown`
    // closes the listener as soon as this future resolves, so we
    // need to (a) publish the new readiness state to any K8s probe
    // that pings us in the next few seconds, and (b) hold the
    // listener open for `drain_grace_secs` so the probe has a chance
    // to observe 503 and remove the pod from the Service endpoints
    // list. Without this grace, the listener slams shut on SIGTERM
    // and any in-flight L7 connections get RST instead of a clean
    // close.
    //
    // The `mark_not_ready` call holds the write lock only long
    // enough to flip the bool — no I/O, no allocation — so a
    // concurrent `/health/ready` probe is delayed by microseconds at
    // most. The subsequent `tokio::time::sleep` is interruptible and
    // yields the runtime, so it does NOT block the listener's accept
    // loop; axum keeps accepting (and immediately rejecting) new
    // connections during the grace period.
    if let Ok(mut checker) = health_checker.write() {
        checker.mark_not_ready();
    } else {
        tracing::warn!("health_checker lock poisoned during shutdown; readiness may not flip");
    }
    tracing::info!(
        drain_grace_secs,
        "readiness flipped to NotReady; waiting for orchestrator drain before closing listener"
    );
    if drain_grace_secs > 0 {
        tokio::time::sleep(std::time::Duration::from_secs(drain_grace_secs)).await;
    }
}

async fn shutdown_signal(
    health_checker: Arc<std::sync::RwLock<HealthChecker>>,
    drain_grace_secs: u64,
) {
    wait_for_shutdown_signal().await;
    tracing::info!("Shutdown signal received");
    drain_gracefully(&health_checker, drain_grace_secs).await;
}
