#![allow(clippy::module_name_repetitions)]
//! Structured logging initialisation for the vllm-lite server.
//!
//! Wraps [`tracing_subscriber`] with a dual-output setup (console + optional
//! daily-rotating JSON file). Honours `RUST_LOG` env override if set; falls
//! back to the `log_level` argument otherwise.
use std::path::PathBuf;
use std::sync::OnceLock;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

/// The configured fallback log level (e.g. `--log-level` / the YAML
/// `server.log_level`), captured at log-init time *before* any `RUST_LOG`
/// override. `/debug/trace` reports the **effective** directive — `RUST_LOG`
/// when set, else this value — which is exactly the precedence
/// [`init_logging`] applies when building the `EnvFilter`, so the endpoint
/// agrees with what the process actually emits (RIL ISS-093).
pub(crate) static CONFIGURED_LOG_LEVEL: OnceLock<String> = OnceLock::new();

/// The effective filter directive: `RUST_LOG` when set and parseable,
/// otherwise the configured fallback.
#[must_use]
pub(crate) fn effective_log_level() -> String {
    std::env::var("RUST_LOG").unwrap_or_else(|_| {
        CONFIGURED_LOG_LEVEL
            .get()
            .cloned()
            .unwrap_or_else(|| "info".to_string())
    })
}

/// Initialise the global tracing subscriber.
///
/// # Arguments
/// * `log_dir` - If `Some`, also writes JSON logs to `<dir>/vllm-lite.log.YYYY-MM-DD`.
///   Missing directories are created on best-effort basis.
/// * `log_level` - Fallback filter directive (e.g. `"info"`, `"debug"`) when
///   `RUST_LOG` is unset.
///
/// Calling this more than once is a no-op for the second call (tracing refuses
/// to re-install the global subscriber).
pub fn init_logging(log_dir: Option<PathBuf>, log_level: &str) {
    // RIL ISS-093: record the configured fallback so `/debug/trace` can
    // report the effective level (RUST_LOG wins when set). First set wins —
    // late OTLP-fallback re-inits keep the first (authoritative) value.
    let _ = CONFIGURED_LOG_LEVEL.set(log_level.to_string());
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));

    let subscriber = tracing_subscriber::registry().with(env_filter);

    if let Some(dir) = log_dir {
        // RIL ISS-150: fail fast when the log directory can't be created
        // instead of panicking deep inside `RollingFileAppender::new` with
        // an init stack trace and exit 101. `create_dir_all(...).ok()` used
        // to swallow the error here, so a typo'd `--log-dir` (parent is a
        // file, or no write permission) turned a config mistake into a raw
        // panic. stderr is the actionable channel (called before/without
        // tracing attached to a file yet).
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!("log dir error: {}", e);
            std::process::exit(1);
        }
        let file_appender = RollingFileAppender::new(Rotation::DAILY, dir, "vllm-lite.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

        // 文件输出: JSON 格式 (用于程序解析)
        let json_layer = fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false)
            .json()
            .with_target(true)
            .with_thread_ids(false)
            .with_file(true)
            .with_line_number(true);

        // 控制台输出: 美化格式 (人类可读)
        let console_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(false)
            .with_ansi(true)
            .compact();

        subscriber.with(json_layer).with(console_layer).init();
    } else {
        // 仅控制台输出: 美化格式
        subscriber
            .with(fmt::layer().with_target(true).with_ansi(true).compact())
            .init();
    }
}
