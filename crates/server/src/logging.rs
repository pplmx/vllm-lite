use std::path::PathBuf;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub fn init_logging(log_dir: Option<PathBuf>, log_level: &str) {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));

    let subscriber = tracing_subscriber::registry().with(env_filter);

    if let Some(dir) = log_dir {
        std::fs::create_dir_all(&dir).ok();
        let file_appender = RollingFileAppender::new(Rotation::DAILY, dir, "vllm-lite.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

        subscriber
            .with(
                fmt::layer()
                    .with_writer(non_blocking)
                    .with_ansi(false)
                    .json()
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_file(true)
                    .with_line_number(true),
            )
            .init();
    } else {
        subscriber
            .with(fmt::layer().with_target(true).with_thread_ids(false).json())
            .init();
    }
}

#[allow(dead_code)]
pub fn log_request(request_id: &str, prompt: &str, params: &str) {
    tracing::info!(
        request_id = %request_id,
        prompt_len = prompt.len(),
        params = %params,
        "Request started"
    );
}

#[allow(dead_code)]
pub fn log_response(request_id: &str, tokens: usize, latency_ms: f64) {
    tracing::info!(
        request_id = %request_id,
        output_tokens = tokens,
        latency_ms = latency_ms,
        "Request completed"
    );
}

#[allow(dead_code)]
pub fn log_error(request_id: &str, error: &str) {
    tracing::error!(
        request_id = %request_id,
        error = %error,
        "Request failed"
    );
}
