use std::path::PathBuf;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

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
