//! CLI argument parsing namespace. The `Args` struct itself lives in `args.rs`; this module exists for future subcommands (e.g., `vllm-server bench`, `vllm-server migrate`).
#![allow(clippy::module_name_repetitions)]
mod args;

pub use args::{CliArgs, CliValidationError, LogLevel, ModelArgs};

/// The compiled server semver (`CARGO_PKG_VERSION`), re-exported for
/// startup logging / `--version` consumers across the crate.
pub use args::SERVER_VERSION;
