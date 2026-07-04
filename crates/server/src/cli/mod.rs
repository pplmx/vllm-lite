//! CLI argument parsing namespace. The `Args` struct itself lives in `args.rs`; this module exists for future subcommands (e.g., `vllm-server bench`, `vllm-server migrate`).
#![allow(clippy::module_name_repetitions)]
mod args;

pub use args::{CliArgs, CliValidationError, LogLevel, ModelArgs};
