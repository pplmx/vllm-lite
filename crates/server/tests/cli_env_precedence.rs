//! RIL ISS-081: environment-variable overrides must populate the resolved
//! `AppConfig` through `CliArgs::to_app_config`.
//!
//! This lives in `tests/` (not `src/cli/args/tests.rs`) on purpose: nextest
//! runs each integration-test file in its own process, so setting a
//! `VLLM_*` process-wide env var here cannot race with the `cli::args` unit
//! tests (which assert that an unset env leaves the args `None`).
//!
//! The precedence contract under test:
//!   explicit CLI flag / env var  >  `--config` YAML  >  built-in defaults.
//! Here we exercise the env leg: a set env var must still flow through
//! `Option<T>` into the config, and an explicit flag must beat the env var.

use clap::Parser;
use vllm_server::cli::CliArgs;

/// Restore the prior `VLLM_PORT` value (or remove it) when this Guard drops,
/// so a shell that already exports the var is left untouched after the test.
struct EnvGuard(Option<std::ffi::OsString>);

// SAFETY helpers for this file: `std::env::set_var` / `remove_var` are
// `unsafe` since Rust 1.80 because they mutate process-wide state. This test
// binary runs as its own process and the two tests below serialize on
// `OnceLock<Mutex>` semantics naturally (sequential execution in one binary),
// so no other thread reads or writes `VLLM_PORT` concurrently.
#[allow(unsafe_code)]
impl EnvGuard {
    /// Capture the prior value and set the test value.
    fn set(value: &str) -> Self {
        let prior = std::env::var_os("VLLM_PORT");
        // SAFETY: single-process test file (see module doc).
        unsafe { std::env::set_var("VLLM_PORT", value) };
        Self(prior)
    }
}

#[allow(unsafe_code)]
impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: same single-process rationale as in `set`; restores the
        // variable this test owns.
        match &self.0 {
            Some(v) => unsafe { std::env::set_var("VLLM_PORT", v) },
            None => unsafe { std::env::remove_var("VLLM_PORT") },
        }
    }
}

#[test]
fn env_var_flows_into_resolved_config() {
    let _guard = EnvGuard::set("9005");

    let config = CliArgs::parse_from(["vllm-server", "-m", "/test/model"]).to_app_config();
    assert_eq!(
        config.server.port, 9005,
        "VLLM_PORT env must flow through to_app_config (env > YAML > defaults)"
    );
}

#[test]
fn explicit_flag_beats_env_var() {
    let _guard = EnvGuard::set("9005");

    let config =
        CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-p", "7777"]).to_app_config();
    assert_eq!(
        config.server.port, 7777,
        "explicit -p flag must beat the VLLM_PORT env (CLI > env)"
    );
}
