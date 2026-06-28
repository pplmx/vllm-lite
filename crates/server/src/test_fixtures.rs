//! Shared fixtures for server unit and integration tests.
//!
//! # Why This Module Lives in `vllm-server` (Not `vllm-testing`)
//!
//! During the v19.0 architecture audit and Phase 31 (v21.1) remediation, the
//! audit recommended moving this fixture module into `vllm-testing` (finding
//! ARCH-F-14 / ML-07). After analysis, the move is **architecturally
//! infeasible** without breaking the dependency layering rule. Rationale:
//!
//! 1. **Circular dependency.** `api_state()` and `api_state_with_mock_engine()`
//!    return `crate::ApiState` and `crate::api::EngineHandle` — both are
//!    `pub` types defined in `vllm-server`. Moving the fixtures into
//!    `vllm-testing` would force `vllm-testing → vllm-server`, while
//!    `vllm-server`'s `dev-dependencies` already include `vllm-testing` for
//!    integration tests. This creates a cycle.
//!
//! 2. **Layering rule.** The project layering is
//!    `vllm-traits ← vllm-core ← {vllm-model, vllm-server, vllm-dist}`.
//!    `vllm-testing` is a sibling of `vllm-server` (not a child). Allowing
//!    `vllm-testing → vllm-server` would invert the rule for the test crate.
//!
//! 3. **No consumer outside server.** A grep of `cargo tree` and the test
//!    surfaces shows no other crate consumes these fixtures. They are
//!    server-test-specific glue.
//!
//! # What `vllm-testing` Does Provide (Server Tests Should Prefer)
//!
//! - `vllm_testing::TestHarness` — engine environment setup
//! - `vllm_testing::RequestFactory` — generates test requests
//! - `vllm_testing::BatchBuilder` / `RequestBuilder` — fixture builders
//! - `vllm_testing::mocks::{FakeModel, StubModel, ...}` — deterministic backends
//! - `vllm_testing::prelude::*` — bulk import of the above
//!
//! Server tests that do not need `ApiState` should use those exports
//! directly. The functions in *this* module exist exclusively for tests
//! that need a wired `ApiState`.
//!
//! # Re-evaluation Triggers
//!
//! - If `ApiState` ever becomes a stable trait object (e.g.,
//!   `Arc<dyn ApiStateLike>`) with a default impl in `vllm-testing`, the
//!   fixtures could move.
//! - If `vllm-server` is split into `vllm-server` + `vllm-server-types`
//!   (with `ApiState` in the types crate), the fixtures can move into
//!   `vllm-testing` which would depend on `vllm-server-types`.
//!
//! See: `.planning/audit/architecture/REPORT.md` (v19.0 finding ARCH-F-14),
//! Phase 31 plan `31-06` for the original analysis.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::types::{EngineMessage, TokenId};
use vllm_model::config::Architecture;
use vllm_model::tokenizer::Tokenizer;

use crate::ApiState;
use crate::api::EngineHandle;
use crate::health::HealthChecker;
use crate::openai::batch::BatchManager;

/// Build [`ApiState`] with defaults suitable for handler tests.
#[must_use]
pub fn api_state(architecture: Architecture) -> ApiState {
    let (engine_tx, _engine_rx) = mpsc::unbounded_channel();
    ApiState {
        engine_tx,
        tokenizer: Arc::new(Tokenizer::new()),
        architecture,
        batch_manager: Arc::new(BatchManager::new()),
        auth: None,
        health: Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true))),
        metrics: Arc::new(EnhancedMetricsCollector::new()),
    }
}

/// Mock engine that replies with the given token stream for each chat request.
#[must_use]
pub fn spawn_mock_engine(reply_tokens: Vec<TokenId>) -> (EngineHandle, JoinHandle<()>) {
    let (engine_tx, mut engine_rx) = mpsc::unbounded_channel();
    let handle = tokio::spawn(async move {
        while let Some(msg) = engine_rx.recv().await {
            if let EngineMessage::AddRequest { response_tx, .. } = msg {
                for token in &reply_tokens {
                    if response_tx.send(*token).await.is_err() {
                        break;
                    }
                }
            }
        }
    });
    (engine_tx, handle)
}

/// [`ApiState`] wired to a mock engine that emits `reply_tokens`.
#[must_use]
pub fn api_state_with_mock_engine(
    architecture: Architecture,
    reply_tokens: Vec<TokenId>,
) -> (ApiState, JoinHandle<()>) {
    let (engine_tx, handle) = spawn_mock_engine(reply_tokens);
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(Tokenizer::new()),
        architecture,
        batch_manager: Arc::new(BatchManager::new()),
        auth: None,
        health: Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true))),
        metrics: Arc::new(EnhancedMetricsCollector::new()),
    };
    (state, handle)
}
