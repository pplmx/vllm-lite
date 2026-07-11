//! `EngineBuilder` — named-method builder for [`Engine`]. Equivalent to
//! calling `Engine::with_config_boxed(...)` then setting the optional
//! fields directly.

use std::sync::Arc;

use crate::engine::Engine;
use crate::engine::SleepPolicy;
use crate::speculative::AdaptiveSpeculativeDecoder;
use crate::speculative::draft_resolver::DraftResolver;
use crate::types::SchedulerConfig;
use vllm_traits::ModelBackend;

/// Builder for [`Engine`] with named methods for all optional fields.
///
/// # Example
///
/// ```ignore
/// use vllm_core::EngineBuilder;
/// use vllm_traits::ModelBackend;
///
/// let target: Box<dyn ModelBackend> = Box::new(/* ... */);
/// let engine = EngineBuilder::new(target)
///     .with_num_kv_blocks(1024)
///     .with_max_draft_tokens(5)
///     .build();
/// ```
pub struct EngineBuilder {
    target_model: Box<dyn ModelBackend>,
    draft_model: Option<Box<dyn ModelBackend>>,
    config: SchedulerConfig,
    max_draft_tokens: usize,
    num_kv_blocks: usize,
    adaptive_decoder: Option<AdaptiveSpeculativeDecoder>,
    draft_resolver: Option<Arc<DraftResolver>>,
    sleep_policy: SleepPolicy,
}

impl std::fmt::Debug for EngineBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineBuilder")
            .field("target_model", &"<dyn ModelBackend>")
            .field(
                "draft_model",
                &self.draft_model.as_ref().map(|_| "<dyn ModelBackend>"),
            )
            .field("config", &self.config)
            .field("max_draft_tokens", &self.max_draft_tokens)
            .field("num_kv_blocks", &self.num_kv_blocks)
            .field("adaptive_decoder", &self.adaptive_decoder)
            .field(
                "draft_resolver",
                &self.draft_resolver.as_ref().map(Arc::strong_count),
            )
            .field("sleep_policy", &self.sleep_policy)
            .finish()
    }
}

impl EngineBuilder {
    /// Create a new builder with a target model. Other fields use defaults:
    /// - `draft_model = None`
    /// - `config = SchedulerConfig::default()`
    /// - `max_draft_tokens = 4`
    /// - `num_kv_blocks = 1024`
    /// - `adaptive_decoder = None`
    /// - `draft_resolver = None`
    /// - `sleep_policy = SleepPolicy::default()`
    #[must_use]
    pub fn new(target_model: Box<dyn ModelBackend>) -> Self {
        Self {
            target_model,
            draft_model: None,
            config: SchedulerConfig::default(),
            max_draft_tokens: 4,
            num_kv_blocks: 1024,
            adaptive_decoder: None,
            draft_resolver: None,
            sleep_policy: SleepPolicy::default(),
        }
    }

    /// Set the optional draft model.
    #[must_use]
    pub fn with_draft_model(mut self, draft_model: Box<dyn ModelBackend>) -> Self {
        self.draft_model = Some(draft_model);
        self
    }

    /// Override the scheduler config.
    #[must_use]
    pub fn with_config(mut self, config: SchedulerConfig) -> Self {
        self.config = config;
        self
    }

    /// Override the max draft tokens per step.
    #[must_use]
    pub const fn with_max_draft_tokens(mut self, n: usize) -> Self {
        self.max_draft_tokens = n;
        self
    }

    /// Override the number of KV-cache blocks.
    #[must_use]
    pub const fn with_num_kv_blocks(mut self, n: usize) -> Self {
        self.num_kv_blocks = n;
        self
    }

    /// Set an optional adaptive speculative decoder.
    #[must_use]
    // Used only by `ctor/builder/tests.rs`; `pub(crate)` is sufficient but
    // rustc emits a dead-code warning during non-test builds because the
    // symbol is reachable only under `cfg(test)`. Allow is intentional.
    #[allow(dead_code)]
    pub(crate) fn with_adaptive_decoder(mut self, decoder: AdaptiveSpeculativeDecoder) -> Self {
        self.adaptive_decoder = Some(decoder);
        self
    }

    /// Set an optional per-request draft resolver (v18+).
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn with_draft_resolver(mut self, resolver: Arc<DraftResolver>) -> Self {
        self.draft_resolver = Some(resolver);
        self
    }

    /// Override the sleep policy.
    #[must_use]
    pub const fn with_sleep_policy(mut self, policy: SleepPolicy) -> Self {
        self.sleep_policy = policy;
        self
    }

    /// Build the [`Engine`]. Equivalent to calling `Engine::with_config_boxed(...)`
    /// then setting the optional fields directly.
    #[must_use]
    pub fn build(self) -> Engine {
        let mut engine = Engine::with_config_boxed(
            self.target_model,
            self.draft_model,
            self.config,
            self.max_draft_tokens,
            self.num_kv_blocks,
        );
        engine.adaptive_decoder = self.adaptive_decoder;
        engine.draft_resolver = self.draft_resolver;
        engine.sleep_policy = self.sleep_policy;
        engine
    }
}
