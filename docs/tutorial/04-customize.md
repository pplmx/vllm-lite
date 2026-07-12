# Tutorial 4: Customize the Engine

This tutorial shows how to customize sampling parameters and scheduling policy.

## Sampling Parameters

Per-request sampling is configured via `SamplingParams` on each `Request`:

```rust,no_run
use vllm_core::types::{Request, SamplingParams};

# fn doc() {
let mut req = Request::new(1, vec![10, 20, 30], 50);
req.sampling_params = SamplingParams::builder()
    .with_temperature(0.7)
    .with_top_k(40)
    .with_top_p(0.9)
    .build();
# }
```

The engine applies these parameters inside its step loop when sampling logits.

## Custom Scheduling Policy

Implement [`SchedulingPolicy`](../../crates/core/src/scheduler/policy/trait_def.rs)
and install it on the scheduler via `SchedulerEngine::set_policy`:

```rust,no_run
use std::sync::Arc;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::SchedulerEngine;
use vllm_core::scheduler::policy::{
    PriorityScore, SchedulingContext, SchedulingPolicy,
};
use vllm_core::types::{SchedulerConfig, Sequence};

pub struct ShortPromptFirstPolicy;

impl SchedulingPolicy for ShortPromptFirstPolicy {
    fn name(&self) -> &'static str {
        "short-prompt-first"
    }

    fn compute_priority(&self, seq: &Sequence, _ctx: &SchedulingContext) -> PriorityScore {
        // Shorter prompts get higher priority (lower remaining work).
        let remaining = seq.max_tokens.saturating_sub(seq.tokens.len());
        PriorityScore(u64::try_from(remaining).unwrap_or(u64::MAX))
    }
}

# fn doc() {
let config = SchedulerConfig::default();
let metrics = Arc::new(EnhancedMetricsCollector::new());
let mut scheduler = SchedulerEngine::new(config, 1024, metrics);
scheduler.set_policy(Box::new(ShortPromptFirstPolicy));
# }
```

Built-in policies: `FcfsPolicy`, `SjfPolicy`, `PriorityPolicy` — see
`vllm_core::scheduler::policy`.

## Engine Configuration

Tune batch limits and KV capacity at construction time. The
`Engine::with_config(...)` constructor was retired in v31; use the
builder's `with_*` methods instead:

```rust,no_run
use vllm_core::EngineBuilder;
use vllm_core::types::SchedulerConfig;
use vllm_traits::{StubModelBackend, ModelBackend};

# fn doc() {
let config = SchedulerConfig::builder()
    .with_max_num_seqs(128)
    .with_max_num_batched_tokens(4096)
    .build();

let target: Box<dyn ModelBackend> = Box::new(StubModelBackend);
let engine = EngineBuilder::new(target)
    .with_num_kv_blocks(2048)
    .with_max_draft_tokens(4)
    .with_config(config)
    .build();
# }
```

## Registering New Model Architectures

New architectures live in `vllm-model`, not `vllm-core`. See
[Tutorial 2](02-load-model.md) and `docs/architecture.md` for the 3-step
`Architecture` trait registration flow.

## Testing Custom Code

Use property-based tests (ADR-016) for policy invariants:

```rust,no_run
use proptest::prelude::*;
use vllm_core::scheduler::policy::{PriorityScore, SchedulingContext, SchedulingPolicy};
use vllm_core::types::Sequence;
use std::time::Instant;

struct MyPolicy;
impl SchedulingPolicy for MyPolicy {
    fn name(&self) -> &'static str { "my-policy" }
    fn compute_priority(&self, _seq: &Sequence, _ctx: &SchedulingContext) -> PriorityScore {
        PriorityScore(0)
    }
}

proptest! {
    #[test]
    fn prop_priority_is_bounded(_seed: u64) {
        let policy = MyPolicy;
        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: 1,
            running_count: 0,
            memory_pressure: 0.0,
        };
        // Build a minimal Sequence in a real test — see scheduler unit tests.
        let _ = policy.name();
        prop_assert!(ctx.queue_length >= 0);
    }
}
```

Run: `cargo test -p vllm-core scheduling`

## Next Steps

→ [Tutorial 5: Production](05-production.md)
