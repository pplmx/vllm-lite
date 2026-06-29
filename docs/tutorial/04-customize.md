# Tutorial 4: Customize the Engine

This tutorial shows how to add custom strategies (scheduling, sampling).

## Custom Sampling Strategy

`SamplingStrategy` is a trait in `vllm-core`. To add a new strategy:

```rust,no_run
use vllm_core::sampling::{SamplingStrategy, SamplingParams, TokenId};

pub struct MyCustomSampler;

impl SamplingStrategy for MyCustomSampler {
    fn sample(&self, logits: &[f32], _params: &SamplingParams) -> TokenId {
        // Your logic here. Example: always return argmax.
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as TokenId)
            .unwrap_or(0)
    }
}
```

Register it with the engine via `EngineBuilder::with_sampling_strategy`.

## Custom Scheduling Policy

`SchedulingPolicy` trait in `vllm-core::scheduler::policy`:

```rust,no_run
use vllm_core::scheduler::policy::SchedulingPolicy;
use vllm_core::types::{Request, PriorityScore};

pub struct MyFairSharePolicy;

impl SchedulingPolicy for MyFairSharePolicy {
    fn name(&self) -> &'static str { "fair-share" }

    fn compute_priority(&self, req: &Request) -> PriorityScore {
        // Fair-share: weight by user_id to prevent starvation
        PriorityScore(req.user_id as i64 * 100 + req.arrival_time as i64)
    }
}
```

## Registering Custom Components

Use the global registry (`vllm_core::arch::ArchitectureRegistry` /
`vllm_core::scheduler::policy::PolicyRegistry`) — see existing examples.

## Testing Custom Code

Use property-based tests (ADR-016) for invariants:

```rust,no_run
use proptest::prelude::*;

proptest! {
    /// My custom policy always returns a non-negative priority.
    #[test]
    fn prop_my_policy_non_negative(req in arbitrary_request()) {
        let policy = MyFairSharePolicy;
        let priority = policy.compute_priority(&req);
        prop_assert!(priority.0 >= 0);
    }
}
```

Run: `cargo test -p vllm-core my_policy`

## Next Steps

→ [Tutorial 5: Production](05-production.md)
