vllm-lite/
├── Cargo.toml
└── src/
    ├── main.rs
    ├── lib.rs
    │
    ├── engine/ # 核心执行引擎
    │ ├── mod.rs
    │ ├── engine.rs
    │
    ├── scheduler/
    │ ├── mod.rs
    │ ├── scheduler.rs
    │ ├── policy.rs
    │
    ├── kv_cache/
    │ ├── mod.rs
    │ ├── cache.rs
    │ ├── block.rs
    │ ├── allocator.rs
    │
    ├── model/
    │ ├── mod.rs
    │ ├── trait.rs
    │ ├── fake.rs # MVP 用
    │
    ├── types/
    │ ├── mod.rs
    │ ├── request.rs
    │ ├── sequence.rs
    │ ├── batch.rs
    │
    ├── sampling/
    │ ├── mod.rs
    │ ├── sampler.rs
    │
    ├── runtime/
    │ ├── mod.rs
    │ ├── worker.rs
