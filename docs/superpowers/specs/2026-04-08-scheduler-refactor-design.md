# SchedulerEngine 架构重构设计

## 目标

将 SchedulerEngine 拆分为独立的 Manager 组件，实现完全解耦，提升可测试性和可维护性。

## 当前状态

### 问题
- SchedulerEngine 有 9 个字段，承担过多职责
- BlockAllocator, EvictionPolicy, PrefixCache 混在一起
- 难以单独测试每个组件
- 代码难以维护

### 现有字段分析

| 字段 | 类型 | 职责 | 拆分方案 |
|------|------|------|----------|
| queue_manager | QueueManager | 队列管理 | 保持 |
| batch_planner | BatchPlanner | 批处理规划 | 保持 |
| kv_allocator | BlockAllocator | KV 块分配 | → MemoryManager |
| prefix_cache | PrefixCache | 前缀缓存 | → CacheManager |
| eviction_policy | EvictionPolicy | 驱逐策略 | → MemoryManager |
| preemption_manager | PreemptionManager | 抢占管理 | 保持或 → MemoryManager |
| config | SchedulerConfig | 配置 | 保持 |
| stats | SchedulerStats | 统计 | 保持 |
| next_seq_id | SeqId | 序列ID生成 | 保持 |
| running | Vec<Sequence> | 运行中序列 | 保持 |
| finished | Vec<Sequence> | 已完成序列 | 保持 |
| observers | SchedulerObservers | 观察者 | 保持 |

## 目标架构（最终版）

**原则**: 彻底解耦，调度器自包含，不依赖 kv_cache 模块

```
crates/core/src/
├── kv_cache/              # 模型层使用的物理KV存储（保留）
│   ├── mod.rs
│   ├── block.rs          # 物理块管理
│   └── paged_tensor.rs   # 模型使用
│
└── scheduler/            # 调度层（完全独立）
    ├── engine.rs
    ├── queue_manager.rs
    ├── batch_planner.rs
    ├── memory/           # 内存管理（新建，从 kv_cache 移动）
    │   ├── mod.rs
    │   ├── allocator.rs  # BlockAllocator (从 kv_cache 移动)
    │   └── eviction.rs   # EvictionPolicy (从 scheduler 移动)
    ├── cache/            # 缓存管理（新建，从 kv_cache 移动）
    │   ├── mod.rs
    │   └── prefix_cache.rs # PrefixCache (从 kv_cache 移动)
    ├── preemption.rs
    └── observer.rs
```

## 核心设计

### scheduler/memory/allocator.rs

从 kv_cache/ 移动，保持原有接口：

```rust
pub struct BlockAllocator {
    free_blocks: VecDeque<BlockId>,
    total: usize,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self;
    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>>;
    pub fn free(&mut self, blocks: &[BlockId]);
    pub fn available(&self) -> usize;
    pub fn total(&self) -> usize;
}
```

### scheduler/memory/eviction.rs

从 scheduler/engine.rs 移动：

```rust
pub struct EvictionPolicy { ... }

impl EvictionPolicy {
    pub fn new() -> Self;
    pub fn select_victims(&self, running: &[Sequence], count: usize) -> Vec<BlockId>;
    pub fn release_blocks(&self, blocks: &[BlockId]);
    pub fn record_blocks(&self, blocks: &[BlockId]);
}
```

### scheduler/memory/mod.rs

封装以上两个 + PreemptionManager：

```rust
pub struct MemoryManager {
    allocator: BlockAllocator,
    eviction_policy: EvictionPolicy,
    preemption_manager: PreemptionManager,
}

impl MemoryManager {
    pub fn new(num_blocks: usize, config: SchedulerConfig) -> Self;
    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>>;
    pub fn free(&mut self, blocks: &[BlockId]);
    pub fn available(&self) -> usize;
    pub fn total(&self) -> usize;
    pub fn select_victims(&self, running: &[Sequence], count: usize) -> Vec<BlockId>;
    pub fn release_blocks(&mut self, blocks: &[BlockId]);
    pub fn should_preempt(&self, running: &[Sequence], waiting: usize) -> bool;
    pub fn execute_preemption(&mut self, running: &mut Vec<Sequence>, queue: &mut QueueManager);
}
```

### scheduler/cache/prefix_cache.rs

从 kv_cache/ 移动，保持原有接口：

```rust
pub struct PrefixCache { ... }

impl PrefixCache {
    pub fn new() -> Self;
    pub fn get(&mut self, key: CacheKey) -> Option<&CachedEntry>;
    pub fn insert(&mut self, key: CacheKey, blocks: Vec<BlockId>, token_count: usize);
    pub fn find_prefix_match(&mut self, tokens: &[TokenId]) -> Option<&CachedEntry>;
    pub fn find_reverse_prefix_match(&self, tokens: &[TokenId]) -> Option<(Arc<Vec<BlockId>>, usize)>;
    pub fn evict(&mut self, allocator: &mut BlockAllocator);
    pub fn stats(&self) -> PrefixCacheStats;
}
```

### scheduler/cache/mod.rs

```rust
pub struct CacheManager {
    prefix_cache: PrefixCache,
}

impl CacheManager {
    pub fn new() -> Self;
    pub fn get(&mut self, key: CacheKey) -> Option<&CachedEntry>;
    pub fn insert(&mut self, key: CacheKey, blocks: Vec<BlockId>, token_count: usize);
    pub fn find_prefix_match(&mut self, tokens: &[TokenId]) -> Option<&CachedEntry>;
    pub fn find_reverse_prefix_match(&self, tokens: &[TokenId]) -> Option<(Arc<Vec<BlockId>>, usize)>;
    pub fn evict(&mut self, memory: &mut MemoryManager);
    pub fn stats(&self) -> PrefixCacheStats;
    pub fn hit_rate(&self) -> f64;
}
```

### SchedulerEngine (重构后)

```rust
pub struct SchedulerEngine {
    queue_manager: QueueManager,
    batch_planner: BatchPlanner,
    memory: MemoryManager,
    cache: CacheManager,
    config: SchedulerConfig,
    stats: SchedulerStats,
    next_seq_id: SeqId,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    observers: SchedulerObservers,
}
```

## 迁移计划

### Phase 1: 创建 scheduler/memory/ 模块
1. 移动 BlockAllocator 到 scheduler/memory/allocator.rs
2. 移动 EvictionPolicy 到 scheduler/memory/eviction.rs
3. 创建 scheduler/memory/mod.rs (MemoryManager)
4. 更新所有 import 引用
5. 运行测试验证

### Phase 2: 创建 scheduler/cache/ 模块
1. 移动 PrefixCache 到 scheduler/cache/prefix_cache.rs
2. 创建 scheduler/cache/mod.rs (CacheManager)
3. 更新所有 import 引用
4. 运行测试验证

### Phase 3: 重构 SchedulerEngine
1. 使用新的 memory 和 cache 模块
2. 移除旧的 kv_allocator, eviction_policy, prefix_cache, preemption_manager 字段
3. 更新所有方法调用

### Phase 4: 清理
1. 删除 kv_cache 模块中的 allocator.rs 和 prefix_cache.rs
2. 更新 kv_cache/mod.rs
3. 运行完整测试套件

### Phase 5: 最终验证
1. 运行 clippy
2. 运行 fmt
3. 性能基准测试

## 待确认

- [x] 彻底解耦方案
- [ ] 确认后开始实现计划
