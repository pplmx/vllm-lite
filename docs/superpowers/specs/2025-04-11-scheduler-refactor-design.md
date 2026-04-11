# Scheduler 架构重构设计文档

**日期**: 2025-04-11  
**作者**: AI Agent  
**状态**: 待评审  
**配置**: 策略D + P/D分离A + 内存优化B + 兼容性B + 范围C

---

## 1. 设计目标

### 1.1 核心目标
- 实现真正的 Prefill/Decode 严格分离，最大化 GPU 计算效率
- 支持可插拔的调度策略（FCFS、SJF、优先级、动态切换）
- 使用 Radix Tree 优化前缀缓存性能
- 重构 traits 层 Batch 接口，适配新架构需求

### 1.2 性能目标
| 指标 | 当前 | 目标 | 提升 |
|------|------|------|------|
| Prefix Cache 查找 | O(n) | O(k) | 10-100x |
| Queue get/remove | O(n) | O(1) | n倍 |
| 批处理构建 | 600+ 行 | <200 行/组件 | 可维护性 |
| P/D 分离效率 | 混合 batch | 纯 batch | 15-30% GPU 利用率 |

---

## 2. 架构设计

### 2.1 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     SchedulerEngine                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Request    │  │    Phase     │  │   Memory     │      │
│  │    Queue     │──│  Scheduler   │──│ Orchestrator │      │
│  │  (Indexed)   │  │              │  │              │      │
│  └──────────────┘  └──────┬───────┘  └──────────────┘      │
│                           │                                  │
│  ┌──────────────┐  ┌──────┴───────┐  ┌──────────────┐      │
│  │   Policy     │  │   Batch      │  │   Prefix     │      │
│  │   Engine     │──│  Composer    │──│    Cache     │      │
│  │ (Strategy)   │  │              │  │ (Radix Tree) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    vllm_traits::Batch                       │
│                    (Refactored Interface)                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

#### 2.2.1 RequestQueue - 索引化请求队列

**职责**: 管理所有等待调度的请求，支持 O(1) 查找和按策略排序

```rust
pub struct RequestQueue {
    // 主存储: SeqId -> Sequence
    sequences: HashMap<SeqId, Sequence>,
    // 按策略排序的优先级队列
    priority_queue: BinaryHeap<ScheduledSequence>,
    // 状态索引: Status -> Vec<SeqId>
    status_index: HashMap<Status, HashSet<SeqId>>,
    // 阶段索引: Phase -> Vec<SeqId>
    phase_index: HashMap<Phase, HashSet<SeqId>>,
}

pub struct ScheduledSequence {
    seq_id: SeqId,
    priority: PriorityScore, // 由策略计算
    arrival_time: Instant,
}

impl Ord for ScheduledSequence {
    fn cmp(&self, other: &Self) -> Ordering {
        // 策略可配置的比较逻辑
        self.priority.cmp(&other.priority)
            .then_with(|| self.arrival_time.cmp(&other.arrival_time))
    }
}
```

**接口**:
- `enqueue(seq: Sequence) -> Result<()>` - 加入队列
- `dequeue() -> Option<Sequence>` - 按策略出队
- `get(seq_id: SeqId) -> Option<&Sequence>` - O(1) 查找
- `remove(seq_id: SeqId) -> Option<Sequence>` - O(1) 删除
- `drain_by_phase(phase: Phase) -> Vec<Sequence>` - 按阶段批量提取

#### 2.2.2 PhaseScheduler - 阶段调度器

**职责**: 管理 Prefill 和 Decode 两个阶段的独立调度

```rust
pub struct PhaseScheduler {
    prefill_scheduler: PrefillScheduler,
    decode_scheduler: DecodeScheduler,
    current_phase: Phase,
    phase_switch_policy: PhaseSwitchPolicy,
}

pub enum Phase {
    Prefill,
    Decode,
}

pub struct PhaseSwitchPolicy {
    /// 连续 decode 多少轮后强制切换
    max_consecutive_decode: u32,
    /// Prefill 队列优先级阈值（队列长度超过此值优先 prefill）
    prefill_priority_threshold: usize,
    /// 最小 decode 批次大小（低于此值考虑切换）
    min_decode_batch_size: usize,
}

impl PhaseScheduler {
    /// 决定当前应该调度哪个阶段
    fn select_phase(&self, state: &SchedulerState) -> Phase {
        match self.current_phase {
            Phase::Decode if self.should_switch_to_prefill(state) => Phase::Prefill,
            Phase::Prefill if self.prefill_complete(state) => Phase::Decode,
            _ => self.current_phase,
        }
    }
    
    /// 从指定阶段提取批次
    fn extract_batch(&mut self, phase: Phase, budget: TokenBudget) -> Batch {
        match phase {
            Phase::Prefill => self.prefill_scheduler.extract_batch(budget),
            Phase::Decode => self.decode_scheduler.extract_batch(budget),
        }
    }
}

pub struct PrefillScheduler {
    /// Prefill 使用 chunked 处理
    chunk_size: usize,
    /// 当前正在处理的序列及其进度
    in_progress: HashMap<SeqId, usize>, // seq_id -> computed_tokens
}

pub struct DecodeScheduler {
    /// Decode 是简单的 1-token 迭代
    max_batch_size: usize,
}
```

**严格 P/D 分离规则**:
1. 一个 batch 只能包含同阶段的序列
2. PhaseScheduler 根据策略决定当前调度哪个阶段
3. 切换阶段时，当前批次必须完全处理完成

#### 2.2.3 PolicyEngine - 调度策略引擎

**职责**: 提供可插拔的调度策略

```rust
/// 调度策略 trait
pub trait SchedulingPolicy: Send + Sync {
    /// 计算序列的优先级分数（越小越高）
    fn compute_priority(&self, seq: &Sequence, context: &SchedulingContext) -> PriorityScore;
    
    /// 策略名称
    fn name(&self) -> &'static str;
}

/// 调度上下文
pub struct SchedulingContext {
    pub current_time: Instant,
    pub queue_length: usize,
    pub running_count: usize,
    pub memory_pressure: f32,
}

/// 优先级分数
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PriorityScore(pub u64);

// ============ 具体策略实现 ============

/// FCFS: 先来先服务
pub struct FcfsPolicy;

impl SchedulingPolicy for FcfsPolicy {
    fn compute_priority(&self, seq: &Sequence, ctx: &SchedulingContext) -> PriorityScore {
        // 使用序列 ID 作为到达顺序的代理（越小越高优先级）
        // 或者使用 ctx.current_time 和序列创建时间的差值
        PriorityScore(seq.id)
    }
    
    fn name(&self) -> &'static str { "FCFS" }
}

/// SJF: 最短作业优先（基于剩余 token 数估计）
pub struct SjfPolicy {
    /// 权重: 用户定义优先级
    sjf_priority_weight: f32,
    /// 权重: 剩余工作量
    sjf_remaining_work_weight: f32,
}

impl SchedulingPolicy for SjfPolicy {
    fn compute_priority(&self, seq: &Sequence, ctx: &SchedulingContext) -> PriorityScore {
        let remaining_tokens = seq.max_tokens.saturating_sub(seq.tokens.len());
        let user_priority = seq.priority.0 as u64;
        
        let score = (self.sjf_priority_weight * user_priority as f32
            + self.sjf_remaining_work_weight * remaining_tokens as f32) as u64;
        
        PriorityScore(score)
    }
    
    fn name(&self) -> &'static str { "SJF" }
}

/// 优先级调度（支持抢占）
pub struct PriorityPolicy {
    /// 老化因子（防止饥饿）
    priority_aging_factor: f32,
    /// 优先级队列数量
    priority_levels: u8,
}

impl SchedulingPolicy for PriorityPolicy {
    fn compute_priority(&self, seq: &Sequence, ctx: &SchedulingContext) -> PriorityScore {
        let wait_time = ctx.current_time.duration_since(seq.arrival_time).as_secs();
        let aging_bonus = (wait_time as f32 * self.priority_aging_factor) as u64;
        
        // 基础优先级 + 老化补偿
        let base_priority = seq.priority.0 as u64;
        let effective_priority = base_priority.saturating_sub(aging_bonus);
        
        PriorityScore(effective_priority)
    }
    
    fn name(&self) -> &'static str { "Priority" }
}

/// 策略工厂
pub struct PolicyFactory;

impl PolicyFactory {
    pub fn create(policy_type: PolicyType) -> Box<dyn SchedulingPolicy> {
        match policy_type {
            PolicyType::Fcfs => Box::new(FcfsPolicy),
            PolicyType::Sjf { priority_weight, remaining_work_weight } => {
                Box::new(SjfPolicy { priority_weight, remaining_work_weight })
            }
            PolicyType::Priority { aging_factor, num_priority_levels } => {
                Box::new(PriorityPolicy { aging_factor, num_priority_levels })
            }
        }
    }
}
```

#### 2.2.4 MemoryOrchestrator - 内存编排器

**职责**: 统一管理 KV Cache 内存分配、抢占和回收

```rust
pub struct MemoryOrchestrator {
    block_allocator: BlockAllocator,
    eviction_policy: EvictionPolicy,
    preemption_manager: PreemptionManager,
    memory_pool: MemoryPool,
}

/// 内存池（支持预分配和池化）
pub struct MemoryPool {
    /// 预分配的小块数组（常用大小）
    small_block_cache: [Vec<Vec<BlockId>>; 8], // 1-8 blocks
    /// 大块分配器
    large_block_allocator: BlockAllocator,
}

impl MemoryOrchestrator {
    /// 分配 blocks，优先从池中获取
    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>> {
        if num_blocks <= self.small_block_cache.len() {
            // 从缓存池获取
            self.allocate_from_pool(num_blocks)
        } else {
            // 从分配器获取
            self.block_allocator.allocate(num_blocks)
        }
    }
    
    /// 需要抢占时选择受害者
    pub fn select_victims(&mut self, running: &[Sequence], blocks_needed: usize) -> Vec<BlockId> {
        self.eviction_policy.select_victims(running, blocks_needed)
    }
    
    /// 执行抢占
    pub fn preempt(&mut self, seq: &mut Sequence) {
        self.preemption_manager.preempt(seq);
        self.block_allocator.free(&seq.kv_blocks);
        seq.kv_blocks = Arc::new(vec![]);
        seq.status = Status::Waiting;
    }
}
```

#### 2.2.5 BatchComposer - 批次组合器

**职责**: 将序列组合成高效的 GPU batch

```rust
pub struct BatchComposer {
    config: BatchCompositionConfig,
}

pub struct BatchCompositionConfig {
    /// 最大 batch 大小
    max_batch_size: usize,
    /// 最大 token 预算
    max_token_budget: usize,
    /// 是否启用相似度分组
    enable_similarity_grouping: bool,
}

impl BatchComposer {
    /// 从序列列表构建 batch
    pub fn compose(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        match phase {
            Phase::Prefill => self.compose_prefill_batch(sequences),
            Phase::Decode => self.compose_decode_batch(sequences),
        }
    }
    
    fn compose_prefill_batch(&self, sequences: Vec<Sequence>) -> Batch {
        // Prefill batch: 可能包含不同长度的序列
        // 按长度分组以优化内存访问模式
        let mut batches: Vec<Batch> = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_tokens = 0;
        
        for seq in sequences {
            let seq_tokens = seq.tokens.len().saturating_sub(seq.num_computed_tokens);
            
            if current_tokens + seq_tokens > self.config.max_token_budget 
                || current_batch.len() >= self.config.max_batch_size {
                // 当前 batch 已满，创建新 batch
                if !current_batch.is_empty() {
                    batches.push(self.build_batch_from_sequences(&current_batch, Phase::Prefill));
                    current_batch.clear();
                    current_tokens = 0;
                }
            }
            
            current_tokens += seq_tokens;
            current_batch.push(seq);
        }
        
        // 处理剩余序列
        if !current_batch.is_empty() {
            batches.push(self.build_batch_from_sequences(&current_batch, Phase::Prefill));
        }
        
        // 返回第一个 batch（后续 batch 由调度器下次迭代处理）
        batches.into_iter().next().unwrap_or_default()
    }
    
    fn compose_decode_batch(&self, sequences: Vec<Sequence>) -> Batch {
        // Decode batch: 所有序列都是 1 token，直接截断到 batch_size
        let batch_size = sequences.len().min(self.config.max_batch_size);
        let batch_seqs: Vec<_> = sequences.into_iter().take(batch_size).collect();
        
        self.build_batch_from_sequences(&batch_seqs, Phase::Decode)
    }
    
    fn build_batch_from_sequences(&self, sequences: &[Sequence], phase: Phase) -> Batch {
        let seq_ids: Vec<_> = sequences.iter().map(|s| s.id).collect();
        let input_tokens: Vec<_> = sequences.iter().map(|s| {
            match phase {
                Phase::Prefill => {
                    // Prefill: 返回剩余未计算的 tokens
                    let start = s.num_computed_tokens;
                    s.tokens[start..].to_vec()
                }
                Phase::Decode => {
                    // Decode: 只返回最后一个 token
                    s.tokens.last().copied().into_iter().collect()
                }
            }
        }).collect();
        
        let positions: Vec<_> = sequences.iter().map(|s| {
            let start = s.num_computed_tokens;
            (start..start + input_tokens[s.id as usize % input_tokens.len()].len()).collect()
        }).collect();
        
        Batch {
            seq_ids,
            input_tokens,
            positions,
            // ... 其他字段
            phase,
        }
    }
}
```

#### 2.2.6 PrefixCache - Radix Tree 实现

**职责**: 高效的前缀匹配和缓存管理

```rust
/// Radix Tree 节点
pub struct RadixNode {
    /// 该节点代表的 token 序列（路径前缀）
    tokens: Vec<TokenId>,
    /// 对应的 KV blocks
    blocks: Option<Arc<Vec<BlockId>>>,
    /// 子节点：key 是下一个 token
    children: HashMap<TokenId, Box<RadixNode>>,
    /// 访问次数（用于 LRU）
    access_count: u64,
    /// 最后访问时间
    last_access: Instant,
    /// 是否是完整缓存条目
    is_complete: bool,
}

pub struct RadixTree {
    root: RadixNode,
    config: PrefixCacheConfig,
    /// 总条目数
    entry_count: usize,
    /// 总 block 数
    total_blocks: usize,
}

impl RadixTree {
    /// 查找最长前缀匹配
    pub fn longest_prefix_match(&self, tokens: &[TokenId]) -> Option<PrefixMatchResult> {
        let mut node = &self.root;
        let mut matched_len = 0;
        let mut matched_blocks = None;
        
        for (i, &token) in tokens.iter().enumerate() {
            if let Some(child) = node.children.get(&token) {
                matched_len = i + 1;
                node = child;
                if node.is_complete {
                    matched_blocks = node.blocks.clone();
                }
            } else {
                break;
            }
        }
        
        matched_blocks.map(|blocks| PrefixMatchResult {
            blocks,
            matched_tokens: matched_len,
        })
    }
    
    /// 插入新的缓存条目
    pub fn insert(&mut self, tokens: &[TokenId], blocks: Vec<BlockId>) {
        let mut node = &mut self.root;
        
        for &token in tokens {
            node = node.children.entry(token)
                .or_insert_with(|| Box::new(RadixNode::new()));
        }
        
        node.blocks = Some(Arc::new(blocks));
        node.is_complete = true;
        node.last_access = Instant::now();
        self.entry_count += 1;
    }
    
    /// 访问缓存（更新 LRU）
    pub fn touch(&mut self, tokens: &[TokenId]) {
        if let Some(node) = self.find_node_mut(tokens) {
            node.access_count += 1;
            node.last_access = Instant::now();
        }
    }
    
    /// 执行 LRU 淘汰
    pub fn evict_lru(&mut self, max_entries: usize) {
        if self.entry_count <= max_entries {
            return;
        }
        
        let to_evict = self.entry_count - max_entries;
        // 按 last_access 排序，淘汰最旧的
        self.evict_oldest(to_evict);
    }
}

pub struct PrefixMatchResult {
    pub blocks: Arc<Vec<BlockId>>,
    pub matched_tokens: usize,
}
```

**复杂度对比**:
| 操作 | HashMap (当前) | Radix Tree (新) |
|------|----------------|-----------------|
| 精确匹配 | O(1) | O(k) |
| 前缀匹配 | O(n) | O(k) |
| 插入 | O(1) | O(k) |
| 空间 | O(n) | O(共享前缀) |

其中 k 是 token 序列长度，n 是缓存条目数。

---

## 3. Traits 层重构

### 3.1 Batch 结构增强

当前 `Batch` 需要支持新架构，修改 `vllm_traits`:

```rust
// crates/traits/src/lib.rs

/// 批次阶段
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchPhase {
    Prefill,
    Decode,
    Mixed, // 向后兼容
}

/// 增强的 Batch 结构
pub struct Batch {
    pub seq_ids: Vec<SeqId>,
    pub input_tokens: Vec<Vec<TokenId>>,
    pub positions: Vec<Vec<usize>>,
    pub kv_block_ids: Vec<Vec<BlockId>>,
    pub num_computed_tokens: Vec<usize>,
    pub is_prefill: Vec<bool>,
    // 新增字段
    pub phase: BatchPhase,
    /// 批次总 token 数（用于性能分析）
    pub total_tokens: usize,
    /// 最大序列长度（用于内存规划）
    pub max_seq_len: usize,
}

/// 批次构建器 trait（可选扩展点）
pub trait BatchBuilder {
    fn add_sequence(&mut self, seq: &Sequence) -> Result<()>;
    fn build(self) -> Batch;
    fn phase(&self) -> BatchPhase;
}

/// 调度器状态 trait（用于监控和诊断）
pub trait SchedulerState {
    fn waiting_count(&self) -> usize;
    fn running_count(&self) -> usize;
    fn current_phase(&self) -> Option<BatchPhase>;
    fn memory_usage(&self) -> (usize, usize); // used, total
}
```

### 3.2 ModelBackend 适配

ModelBackend 需要支持分阶段处理：

```rust
/// 模型后端扩展 trait
pub trait ModelBackendExt: ModelBackend {
    /// 批量前向传播（自动处理 prefill/decode）
    fn forward_batch(&self, batch: &Batch) -> Result<BatchOutput> {
        match batch.phase {
            BatchPhase::Prefill => self.forward_prefill(batch),
            BatchPhase::Decode => self.forward_decode(batch),
            BatchPhase::Mixed => self.forward(batch), // 向后兼容
        }
    }
    
    /// Prefill 阶段专用（可并行处理长序列）
    fn forward_prefill(&self, batch: &Batch) -> Result<BatchOutput>;
    
    /// Decode 阶段专用（优化单 token 生成）
    fn forward_decode(&self, batch: &Batch) -> Result<BatchOutput>;
}
```

---

## 4. 数据流

### 4.1 请求处理流程

```
Request Arrives
      │
      ▼
┌─────────────┐
│ PrefixCache │◄──── Radix Tree 查找
│ (Radix Tree)│
└──────┬──────┘
       │ 命中: 复用 blocks
       │ 未命中: 分配新 blocks
       ▼
┌─────────────┐
│ RequestQueue│◄──── 按策略计算优先级
│  (Indexed)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│PhaseScheduler│◄──── 选择 Prefill/Decode
│             │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│BatchComposer │◄---- 构建同阶段 batch
│             │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ModelBackend │◄---- forward_batch()
│             │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Update    │◄---- 更新序列状态，回写 KV cache
│             │
└─────────────┘
```

### 4.2 调度决策流程

```rust
fn schedule_next_batch(&mut self) -> Option<Batch> {
    // 1. 检查当前阶段是否应该切换
    let phase = self.phase_scheduler.select_phase(&self.state);
    
    // 2. 从对应阶段提取候选序列
    let candidates = self.request_queue.drain_by_phase(phase);
    
    // 3. 应用调度策略排序
    let sorted = self.policy_engine.sort(candidates, &self.context);
    
    // 4. 检查内存是否足够
    let memory_ok = self.memory_orchestrator.check_capacity(&sorted);
    if !memory_ok {
        // 执行抢占
        let victims = self.memory_orchestrator.select_victims(&sorted);
        self.preempt_sequences(victims);
    }
    
    // 5. 使用 BatchComposer 构建 batch
    let batch = self.batch_composer.compose(sorted, phase);
    
    // 6. 更新调度器状态
    self.state.record_batch(&batch);
    
    Some(batch)
}
```

---

## 5. 配置变更

### 5.1 新增配置项

```rust
// crates/core/src/types.rs

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    // 现有字段...
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_consecutive_decode: u32,
    pub enable_pd_separation: bool,
    pub prefill_chunk_size: usize,
    pub decode_preference_ratio: f32,
    pub enable_priority_scheduling: bool,
    pub enable_dynamic_batching: bool,
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    
    // 新增字段
    /// 调度策略类型
    pub scheduling_policy: SchedulingPolicyType,
    /// 策略配置（序列化存储）
    pub policy_config: PolicyConfig,
    /// Phase 切换策略
    pub phase_switch_policy: PhaseSwitchPolicy,
    /// 前缀缓存配置
    pub prefix_cache_config: PrefixCacheConfig,
    /// 内存池配置
    pub memory_pool_config: MemoryPoolConfig,
}

#[derive(Clone, Debug)]
pub enum SchedulingPolicyType {
    Fcfs,
    Sjf,
    Priority,
}

#[derive(Clone, Debug, Default)]
pub struct PolicyConfig {
    /// SJF: 用户优先级权重
    pub sjf_priority_weight: f32,
    /// SJF: 剩余工作量权重
    pub sjf_remaining_work_weight: f32,
    /// Priority: 老化因子
    pub priority_aging_factor: f32,
    /// Priority: 优先级级别数
    pub priority_levels: u8,
}

#[derive(Clone, Debug)]
pub struct PhaseSwitchPolicy {
    /// 连续 decode 多少轮后强制切换
    pub max_consecutive_decode: u32,
    /// Prefill 队列优先级阈值
    pub prefill_priority_threshold: usize,
    /// 最小 decode 批次大小
    pub min_decode_batch_size: usize,
}

#[derive(Clone, Debug)]
pub struct PrefixCacheConfig {
    /// 最大条目数
    pub max_entries: usize,
    /// 最大 block 数
    pub max_blocks: usize,
    /// 启用 Radix Tree
    pub enable_radix_tree: bool,
}

#[derive(Clone, Debug)]
pub struct MemoryPoolConfig {
    /// 预分配的小块大小列表
    pub preallocated_sizes: Vec<usize>,
    /// 每种大小的预分配数量
    pub preallocated_count: usize,
}
```

---

## 6. 错误处理

### 6.1 错误类型

```rust
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("内存不足，需要 {needed} blocks，可用 {available}")]
    OutOfMemory { needed: usize, available: usize },
    
    #[error("序列 {seq_id} 不存在")]
    SequenceNotFound { seq_id: SeqId },
    
    #[error("无效的调度策略配置: {0}")]
    InvalidPolicyConfig(String),
    
    #[error("批处理构建失败: {0}")]
    BatchCompositionError(String),
    
    #[error("前缀缓存操作失败: {0}")]
    PrefixCacheError(String),
}

pub type Result<T> = std::result::Result<T, SchedulerError>;
```

### 6.2 错误恢复策略

| 错误场景 | 恢复策略 |
|----------|----------|
| 内存不足 | 触发抢占，释放低优先级序列 |
| 无效策略配置 | 回退到默认 FCFS 策略，记录警告 |
| Batch 构建失败 | 减小 batch 大小重试 |
| Prefix Cache 失败 | 禁用缓存，直接分配新 blocks |

---

## 7. 测试策略

### 7.1 单元测试

```rust
#[cfg(test)]
mod tests {
    // RequestQueue 测试
    #[test]
    fn test_request_queue_o1_operations() {
        // 验证 O(1) 查找和删除
    }
    
    // PhaseScheduler 测试
    #[test]
    fn test_phase_switch_policy() {
        // 验证阶段切换逻辑
    }
    
    // PolicyEngine 测试
    #[test]
    fn test_scheduling_policies() {
        // 测试 FCFS、SJF、Priority 的正确性
    }
    
    // Radix Tree 测试
    #[test]
    fn test_radix_tree_prefix_match() {
        // 验证前缀匹配正确性和复杂度
    }
    
    // BatchComposer 测试
    #[test]
    fn test_batch_phase_isolation() {
        // 验证 batch 不包含混合阶段
    }
}
```

### 7.2 集成测试

```rust
// tests/scheduler_integration.rs

#[test]
fn test_strict_pd_separation() {
    // 验证 Prefill batch 只包含 prefill 序列
    // 验证 Decode batch 只包含 decode 序列
}

#[test]
fn test_policy_switching() {
    // 验证运行时策略切换
}

#[test]
fn test_prefix_cache_performance() {
    // 对比 Radix Tree vs HashMap 性能
}

#[test]
fn test_memory_preemption() {
    // 验证内存压力下抢占行为
}
```

### 7.3 性能基准

```rust
// benches/scheduler_bench.rs

fn bench_prefix_cache_lookup(c: &mut Criterion) {
    // 对比新旧实现
}

fn bench_batch_building(c: &mut Criterion) {
    // 测量 batch 构建时间
}

fn bench_policy_sorting(c: &mut Criterion) {
    // 测量策略排序性能
}
```

---

## 8. 迁移计划

### 8.1 阶段划分

**Phase 1: 基础设施** (2-3 天)
- [ ] 创建新的模块结构
- [ ] 实现 RequestQueue (索引化)
- [ ] 实现基础 Policy trait 和 FCFS
- [ ] 添加单元测试

**Phase 2: 核心调度器** (3-4 天)
- [ ] 实现 PhaseScheduler
- [ ] 实现 BatchComposer
- [ ] 严格 P/D 分离
- [ ] 集成测试

**Phase 3: 高级特性** (2-3 天)
- [ ] 实现 SJF 和 Priority 策略
- [ ] 实现 Radix Tree PrefixCache
- [ ] 策略动态切换
- [ ] 性能基准

**Phase 4: Traits 重构** (2-3 天)
- [ ] 重构 Batch 结构
- [ ] 添加 BatchPhase
- [ ] 更新 ModelBackend trait
- [ ] 跨 crate 集成测试

**Phase 5: 替换与清理** (1-2 天)
- [ ] 替换旧 SchedulerEngine
- [ ] 删除废弃代码
- [ ] 更新文档
- [ ] CI 验证

### 8.2 回滚策略

- 每个 Phase 独立分支，可单独回滚
- 保持旧实现并行运行（通过 feature flag）
- 关键路径添加兼容性适配层

---

## 9. 风险与缓解

| 风险 | 影响 | 可能性 | 缓解措施 |
|------|------|--------|----------|
| Radix Tree 实现复杂 | 高 | 中 | 使用成熟库或参考开源实现 |
| P/D 分离导致饥饿 | 中 | 中 | PhaseSwitchPolicy 参数调优 |
| Traits Breaking Change | 中 | 高 | 提前通知，提供迁移指南 |
| 性能回归 | 高 | 低 | 完善基准测试，分阶段验证 |
| 内存泄漏 | 高 | 低 | 强化测试，使用 sanitizer |

---

## 10. 参考实现

### 10.1 Radix Tree 参考
- vLLM 官方 Prefix Caching 实现
- Linux Kernel Radix Tree

### 10.2 P/D 分离参考
- Orca: Continuous Batching
- vLLM: PagedAttention

---

## 11. 待决策项

1. **Radix Tree 实现**: 自建 vs 使用现有 crate（如 `radix_trie`）
2. **策略序列化**: JSON vs 自定义格式
3. **监控指标**: 需要暴露哪些 Prometheus 指标
4. **配置热加载**: 是否支持运行时配置更新

---

**文档版本**: 1.0  
**下次评审日期**: 2025-04-12
