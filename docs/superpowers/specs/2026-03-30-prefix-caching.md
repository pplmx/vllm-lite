# vLLM-lite Prefix Caching

## 1. Overview

实现 Prefix Caching, 让相同 token 序列的请求共享 KV cache, 避免重复计算。

**目标:**
- 多个请求如果有相同的前缀 (prompt + generated tokens), 共享 KV blocks
- 基于 hash 的快速查找
- LRU 淘汰策略

## 2. 核心概念

### CacheKey

```rust
pub type CacheKey = u64;

fn hash_tokens(tokens: &[TokenId]) -> CacheKey {
    // FxHash 或 AHash, 简单高效
    tokens.iter().fold(0u64, |acc, &t| acc.wrapping_mul(31).wrapping_add(t as u64))
}
```

### CachedEntry

```rust
pub struct CachedEntry {
    pub key: CacheKey,
    pub blocks: Vec<BlockId>,
    pub token_count: usize,
    pub last_access: Instant,
}

pub struct PrefixCache {
    entries: HashMap<CacheKey, CachedEntry>,
    lru_order: Vec<CacheKey>,  // 最近使用排最前
    block_refs: HashMap<BlockId, usize>,  // 每个 block 的引用计数
}
```

## 3. 核心操作

### 3.1 查找

```rust
impl PrefixCache {
    pub fn get(&self, key: CacheKey) -> Option<&CachedEntry> {
        // 找到后更新 LRU
        if let Some(entry) = self.entries.get(&key) {
            self.move_to_front(key);
            Some(entry)
        } else {
            None
        }
    }
}
```

### 3.2 插入

```rust
pub fn insert(&mut self, key: CacheKey, blocks: Vec<BlockId>, token_count: usize) {
    // 增加引用计数
    for &block in &blocks {
        *self.block_refs.entry(block).or_insert(0) += 1;
    }
    
    // 插入缓存
    let entry = CachedEntry {
        key,
        blocks,
        token_count,
        last_access: Instant::now(),
    };
    self.entries.insert(key, entry);
    self.lru_order.push(key);
}
```

### 3.3 淘汰 (LRU)

```rust
pub fn evict(&mut self, allocator: &mut BlockAllocator) {
    while let Some(oldest_key) = self.lru_order.pop_front() {
        if let Some(entry) = self.entries.remove(&oldest_key) {
            // 引用计数减 1
            for &block in &entry.blocks {
                if let Some(count) = self.block_refs.get_mut(&block) {
                    *count -= 1;
                    if *count == 0 {
                        allocator.free(&[block]);
                        self.block_refs.remove(&block);
                    }
                }
            }
            break;  // 淘汰一个就够
        }
    }
}
```

## 4. Scheduler 集成

### 4.1 数据结构变更

```rust
pub struct Scheduler {
    // ... existing fields
    pub prefix_cache: PrefixCache,  // 新增
}
```

### 4.2 add_request 变更

```rust
pub fn add_request(&mut self, req: Request) -> SeqId {
    // 计算 prompt hash
    let key = hash_tokens(&req.prompt);
    
    // 查缓存
    if let Some(entry) = self.prefix_cache.get(key) {
        // 命中! 复用 blocks
        let seq = Sequence {
            id,
            tokens: req.prompt,
            kv_blocks: entry.blocks.clone(),
            num_computed_tokens: entry.token_count,
            status: Status::Decoding,  // 直接 decode
            // ...
        };
    } else {
        // 未命中, 正常处理
        // ...
    }
}
```

### 4.3 update 变更

```rust
pub fn update(&mut self, seq_ids: &[SeqId], next_tokens: &[TokenId], input_counts: &[usize]) {
    // ... existing logic
    
    // 对于新完成的序列, 存入缓存
    for seq in finished_sequences {
        let key = hash_tokens(&seq.tokens);
        self.prefix_cache.insert(key, seq.kv_blocks.clone(), seq.tokens.len());
    }
}
```

## 5. 完整流程

```
1. 请求到达
   └─ compute hash(prompt)
   └─ check prefix_cache.get(hash)

2. 缓存命中
   └─ 复用 blocks, 直接 decode
   └─ 更新 LRU

3. 缓存未命中
   └─ 正常 prefill
   └─ build_batch, forward, update

4. 序列完成
   └─ compute hash(full_tokens)
   └─ prefix_cache.insert(hash, blocks)

5. OOM 时
   └─ prefix_cache.evict() → 释放 blocks
```

## 6. 测试场景

### 测试 1: 相同 prompt 命中

```
请求1: "Hello" → 缓存 miss → prefill → cache ["Hello"]
请求2: "Hello" → 缓存 hit! → 直接 decode
```

### 测试 2: 前缀命中

```
请求1: "Hello world" → cache ["Hello world"]
请求2: "Hello world how are you" → 复用 ["Hello world"] 的 blocks，只需 prefill " how are you"
```

**实现方案: 前缀匹配**

当完整 hash 未命中时，尝试查找最长前缀匹配：

1. 尝试对 prompt 的所有前缀计算 hash
2. 从长到短，找到第一个匹配的缓存条目
3. 复用已分配的 KV blocks，只需 prefill 额外部分

```rust
fn find_prefix_match(&self, tokens: &[TokenId]) -> Option<&CachedEntry> {
    // 从长到短尝试所有前缀
    for prefix_len in (1..=tokens.len()).rev() {
        let prefix = &tokens[..prefix_len];
        let key = hash_tokens(prefix);
        if let Some(entry) = self.entries.get(&key) {
            return Some(entry);
        }
    }
    None
}
```

**关键点:**
- 缓存的 key 是完整序列的 hash
- 从最长前缀开始匹配，确保复用最多已计算的 KV
- 复用已分配的 KV blocks，只需 prefill 额外部分

### 测试 3: LRU 淘汰

```
缓存 100 个序列
新请求进来, OOM
→ 淘汰最老的缓存
→ 新请求正常执行
```

## 7. 实现计划

### Phase 1 (已完成)
- [x] 修改 BlockAllocator: 增加引用计数
- [x] 实现 PrefixCache 结构
- [x] 实现 get/insert/evict
- [x] Scheduler 集成
- [x] 测试

### 前缀命中 (当前)
- [ ] PrefixCache 添加 find_prefix_match 方法
- [ ] Scheduler add_request 支持前缀命中
- [ ] 处理 num_computed_tokens 正确设置
- [ ] 测试前缀命中场景

## 8. 边界情况

1. **Hash 碰撞**: 暂不处理, 概率极低
2. **空序列**: 不缓存
3. **单 token**: 可以缓存, 但收益小
4. **并发**: Scheduler 是单线程, 不需要锁