# vLLM-lite Phase 2+3: Continuous Batching

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Status:** ⚠️ Partially implemented (simplified version in commit d22d3e3, see `2026-03-30-continuous-batching.md`)

**Goal:** Multi-sequence continuous batching with chunked prefill, decode-priority scheduling, and SSE streaming API.

**Architecture:** Refactor Scheduler to support budget-driven batching (max_num_seqs + max_num_batched_tokens). Engine gains channel-based streaming. Server converts to SSE.

**Tech Stack:** Rust, tokio, axum, eventsource-stream

**Spec:** `docs/superpowers/specs/2026-03-29-vllm-lite-design.md` Sections 5, 12

---

## Key Design: Budget-Driven Batching

Each step has a token budget (`max_num_batched_tokens`, default 4096):
1. **Decode first**: Each running decode sequence costs 1 token. Add all decode sequences.
2. **Prefill with remaining budget**: Each prefill sequence costs `num_uncomputed_tokens`. Add prefill sequences chunked to fit remaining budget.
3. **New sequences**: Waiting sequences join when budget allows.

---

### Task P2-1: Refactor Scheduler for continuous batching

**Files:**
- Modify: `crates/core/src/scheduler.rs`
- Modify: `crates/core/src/types.rs` (add SchedulerConfig)

- [ ] **Step 1: Add SchedulerConfig to types.rs**

Append to `crates/core/src/types.rs`:
```rust
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 4096,
        }
    }
}
```

- [ ] **Step 2: Refactor Scheduler**

Rewrite `crates/core/src/scheduler.rs`:
```rust
use crate::types::{Batch, Request, SchedulerConfig, SeqId, Sequence, Status, TokenId};
use std::collections::VecDeque;

pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    next_seq_id: SeqId,
    config: SchedulerConfig,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler {
    pub fn new() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    pub fn with_config(config: SchedulerConfig) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            finished: Vec::new(),
            next_seq_id: 1,
            config,
        }
    }

    pub fn add_request(&mut self, req: Request) -> SeqId {
        let id = if req.id == 0 {
            let id = self.next_seq_id;
            self.next_seq_id += 1;
            id
        } else {
            req.id
        };

        let seq = Sequence {
            id,
            tokens: req.prompt,
            num_computed_tokens: 0,
            status: Status::Waiting,
            max_tokens: req.max_tokens,
            sampling_params: req.sampling_params,
        };
        self.waiting.push_back(seq);
        id
    }

    /// Admit waiting sequences into running, respecting max_num_seqs.
    fn admit_waiting(&mut self) {
        while self.running.len() < self.config.max_num_seqs {
            match self.waiting.pop_front() {
                Some(mut seq) => {
                    seq.status = Status::Prefilling;
                    self.running.push(seq);
                }
                None => break,
            }
        }
    }

    /// How many uncomputed tokens a sequence has.
    fn pending_tokens(seq: &Sequence) -> usize {
        if seq.status == Status::Prefilling {
            seq.tokens.len() - seq.num_computed_tokens
        } else if seq.status == Status::Decoding {
            1
        } else {
            0
        }
    }

    pub fn build_batch(&mut self) -> Batch {
        self.admit_waiting();

        let mut seq_ids = vec![];
        let mut input_tokens = vec![];
        let mut positions = vec![];
        let mut budget = self.config.max_num_batched_tokens;

        // Phase 1: Decode sequences first (1 token each)
        for seq in &self.running {
            if seq.status != Status::Decoding {
                continue;
            }
            if budget == 0 {
                break;
            }

            let last = *seq.tokens.last().unwrap();
            let pos = seq.tokens.len() - 1;
            seq_ids.push(seq.id);
            input_tokens.push(vec![last]);
            positions.push(vec![pos]);
            budget = budget.saturating_sub(1);
        }

        // Phase 2: Prefill sequences with remaining budget
        for seq in &self.running {
            if seq.status != Status::Prefilling {
                continue;
            }
            if budget == 0 {
                break;
            }

            let start = seq.num_computed_tokens;
            let remaining = seq.tokens.len() - start;
            let chunk_size = remaining.min(budget);

            let tokens = seq.tokens[start..start + chunk_size].to_vec();
            let pos: Vec<usize> = (start..start + chunk_size).collect();

            seq_ids.push(seq.id);
            input_tokens.push(tokens);
            positions.push(pos);
            budget = budget.saturating_sub(chunk_size);
        }

        Batch {
            seq_ids,
            input_tokens,
            positions,
        }
    }

    pub fn update(&mut self, seq_ids: &[SeqId], next_tokens: &[TokenId]) {
        for (seq_id, token) in seq_ids.iter().zip(next_tokens) {
            if let Some(seq) = self.running.iter_mut().find(|s| s.id == *seq_id) {
                if seq.status == Status::Prefilling {
                    // Count how many tokens were in the batch for this seq
                    // The token was appended, so num_computed = tokens.len() - 1
                    // (the -1 because the new token hasn't been counted yet)
                    // Actually: tokens already has the prompt + appended token
                    // num_computed should track what the model has seen
                    seq.num_computed_tokens = seq.tokens.len();
                    // Check if prefill is complete
                    if seq.num_computed_tokens >= seq.tokens.len() {
                        seq.status = Status::Decoding;
                    }
                    // else: still chunked prefill, stays Prefilling
                }

                seq.tokens.push(*token);

                if seq.tokens.len() >= seq.max_tokens {
                    seq.status = Status::Finished;
                }
            }
        }

        // Move finished → finished list
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].status == Status::Finished {
                let seq = self.running.remove(i);
                self.finished.push(seq);
            } else {
                i += 1;
            }
        }
    }

    pub fn has_pending(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    pub fn finished_sequences(&self) -> &[Sequence] {
        &self.finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_request_prefill() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![10, 20, 30], 5));

        let batch = sched.build_batch();
        assert_eq!(batch.seq_ids.len(), 1);
        assert_eq!(batch.input_tokens[0], vec![10, 20, 30]);
        assert_eq!(batch.positions[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_decode_after_prefill() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![10, 20, 30], 5));

        let batch = sched.build_batch();
        sched.update(&batch.seq_ids, &[99]);

        let batch2 = sched.build_batch();
        assert_eq!(batch2.input_tokens[0], vec![99]);
        assert_eq!(batch2.positions[0], vec![3]);
    }

    #[test]
    fn test_multi_sequence_batch() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![10, 20], 5));
        sched.add_request(Request::new(2, vec![30, 40], 5));

        let batch = sched.build_batch();
        assert_eq!(batch.seq_ids.len(), 2);
        assert_eq!(batch.input_tokens[0], vec![10, 20]);
        assert_eq!(batch.input_tokens[1], vec![30, 40]);
    }

    #[test]
    fn test_decode_priority() {
        let mut sched = Scheduler::new();
        // First request: already in decode
        sched.add_request(Request::new(1, vec![10], 5));
        let batch = sched.build_batch();
        sched.update(&batch.seq_ids, &[99]); // now decoding

        // Second request: just added, still prefill
        sched.add_request(Request::new(2, vec![20, 30, 40], 5));

        let batch = sched.build_batch();
        // Decode seq (id=1) should come first
        assert_eq!(batch.seq_ids[0], 1);
        assert_eq!(batch.input_tokens[0], vec![99]);
        // Prefill seq (id=2) should come second
        assert_eq!(batch.seq_ids[1], 2);
        assert_eq!(batch.input_tokens[1], vec![20, 30, 40]);
    }

    #[test]
    fn test_chunked_prefill() {
        let config = SchedulerConfig {
            max_num_seqs: 256,
            max_num_batched_tokens: 3, // very small budget
        };
        let mut sched = Scheduler::with_config(config);
        sched.add_request(Request::new(1, vec![10, 20, 30, 40, 50], 10));

        // Step 1: chunk 3 tokens
        let batch = sched.build_batch();
        assert_eq!(batch.input_tokens[0], vec![10, 20, 30]);
        sched.update(&batch.seq_ids, &[99]);

        // Step 2: still prefill, 2 remaining tokens (prompt [10,20,30,40,50], computed 3)
        let batch = sched.build_batch();
        // After update: tokens are [10,20,30,40,50,99], num_computed = 5
        // So it should now be decoding
        assert_eq!(batch.input_tokens[0], vec![99]); // decode mode
    }

    #[test]
    fn test_max_num_seqs_limit() {
        let config = SchedulerConfig {
            max_num_seqs: 1,
            max_num_batched_tokens: 4096,
        };
        let mut sched = Scheduler::with_config(config);
        sched.add_request(Request::new(1, vec![10], 5));
        sched.add_request(Request::new(2, vec![20], 5));

        let batch = sched.build_batch();
        // Only 1 seq admitted due to max_num_seqs=1
        assert_eq!(batch.seq_ids.len(), 1);
        assert_eq!(sched.running_count(), 1);
        assert_eq!(sched.waiting_count(), 1);
    }

    #[test]
    fn test_finished_removed() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![10], 3)); // max_tokens=3

        let batch = sched.build_batch();
        sched.update(&batch.seq_ids, &[20]); // tokens: [10, 20]

        let batch = sched.build_batch();
        sched.update(&batch.seq_ids, &[30]); // tokens: [10, 20, 30] → finished

        assert!(!sched.has_pending());
        assert_eq!(sched.finished_sequences().len(), 1);
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p vllm-core -- scheduler`
Expected: 7 passed

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(core): refactor Scheduler for continuous batching with budget limits"
```

---

### Task P2-2: Engine streaming with channels

**Files:**
- Modify: `crates/core/src/engine.rs`
- Modify: `crates/core/src/types.rs` (add EngineMessage)

- [ ] **Step 1: Add EngineMessage to types.rs**

Append to `crates/core/src/types.rs`:
```rust
use tokio::sync::mpsc;

pub enum EngineMessage {
    AddRequest {
        request: Request,
        response_tx: mpsc::UnboundedSender<TokenId>,
    },
    Shutdown,
}
```

Note: This adds `tokio` as a dependency to core. Add to `crates/core/Cargo.toml`:
```toml
[dependencies]
thiserror = "2"
tokio = { version = "1", features = ["sync"] }
```

- [ ] **Step 2: Refactor Engine for streaming**

Rewrite `crates/core/src/engine.rs`:
```rust
use crate::error::Result;
use crate::scheduler::Scheduler;
use crate::types::{BatchOutput, EngineMessage, Request, SchedulerConfig, SeqId, TokenId};
use std::collections::HashMap;
use tokio::sync::mpsc;

pub trait ModelBackend: Send + Sync {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> Result<BatchOutput>;
}

pub struct Engine<M: ModelBackend> {
    pub scheduler: Scheduler,
    pub model: M,
    /// Per-sequence response channels
    response_txs: HashMap<SeqId, mpsc::UnboundedSender<TokenId>>,
}

impl<M: ModelBackend> Engine<M> {
    pub fn new(model: M) -> Self {
        Self {
            scheduler: Scheduler::new(),
            model,
            response_txs: HashMap::new(),
        }
    }

    pub fn with_config(model: M, config: SchedulerConfig) -> Self {
        Self {
            scheduler: Scheduler::with_config(config),
            model,
            response_txs: HashMap::new(),
        }
    }

    pub fn add_request(&mut self, req: Request, response_tx: mpsc::UnboundedSender<TokenId>) -> SeqId {
        let seq_id = self.scheduler.add_request(req);
        self.response_txs.insert(seq_id, response_tx);
        seq_id
    }

    /// Single step: build batch → forward → update → send tokens
    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let output = self
            .model
            .forward(&batch.seq_ids, &batch.input_tokens, &batch.positions)?;

        self.scheduler.update(&batch.seq_ids, &output.next_tokens);

        // Send tokens to response channels
        let mut results = Vec::new();
        for (seq_id, token) in batch.seq_ids.iter().zip(output.next_tokens.iter()) {
            if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.send(*token);
            }
            results.push((*seq_id, *token));
        }

        // Clean up channels for finished sequences
        for seq in self.scheduler.finished_sequences() {
            self.response_txs.remove(&seq.id);
        }

        Ok(results)
    }

    /// Process messages from channel, run steps until shutdown
    pub fn run(&mut self, mut msg_rx: mpsc::UnboundedReceiver<EngineMessage>) {
        loop {
            // Non-blocking: collect all pending messages
            while let Ok(msg) = msg_rx.try_recv() {
                match msg {
                    EngineMessage::AddRequest { request, response_tx } => {
                        self.add_request(request, response_tx);
                    }
                    EngineMessage::Shutdown => return,
                }
            }

            if self.scheduler.has_pending() {
                if let Err(e) = self.step() {
                    eprintln!("Engine step error: {}", e);
                }
            }

            // Small sleep to avoid busy-loop when idle
            // In production, use tokio::select! with notify
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    pub fn has_pending(&self) -> bool {
        self.scheduler.has_pending()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;

    struct StubModel {
        token_to_return: TokenId,
    }

    impl ModelBackend for StubModel {
        fn forward(
            &self,
            seq_ids: &[SeqId],
            _input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
        ) -> Result<BatchOutput> {
            Ok(BatchOutput {
                seq_ids: seq_ids.to_vec(),
                next_tokens: seq_ids.iter().map(|_| self.token_to_return).collect(),
            })
        }
    }

    #[test]
    fn test_engine_streaming() {
        let mut engine = Engine::new(StubModel { token_to_return: 42 });
        let (tx, mut rx) = mpsc::unbounded_channel();

        engine.add_request(Request::new(1, vec![10, 20], 4), tx);

        // Step 1: prefill
        let out = engine.step().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(rx.try_recv().unwrap(), 42);

        // Step 2: decode
        let out = engine.step().unwrap();
        assert_eq!(out[0], (1, 42));
        assert_eq!(rx.try_recv().unwrap(), 42);

        // Step 3: reaches max_tokens
        let out = engine.step().unwrap();
        assert_eq!(out[0], (1, 42));
        assert_eq!(rx.try_recv().unwrap(), 42);

        // Done
        assert!(!engine.has_pending());
    }

    #[test]
    fn test_engine_multi_request() {
        let mut engine = Engine::new(StubModel { token_to_return: 10 });
        let (tx1, mut rx1) = mpsc::unbounded_channel();
        let (tx2, mut rx2) = mpsc::unbounded_channel();

        engine.add_request(Request::new(1, vec![10], 3), tx1);
        engine.add_request(Request::new(2, vec![20], 3), tx2);

        // Step 1: both prefill
        engine.step().unwrap();
        assert_eq!(rx1.try_recv().unwrap(), 10);
        assert_eq!(rx2.try_recv().unwrap(), 10);

        // Step 2: both decode
        engine.step().unwrap();
        assert_eq!(rx1.try_recv().unwrap(), 10);
        assert_eq!(rx2.try_recv().unwrap(), 10);

        // Step 3: both finish
        engine.step().unwrap();
        assert_eq!(rx1.try_recv().unwrap(), 10);
        assert_eq!(rx2.try_recv().unwrap(), 10);

        assert!(!engine.has_pending());
    }
}
```

- [ ] **Step 3: Update types.rs**

Remove `EngineMessage` from types.rs if it was added there — it's now in engine.rs with the tokio import.

Actually, keep `EngineMessage` in types.rs and add tokio dep to core.

- [ ] **Step 4: Run tests**

Run: `cargo test -p vllm-core`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): add channel-based streaming to Engine"
```

---

### Task P2-3: Server SSE streaming

**Files:**
- Modify: `crates/server/src/api.rs`
- Modify: `crates/server/src/main.rs`
- Modify: `crates/server/Cargo.toml` (add eventsource-stream or use manual SSE)

- [ ] **Step 1: Rewrite api.rs for SSE**

`crates/server/src/api.rs`:
```rust
use axum::{
    extract::State,
    response::sse::{Event, Sse},
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::convert::Infallible;
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{EngineMessage, Request};
use vllm_model::fake::FakeModel;

pub type EngineHandle = mpsc::UnboundedSender<EngineMessage>;

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    100
}

#[derive(Serialize)]
struct CompletionChunk {
    id: String,
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    text: String,
    index: usize,
}

pub async fn completions(
    State(engine_tx): State<EngineHandle>,
    Json(req): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let prompt_tokens: Vec<u32> = req
        .prompt
        .split_whitespace()
        .enumerate()
        .map(|(i, _)| (i + 1) as u32)
        .collect();

    let max_tokens = prompt_tokens.len() + req.max_tokens;
    let request = Request::new(0, prompt_tokens, max_tokens);

    let (response_tx, mut response_rx) = mpsc::unbounded_channel();

    // Send request to engine
    engine_tx
        .send(EngineMessage::AddRequest {
            request,
            response_tx,
        })
        .unwrap();

    let stream = stream::unfold(response_rx, |mut rx| async move {
        match rx.recv().await {
            Some(token) => {
                let chunk = CompletionChunk {
                    id: "cmpl-0".to_string(),
                    choices: vec![Choice {
                        text: format!("token_{}", token),
                        index: 0,
                    }],
                };
                let data = serde_json::to_string(&chunk).unwrap();
                Some((Ok(Event::default().data(data)), rx))
            }
            None => {
                // Stream ended
                Some((Ok(Event::default().data("[DONE]")), rx))
            }
        }
    });

    Sse::new(stream)
}
```

- [ ] **Step 2: Update main.rs**

`crates/server/src/main.rs`:
```rust
mod api;

use axum::{routing::post, Router};
use std::collections::HashMap;
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{EngineMessage, SchedulerConfig};
use vllm_model::fake::FakeModel;

#[tokio::main]
async fn main() {
    let model = FakeModel::new(32000);
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
    };
    let mut engine = Engine::with_config(model, config);

    let (msg_tx, msg_rx) = mpsc::unbounded_channel::<EngineMessage>();

    // Spawn engine worker thread
    std::thread::spawn(move || {
        engine.run(msg_rx);
    });

    let app = Router::new()
        .route("/v1/completions", post(api::completions))
        .with_state(msg_tx);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("vllm-lite listening on http://0.0.0.0:8000");
    axum::serve(listener, app).await.unwrap();
}
```

- [ ] **Step 3: Update Cargo.toml dependencies**

`crates/server/Cargo.toml`:
```toml
[package]
name = "vllm-server"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "vllm-server"
path = "src/main.rs"

[dependencies]
vllm-core = { path = "../core" }
vllm-model = { path = "../model" }
tokio = { version = "1", features = ["full"] }
axum = { version = "0.8", features = ["sse"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
futures = "0.3"
```

- [ ] **Step 4: Verify compiles**

Run: `cargo check --workspace`
Expected: PASS

- [ ] **Step 5: Manual test**

```bash
cargo run -p vllm-server

# In another terminal:
curl -N -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello world", "max_tokens": 3, "stream": true}'
```

Expected: SSE stream with token events followed by `[DONE]`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(server): SSE streaming for /v1/completions"
```

---

### Task P2-4: Integration test + cleanup

**Files:**
- Create: `crates/core/tests/integration.rs`

- [ ] **Step 1: Write end-to-end integration test**

`crates/core/tests/integration.rs`:
```rust
use vllm_core::engine::{Engine, ModelBackend};
use vllm_core::error::Result;
use vllm_core::types::{BatchOutput, Request, SchedulerConfig, SeqId, TokenId};
use tokio::sync::mpsc;

struct IncrementModel;

impl ModelBackend for IncrementModel {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<BatchOutput> {
        // Returns incrementing token IDs per sequence
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|id| *id as TokenId).collect(),
        })
    }
}

#[test]
fn test_continuous_batching_with_streaming() {
    let config = SchedulerConfig {
        max_num_seqs: 2,
        max_num_batched_tokens: 100,
    };
    let mut engine = Engine::with_config(IncrementModel, config);

    let (tx1, mut rx1) = mpsc::unbounded_channel();
    let (tx2, mut rx2) = mpsc::unbounded_channel();

    // Add two requests
    engine.add_request(Request::new(1, vec![10, 20], 4), tx1);
    engine.add_request(Request::new(2, vec![30, 40, 50], 5), tx2);

    // Step 1: both prefill
    engine.step().unwrap();
    assert!(rx1.try_recv().is_ok());
    assert!(rx2.try_recv().is_ok());

    // Step 2: both decode
    engine.step().unwrap();
    assert!(rx1.try_recv().is_ok());
    assert!(rx2.try_recv().is_ok());

    // Step 3: req1 finishes (max_tokens=4), req2 continues
    engine.step().unwrap();
    assert!(rx1.try_recv().is_ok());
    assert!(rx2.try_recv().is_ok());

    // Step 4: req2 continues
    assert!(engine.has_pending());
    engine.step().unwrap();
    assert!(rx2.try_recv().is_ok());

    // Done
    assert!(!engine.has_pending());
}

#[test]
fn test_chunked_prefill_integration() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 2, // tiny budget
    };
    let mut engine = Engine::with_config(IncrementModel, config);

    let (tx, mut rx) = mpsc::unbounded_channel();
    engine.add_request(Request::new(1, vec![10, 20, 30, 40], 6), tx);

    // Step 1: chunk 2 tokens (budget=2)
    engine.step().unwrap();
    assert!(rx.try_recv().is_ok());

    // Step 2: chunk 2 more tokens
    engine.step().unwrap();
    assert!(rx.try_recv().is_ok());

    // Step 3: now decoding
    engine.step().unwrap();
    assert!(rx.try_recv().is_ok());

    // Step 4: decode
    engine.step().unwrap();
    assert!(rx.try_recv().is_ok());

    assert!(!engine.has_pending());
}
```

- [ ] **Step 2: Run full test suite**

Run: `cargo test --workspace`
Expected: all tests pass

- [ ] **Step 3: Clippy**

Run: `cargo clippy --workspace -- -D warnings`
Fix any warnings.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: add integration tests for continuous batching"
```
