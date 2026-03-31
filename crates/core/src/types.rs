use tokio::sync::mpsc;

pub type TokenId = u32;
pub type SeqId = u64;
pub type BlockId = usize;
pub use crate::kv_cache::BLOCK_SIZE;

#[derive(Clone, Debug)]
pub struct Request {
    pub id: SeqId,
    pub prompt: Vec<TokenId>,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
}

impl Request {
    pub fn new(id: SeqId, prompt: Vec<TokenId>, max_tokens: usize) -> Self {
        Self {
            id,
            prompt,
            max_tokens,
            sampling_params: SamplingParams::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub beam_width: usize,
    pub length_penalty: f32,
    pub max_retries: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            beam_width: 1,
            length_penalty: 0.6,
            max_retries: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sequence {
    pub id: SeqId,
    pub tokens: Vec<TokenId>,
    pub kv_blocks: Vec<BlockId>,
    pub num_computed_tokens: usize,
    pub prompt_len: usize,
    pub status: Status,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
    pub consecutive_decode_rounds: u32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Status {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

#[derive(Clone, Debug)]
pub struct Batch {
    pub seq_ids: Vec<SeqId>,
    pub input_tokens: Vec<Vec<TokenId>>,
    pub positions: Vec<Vec<usize>>,
}

impl Batch {
    pub fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct BatchOutput {
    pub seq_ids: Vec<SeqId>,
    pub next_tokens: Vec<TokenId>,
}

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_consecutive_decode: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 4096,
            max_consecutive_decode: 10,
        }
    }
}

use crate::metrics::MetricsSnapshot;

pub enum EngineMessage {
    AddRequest {
        request: Request,
        response_tx: mpsc::UnboundedSender<TokenId>,
    },
    GetMetrics {
        response_tx: mpsc::UnboundedSender<MetricsSnapshot>,
    },
    Shutdown,
}
