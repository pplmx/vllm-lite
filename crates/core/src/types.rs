pub type TokenId = u32;
pub type SeqId = u64;
pub type BlockId = usize;

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
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sequence {
    pub id: SeqId,
    pub tokens: Vec<TokenId>,
    pub num_computed_tokens: usize,
    pub status: Status,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Status {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

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

pub struct BatchOutput {
    pub seq_ids: Vec<SeqId>,
    pub next_tokens: Vec<TokenId>,
}
