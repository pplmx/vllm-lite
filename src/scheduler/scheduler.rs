use crate::types::{
    request::Request,
    sequence::{Sequence, Status},
    batch::Batch,
};

pub struct Scheduler {
    pub sequences: Vec<Sequence>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self { sequences: vec![] }
    }

    pub fn add_request(&mut self, req: Request) {
        self.sequences.push(Sequence {
            id: req.id,
            tokens: req.prompt,
            kv_blocks: vec![],
            status: Status::Prefill,
        });
    }

    pub fn build_batch(&mut self) -> Batch {
        let mut input_tokens = vec![];
        let mut seq_ids = vec![];
        let mut kv_map = vec![];

        for seq in &self.sequences {
            if seq.status != Status::Finished {
                let token = *seq.tokens.last().unwrap();
                input_tokens.push(token);
                seq_ids.push(seq.id);
                kv_map.push(seq.kv_blocks.clone());
            }
        }

        Batch {
            input_tokens,
            seq_ids,
            kv_map,
        }
    }

    pub fn update(&mut self, tokens: Vec<u32>) {
        for (seq, token) in self.sequences.iter_mut().zip(tokens) {
            if seq.status == Status::Finished {
                continue;
            }

            seq.tokens.push(token);

            if seq.tokens.len() > 50 {
                seq.status = Status::Finished;
            } else {
                seq.status = Status::Decoding;
            }
        }
    }
}
