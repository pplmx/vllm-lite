use rand::Rng;
use super::r#trait::{Model, ModelOutput};
use crate::types::batch::Batch;

pub struct FakeModel {
    vocab_size: usize,
}

impl FakeModel {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl Model for FakeModel {
    fn forward(&self, batch: &Batch) -> ModelOutput {
        let mut rng = rand::thread_rng();
        let logits = (0..batch.input_tokens.len())
            .map(|_| rng.gen::<f32>())
            .collect();

        ModelOutput { logits }
    }
}
