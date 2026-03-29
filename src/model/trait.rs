use crate::types::batch::Batch;

pub struct ModelOutput {
    pub logits: Vec<f32>,
}

pub trait Model: Send + Sync {
    fn forward(&self, batch: &Batch) -> ModelOutput;
}
