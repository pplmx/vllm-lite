use crate::{
    scheduler::scheduler::Scheduler,
    model::r#trait::Model,
    sampling::sampler::greedy_sample,
};

pub struct Engine<M: Model> {
    pub scheduler: Scheduler,
    pub model: M,
}

impl<M: Model> Engine<M> {
    pub fn step(&mut self) {
        let batch = self.scheduler.build_batch();
        let output = self.model.forward(&batch);
        let tokens = greedy_sample(&output.logits);
        self.scheduler.update(tokens);
    }
}
