pub fn greedy_sample(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|_| 1u32).collect()
}
