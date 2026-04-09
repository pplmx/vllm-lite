use std::time::Instant;

#[derive(Clone)]
pub struct SchedulerStats {
    pub total_batches: usize,
    pub total_prefill_requests: usize,
    pub total_decode_requests: usize,
    pub total_preemptions: usize,
    pub total_evictions: usize,
    pub avg_batch_size: f64,
    pub last_batch_size: usize,
    pub batch_size_sum: u64,
    pub last_update: Instant,
}

impl Default for SchedulerStats {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulerStats {
    pub fn new() -> Self {
        Self {
            total_batches: 0,
            total_prefill_requests: 0,
            total_decode_requests: 0,
            total_preemptions: 0,
            total_evictions: 0,
            avg_batch_size: 0.0,
            last_batch_size: 0,
            batch_size_sum: 0,
            last_update: Instant::now(),
        }
    }

    pub fn record_batch(&mut self, batch_size: usize) {
        self.total_batches += 1;
        self.last_batch_size = batch_size;
        self.batch_size_sum += batch_size as u64;
        self.avg_batch_size = self.batch_size_sum as f64 / self.total_batches as f64;
        self.last_update = Instant::now();
    }

    pub fn record_prefill(&mut self) {
        self.total_prefill_requests += 1;
    }

    pub fn record_decode(&mut self) {
        self.total_decode_requests += 1;
    }

    pub fn record_preemption(&mut self) {
        self.total_preemptions += 1;
    }

    pub fn record_eviction(&mut self) {
        self.total_evictions += 1;
    }
}
