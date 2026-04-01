use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::types::{BatchJob, BatchResultItem, BatchStatus};

pub struct BatchManager {
    jobs: Arc<RwLock<HashMap<String, BatchJob>>>,
}

impl BatchManager {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create_job(
        &self,
        endpoint: String,
        prompts: Vec<String>,
        model: Option<String>,
        max_tokens: Option<i64>,
        temperature: Option<f32>,
    ) -> String {
        let id = format!("batch_{}", Uuid::new_v4());
        let job = BatchJob::new(
            id.clone(),
            endpoint,
            prompts,
            model,
            max_tokens,
            temperature,
        );
        self.jobs.write().await.insert(id.clone(), job);
        id
    }

    pub async fn get_job(&self, id: &str) -> Option<BatchJob> {
        self.jobs.read().await.get(id).cloned()
    }

    pub async fn get_all_jobs(&self) -> Vec<BatchJob> {
        self.jobs.read().await.values().cloned().collect()
    }

    pub async fn update_job(&self, job: BatchJob) {
        self.jobs.write().await.insert(job.id.clone(), job);
    }

    pub async fn add_result(&self, job_id: &str, result: BatchResultItem) {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.results.push(result);
        }
    }

    pub async fn set_completed(&self, job_id: &str) {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.status = BatchStatus::Completed;
            job.completed_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64,
            );
        }
    }
}

impl Default for BatchManager {
    fn default() -> Self {
        Self::new()
    }
}
