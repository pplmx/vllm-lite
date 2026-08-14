//! `OpenAI` Batch API request lifecycle: enqueue, dispatch to engine, poll status, cancel, collect results.
//!
//! `BatchManager` owns the per-batch state machine (`Pending` →
//! `InProgress` → `Completed`/`Failed`/`Cancelled`) and is shared across
//! the `/v1/batches` handlers via `Arc`.
#![allow(clippy::module_name_repetitions)]
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::types::{BatchEndpoint, BatchJob, BatchResultItem, BatchStatus};

#[derive(Debug)]
/// Manager for Batch. Owns the underlying resource, coordinates concurrent access, and exposes a thread-safe public API.
pub struct BatchManager {
    jobs: Arc<RwLock<HashMap<String, BatchJob>>>,
}

impl BatchManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create_job(
        &self,
        endpoint: BatchEndpoint,
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

    /// Set completed.
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    pub async fn set_completed(&self, job_id: &str) {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.status = BatchStatus::Completed;
            job.completed_at = Some(
                // invariant: monotonic clock is always >= UNIX_EPOCH.
                i64::try_from(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        // invariant: pre-conditions make this infallible at this call site.
                        .unwrap()
                        .as_secs(),
                )
                .unwrap_or(i64::MAX),
            );
        }
    }

    /// Transition a job to `InProgress` the first time its worker picks it
    /// up. Returns `false` (and leaves the status untouched) when the job is
    /// missing **or already cancelled** — a race where
    /// [`Self::request_cancel`] won before the worker picked the job up must
    /// not be overwritten back to `InProgress` (the worker then sees the
    /// cancel flag, skips all prompts, and the job stays `Cancelled`).
    #[must_use]
    pub async fn mark_in_progress(&self, job_id: &str) -> bool {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            if job
                .cancel_requested
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                return false;
            }
            job.status = BatchStatus::InProgress;
            true
        } else {
            false
        }
    }

    /// Set a job to `Failed` (one or more requests errored). Records the
    /// completion timestamp so `get_batch` reports a terminal state; partial
    /// results remain readable via `get_batch_results`.
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    pub async fn set_failed(&self, job_id: &str) {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.status = BatchStatus::Failed;
            if job.completed_at.is_none() {
                job.completed_at = Some(
                    // invariant: monotonic clock is always >= UNIX_EPOCH.
                    i64::try_from(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            // invariant: pre-conditions make this infallible at this call site.
                            .unwrap()
                            .as_secs(),
                    )
                    .unwrap_or(i64::MAX),
                );
            }
        }
    }

    /// Signal a job to stop. Sets the cooperative `cancel_requested` flag
    /// so the worker (if running) stops dispatching new prompts and
    /// cancels its in-flight engine sequence; also transitions the status
    /// to `Cancelled` so `GET /v1/batches/{id}` reports it immediately.
    /// No-op for jobs already in a terminal state.
    pub async fn request_cancel(&self, job_id: &str) -> bool {
        let mut jobs = self.jobs.write().await;
        let result = match jobs.get_mut(job_id) {
            None => false,
            Some(job) => match job.status {
                BatchStatus::Pending | BatchStatus::InProgress => {
                    job.cancel_requested
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    job.status = BatchStatus::Cancelled;
                    true
                }
                BatchStatus::Completed | BatchStatus::Failed | BatchStatus::Cancelled => false,
            },
        };
        drop(jobs);
        result
    }

    /// Whether a user requested cancellation for `job_id` (regardless of
    /// the job's current status — the worker uses this as its stop signal).
    #[must_use]
    pub async fn is_cancelled(&self, job_id: &str) -> bool {
        let jobs = self.jobs.read().await;
        let cancelled = jobs.get(job_id).is_some_and(|j| {
            j.cancel_requested
                .load(std::sync::atomic::Ordering::Relaxed)
        });
        drop(jobs);
        cancelled
    }
}

impl Default for BatchManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_manager() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let jobs = rt.block_on(mgr.get_all_jobs());
        assert!(jobs.is_empty());
    }

    #[test]
    fn create_job_returns_unique_id() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let id1 = rt.block_on(mgr.create_job(
            BatchEndpoint::Chat,
            vec!["hello".into()],
            None,
            None,
            None,
        ));
        let id2 = rt.block_on(mgr.create_job(
            BatchEndpoint::Completion,
            vec!["world".into()],
            Some("gpt-3".into()),
            Some(100),
            Some(0.7),
        ));
        assert_ne!(id1, id2);
    }

    #[test]
    fn get_job_returns_none_for_missing() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let job = rt.block_on(mgr.get_job("nonexistent"));
        assert!(job.is_none());
    }

    #[test]
    fn create_and_get_job_round_trip() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let id = rt.block_on(mgr.create_job(
            BatchEndpoint::Chat,
            vec!["test prompt".into()],
            None,
            None,
            None,
        ));
        let job = rt.block_on(mgr.get_job(&id)).unwrap();
        assert_eq!(job.id, id);
        assert_eq!(job.status, BatchStatus::Pending);
    }

    #[test]
    fn get_all_jobs_returns_all() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let id1 =
            rt.block_on(mgr.create_job(BatchEndpoint::Chat, vec!["a".into()], None, None, None));
        let id2 = rt.block_on(mgr.create_job(
            BatchEndpoint::Completion,
            vec!["b".into()],
            None,
            None,
            None,
        ));
        let all = rt.block_on(mgr.get_all_jobs());
        assert_eq!(all.len(), 2);
        let ids: Vec<&str> = all.iter().map(|j| j.id.as_str()).collect();
        assert!(ids.contains(&id1.as_str()));
        assert!(ids.contains(&id2.as_str()));
    }

    #[test]
    fn update_job_modifies_existing() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let id = rt.block_on(mgr.create_job(
            BatchEndpoint::Chat,
            vec!["update me".into()],
            None,
            None,
            None,
        ));
        let mut job = rt.block_on(mgr.get_job(&id)).unwrap();
        job.status = BatchStatus::InProgress;
        rt.block_on(mgr.update_job(job));
        let updated = rt.block_on(mgr.get_job(&id)).unwrap();
        assert_eq!(updated.status, BatchStatus::InProgress);
    }

    #[test]
    fn add_result_appends_to_job() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let id = rt.block_on(mgr.create_job(
            BatchEndpoint::Chat,
            vec!["result test".into()],
            None,
            None,
            None,
        ));
        let result = BatchResultItem {
            index: 0,
            status: "succeeded".to_string(),
            content: Some("output".into()),
            error: None,
        };
        rt.block_on(mgr.add_result(&id, result));
        let job = rt.block_on(mgr.get_job(&id)).unwrap();
        assert_eq!(job.results.len(), 1);
        assert_eq!(job.results[0].index, 0);
        assert_eq!(job.results[0].content.as_deref(), Some("output"));
    }

    #[test]
    fn set_completed_updates_status_and_timestamp() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let id = rt.block_on(mgr.create_job(
            BatchEndpoint::Chat,
            vec!["complete".into()],
            None,
            None,
            None,
        ));
        rt.block_on(mgr.set_completed(&id));
        let job = rt.block_on(mgr.get_job(&id)).unwrap();
        assert_eq!(job.status, BatchStatus::Completed);
        assert!(job.completed_at.is_some());
    }

    #[test]
    fn add_result_to_missing_job_is_noop() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = BatchResultItem {
            index: 0,
            status: "succeeded".to_string(),
            content: None,
            error: None,
        };
        // Should not panic
        rt.block_on(mgr.add_result("nonexistent", result));
    }

    #[test]
    fn set_completed_on_missing_job_is_noop() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        // Should not panic
        rt.block_on(mgr.set_completed("nonexistent"));
    }

    #[test]
    fn mark_in_progress_transitions_pending_job() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let id =
            rt.block_on(mgr.create_job(BatchEndpoint::Chat, vec!["go".into()], None, None, None));
        assert!(rt.block_on(mgr.mark_in_progress(&id)));
        let job = rt.block_on(mgr.get_job(&id)).unwrap();
        assert_eq!(job.status, BatchStatus::InProgress);
        // Idempotent — a second transition keeps InProgress.
        assert!(rt.block_on(mgr.mark_in_progress(&id)));
    }

    #[test]
    fn mark_in_progress_on_missing_job_is_noop() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(!rt.block_on(mgr.mark_in_progress("nonexistent")));
    }

    #[test]
    fn set_failed_marks_terminal_with_timestamp() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let id = rt.block_on(mgr.create_job(
            BatchEndpoint::Completion,
            vec!["boom".into()],
            None,
            None,
            None,
        ));
        rt.block_on(mgr.set_failed(&id));
        let job = rt.block_on(mgr.get_job(&id)).unwrap();
        assert_eq!(job.status, BatchStatus::Failed);
        assert!(job.completed_at.is_some());
    }

    #[test]
    fn set_failed_on_missing_job_is_noop() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        // Should not panic
        rt.block_on(mgr.set_failed("nonexistent"));
    }

    #[test]
    fn request_cancel_transitions_in_progress_to_cancelled() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let id = rt.block_on(mgr.create_job(
            BatchEndpoint::Chat,
            vec!["a".into(), "b".into()],
            None,
            None,
            None,
        ));
        rt.block_on(mgr.mark_in_progress(&id));

        assert!(rt.block_on(mgr.request_cancel(&id)));
        let job = rt.block_on(mgr.get_job(&id)).unwrap();
        assert_eq!(job.status, BatchStatus::Cancelled);
        assert!(
            job.cancel_requested
                .load(std::sync::atomic::Ordering::Relaxed),
            "cancel flag must be observable by the worker"
        );
    }

    #[test]
    fn request_cancel_on_pending_job_succeeds() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let id = rt.block_on(mgr.create_job(
            BatchEndpoint::Completion,
            vec!["x".into()],
            None,
            None,
            None,
        ));
        assert!(rt.block_on(mgr.request_cancel(&id)));
        assert!(rt.block_on(mgr.is_cancelled(&id)));
    }

    #[test]
    fn request_cancel_on_terminal_job_is_noop() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let id =
            rt.block_on(mgr.create_job(BatchEndpoint::Chat, vec!["done".into()], None, None, None));
        rt.block_on(mgr.set_completed(&id));
        // Terminal — nothing to cancel.
        assert!(!rt.block_on(mgr.request_cancel(&id)));
    }

    #[test]
    fn request_cancel_on_missing_job_is_noop() {
        let mgr = BatchManager::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(!rt.block_on(mgr.request_cancel("nonexistent")));
        assert!(!rt.block_on(mgr.is_cancelled("nonexistent")));
    }
}
