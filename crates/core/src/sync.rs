//! Internal synchronization helpers.

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::error::{EngineError, Result};

/// Acquire a mutex guard, mapping poison to [`EngineError::LockPoisoned`].
#[allow(dead_code)]
pub(crate) fn lock_mutex<T: ?Sized>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|PoisonError { .. }| EngineError::LockPoisoned)
}
