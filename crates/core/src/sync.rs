//! sync: sync.

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::error::{EngineError, Result};

/// Acquire a mutex guard, mapping poison to [`EngineError::LockPoisoned`].
pub fn lock_mutex<'a, T: ?Sized>(mutex: &'a Mutex<T>) -> Result<MutexGuard<'a, T>> {
    mutex
        .lock()
        .map_err(|PoisonError { .. }| EngineError::LockPoisoned)
}
