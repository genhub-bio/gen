use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// Retains a related model for one model instance without writing to the database.
///
/// Cloning the cache creates an independent cache cell that initially holds the same model Arc.
#[derive(Debug)]
pub struct ForeignKeyCache<T> {
    value: Mutex<Option<Arc<T>>>,
}

impl<T> Default for ForeignKeyCache<T> {
    fn default() -> Self {
        Self {
            value: Mutex::new(None),
        }
    }
}

impl<T> Clone for ForeignKeyCache<T> {
    fn clone(&self) -> Self {
        Self {
            value: Mutex::new(self.snapshot()),
        }
    }
}

impl<T> ForeignKeyCache<T> {
    /// Returns the currently retained model, if any.
    pub fn snapshot(&self) -> Option<Arc<T>> {
        self.lock().as_ref().map(Arc::clone)
    }

    /// Replaces or clears the runtime-only cached model.
    pub fn replace(&self, value: Option<Arc<T>>) {
        *self.lock() = value;
    }

    /// Returns a matching cached value or loads and retains one while holding the cache lock.
    pub fn get_or_load<E>(
        &self,
        matches: impl FnOnce(&T) -> bool,
        load: impl FnOnce() -> Result<Arc<T>, E>,
    ) -> Result<Arc<T>, E> {
        let mut cached = self.lock();
        if let Some(value) = cached.as_ref()
            && matches(value)
        {
            return Ok(Arc::clone(value));
        }

        *cached = None;
        let value = load()?;
        *cached = Some(Arc::clone(&value));
        Ok(value)
    }

    fn lock(&self) -> MutexGuard<'_, Option<Arc<T>>> {
        self.value.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
