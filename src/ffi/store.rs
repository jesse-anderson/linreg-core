//! Global handle store for FFI results.
//!
//! VBA receives opaque `usize` handles.  Rust stores the actual results here
//! behind a `Mutex`.  Handles are always >= 1; 0 signals an error to the caller.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use super::types::FitResult;

// ── Handle counter ────────────────────────────────────────────────────────────

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

// ── Global store ──────────────────────────────────────────────────────────────

static STORE: OnceLock<Mutex<HashMap<usize, FitResult>>> = OnceLock::new();

fn store() -> &'static Mutex<HashMap<usize, FitResult>> {
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Insert a result into the store and return its handle (always >= 1).
pub fn insert(result: FitResult) -> usize {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    store().lock().unwrap().insert(id, result);
    id
}

/// Run a closure against the stored result for `id`, returning `None` if the
/// handle does not exist.
pub fn with<T>(id: usize, f: impl FnOnce(&FitResult) -> T) -> Option<T> {
    let guard = store().lock().unwrap();
    guard.get(&id).map(f)
}

/// Remove the result for `id` from the store (idempotent).
pub fn remove(id: usize) {
    store().lock().unwrap().remove(&id);
}

// ── Thread-local last-error ───────────────────────────────────────────────────

thread_local! {
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

/// Store an error message for later retrieval via `LR_GetLastError`.
pub fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.to_string());
}

/// Retrieve the last error message set on this thread.
pub fn get_last_error() -> String {
    LAST_ERROR.with(|e| e.borrow().clone())
}
