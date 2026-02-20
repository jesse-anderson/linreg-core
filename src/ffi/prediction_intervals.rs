//! Prediction interval FFI bindings.
//!
//! # Usage from VBA
//!
//! ```vba
//! Dim h As LongPtr
//! h = LR_PredictionIntervals(VarPtr(y(0)), n, VarPtr(x(0)), n, p, _
//!                             VarPtr(newX(0)), n_new, 0.05)
//! If h = 0 Then MsgBox LR_GetLastError() : Exit Sub
//!
//! Dim pred(n_new - 1) As Double
//! Dim lower(n_new - 1) As Double
//! Dim upper(n_new - 1) As Double
//! LR_GetPredicted  h, VarPtr(pred(0)),  n_new
//! LR_GetLowerBound h, VarPtr(lower(0)), n_new
//! LR_GetUpperBound h, VarPtr(upper(0)), n_new
//! LR_Free h
//! ```

use std::slice;

use super::store::{insert, set_last_error, with};
use super::types::FitResult;

// ── Helper: copy a Vec<f64> from a PredictionIntervalOutput field ─────────────

unsafe fn copy_pi_field(
    handle: usize,
    selector: impl Fn(&crate::prediction_intervals::PredictionIntervalOutput) -> &Vec<f64>,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    if out_ptr.is_null() || out_len < 0 {
        return -1;
    }
    with(handle, |r| match r {
        FitResult::PredictionInterval(pi) => {
            let src = selector(pi);
            let count = src.len().min(out_len as usize);
            let dst = unsafe { slice::from_raw_parts_mut(out_ptr, count) };
            dst.copy_from_slice(&src[..count]);
            count as i32
        }
        _ => -1,
    })
    .unwrap_or(-1)
}

// ── Fit ───────────────────────────────────────────────────────────────────────

/// Compute OLS prediction intervals for new observations.
///
/// - `y_ptr`      – training response values (n)
/// - `n`          – number of training observations
/// - `x_ptr`      – flat row-major training predictors (n × p, no intercept)
/// - `p`          – number of predictors
/// - `new_x_ptr`  – flat row-major new predictors (n_new × p, no intercept)
/// - `n_new`      – number of new observations to predict
/// - `alpha`      – significance level (e.g. 0.05 for 95% intervals)
///
/// Returns a handle >= 1, or 0 on error.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn LR_PredictionIntervals(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    new_x_ptr: *const f64,
    n_new: i32,
    alpha: f64,
) -> usize {
    if y_ptr.is_null() || x_ptr.is_null() || new_x_ptr.is_null()
        || n <= 0 || p <= 0 || n_new <= 0
    {
        set_last_error("LR_PredictionIntervals: null pointer or non-positive dimensions");
        return 0;
    }
    let n = n as usize;
    let p = p as usize;
    let n_new = n_new as usize;

    let y = unsafe { slice::from_raw_parts(y_ptr, n) }.to_vec();

    // Build column-oriented training predictors
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n * p) };
    let mut x_vars: Vec<Vec<f64>> = vec![Vec::with_capacity(n); p];
    for row in 0..n {
        for col in 0..p {
            x_vars[col].push(x_flat[row * p + col]);
        }
    }

    // Build column-oriented new predictors
    let new_x_flat = unsafe { slice::from_raw_parts(new_x_ptr, n_new * p) };
    let mut new_x_cols: Vec<Vec<f64>> = vec![Vec::with_capacity(n_new); p];
    for row in 0..n_new {
        for col in 0..p {
            new_x_cols[col].push(new_x_flat[row * p + col]);
        }
    }
    let new_x_refs: Vec<&[f64]> = new_x_cols.iter().map(|v| v.as_slice()).collect();

    match crate::prediction_intervals::prediction_intervals(&y, &x_vars, &new_x_refs, alpha) {
        Ok(result) => insert(FitResult::PredictionInterval(result)),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

// ── Getters ───────────────────────────────────────────────────────────────────

/// Copy point predictions into the caller's buffer.
/// Returns the number of values written, or -1 on error.
#[no_mangle]
pub extern "system" fn LR_GetPredicted(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    unsafe { copy_pi_field(handle, |pi| &pi.predicted, out_ptr, out_len) }
}

/// Copy lower interval bounds into the caller's buffer.
#[no_mangle]
pub extern "system" fn LR_GetLowerBound(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    unsafe { copy_pi_field(handle, |pi| &pi.lower_bound, out_ptr, out_len) }
}

/// Copy upper interval bounds into the caller's buffer.
#[no_mangle]
pub extern "system" fn LR_GetUpperBound(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    unsafe { copy_pi_field(handle, |pi| &pi.upper_bound, out_ptr, out_len) }
}

/// Copy prediction standard errors into the caller's buffer.
#[no_mangle]
pub extern "system" fn LR_GetSEPred(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    unsafe { copy_pi_field(handle, |pi| &pi.se_pred, out_ptr, out_len) }
}
