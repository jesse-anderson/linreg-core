//! OLS regression FFI bindings.
//!
//! # Usage from VBA
//!
//! ```vba
//! Dim h As LongPtr
//! h = LR_OLS(VarPtr(y(0)), n, VarPtr(x(0)), n, p)
//! If h = 0 Then MsgBox LR_GetLastError() : Exit Sub
//!
//! Dim coefs(p) As Double
//! LR_GetCoefficients h, VarPtr(coefs(0)), p + 1
//! Debug.Print "R² = "; LR_GetRSquared(h)
//! LR_Free h
//! ```

use std::slice;

use super::store::{get_last_error, insert, remove, set_last_error, with};
use super::types::FitResult;
use crate::core::ols_regression;

// ── Helper: copy a Vec<f64> into a caller-supplied buffer ────────────────────

/// Copy up to `out_len` values from `src` into the buffer at `out_ptr`.
/// Returns the number of values written, or -1 on invalid arguments.
unsafe fn copy_doubles(src: &[f64], out_ptr: *mut f64, out_len: i32) -> i32 {
    if out_ptr.is_null() || out_len < 0 {
        return -1;
    }
    let count = src.len().min(out_len as usize);
    let dst = unsafe { slice::from_raw_parts_mut(out_ptr, count) };
    dst.copy_from_slice(&src[..count]);
    count as i32
}

// ── Fit ───────────────────────────────────────────────────────────────────────

/// Fit an OLS regression model.
///
/// # Arguments
/// - `y_ptr`   – pointer to array of `n` response values
/// - `n`       – number of observations
/// - `x_ptr`   – pointer to flat row-major matrix of shape (n, p)
///               (each row has p predictor values, NO intercept column)
/// - `p`       – number of predictor columns (not counting the intercept)
///
/// # Returns
/// An opaque handle >= 1 on success, or 0 on failure.
/// Retrieve the error message with `LR_GetLastError`.
#[no_mangle]
pub extern "system" fn LR_OLS(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
) -> usize {
    if y_ptr.is_null() || x_ptr.is_null() || n <= 0 || p <= 0 {
        set_last_error("LR_OLS: null pointer or non-positive dimensions");
        return 0;
    }
    let n = n as usize;
    let p = p as usize;

    let y = unsafe { slice::from_raw_parts(y_ptr, n) }.to_vec();

    // x_ptr is row-major (n × p).  Split into p column vectors.
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n * p) };
    let mut x_vars: Vec<Vec<f64>> = vec![Vec::with_capacity(n); p];
    for row in 0..n {
        for col in 0..p {
            x_vars[col].push(x_flat[row * p + col]);
        }
    }

    // Build variable names: Intercept, X1 … Xp
    let mut names = vec!["Intercept".to_string()];
    for i in 1..=p {
        names.push(format!("X{}", i));
    }

    match ols_regression(&y, &x_vars, &names) {
        Ok(result) => insert(FitResult::Ols(result)),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

// ── Scalar getters ────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "system" fn LR_GetRSquared(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Ols(o) => o.r_squared,
        FitResult::Ridge(r) => r.r_squared,
        FitResult::Lasso(l) => l.r_squared,
        FitResult::ElasticNet(e) => e.r_squared,
        FitResult::Wls(w) => w.r_squared,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

#[no_mangle]
pub extern "system" fn LR_GetAdjRSquared(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Ols(o) => o.adj_r_squared,
        FitResult::Ridge(r) => r.adj_r_squared,
        FitResult::Lasso(l) => l.adj_r_squared,
        FitResult::ElasticNet(e) => e.adj_r_squared,
        FitResult::Wls(w) => w.adj_r_squared,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

#[no_mangle]
pub extern "system" fn LR_GetFStatistic(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Ols(o) => o.f_statistic,
        FitResult::Wls(w) => w.f_statistic,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

#[no_mangle]
pub extern "system" fn LR_GetFPValue(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Ols(o) => o.f_p_value,
        FitResult::Wls(w) => w.f_p_value,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

#[no_mangle]
pub extern "system" fn LR_GetMSE(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Ols(o) => o.mse,
        FitResult::Ridge(r) => r.mse,
        FitResult::Lasso(l) => l.mse,
        FitResult::ElasticNet(e) => e.mse,
        FitResult::Wls(w) => w.mse,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

#[no_mangle]
pub extern "system" fn LR_GetNumCoefficients(handle: usize) -> i32 {
    with(handle, |r| match r {
        // OLS / WLS coefficients include the intercept
        FitResult::Ols(o) => o.coefficients.len() as i32,
        FitResult::Wls(w) => w.coefficients.len() as i32,
        // Regularized: slopes only (intercept via LR_GetIntercept)
        FitResult::Ridge(r) => r.coefficients.len() as i32,
        FitResult::Lasso(l) => l.coefficients.len() as i32,
        FitResult::ElasticNet(e) => e.coefficients.len() as i32,
        _ => -1,
    })
    .unwrap_or(-1)
}

#[no_mangle]
pub extern "system" fn LR_GetNumObservations(handle: usize) -> i32 {
    with(handle, |r| match r {
        FitResult::Ols(o) => o.n as i32,
        FitResult::Wls(w) => w.n as i32,
        FitResult::Ridge(r) => r.fitted_values.len() as i32,
        FitResult::Lasso(l) => l.fitted_values.len() as i32,
        FitResult::ElasticNet(e) => e.fitted_values.len() as i32,
        _ => -1,
    })
    .unwrap_or(-1)
}

// ── Vector getters ────────────────────────────────────────────────────────────

/// Copy coefficients into a caller-allocated buffer.
/// For OLS / WLS the buffer must hold k+1 values (including intercept).
/// For regularized models it holds k values (slopes only; use LR_GetIntercept).
/// Returns the number of values written, or -1 on error.
#[no_mangle]
pub extern "system" fn LR_GetCoefficients(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    with(handle, |r| {
        let coefs: &[f64] = match r {
            FitResult::Ols(o) => &o.coefficients,
            FitResult::Wls(w) => &w.coefficients,
            FitResult::Ridge(r) => &r.coefficients,
            FitResult::Lasso(l) => &l.coefficients,
            FitResult::ElasticNet(e) => &e.coefficients,
            _ => return -1,
        };
        unsafe { copy_doubles(coefs, out_ptr, out_len) }
    })
    .unwrap_or(-1)
}

/// Copy standard errors into the buffer. OLS and WLS.
#[no_mangle]
pub extern "system" fn LR_GetStdErrors(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    with(handle, |r| match r {
        FitResult::Ols(o) => unsafe { copy_doubles(&o.std_errors, out_ptr, out_len) },
        FitResult::Wls(w) => unsafe { copy_doubles(&w.standard_errors, out_ptr, out_len) },
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Copy t-statistics into the buffer. OLS and WLS.
#[no_mangle]
pub extern "system" fn LR_GetTStats(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    with(handle, |r| match r {
        FitResult::Ols(o) => unsafe { copy_doubles(&o.t_stats, out_ptr, out_len) },
        FitResult::Wls(w) => unsafe { copy_doubles(&w.t_statistics, out_ptr, out_len) },
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Copy p-values into the buffer. OLS and WLS.
#[no_mangle]
pub extern "system" fn LR_GetPValues(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    with(handle, |r| match r {
        FitResult::Ols(o) => unsafe { copy_doubles(&o.p_values, out_ptr, out_len) },
        FitResult::Wls(w) => unsafe { copy_doubles(&w.p_values, out_ptr, out_len) },
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Copy raw residuals into the buffer. Works on all regression types.
#[no_mangle]
pub extern "system" fn LR_GetResiduals(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    with(handle, |r| {
        let v: &[f64] = match r {
            FitResult::Ols(o) => &o.residuals,
            FitResult::Wls(w) => &w.residuals,
            FitResult::Ridge(r) => &r.residuals,
            FitResult::Lasso(l) => &l.residuals,
            FitResult::ElasticNet(e) => &e.residuals,
            _ => return -1,
        };
        unsafe { copy_doubles(v, out_ptr, out_len) }
    })
    .unwrap_or(-1)
}

/// Copy fitted values into the buffer. Works on all regression types.
#[no_mangle]
pub extern "system" fn LR_GetFittedValues(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    with(handle, |r| {
        let v: &[f64] = match r {
            FitResult::Ols(o) => &o.predictions,
            FitResult::Wls(w) => &w.fitted_values,
            FitResult::Ridge(r) => &r.fitted_values,
            FitResult::Lasso(l) => &l.fitted_values,
            FitResult::ElasticNet(e) => &e.fitted_values,
            _ => return -1,
        };
        unsafe { copy_doubles(v, out_ptr, out_len) }
    })
    .unwrap_or(-1)
}

// ── Generic vector / matrix getters ──────────────────────────────────────────
//
// Used for Cook's distances, DFFITS values, VIF values, DFBETAS matrices,
// and the lambda path — anything stored as FitResult::Vector or ::Matrix.

/// Return the number of elements in a `Vector` result.
/// Returns -1 if the handle does not hold a vector.
#[no_mangle]
pub extern "system" fn LR_GetVectorLength(handle: usize) -> i32 {
    with(handle, |r| match r {
        FitResult::Vector(v) => v.len() as i32,
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Copy a `Vector` result into a caller-supplied buffer.
/// Returns the number of values written, or -1 on error.
#[no_mangle]
pub extern "system" fn LR_GetVector(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    if out_ptr.is_null() || out_len < 0 {
        return -1;
    }
    with(handle, |r| match r {
        FitResult::Vector(v) => {
            let count = v.len().min(out_len as usize);
            let dst = unsafe { slice::from_raw_parts_mut(out_ptr, count) };
            dst.copy_from_slice(&v[..count]);
            count as i32
        }
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Return the number of rows in a `Matrix` result (DFBETAS: = n observations).
/// Returns -1 if the handle does not hold a matrix.
#[no_mangle]
pub extern "system" fn LR_GetMatrixRows(handle: usize) -> i32 {
    with(handle, |r| match r {
        FitResult::Matrix { rows, .. } => *rows as i32,
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Return the number of columns in a `Matrix` result (DFBETAS: = p parameters).
/// Returns -1 if the handle does not hold a matrix.
#[no_mangle]
pub extern "system" fn LR_GetMatrixCols(handle: usize) -> i32 {
    with(handle, |r| match r {
        FitResult::Matrix { cols, .. } => *cols as i32,
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Copy a `Matrix` result into a caller-supplied buffer (row-major order).
/// `out_len` must be >= rows × cols.
/// Returns the number of values written, or -1 on error.
#[no_mangle]
pub extern "system" fn LR_GetMatrix(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    if out_ptr.is_null() || out_len < 0 {
        return -1;
    }
    with(handle, |r| match r {
        FitResult::Matrix { data, rows, cols } => {
            let total = rows * cols;
            let count = total.min(out_len as usize);
            let dst = unsafe { slice::from_raw_parts_mut(out_ptr, count) };
            dst.copy_from_slice(&data[..count]);
            count as i32
        }
        _ => -1,
    })
    .unwrap_or(-1)
}

// ── Handle management ─────────────────────────────────────────────────────────

/// Release the result associated with `handle`. Idempotent.
#[no_mangle]
pub extern "system" fn LR_Free(handle: usize) {
    remove(handle);
}

/// Copy the last error message into a caller-supplied buffer.
/// Returns the number of bytes written (not including null terminator).
/// The buffer is always null-terminated if `out_len` > 0.
#[no_mangle]
pub extern "system" fn LR_GetLastError(out_ptr: *mut u8, out_len: i32) -> i32 {
    if out_ptr.is_null() || out_len <= 0 {
        return -1;
    }
    let msg = get_last_error();
    let bytes = msg.as_bytes();
    let cap = (out_len as usize).saturating_sub(1); // leave room for null
    let count = bytes.len().min(cap);
    unsafe {
        let dst = slice::from_raw_parts_mut(out_ptr, count + 1);
        dst[..count].copy_from_slice(&bytes[..count]);
        dst[count] = 0; // null-terminate
    }
    count as i32
}
