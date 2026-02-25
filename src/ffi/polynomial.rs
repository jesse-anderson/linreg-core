//! Polynomial regression FFI bindings.
//!
//! Polynomial regression fits a polynomial relationship between a single predictor
//! and response: y = β₀ + β₁x + β₂x² + ... + β_d·x^d.
//!
//! This module provides functions to fit polynomial models and make predictions.
//! Centering (subtracting the mean of x before raising to powers) is recommended
//! for degree ≥ 3 to reduce multicollinearity.

use std::slice;

use super::store::{insert, set_last_error, with};
use super::types::FitResult;
use crate::polynomial::{polynomial_regression, predict, PolynomialOptions};

// ── Helper ────────────────────────────────────────────────────────────────────

/// Convert a boolean to an integer (0 or 1).
#[inline]
fn bool_to_int(b: bool) -> i32 {
    if b { 1 } else { 0 }
}

// ── Fit function ──────────────────────────────────────────────────────────────

/// Fit a polynomial regression model.
///
/// # Arguments
///
/// - `y_ptr`   – pointer to n response values
/// - `n`       – number of observations
/// - `x_ptr`   – pointer to n predictor values (single variable)
/// - `degree`  – polynomial degree (>= 1)
/// - `center`  – 1 = center the predictor (subtract mean) before raising to powers,
///               0 = use raw predictor values. Centering is recommended for degree >= 3.
///
/// # Returns
///
/// A handle on success, 0 on error.
///
/// # Safety
///
/// Caller must ensure:
/// - `y_ptr` is valid for reading `n` f64 values
/// - `x_ptr` is valid for reading `n` f64 values
/// - Both pointers are properly aligned for f64
/// - The memory regions will not be modified by any other thread during this call
#[no_mangle]
pub extern "system" fn LR_PolynomialFit(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    degree: i32,
    center: i32,
) -> usize {
    if y_ptr.is_null() || x_ptr.is_null() || n <= 0 || degree < 1 {
        set_last_error("LR_PolynomialFit: invalid arguments (null pointer, non-positive n, or degree < 1)");
        return 0;
    }

    let n = n as usize;
    let degree = degree as usize;

    // SAFETY: Caller has ensured pointers are valid per safety contract
    let y = unsafe { slice::from_raw_parts(y_ptr, n) }.to_vec();
    let x = unsafe { slice::from_raw_parts(x_ptr, n) }.to_vec();

    let options = PolynomialOptions {
        degree,
        center: center != 0,
        ..Default::default()
    };

    match polynomial_regression(&y, &x, &options) {
        Ok(fit) => insert(FitResult::Polynomial(fit)),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

// ── Prediction ─────────────────────────────────────────────────────────────────

/// Predict using a fitted polynomial model.
///
/// # Arguments
///
/// - `handle`  – handle returned by `LR_PolynomialFit`
/// - `x_ptr`   – pointer to n_new predictor values
/// - `n_new`   – number of new observations
/// - `out_ptr` – pointer to output array (must have space for n_new f64 values)
///
/// # Returns
///
/// Number of values written on success, 0 on error.
///
/// # Safety
///
/// Caller must ensure:
/// - `handle` is a valid handle returned by `LR_PolynomialFit`
/// - `x_ptr` is valid for reading `n_new` f64 values
/// - `out_ptr` is valid for writing `n_new` f64 values
/// - All pointers are properly aligned for f64
/// - The memory regions will not be accessed by any other thread during this call
#[no_mangle]
pub extern "system" fn LR_PolynomialPredict(
    handle: usize,
    x_ptr: *const f64,
    n_new: i32,
    out_ptr: *mut f64,
) -> i32 {
    if x_ptr.is_null() || out_ptr.is_null() || n_new <= 0 {
        set_last_error("LR_PolynomialPredict: invalid arguments (null pointer or non-positive n_new)");
        return 0;
    }

    let n_new = n_new as usize;

    // SAFETY: Caller has ensured pointers are valid per safety contract
    let x_new = unsafe { slice::from_raw_parts(x_ptr, n_new) };

    let predictions: Vec<f64> = match with(handle, |r| match r {
        FitResult::Polynomial(fit) => predict(fit, x_new),
        _ => Err(crate::Error::InvalidInput("Not a PolynomialFit handle".into())),
    }) {
        Some(Ok(preds)) => preds,
        Some(Err(e)) => {
            set_last_error(&e.to_string());
            return 0;
        }
        None => {
            set_last_error("LR_PolynomialPredict: invalid handle");
            return 0;
        }
    };

    if predictions.len() != n_new {
        set_last_error(&format!("LR_PolynomialPredict: expected {} predictions, got {}", n_new, predictions.len()));
        return 0;
    }

    // SAFETY: Caller has ensured out_ptr is valid for writing n_new f64 values
    unsafe {
        let out_slice = slice::from_raw_parts_mut(out_ptr, n_new);
        out_slice.copy_from_slice(&predictions);
    }

    n_new as i32
}

// ── Scalar getters ─────────────────────────────────────────────────────────────

/// Get the polynomial degree used in fitting.
#[no_mangle]
pub extern "system" fn LR_GetPolynomialDegree(handle: usize) -> i32 {
    with(handle, |r| match r {
        FitResult::Polynomial(fit) => fit.degree as i32,
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Get whether the predictor was centered during fitting.
#[no_mangle]
pub extern "system" fn LR_GetPolynomialCenter(handle: usize) -> i32 {
    with(handle, |r| match r {
        FitResult::Polynomial(fit) => bool_to_int(fit.centered),
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Get the center value used (mean of x if centered, 0 otherwise).
#[no_mangle]
pub extern "system" fn LR_GetPolynomialCenterValue(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Polynomial(fit) => fit.x_mean,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

// ── OLS output delegation ───────────────────────────────────────────────────────

/// Get R-squared from the polynomial fit.
#[no_mangle]
pub extern "system" fn LR_GetPolynomialRSquared(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Polynomial(fit) => fit.ols_output.r_squared,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Get adjusted R-squared from the polynomial fit.
#[no_mangle]
pub extern "system" fn LR_GetPolynomialAdjRSquared(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Polynomial(fit) => fit.ols_output.adj_r_squared,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Get mean squared error from the polynomial fit.
#[no_mangle]
pub extern "system" fn LR_GetPolynomialMSE(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Polynomial(fit) => fit.ols_output.mse,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Get the number of coefficients (degree + 1, including intercept).
#[no_mangle]
pub extern "system" fn LR_GetPolynomialNumCoefficients(handle: usize) -> i32 {
    with(handle, |r| match r {
        FitResult::Polynomial(fit) => fit.ols_output.coefficients.len() as i32,
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Get coefficients from the polynomial fit.
///
/// Returns number of coefficients written on success, 0 on error.
#[no_mangle]
pub extern "system" fn LR_GetPolynomialCoefficients(
    handle: usize,
    out_ptr: *mut f64,
    out_len: i32,
) -> i32 {
    if out_ptr.is_null() || out_len <= 0 {
        return 0;
    }

    let out_len = out_len as usize;

    with(handle, |r| match r {
        FitResult::Polynomial(fit) => {
            let coefs = &fit.ols_output.coefficients;
            let n = coefs.len().min(out_len);
            // SAFETY: Caller has ensured out_ptr is valid for writing out_len f64 values
            unsafe {
                let out_slice = slice::from_raw_parts_mut(out_ptr, n);
                out_slice.copy_from_slice(&coefs[..n]);
            }
            n as i32
        }
        _ => 0,
    })
    .unwrap_or(0)
}
