//! K-Fold Cross Validation FFI bindings.
//!
//! Four fit functions cover OLS, Ridge, Lasso, and Elastic Net.  Each returns
//! a `CV` handle storing a [`CVResult`].  Retrieve aggregate metrics with the
//! `LR_GetCV*` scalar getters below.  Per-fold details are not exposed via FFI.
//!
//! [`CVResult`]: crate::cross_validation::CVResult

use std::slice;

use super::store::{insert, set_last_error, with};
use super::types::FitResult;
use crate::cross_validation::{
    kfold_cv_elastic_net, kfold_cv_lasso, kfold_cv_ols, kfold_cv_ridge, KFoldOptions,
};

// ── Helper ────────────────────────────────────────────────────────────────────

unsafe fn unpack(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    if y_ptr.is_null() || x_ptr.is_null() || n <= 0 || p <= 0 {
        return None;
    }
    let n = n as usize;
    let p = p as usize;
    let y = slice::from_raw_parts(y_ptr, n).to_vec();
    let x_flat = slice::from_raw_parts(x_ptr, n * p);
    let mut x_vars: Vec<Vec<f64>> = vec![Vec::with_capacity(n); p];
    for row in 0..n {
        for col in 0..p {
            x_vars[col].push(x_flat[row * p + col]);
        }
    }
    Some((y, x_vars))
}

// ── Fit functions ─────────────────────────────────────────────────────────────

/// K-Fold Cross Validation for OLS regression.
///
/// - `y_ptr`   – pointer to n response values
/// - `n`       – number of observations
/// - `x_ptr`   – flat row-major predictor matrix (n × p, no intercept column)
/// - `p`       – number of predictors
/// - `n_folds` – number of folds (>= 2)
///
/// Returns a handle.  Use `LR_GetCV*` getters to retrieve aggregate metrics.
#[no_mangle]
pub extern "system" fn LR_KFoldOLS(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    n_folds: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_KFoldOLS: null pointer or non-positive dimensions");
            return 0;
        }
    };
    let p_us = p as usize;
    let mut names = vec!["Intercept".to_string()];
    for i in 1..=p_us {
        names.push(format!("X{}", i));
    }
    let opts = KFoldOptions { n_folds: n_folds.max(2) as usize, shuffle: false, seed: None };
    match kfold_cv_ols(&y, &x_vars, &names, &opts) {
        Ok(r) => insert(FitResult::CV(r)),
        Err(e) => { set_last_error(&e.to_string()); 0 }
    }
}

/// K-Fold Cross Validation for Ridge regression.
///
/// - `lambda`      – L2 regularisation strength
/// - `standardize` – 1 = standardize predictors, 0 = do not
/// - `n_folds`     – number of folds
#[no_mangle]
pub extern "system" fn LR_KFoldRidge(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    lambda: f64,
    standardize: i32,
    n_folds: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_KFoldRidge: null pointer or non-positive dimensions");
            return 0;
        }
    };
    let opts = KFoldOptions { n_folds: n_folds.max(2) as usize, shuffle: false, seed: None };
    match kfold_cv_ridge(&x_vars, &y, lambda, standardize != 0, &opts) {
        Ok(r) => insert(FitResult::CV(r)),
        Err(e) => { set_last_error(&e.to_string()); 0 }
    }
}

/// K-Fold Cross Validation for Lasso regression.
///
/// - `lambda`      – L1 regularisation strength
/// - `standardize` – 1 = standardize predictors, 0 = do not
/// - `n_folds`     – number of folds
#[no_mangle]
pub extern "system" fn LR_KFoldLasso(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    lambda: f64,
    standardize: i32,
    n_folds: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_KFoldLasso: null pointer or non-positive dimensions");
            return 0;
        }
    };
    let opts = KFoldOptions { n_folds: n_folds.max(2) as usize, shuffle: false, seed: None };
    match kfold_cv_lasso(&x_vars, &y, lambda, standardize != 0, &opts) {
        Ok(r) => insert(FitResult::CV(r)),
        Err(e) => { set_last_error(&e.to_string()); 0 }
    }
}

/// K-Fold Cross Validation for Elastic Net regression.
///
/// - `lambda`      – regularisation strength
/// - `alpha`       – mixing parameter (0 = Ridge, 1 = Lasso)
/// - `standardize` – 1 = standardize predictors, 0 = do not
/// - `n_folds`     – number of folds
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn LR_KFoldElasticNet(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    lambda: f64,
    alpha: f64,
    standardize: i32,
    n_folds: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_KFoldElasticNet: null pointer or non-positive dimensions");
            return 0;
        }
    };
    let opts = KFoldOptions { n_folds: n_folds.max(2) as usize, shuffle: false, seed: None };
    match kfold_cv_elastic_net(&x_vars, &y, lambda, alpha, standardize != 0, &opts) {
        Ok(r) => insert(FitResult::CV(r)),
        Err(e) => { set_last_error(&e.to_string()); 0 }
    }
}

// ── CV scalar getters ─────────────────────────────────────────────────────────

/// Number of folds used.
#[no_mangle]
pub extern "system" fn LR_GetCVNFolds(handle: usize) -> i32 {
    with(handle, |r| match r {
        FitResult::CV(cv) => cv.n_folds as i32,
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Mean MSE across all folds.
#[no_mangle]
pub extern "system" fn LR_GetCVMeanMSE(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::CV(cv) => cv.mean_mse,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Standard deviation of MSE across folds.
#[no_mangle]
pub extern "system" fn LR_GetCVStdMSE(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::CV(cv) => cv.std_mse,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Mean RMSE across all folds.
#[no_mangle]
pub extern "system" fn LR_GetCVMeanRMSE(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::CV(cv) => cv.mean_rmse,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Standard deviation of RMSE across folds.
#[no_mangle]
pub extern "system" fn LR_GetCVStdRMSE(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::CV(cv) => cv.std_rmse,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Mean test-set R² across all folds.
#[no_mangle]
pub extern "system" fn LR_GetCVMeanR2(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::CV(cv) => cv.mean_r_squared,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}
