//! Regularized and weighted regression FFI bindings.
//!
//! Covers Ridge, Lasso, Elastic Net, WLS, and lambda-path generation.
//! All fit functions share the generic getters defined in `ols.rs`
//! (`LR_GetCoefficients`, `LR_GetRSquared`, etc.).  The functions defined
//! here add regularized-specific getters (`LR_GetIntercept`, `LR_GetDF`,
//! `LR_GetNNonzero`, `LR_GetConverged`) and the WLS / lambda-path entry points.

use std::slice;

use super::store::{insert, set_last_error, with};
use super::types::FitResult;
use crate::linalg::Matrix;
use crate::regularized::elastic_net::{elastic_net_fit, ElasticNetOptions};
use crate::regularized::lasso::{lasso_fit, LassoFitOptions};
use crate::regularized::path::{make_lambda_path, LambdaPathOptions};
use crate::regularized::ridge::{ridge_fit, RidgeFitOptions};
use crate::weighted_regression::wls_regression;

// ── Helper: build design matrix from row-major predictor array ────────────────

/// Build an (n × p+1) design matrix with a leading intercept column from a
/// flat row-major predictor array of shape (n × p).
fn build_design_matrix(x_flat: &[f64], n: usize, p: usize) -> Matrix {
    let mut data = Vec::with_capacity(n * (p + 1));
    for row in 0..n {
        data.push(1.0); // intercept
        for col in 0..p {
            data.push(x_flat[row * p + col]);
        }
    }
    Matrix::new(n, p + 1, data)
}

// ── Ridge ─────────────────────────────────────────────────────────────────────

/// Fit a Ridge regression model.
///
/// - `y_ptr`       – pointer to n response values
/// - `n`           – number of observations
/// - `x_ptr`       – flat row-major predictor matrix (n × p, no intercept column)
/// - `p`           – number of predictors
/// - `lambda`      – L2 regularisation strength (>= 0)
/// - `standardize` – 1 = standardize predictors before fitting, 0 = do not
///
/// Returns a handle >= 1, or 0 on error.
#[no_mangle]
pub extern "system" fn LR_Ridge(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    lambda: f64,
    standardize: i32,
) -> usize {
    if y_ptr.is_null() || x_ptr.is_null() || n <= 0 || p <= 0 {
        set_last_error("LR_Ridge: null pointer or non-positive dimensions");
        return 0;
    }
    let n = n as usize;
    let p = p as usize;
    let y = unsafe { slice::from_raw_parts(y_ptr, n) }.to_vec();
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n * p) };
    let x = build_design_matrix(x_flat, n, p);

    let options = RidgeFitOptions {
        lambda,
        standardize: standardize != 0,
        intercept: true,
        ..Default::default()
    };

    match ridge_fit(&x, &y, &options) {
        Ok(result) => insert(FitResult::Ridge(result)),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

// ── Lasso ─────────────────────────────────────────────────────────────────────

/// Fit a Lasso regression model.
///
/// - `max_iter` – maximum coordinate descent iterations
/// - `tol`      – convergence tolerance
///
/// Returns a handle >= 1, or 0 on error.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn LR_Lasso(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    lambda: f64,
    standardize: i32,
    max_iter: i32,
    tol: f64,
) -> usize {
    if y_ptr.is_null() || x_ptr.is_null() || n <= 0 || p <= 0 {
        set_last_error("LR_Lasso: null pointer or non-positive dimensions");
        return 0;
    }
    let n = n as usize;
    let p = p as usize;
    let y = unsafe { slice::from_raw_parts(y_ptr, n) }.to_vec();
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n * p) };
    let x = build_design_matrix(x_flat, n, p);

    let options = LassoFitOptions {
        lambda,
        standardize: standardize != 0,
        intercept: true,
        max_iter: max_iter.max(1) as usize,
        tol,
        ..Default::default()
    };

    match lasso_fit(&x, &y, &options) {
        Ok(result) => insert(FitResult::Lasso(result)),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

// ── Elastic Net ───────────────────────────────────────────────────────────────

/// Fit an Elastic Net regression model.
///
/// - `alpha` – mixing parameter (0 = Ridge, 1 = Lasso)
///
/// Returns a handle >= 1, or 0 on error.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn LR_ElasticNet(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    lambda: f64,
    alpha: f64,
    standardize: i32,
    max_iter: i32,
    tol: f64,
) -> usize {
    if y_ptr.is_null() || x_ptr.is_null() || n <= 0 || p <= 0 {
        set_last_error("LR_ElasticNet: null pointer or non-positive dimensions");
        return 0;
    }
    let n = n as usize;
    let p = p as usize;
    let y = unsafe { slice::from_raw_parts(y_ptr, n) }.to_vec();
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n * p) };
    let x = build_design_matrix(x_flat, n, p);

    let options = ElasticNetOptions {
        lambda,
        alpha,
        standardize: standardize != 0,
        intercept: true,
        max_iter: max_iter.max(1) as usize,
        tol,
        ..Default::default()
    };

    match elastic_net_fit(&x, &y, &options) {
        Ok(result) => insert(FitResult::ElasticNet(result)),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

// ── Regularized-specific getters ──────────────────────────────────────────────

/// Return the intercept for a regularized model.
/// For OLS, returns `coefficients[0]`.
#[no_mangle]
pub extern "system" fn LR_GetIntercept(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Ols(o) => o.coefficients.first().copied().unwrap_or(f64::NAN),
        FitResult::Ridge(r) => r.intercept,
        FitResult::Lasso(l) => l.intercept,
        FitResult::ElasticNet(e) => e.intercept,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Return the effective degrees of freedom (Ridge only; NAN for others).
#[no_mangle]
pub extern "system" fn LR_GetDF(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Ridge(r) => r.df,
        FitResult::Ols(o) => o.df as f64,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Return the number of non-zero coefficients (Lasso / Elastic Net only; -1 for others).
#[no_mangle]
pub extern "system" fn LR_GetNNonzero(handle: usize) -> i32 {
    with(handle, |r| match r {
        FitResult::Lasso(l) => l.n_nonzero as i32,
        FitResult::ElasticNet(e) => e.n_nonzero as i32,
        _ => -1,
    })
    .unwrap_or(-1)
}

/// Return 1 if the coordinate descent converged, 0 if not, -1 if not applicable.
#[no_mangle]
pub extern "system" fn LR_GetConverged(handle: usize) -> i32 {
    with(handle, |r| match r {
        FitResult::Lasso(l) => l.converged as i32,
        FitResult::ElasticNet(e) => e.converged as i32,
        _ => -1,
    })
    .unwrap_or(-1)
}

// ── WLS ───────────────────────────────────────────────────────────────────────

/// Fit a Weighted Least Squares regression model.
///
/// - `y_ptr`  – pointer to n response values
/// - `n`      – number of observations
/// - `x_ptr`  – flat row-major predictor matrix (n × p, no intercept column)
/// - `p`      – number of predictors
/// - `w_ptr`  – pointer to n non-negative observation weights
///
/// All scalar and vector getters from OLS (`LR_GetRSquared`, `LR_GetCoefficients`,
/// `LR_GetResiduals`, `LR_GetFStatistic`, etc.) work on the returned handle.
///
/// Returns a handle >= 1, or 0 on error.
#[no_mangle]
pub extern "system" fn LR_WLS(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    w_ptr: *const f64,
) -> usize {
    if y_ptr.is_null() || x_ptr.is_null() || w_ptr.is_null() || n <= 0 || p <= 0 {
        set_last_error("LR_WLS: null pointer or non-positive dimensions");
        return 0;
    }
    let n_us = n as usize;
    let p_us = p as usize;
    let y = unsafe { slice::from_raw_parts(y_ptr, n_us) }.to_vec();
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n_us * p_us) };
    let weights = unsafe { slice::from_raw_parts(w_ptr, n_us) }.to_vec();

    // Split flat row-major matrix into column vectors (matching wls_regression API)
    let mut x_vars: Vec<Vec<f64>> = vec![Vec::with_capacity(n_us); p_us];
    for row in 0..n_us {
        for col in 0..p_us {
            x_vars[col].push(x_flat[row * p_us + col]);
        }
    }

    match wls_regression(&y, &x_vars, &weights) {
        Ok(result) => insert(FitResult::Wls(result)),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

// ── Lambda path ───────────────────────────────────────────────────────────────

/// Generate a glmnet-style lambda sequence for regularized regression.
///
/// - `y_ptr`            – pointer to n response values
/// - `n`                – number of observations
/// - `x_ptr`            – flat row-major predictor matrix (n × p, no intercept column)
/// - `p`                – number of predictors
/// - `nlambda`          – number of lambda values to generate (e.g. 100)
/// - `lambda_min_ratio` – ratio of min to max lambda (e.g. 0.01); pass 0.0 for auto
/// - `alpha`            – elastic-net mixing parameter (0 = ridge, 1 = lasso)
///
/// Returns a handle to a `Vector` of descending lambda values.
/// Use `LR_GetVectorLength` and `LR_GetVector` to read the sequence.
/// Returns 0 on error.
#[no_mangle]
pub extern "system" fn LR_LambdaPath(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    nlambda: i32,
    lambda_min_ratio: f64,
    alpha: f64,
) -> usize {
    if y_ptr.is_null() || x_ptr.is_null() || n <= 0 || p <= 0 || nlambda <= 0 {
        set_last_error("LR_LambdaPath: null pointer or non-positive dimensions");
        return 0;
    }
    let n_us = n as usize;
    let p_us = p as usize;
    let y = unsafe { slice::from_raw_parts(y_ptr, n_us) }.to_vec();
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n_us * p_us) };
    let x = build_design_matrix(x_flat, n_us, p_us);

    let opts = LambdaPathOptions {
        nlambda: nlambda as usize,
        lambda_min_ratio: if lambda_min_ratio > 0.0 { Some(lambda_min_ratio) } else { None },
        alpha,
        eps_for_ridge: 0.001,
    };

    let lambdas = make_lambda_path(&x, &y, &opts, None, Some(0));
    insert(FitResult::Vector(lambdas))
}
