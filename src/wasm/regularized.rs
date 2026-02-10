//! Regularized regression methods for WASM
//!
//! Provides WASM bindings for Ridge, Lasso, and Elastic Net regression.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use super::domain::check_domain;
use crate::error::{error_json, error_to_json};
use crate::linalg;
use crate::regularized;

/// Performs Ridge regression via WASM.
///
/// Ridge regression adds an L2 penalty to the coefficients, which helps with
/// multicollinearity and overfitting. The intercept is never penalized.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `variable_names` - JSON array of variable names
/// * `lambda` - Regularization strength (>= 0, typical range 0.01 to 100)
/// * `standardize` - Whether to standardize predictors (recommended: true)
///
/// # Returns
///
/// JSON string containing:
/// - `lambda` - Lambda value used
/// - `intercept` - Intercept coefficient
/// - `coefficients` - Slope coefficients
/// - `fitted_values` - Predictions on training data
/// - `residuals` - Residuals (y - fitted_values)
/// - `df` - Effective degrees of freedom
///
/// # Errors
///
/// Returns a JSON error object if parsing fails, lambda is negative,
/// or domain check fails.
#[wasm_bindgen]
pub fn ridge_regression(
    y_json: &str,
    x_vars_json: &str,
    _variable_names: &str,
    lambda: f64,
    standardize: bool,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    // Parse JSON input
    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x_vars: Vec<Vec<f64>> = match serde_json::from_str(x_vars_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x_vars: {}", e)),
    };

    // Build design matrix with intercept column
    let (x, n, p) = build_design_matrix(&y, &x_vars);

    if n <= p + 1 {
        return error_json(&format!(
            "Insufficient data: need at least {} observations for {} predictors",
            p + 2,
            p
        ));
    }

    // Configure ridge options
    let options = regularized::ridge::RidgeFitOptions {
        lambda,
        intercept: true,
        standardize,
        max_iter: 100000,
        tol: 1e-7,
        warm_start: None,
        weights: None,
    };

    match regularized::ridge::ridge_fit(&x, &y, &options) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize ridge regression result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs Lasso regression via WASM.
///
/// Lasso regression adds an L1 penalty to the coefficients, which performs
/// automatic variable selection by shrinking some coefficients to exactly zero.
/// The intercept is never penalized.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `variable_names` - JSON array of variable names
/// * `lambda` - Regularization strength (>= 0, typical range 0.01 to 10)
/// * `standardize` - Whether to standardize predictors (recommended: true)
/// * `max_iter` - Maximum coordinate descent iterations (default: 100000)
/// * `tol` - Convergence tolerance (default: 1e-7)
///
/// # Returns
///
/// JSON string containing:
/// - `lambda` - Lambda value used
/// - `intercept` - Intercept coefficient
/// - `coefficients` - Slope coefficients (some may be exactly zero)
/// - `fitted_values` - Predictions on training data
/// - `residuals` - Residuals (y - fitted_values)
/// - `n_nonzero` - Number of non-zero coefficients (excluding intercept)
/// - `iterations` - Number of coordinate descent iterations
/// - `converged` - Whether the algorithm converged
///
/// # Errors
///
/// Returns a JSON error object if parsing fails, lambda is negative,
/// or domain check fails.
#[wasm_bindgen]
pub fn lasso_regression(
    y_json: &str,
    x_vars_json: &str,
    _variable_names: &str,
    lambda: f64,
    standardize: bool,
    max_iter: usize,
    tol: f64,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    // Parse JSON input
    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x_vars: Vec<Vec<f64>> = match serde_json::from_str(x_vars_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x_vars: {}", e)),
    };

    // Build design matrix with intercept column
    let (x, n, p) = build_design_matrix(&y, &x_vars);

    if n <= p + 1 {
        return error_json(&format!(
            "Insufficient data: need at least {} observations for {} predictors",
            p + 2,
            p
        ));
    }

    // Configure lasso options
    let options = regularized::lasso::LassoFitOptions {
        lambda,
        intercept: true,
        standardize,
        max_iter,
        tol,
        ..Default::default()
    };

    match regularized::lasso::lasso_fit(&x, &y, &options) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize lasso regression result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs Elastic Net regression via WASM.
///
/// Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `variable_names` - JSON array of variable names
/// * `lambda` - Regularization strength (>= 0)
/// * `alpha` - Elastic net mixing parameter (0 = Ridge, 1 = Lasso)
/// * `standardize` - Whether to standardize predictors (recommended: true)
/// * `max_iter` - Maximum coordinate descent iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// JSON string containing regression results (same fields as Lasso).
///
/// # Errors
///
/// Returns a JSON error object if parsing fails, parameters are invalid,
/// or domain check fails.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn elastic_net_regression(
    y_json: &str,
    x_vars_json: &str,
    _variable_names: &str,
    lambda: f64,
    alpha: f64,
    standardize: bool,
    max_iter: usize,
    tol: f64,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    // Parse JSON input
    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x_vars: Vec<Vec<f64>> = match serde_json::from_str(x_vars_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x_vars: {}", e)),
    };

    // Build design matrix with intercept column
    let (x, n, p) = build_design_matrix(&y, &x_vars);

    if n <= p + 1 {
        return error_json(&format!(
            "Insufficient data: need at least {} observations for {} predictors",
            p + 2,
            p
        ));
    }

    // Configure elastic net options
    let options = regularized::elastic_net::ElasticNetOptions {
        lambda,
        alpha,
        intercept: true,
        standardize,
        max_iter,
        tol,
        ..Default::default()
    };

    match regularized::elastic_net::elastic_net_fit(&x, &y, &options) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize elastic net regression result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Generates a lambda path for regularized regression via WASM.
///
/// Creates a logarithmically-spaced sequence of lambda values from lambda_max
/// (where all penalized coefficients are zero) down to lambda_min. This is
/// useful for fitting regularization paths and selecting optimal lambda via
/// cross-validation.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `n_lambda` - Number of lambda values to generate (default: 100)
/// * `lambda_min_ratio` - Ratio for smallest lambda (default: 0.0001 if n >= p, else 0.01)
///
/// # Returns
///
/// JSON string containing:
/// - `lambda_path` - Array of lambda values in decreasing order
/// - `lambda_max` - Maximum lambda value
/// - `lambda_min` - Minimum lambda value
/// - `n_lambda` - Number of lambda values
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn make_lambda_path(
    y_json: &str,
    x_vars_json: &str,
    n_lambda: usize,
    lambda_min_ratio: f64,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    // Parse JSON input
    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x_vars: Vec<Vec<f64>> = match serde_json::from_str(x_vars_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x_vars: {}", e)),
    };

    // Build design matrix with intercept column
    let (x, n, p) = build_design_matrix(&y, &x_vars);

    // Standardize X for lambda path computation
    let x_mean: Vec<f64> = (0..x.cols)
        .map(|j| {
            if j == 0 {
                1.0 // Intercept column
            } else {
                (0..n).map(|i| x.get(i, j)).sum::<f64>() / n as f64
            }
        })
        .collect();

    let x_standardized: Vec<f64> = (0..x.cols)
        .map(|j| {
            if j == 0 {
                0.0 // Intercept column - no centering
            } else {
                let mean = x_mean[j];
                let variance =
                    (0..n).map(|i| (x.get(i, j) - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                variance.sqrt()
            }
        })
        .collect();

    // Build standardized X matrix
    let mut x_standardized_data = vec![1.0; n * (p + 1)];
    for j in 0..x.cols {
        for i in 0..n {
            if j == 0 {
                x_standardized_data[i * (p + 1)] = 1.0; // Intercept
            } else {
                let std = x_standardized[j];
                if std > 1e-10 {
                    x_standardized_data[i * (p + 1) + j] = (x.get(i, j) - x_mean[j]) / std;
                } else {
                    x_standardized_data[i * (p + 1) + j] = 0.0;
                }
            }
        }
    }
    let x_standardized = linalg::Matrix::new(n, p + 1, x_standardized_data);

    // Center y
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

    // Configure lambda path options
    let options = regularized::path::LambdaPathOptions {
        nlambda: n_lambda.max(1),
        lambda_min_ratio: if lambda_min_ratio > 0.0 {
            Some(lambda_min_ratio)
        } else {
            None
        },
        alpha: 1.0, // Lasso
        ..Default::default()
    };

    let lambda_path =
        regularized::path::make_lambda_path(&x_standardized, &y_centered, &options, None, Some(0));

    let lambda_max = lambda_path.first().copied().unwrap_or(0.0);
    let lambda_min = lambda_path.last().copied().unwrap_or(0.0);

    // Return as JSON (note: infinity serializes as null in JSON, handled in JS)
    let result = serde_json::json!({
        "lambda_path": lambda_path,
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "n_lambda": lambda_path.len()
    });

    result.to_string()
}

/// Helper function to build a design matrix from column vectors.
///
/// # Arguments
///
/// * `y` - Response variable (used to determine n)
/// * `x_vars` - Predictor column vectors
///
/// # Returns
///
/// A tuple of (Matrix, n, p) where p is the number of predictors (excluding intercept)
fn build_design_matrix(y: &[f64], x_vars: &[Vec<f64>]) -> (linalg::Matrix, usize, usize) {
    let n = y.len();
    let p = x_vars.len();

    let mut x_data = vec![1.0; n * (p + 1)]; // Intercept column
    for (j, x_var) in x_vars.iter().enumerate() {
        for (i, &val) in x_var.iter().enumerate() {
            x_data[i * (p + 1) + j + 1] = val;
        }
    }

    (linalg::Matrix::new(n, p + 1, x_data), n, p)
}
