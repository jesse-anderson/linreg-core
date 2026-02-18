//! Prediction intervals for WASM
//!
//! Provides WASM bindings for computing prediction intervals from
//! OLS, Ridge, Lasso, and Elastic Net regression models.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use super::domain::check_domain;
use crate::error::{error_json, error_to_json};
use crate::linalg;
use crate::regularized;

/// Computes OLS prediction intervals via WASM.
///
/// Fits an OLS model to the training data and computes prediction intervals
/// for the new observations.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays (training data)
/// * `new_x_json` - JSON array of predictor arrays (new observations)
/// * `alpha` - Significance level (e.g., 0.05 for 95% PI)
///
/// # Returns
///
/// JSON string containing predicted values, lower/upper bounds, SE, leverage.
#[wasm_bindgen]
pub fn ols_prediction_intervals(
    y_json: &str,
    x_vars_json: &str,
    new_x_json: &str,
    alpha: f64,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x_vars: Vec<Vec<f64>> = match serde_json::from_str(x_vars_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x_vars: {}", e)),
    };

    let new_x_vecs: Vec<Vec<f64>> = match serde_json::from_str(new_x_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse new_x: {}", e)),
    };

    let new_x_refs: Vec<&[f64]> = new_x_vecs.iter().map(|v| v.as_slice()).collect();

    match crate::prediction_intervals::prediction_intervals(&y, &x_vars, &new_x_refs, alpha) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize prediction intervals")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Computes approximate Ridge regression prediction intervals via WASM.
///
/// Fits a Ridge model and computes conservative prediction intervals using
/// leverage from unpenalized X'X and MSE from the ridge fit.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays (training data)
/// * `new_x_json` - JSON array of predictor arrays (new observations)
/// * `alpha` - Significance level (e.g., 0.05 for 95% PI)
/// * `lambda` - Regularization strength
/// * `standardize` - Whether to standardize predictors
#[wasm_bindgen]
pub fn ridge_prediction_intervals(
    y_json: &str,
    x_vars_json: &str,
    new_x_json: &str,
    alpha: f64,
    lambda: f64,
    standardize: bool,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x_vars: Vec<Vec<f64>> = match serde_json::from_str(x_vars_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x_vars: {}", e)),
    };

    let new_x_vecs: Vec<Vec<f64>> = match serde_json::from_str(new_x_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse new_x: {}", e)),
    };

    // Build design matrix with intercept column
    let (x, _n, _p) = build_design_matrix(&y, &x_vars);

    let options = regularized::ridge::RidgeFitOptions {
        lambda,
        intercept: true,
        standardize,
        ..Default::default()
    };

    let fit = match regularized::ridge::ridge_fit(&x, &y, &options) {
        Ok(f) => f,
        Err(e) => return error_json(&e.to_string()),
    };

    let new_x_refs: Vec<&[f64]> = new_x_vecs.iter().map(|v| v.as_slice()).collect();

    match crate::prediction_intervals::ridge_prediction_intervals(&fit, &x_vars, &new_x_refs, alpha) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize prediction intervals")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Computes approximate Lasso regression prediction intervals via WASM.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays (training data)
/// * `new_x_json` - JSON array of predictor arrays (new observations)
/// * `alpha` - Significance level (e.g., 0.05 for 95% PI)
/// * `lambda` - Regularization strength
/// * `standardize` - Whether to standardize predictors
/// * `max_iter` - Maximum coordinate descent iterations
/// * `tol` - Convergence tolerance
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn lasso_prediction_intervals(
    y_json: &str,
    x_vars_json: &str,
    new_x_json: &str,
    alpha: f64,
    lambda: f64,
    standardize: bool,
    max_iter: usize,
    tol: f64,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x_vars: Vec<Vec<f64>> = match serde_json::from_str(x_vars_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x_vars: {}", e)),
    };

    let new_x_vecs: Vec<Vec<f64>> = match serde_json::from_str(new_x_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse new_x: {}", e)),
    };

    let (x, _n, _p) = build_design_matrix(&y, &x_vars);

    let options = regularized::lasso::LassoFitOptions {
        lambda,
        intercept: true,
        standardize,
        max_iter,
        tol,
        ..Default::default()
    };

    let fit = match regularized::lasso::lasso_fit(&x, &y, &options) {
        Ok(f) => f,
        Err(e) => return error_json(&e.to_string()),
    };

    let new_x_refs: Vec<&[f64]> = new_x_vecs.iter().map(|v| v.as_slice()).collect();

    match crate::prediction_intervals::lasso_prediction_intervals(&fit, &x_vars, &new_x_refs, alpha) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize prediction intervals")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Computes approximate Elastic Net regression prediction intervals via WASM.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays (training data)
/// * `new_x_json` - JSON array of predictor arrays (new observations)
/// * `alpha` - Significance level (e.g., 0.05 for 95% PI)
/// * `lambda` - Regularization strength
/// * `enet_alpha` - Elastic net mixing parameter (0 = Ridge, 1 = Lasso)
/// * `standardize` - Whether to standardize predictors
/// * `max_iter` - Maximum coordinate descent iterations
/// * `tol` - Convergence tolerance
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn elastic_net_prediction_intervals(
    y_json: &str,
    x_vars_json: &str,
    new_x_json: &str,
    alpha: f64,
    lambda: f64,
    enet_alpha: f64,
    standardize: bool,
    max_iter: usize,
    tol: f64,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x_vars: Vec<Vec<f64>> = match serde_json::from_str(x_vars_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x_vars: {}", e)),
    };

    let new_x_vecs: Vec<Vec<f64>> = match serde_json::from_str(new_x_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse new_x: {}", e)),
    };

    let (x, _n, _p) = build_design_matrix(&y, &x_vars);

    let options = regularized::elastic_net::ElasticNetOptions {
        lambda,
        alpha: enet_alpha,
        intercept: true,
        standardize,
        max_iter,
        tol,
        ..Default::default()
    };

    let fit = match regularized::elastic_net::elastic_net_fit(&x, &y, &options) {
        Ok(f) => f,
        Err(e) => return error_json(&e.to_string()),
    };

    let new_x_refs: Vec<&[f64]> = new_x_vecs.iter().map(|v| v.as_slice()).collect();

    match crate::prediction_intervals::elastic_net_prediction_intervals(&fit, &x_vars, &new_x_refs, alpha) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize prediction intervals")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Helper function to build a design matrix from column vectors.
fn build_design_matrix(y: &[f64], x_vars: &[Vec<f64>]) -> (linalg::Matrix, usize, usize) {
    let n = y.len();
    let p = x_vars.len();

    let mut x_data = vec![1.0; n * (p + 1)];
    for (j, x_var) in x_vars.iter().enumerate() {
        for (i, &val) in x_var.iter().enumerate() {
            x_data[i * (p + 1) + j + 1] = val;
        }
    }

    (linalg::Matrix::new(n, p + 1, x_data), n, p)
}
