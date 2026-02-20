//! Polynomial Regression WASM Bindings
//!
//! JavaScript-accessible polynomial regression functions.
//! All functions accept and return JSON strings.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use super::domain::check_domain;
use crate::error::{error_json, error_to_json};
use crate::polynomial::{
    predict, polynomial_elastic_net, polynomial_lasso, polynomial_regression, polynomial_ridge,
    PolynomialFit, PolynomialOptions,
};

/// Fit polynomial regression via WASM.
///
/// # Arguments
///
/// * `y_json` - JSON array of response values, e.g. `[1.0, 4.0, 9.0]`
/// * `x_json` - JSON array of predictor values, e.g. `[1.0, 2.0, 3.0]`
/// * `degree` - Polynomial degree (≥ 1)
/// * `center` - Whether to center x before expanding (reduces multicollinearity)
/// * `standardize` - Whether to standardize polynomial features
///
/// # Returns
///
/// JSON string of the complete [`PolynomialFit`], which includes:
/// - `ols_output` — full OLS regression output (coefficients, R², F-stat, etc.)
/// - `degree`, `centered`, `x_mean`, `x_std`, `standardized`
/// - `feature_names`, `feature_means`, `feature_stds`
///
/// The returned JSON can be passed directly to [`polynomial_predict_wasm`].
///
/// # Errors
///
/// Returns a JSON error object `{"error": "…"}` if:
/// - JSON parsing fails
/// - `degree` is 0
/// - `y` and `x` have different lengths
/// - Insufficient data
/// - Domain check fails
#[wasm_bindgen]
pub fn polynomial_regression_wasm(
    y_json: &str,
    x_json: &str,
    degree: usize,
    center: bool,
    standardize: bool,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x: Vec<f64> = match serde_json::from_str(x_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x: {}", e)),
    };

    let options = PolynomialOptions {
        degree,
        center,
        standardize,
        intercept: true,
    };

    match polynomial_regression(&y, &x, &options) {
        Ok(fit) => serde_json::to_string(&fit)
            .unwrap_or_else(|_| error_json("Failed to serialize polynomial fit")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Predict using a fitted polynomial model via WASM.
///
/// # Arguments
///
/// * `fit_json` - JSON string of the `PolynomialFit` returned by [`polynomial_regression_wasm`]
/// * `x_new_json` - JSON array of new predictor values, e.g. `[6.0, 7.0]`
///
/// # Returns
///
/// JSON array of predicted values, or a JSON error object on failure.
#[wasm_bindgen]
pub fn polynomial_predict_wasm(fit_json: &str, x_new_json: &str) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let fit: PolynomialFit = match serde_json::from_str(fit_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse polynomial fit: {}", e)),
    };

    let x_new: Vec<f64> = match serde_json::from_str(x_new_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x_new: {}", e)),
    };

    match predict(&fit, &x_new) {
        Ok(preds) => serde_json::to_string(&preds)
            .unwrap_or_else(|_| error_json("Failed to serialize predictions")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Fit polynomial Ridge regression via WASM.
///
/// Ridge regularization helps with multicollinearity in polynomial features.
///
/// # Arguments
///
/// * `y_json` - JSON array of response values, e.g. `[1.0, 4.0, 9.0]`
/// * `x_json` - JSON array of predictor values, e.g. `[1.0, 2.0, 3.0]`
/// * `degree` - Polynomial degree (≥ 1)
/// * `lambda` - Regularization strength (≥ 0)
/// * `center` - Whether to center x before expansion (reduces multicollinearity)
/// * `standardize` - Whether to standardize features (recommended)
///
/// # Returns
///
/// JSON string of the RidgeFit result, which includes:
/// - `intercept`, `coefficients`
/// - `fitted_values`, `residuals`
/// - `r_squared`, `adj_r_squared`, `mse`, `rmse`, `mae`
/// - `effective_df`, `log_likelihood`, `aic`, `bic`
#[wasm_bindgen]
pub fn polynomial_ridge_wasm(
    y_json: &str,
    x_json: &str,
    degree: usize,
    lambda: f64,
    center: bool,
    standardize: bool,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x: Vec<f64> = match serde_json::from_str(x_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x: {}", e)),
    };

    match polynomial_ridge(&y, &x, degree, lambda, center, standardize) {
        Ok(fit) => serde_json::to_string(&fit)
            .unwrap_or_else(|_| error_json("Failed to serialize ridge fit")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Fit polynomial Lasso regression via WASM.
///
/// Lasso can perform variable selection among polynomial terms,
/// potentially eliminating higher-order terms.
///
/// # Arguments
///
/// * `y_json` - JSON array of response values, e.g. `[1.0, 4.0, 9.0]`
/// * `x_json` - JSON array of predictor values, e.g. `[1.0, 2.0, 3.0]`
/// * `degree` - Polynomial degree (≥ 1)
/// * `lambda` - Regularization strength (≥ 0)
/// * `center` - Whether to center x before expansion (reduces multicollinearity)
/// * `standardize` - Whether to standardize features (recommended)
///
/// # Returns
///
/// JSON string of the LassoFit result, which includes:
/// - `intercept`, `coefficients`
/// - `fitted_values`, `residuals`
/// - `r_squared`, `adj_r_squared`, `mse`, `rmse`, `mae`
/// - `n_nonzero`, `converged`, `n_iterations`
/// - `log_likelihood`, `aic`, `bic`
#[wasm_bindgen]
pub fn polynomial_lasso_wasm(
    y_json: &str,
    x_json: &str,
    degree: usize,
    lambda: f64,
    center: bool,
    standardize: bool,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x: Vec<f64> = match serde_json::from_str(x_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x: {}", e)),
    };

    match polynomial_lasso(&y, &x, degree, lambda, center, standardize) {
        Ok(fit) => serde_json::to_string(&fit)
            .unwrap_or_else(|_| error_json("Failed to serialize lasso fit")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Fit polynomial Elastic Net regression via WASM.
///
/// Elastic Net combines L1 and L2 penalties, balancing variable selection
/// with multicollinearity handling.
///
/// # Arguments
///
/// * `y_json` - JSON array of response values, e.g. `[1.0, 4.0, 9.0]`
/// * `x_json` - JSON array of predictor values, e.g. `[1.0, 2.0, 3.0]`
/// * `degree` - Polynomial degree (≥ 1)
/// * `lambda` - Regularization strength (≥ 0)
/// * `alpha` - Mixing parameter: 0 = Ridge, 1 = Lasso
/// * `center` - Whether to center x before expansion (reduces multicollinearity)
/// * `standardize` - Whether to standardize features (recommended)
///
/// # Returns
///
/// JSON string of the ElasticNetFit result, which includes:
/// - `intercept`, `coefficients`
/// - `fitted_values`, `residuals`
/// - `r_squared`, `adj_r_squared`, `mse`, `rmse`, `mae`
/// - `n_nonzero`, `converged`, `n_iterations`
/// - `log_likelihood`, `aic`, `bic`
#[wasm_bindgen]
pub fn polynomial_elastic_net_wasm(
    y_json: &str,
    x_json: &str,
    degree: usize,
    lambda: f64,
    alpha: f64,
    center: bool,
    standardize: bool,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let y: Vec<f64> = match serde_json::from_str(y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse y: {}", e)),
    };

    let x: Vec<f64> = match serde_json::from_str(x_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse x: {}", e)),
    };

    match polynomial_elastic_net(&y, &x, degree, lambda, alpha, center, standardize) {
        Ok(fit) => serde_json::to_string(&fit)
            .unwrap_or_else(|_| error_json("Failed to serialize elastic net fit")),
        Err(e) => error_json(&e.to_string()),
    }
}
