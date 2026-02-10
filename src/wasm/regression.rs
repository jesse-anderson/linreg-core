//! OLS and Weighted Least Squares regression for WASM
//!
//! Provides WASM bindings for ordinary least squares and weighted least squares regression.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use super::domain::check_domain;
use crate::core;
use crate::error::{error_json, error_to_json};

/// Performs OLS regression via WASM.
///
/// All parameters and return values are JSON-encoded strings for JavaScript
/// interoperability. Returns regression output including coefficients,
/// standard errors, diagnostic statistics, and VIF analysis.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values: `[1.0, 2.0, 3.0]`
/// * `x_vars_json` - JSON array of predictor arrays: `[[1.0, 2.0], [0.5, 1.0]]`
/// * `variable_names` - JSON array of variable names: `["Intercept", "X1", "X2"]`
///
/// # Returns
///
/// JSON string containing the complete regression output with coefficients,
/// standard errors, t-statistics, p-values, R², F-statistic, residuals, leverage, VIF, etc.
///
/// # Errors
///
/// Returns a JSON error object if:
/// - JSON parsing fails
/// - Insufficient data (n ≤ k + 1)
/// - Matrix is singular
/// - Domain check fails
#[wasm_bindgen]
pub fn ols_regression(y_json: &str, x_vars_json: &str, variable_names: &str) -> String {
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

    let names: Vec<String> = match serde_json::from_str(variable_names) {
        Ok(v) => v,
        Err(_) => vec!["Intercept".to_string()],
    };

    // Call core function
    match core::ols_regression(&y, &x_vars, &names) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize output")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs Weighted Least Squares (WLS) regression via WASM.
///
/// WLS regression allows each observation to have a different weight, which is
/// useful for handling heteroscedasticity or when observations have different
/// precision/variances.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `weights_json` - JSON array of observation weights (must be non-negative)
///
/// # Returns
///
/// JSON string containing:
/// - `coefficients` - Coefficient values (including intercept as first element)
/// - `standard_errors` - Standard errors of the coefficients
/// - `t_statistics` - t-statistics for coefficient significance tests
/// - `p_values` - Two-tailed p-values for coefficients
/// - `r_squared` - R-squared (coefficient of determination)
/// - `adj_r_squared` - Adjusted R-squared
/// - `f_statistic` - F-statistic for overall model significance
/// - `f_p_value` - p-value for F-statistic
/// - `residual_std_error` - Residual standard error (sigma-hat estimate)
/// - `df_residuals` - Degrees of freedom for residuals
/// - `df_model` - Degrees of freedom for the model
/// - `fitted_values` - Fitted values (predicted values)
/// - `residuals` - Residuals (y - ŷ)
/// - `mse` - Mean squared error
/// - `rmse` - Root mean squared error
/// - `mae` - Mean absolute error
/// - `n` - Number of observations
/// - `k` - Number of predictors (excluding intercept)
///
/// # Errors
///
/// Returns a JSON error object if parsing fails, parameters are invalid,
/// or domain check fails.
#[wasm_bindgen]
pub fn wls_regression(y_json: &str, x_vars_json: &str, weights_json: &str) -> String {
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

    let weights: Vec<f64> = match serde_json::from_str(weights_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse weights: {}", e)),
    };

    match crate::weighted_regression::wls_regression(&y, &x_vars, &weights) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize WLS regression result")),
        Err(e) => error_json(&e.to_string()),
    }
}
