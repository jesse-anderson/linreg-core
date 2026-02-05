//! WASM-specific bindings for linreg-core
//!
//! This module contains all WebAssembly bindings using wasm-bindgen.
//! These functions are only compiled when the "wasm" feature is enabled.
//!
//! All WASM functions accept and return JSON strings for JavaScript interoperability.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;
use std::collections::HashSet;
use serde::Serialize;

// Re-export from parent modules
use crate::core;
use crate::diagnostics;
use crate::distributions::{normal_inverse_cdf, student_t_cdf};
use crate::error::{error_json, error_to_json, Error, Result};
use crate::linalg;
use crate::loess;
use crate::regularized;
use crate::stats;

// ============================================================================
// CSV Parsing
// ============================================================================

#[derive(Serialize)]
struct ParsedCsv {
    headers: Vec<String>,
    data: Vec<serde_json::Map<String, serde_json::Value>>,
    numeric_columns: Vec<String>,
}

#[wasm_bindgen]
/// Parses CSV data and returns it as a JSON string.
///
/// Parses the CSV content and identifies numeric columns. Returns a JSON object
/// with headers, data rows, and a list of numeric column names.
///
/// # Arguments
///
/// * `content` - CSV content as a string
///
/// # Returns
///
/// JSON string with structure:
/// ```json
/// {
///   "headers": ["col1", "col2", ...],
///   "data": [{"col1": 1.0, "col2": "text"}, ...],
///   "numeric_columns": ["col1", ...]
/// }
/// ```
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
pub fn parse_csv(content: &str) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(content.as_bytes());

    // Get headers
    let headers: Vec<String> = match reader.headers() {
        Ok(h) => h.iter().map(|s| s.to_string()).collect(),
        Err(e) => return error_json(&format!("Failed to read headers: {}", e)),
    };

    let mut data = Vec::new();
    let mut numeric_col_set = HashSet::new();

    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(e) => return error_json(&format!("Failed to parse CSV record: {}", e)),
        };

        if record.len() != headers.len() {
            continue;
        }

        let mut row_map = serde_json::Map::new();

        for (i, field) in record.iter().enumerate() {
            if i >= headers.len() {
                continue;
            }

            let header = &headers[i];
            let val_trimmed = field.trim();

            // Try to parse as f64
            if let Ok(num) = val_trimmed.parse::<f64>() {
                if num.is_finite() {
                    row_map.insert(
                        header.clone(),
                        serde_json::Value::Number(serde_json::Number::from_f64(num).unwrap()),
                    );
                    numeric_col_set.insert(header.clone());
                    continue;
                }
            }

            // Fallback to string
            row_map.insert(
                header.clone(),
                serde_json::Value::String(val_trimmed.to_string()),
            );
        }
        data.push(row_map);
    }

    let mut numeric_columns: Vec<String> = numeric_col_set.into_iter().collect();
    numeric_columns.sort();

    let output = ParsedCsv {
        headers,
        data,
        numeric_columns,
    };

    serde_json::to_string(&output).unwrap_or_else(|_| error_json("Failed to serialize CSV output"))
}

// ============================================================================
// OLS Regression WASM Wrapper
// ============================================================================

#[wasm_bindgen]
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

// ============================================================================
// Diagnostic Tests WASM Wrappers
// ============================================================================

/// Performs the Rainbow test for linearity via WASM.
///
/// The Rainbow test checks whether the relationship between predictors and response
/// is linear. A significant p-value suggests non-linearity.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `fraction` - Fraction of data to use in the central subset (0.0 to 1.0, typically 0.5)
/// * `method` - Method to use: "r", "python", or "both" (case-insensitive, defaults to "r")
///
/// # Returns
///
/// JSON string containing test statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn rainbow_test(y_json: &str, x_vars_json: &str, fraction: f64, method: &str) -> String {
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

    // Parse method parameter (default to "r" for R)
    let method = match method.to_lowercase().as_str() {
        "python" => diagnostics::RainbowMethod::Python,
        "both" => diagnostics::RainbowMethod::Both,
        _ => diagnostics::RainbowMethod::R, // Default to R
    };

    match diagnostics::rainbow_test(&y, &x_vars, fraction, method) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Rainbow test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs the Harvey-Collier test for linearity via WASM.
///
/// The Harvey-Collier test checks whether the residuals exhibit a linear trend,
/// which would indicate that the model's functional form is misspecified.
/// A significant p-value suggests non-linearity.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
///
/// # Returns
///
/// JSON string containing test statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn harvey_collier_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::harvey_collier_test(&y, &x_vars, diagnostics::HarveyCollierMethod::R) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Harvey-Collier test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs the Breusch-Pagan test for heteroscedasticity via WASM.
///
/// The Breusch-Pagan test checks whether the variance of residuals is constant
/// across the range of predicted values (homoscedasticity assumption).
/// A significant p-value suggests heteroscedasticity.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
///
/// # Returns
///
/// JSON string containing test statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn breusch_pagan_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::breusch_pagan_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Breusch-Pagan test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs the White test for heteroscedasticity via WASM.
///
/// The White test is a more general test for heteroscedasticity that does not
/// assume a specific form of heteroscedasticity. A significant p-value suggests
/// that the error variance is not constant.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `method` - Method to use: "r", "python", or "both" (case-insensitive, defaults to "r")
///
/// # Returns
///
/// JSON string containing test statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn white_test(y_json: &str, x_vars_json: &str, method: &str) -> String {
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

    // Parse method parameter (default to "r" for R)
    let method = match method.to_lowercase().as_str() {
        "python" => diagnostics::WhiteMethod::Python,
        "both" => diagnostics::WhiteMethod::Both,
        _ => diagnostics::WhiteMethod::R, // Default to R
    };

    match diagnostics::white_test(&y, &x_vars, method) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize White test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs the R method White test for heteroscedasticity via WASM.
///
/// This implementation matches R's `skedastic::white()` function behavior.
/// Uses the standard QR decomposition and the R-specific auxiliary matrix
/// structure (intercept, X, X² only - no cross-products).
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays (each array is a column)
///
/// # Returns
///
/// JSON string containing test statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn r_white_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::r_white_method(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize R White test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs the Python method White test for heteroscedasticity via WASM.
///
/// This implementation matches Python's `statsmodels.stats.diagnostic.het_white()` function.
/// Uses the LINPACK QR decomposition with column pivoting and the Python-specific
/// auxiliary matrix structure (intercept, X, X², and cross-products).
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays (each array is a column)
///
/// # Returns
///
/// JSON string containing test statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn python_white_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::python_white_method(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Python White test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs the Jarque-Bera test for normality via WASM.
///
/// The Jarque-Bera test checks whether the residuals are normally distributed
/// by examining skewness and kurtosis. A significant p-value suggests that
/// the residuals deviate from normality.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
///
/// # Returns
///
/// JSON string containing test statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn jarque_bera_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::jarque_bera_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Jarque-Bera test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

// ============================================================================
// Durbin-Watson Test (WASM wrapper)
// ============================================================================

/// Performs the Durbin-Watson test for autocorrelation via WASM.
///
/// The Durbin-Watson test checks for autocorrelation in the residuals.
/// Values near 2 indicate no autocorrelation, values near 0 suggest positive
/// autocorrelation, and values near 4 suggest negative autocorrelation.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
///
/// # Returns
///
/// JSON string containing the DW statistic and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn durbin_watson_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::durbin_watson_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Durbin-Watson test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

// ============================================================================
// Shapiro-Wilk Test (WASM wrapper)
// ============================================================================

/// Performs the Shapiro-Wilk test for normality via WASM.
///
/// The Shapiro-Wilk test is a powerful test for normality,
/// especially for small to moderate sample sizes (3 ≤ n ≤ 5000). It tests
/// the null hypothesis that the residuals are normally distributed.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
///
/// # Returns
///
/// JSON string containing the W statistic (ranges from 0 to 1), p-value,
/// and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn shapiro_wilk_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::shapiro_wilk_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Shapiro-Wilk test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs the Anderson-Darling test for normality via WASM.
///
/// The Anderson-Darling test checks whether the residuals are normally distributed
/// by comparing the empirical distribution to the expected normal distribution.
/// This test is particularly sensitive to deviations in the tails of the distribution.
/// A significant p-value suggests that the residuals deviate from normality.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
///
/// # Returns
///
/// JSON string containing the A² statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn anderson_darling_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::anderson_darling_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Anderson-Darling test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

// ============================================================================
// Cook's Distance (WASM wrapper)
// ============================================================================

/// Computes Cook's distance for identifying influential observations via WASM.
///
/// Cook's distance measures how much each observation influences the regression
/// model by comparing coefficient estimates with and without that observation.
/// Unlike hypothesis tests, this is an influence measure - not a test with p-values.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
///
/// # Returns
///
/// JSON string containing:
/// - Vector of Cook's distances (one per observation)
/// - Thresholds for identifying influential observations
/// - Indices of potentially influential observations
/// - Interpretation and guidance
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn cooks_distance_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::cooks_distance_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Cook's distance result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs DFBETAS analysis via WASM.
///
/// DFBETAS measures the influence of each observation on each regression coefficient.
/// For each observation and each coefficient, it computes the standardized change
/// in the coefficient when that observation is omitted.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
///
/// # Returns
///
/// JSON string containing the DFBETAS matrix, threshold, and influential observations.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn dfbetas_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::dfbetas_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize DFBETAS result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs DFFITS analysis via WASM.
///
/// DFFITS measures the influence of each observation on its own fitted value.
/// It is the standardized change in the fitted value when that observation
/// is omitted from the model.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
///
/// # Returns
///
/// JSON string containing the DFFITS vector, threshold, and influential observations.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn dffits_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::dffits_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize DFFITS result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs Variance Inflation Factor (VIF) analysis via WASM.
///
/// VIF measures how much the variance of regression coefficients is inflated
/// due to multicollinearity among predictor variables. High VIF values indicate
/// that a predictor is highly correlated with other predictors.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
///
/// # Returns
///
/// JSON string containing the maximum VIF, detailed VIF results for each predictor,
/// interpretation, and guidance.
///
/// # Interpretation
///
/// - VIF = 1: No correlation with other predictors
/// - VIF > 5: Moderate multicollinearity (concerning)
/// - VIF > 10: High multicollinearity (severe)
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn vif_test(y_json: &str, x_vars_json: &str) -> String {
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

    match diagnostics::vif_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize VIF result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs the RESET test for model specification error via WASM.
///
/// The RESET (Regression Specification Error Test) test checks whether the model
/// is correctly specified by testing if additional terms (powers of fitted values,
/// regressors, or first principal component) significantly improve the model fit.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `powers_json` - JSON array of powers to use (e.g., [2, 3] for ŷ², ŷ³)
/// * `type_` - Type of terms to add: "fitted", "regressor", or "princomp"
///
/// # Returns
///
/// JSON string containing the F-statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn reset_test(y_json: &str, x_vars_json: &str, powers_json: &str, type_: &str) -> String {
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

    let powers: Vec<usize> = match serde_json::from_str(powers_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse powers: {}", e)),
    };

    // Parse reset type (default to "fitted")
    let reset_type = match type_.to_lowercase().as_str() {
        "regressor" => diagnostics::ResetType::Regressor,
        "princomp" => diagnostics::ResetType::PrincipalComponent,
        _ => diagnostics::ResetType::Fitted,
    };

    match diagnostics::reset_test(&y, &x_vars, &powers, reset_type) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize RESET test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs the Breusch-Godfrey test for higher-order serial correlation via WASM.
///
/// Unlike the Durbin-Watson test which only detects first-order autocorrelation,
/// the Breusch-Godfrey test can detect serial correlation at any lag order.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `order` - Maximum order of serial correlation to test (default: 1)
/// * `test_type` - Type of test statistic: "chisq" or "f" (default: "chisq")
///
/// # Returns
///
/// JSON string containing test statistic, p-value, degrees of freedom, and interpretation.
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn breusch_godfrey_test(y_json: &str, x_vars_json: &str, order: usize, test_type: &str) -> String {
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

    // Parse test type (default to "chisq")
    let bg_test_type = match test_type.to_lowercase().as_str() {
        "f" => diagnostics::BGTestType::F,
        _ => diagnostics::BGTestType::Chisq,
    };

    match diagnostics::breusch_godfrey_test(&y, &x_vars, order, bg_test_type) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Breusch-Godfrey test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

// ============================================================================
// LOESS (Locally Estimated Scatterplot Smoothing) WASM Wrappers
// ============================================================================

/// Performs LOESS regression via WASM.
///
/// LOESS (Locally Estimated Scatterplot Smoothing) is a non-parametric
/// regression method that fits multiple regressions in local subsets
/// of data to create a smooth curve through the data points.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `span` - Fraction of data used in each local fit (0.0 to 1.0)
/// * `degree` - Degree of local polynomial: 0 (constant), 1 (linear), or 2 (quadratic)
/// * `robust_iterations` - Number of robustness iterations (0 for non-robust fit)
/// * `surface` - Surface computation method: "direct" or "interpolate"
///
/// # Returns
///
/// JSON string containing:
/// - `fitted` - Fitted values at each observation point
/// - `span` - Span parameter used
/// - `degree` - Degree of polynomial used
/// - `robust_iterations` - Number of robustness iterations performed
/// - `surface` - Surface computation method used
///
/// # Errors
///
/// Returns a JSON error object if parsing fails or domain check fails.
#[wasm_bindgen]
pub fn loess_fit(
    y_json: &str,
    x_vars_json: &str,
    span: f64,
    degree: usize,
    robust_iterations: usize,
    surface: &str,
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

    let n_predictors = x_vars.len();

    // Parse surface parameter (default to "direct")
    let surface = match surface.to_lowercase().as_str() {
        "interpolate" => loess::LoessSurface::Interpolate,
        _ => loess::LoessSurface::Direct,
    };

    let options = loess::LoessOptions {
        span,
        degree,
        robust_iterations,
        n_predictors,
        surface,
    };

    match loess::loess_fit(&y, &x_vars, &options) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize LOESS result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs LOESS prediction at new query points via WASM.
///
/// Predicts LOESS fitted values at arbitrary new points by redoing the
/// local fitting at each query point using the original training data.
///
/// # Arguments
///
/// * `new_x_json` - JSON array of new predictor values (p vectors, each of length m)
/// * `original_x_json` - JSON array of original training predictors
/// * `original_y_json` - JSON array of original training response values
/// * `span` - Span parameter (must match the original fit)
/// * `degree` - Degree of polynomial (must match the original fit)
/// * `robust_iterations` - Robustness iterations (must match the original fit)
/// * `surface` - Surface computation method: "direct" or "interpolate"
///
/// # Returns
///
/// JSON string containing:
/// - `predictions` - Vector of predicted values at query points
///
/// # Errors
///
/// Returns a JSON error object if parsing fails, parameters don't match
/// the original fit, or domain check fails.
#[wasm_bindgen]
pub fn loess_predict(
    new_x_json: &str,
    original_x_json: &str,
    original_y_json: &str,
    span: f64,
    degree: usize,
    robust_iterations: usize,
    surface: &str,
) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    let new_x: Vec<Vec<f64>> = match serde_json::from_str(new_x_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse new_x: {}", e)),
    };

    let original_x: Vec<Vec<f64>> = match serde_json::from_str(original_x_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse original_x: {}", e)),
    };

    let original_y: Vec<f64> = match serde_json::from_str(original_y_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse original_y: {}", e)),
    };

    let n_predictors = original_x.len();

    // Create a LoessFit with the same parameters (for validation)
    // Parse surface parameter (default to "direct")
    let surface = match surface.to_lowercase().as_str() {
        "interpolate" => loess::LoessSurface::Interpolate,
        _ => loess::LoessSurface::Direct,
    };

    let fit = loess::LoessFit {
        fitted: vec![0.0; original_y.len()], // Dummy fitted values
        predictions: None,
        span,
        degree,
        robust_iterations,
        surface,
    };

    let options = loess::LoessOptions {
        span,
        degree,
        robust_iterations,
        n_predictors,
        surface,
    };

    match fit.predict(&new_x, &original_x, &original_y, &options) {
        Ok(predictions) => {
            let result = serde_json::json!({
                "predictions": predictions
            });
            result.to_string()
        },
        Err(e) => error_json(&e.to_string()),
    }
}

// ============================================================================
// Regularized Regression WASM Wrappers
// ============================================================================

#[wasm_bindgen]
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
    let n = y.len();
    let p = x_vars.len();

    if n <= p + 1 {
        return error_json(&format!(
            "Insufficient data: need at least {} observations for {} predictors",
            p + 2,
            p
        ));
    }

    let mut x_data = vec![1.0; n * (p + 1)]; // Intercept column
    for (j, x_var) in x_vars.iter().enumerate() {
        if x_var.len() != n {
            return error_json(&format!(
                "x_vars[{}] has {} elements, expected {}",
                j,
                x_var.len(),
                n
            ));
        }
        for (i, &val) in x_var.iter().enumerate() {
            x_data[i * (p + 1) + j + 1] = val;
        }
    }

    let x = linalg::Matrix::new(n, p + 1, x_data);

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

#[wasm_bindgen]
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
    let n = y.len();
    let p = x_vars.len();

    if n <= p + 1 {
        return error_json(&format!(
            "Insufficient data: need at least {} observations for {} predictors",
            p + 2,
            p
        ));
    }

    let mut x_data = vec![1.0; n * (p + 1)]; // Intercept column
    for (j, x_var) in x_vars.iter().enumerate() {
        if x_var.len() != n {
            return error_json(&format!(
                "x_vars[{}] has {} elements, expected {}",
                j,
                x_var.len(),
                n
            ));
        }
        for (i, &val) in x_var.iter().enumerate() {
            x_data[i * (p + 1) + j + 1] = val;
        }
    }

    let x = linalg::Matrix::new(n, p + 1, x_data);

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

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
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
    let n = y.len();
    let p = x_vars.len();

    if n <= p + 1 {
        return error_json(&format!(
            "Insufficient data: need at least {} observations for {} predictors",
            p + 2,
            p
        ));
    }

    let mut x_data = vec![1.0; n * (p + 1)]; // Intercept column
    for (j, x_var) in x_vars.iter().enumerate() {
        if x_var.len() != n {
            return error_json(&format!(
                "x_vars[{}] has {} elements, expected {}",
                j,
                x_var.len(),
                n
            ));
        }
        for (i, &val) in x_var.iter().enumerate() {
            x_data[i * (p + 1) + j + 1] = val;
        }
    }

    let x = linalg::Matrix::new(n, p + 1, x_data);

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

#[wasm_bindgen]
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
    let n = y.len();
    let p = x_vars.len();

    let mut x_data = vec![1.0; n * (p + 1)]; // Intercept column
    for (j, x_var) in x_vars.iter().enumerate() {
        if x_var.len() != n {
            return error_json(&format!(
                "x_vars[{}] has {} elements, expected {}",
                j,
                x_var.len(),
                n
            ));
        }
        for (i, &val) in x_var.iter().enumerate() {
            x_data[i * (p + 1) + j + 1] = val;
        }
    }

    let x = linalg::Matrix::new(n, p + 1, x_data);

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

// ============================================================================
// Statistical Utility Functions (WASM wrappers)
// ============================================================================

#[wasm_bindgen]
/// Computes the Student's t-distribution cumulative distribution function.
///
/// Returns P(T ≤ t) for a t-distribution with the given degrees of freedom.
///
/// # Arguments
///
/// * `t` - t-statistic value
/// * `df` - Degrees of freedom
///
/// # Returns
///
/// The CDF value, or `NaN` if domain check fails.
pub fn get_t_cdf(t: f64, df: f64) -> f64 {
    if check_domain().is_err() {
        return f64::NAN;
    }

    student_t_cdf(t, df)
}

#[wasm_bindgen]
/// Computes the critical t-value for a given significance level.
///
/// Returns the t-value such that the area under the t-distribution curve
/// to the right equals alpha/2 (two-tailed test).
///
/// # Arguments
///
/// * `alpha` - Significance level (typically 0.05 for 95% confidence)
/// * `df` - Degrees of freedom
///
/// # Returns
///
/// The critical t-value, or `NaN` if domain check fails.
pub fn get_t_critical(alpha: f64, df: f64) -> f64 {
    if check_domain().is_err() {
        return f64::NAN;
    }

    core::t_critical_quantile(df, alpha)
}

#[wasm_bindgen]
/// Computes the inverse of the standard normal CDF (probit function).
///
/// Returns the z-score such that P(Z ≤ z) = p for a standard normal distribution.
///
/// # Arguments
///
/// * `p` - Probability (0 < p < 1)
///
/// # Returns
///
/// The z-score, or `NaN` if domain check fails.
pub fn get_normal_inverse(p: f64) -> f64 {
    if check_domain().is_err() {
        return f64::NAN;
    }

    normal_inverse_cdf(p)
}

// ============================================================================
// Statistical Utilities (WASM-only)
// ============================================================================

#[wasm_bindgen]
/// Computes the arithmetic mean of a JSON array of f64 values.
///
/// # Arguments
///
/// * `data_json` - JSON string representing an array of f64 values
///
/// # Returns
///
/// JSON string with the mean, or "null" if input is invalid/empty
pub fn stats_mean(data_json: String) -> String {
    if check_domain().is_err() {
        return "null".to_string();
    }

    let data: Vec<f64> = match serde_json::from_str(&data_json) {
        Ok(d) => d,
        Err(_) => return "null".to_string(),
    };

    serde_json::to_string(&stats::mean(&data)).unwrap_or("null".to_string())
}

#[wasm_bindgen]
/// Computes the sample standard deviation of a JSON array of f64 values.
///
/// Uses the (n-1) denominator for unbiased estimation.
///
/// # Arguments
///
/// * `data_json` - JSON string representing an array of f64 values
///
/// # Returns
///
/// JSON string with the standard deviation, or "null" if input is invalid
pub fn stats_stddev(data_json: String) -> String {
    if check_domain().is_err() {
        return "null".to_string();
    }

    let data: Vec<f64> = match serde_json::from_str(&data_json) {
        Ok(d) => d,
        Err(_) => return "null".to_string(),
    };

    serde_json::to_string(&stats::stddev(&data)).unwrap_or("null".to_string())
}

#[wasm_bindgen]
/// Computes the sample variance of a JSON array of f64 values.
///
/// Uses the (n-1) denominator for unbiased estimation.
///
/// # Arguments
///
/// * `data_json` - JSON string representing an array of f64 values
///
/// # Returns
///
/// JSON string with the variance, or "null" if input is invalid
pub fn stats_variance(data_json: String) -> String {
    if check_domain().is_err() {
        return "null".to_string();
    }

    let data: Vec<f64> = match serde_json::from_str(&data_json) {
        Ok(d) => d,
        Err(_) => return "null".to_string(),
    };

    serde_json::to_string(&stats::variance(&data)).unwrap_or("null".to_string())
}

#[wasm_bindgen]
/// Computes the median of a JSON array of f64 values.
///
/// # Arguments
///
/// * `data_json` - JSON string representing an array of f64 values
///
/// # Returns
///
/// JSON string with the median, or "null" if input is invalid/empty
pub fn stats_median(data_json: String) -> String {
    if check_domain().is_err() {
        return "null".to_string();
    }

    let data: Vec<f64> = match serde_json::from_str(&data_json) {
        Ok(d) => d,
        Err(_) => return "null".to_string(),
    };

    serde_json::to_string(&stats::median(&data)).unwrap_or("null".to_string())
}

#[wasm_bindgen]
/// Computes a quantile of a JSON array of f64 values.
///
/// # Arguments
///
/// * `data_json` - JSON string representing an array of f64 values
/// * `q` - Quantile to calculate (0.0 to 1.0)
///
/// # Returns
///
/// JSON string with the quantile value, or "null" if input is invalid
pub fn stats_quantile(data_json: String, q: f64) -> String {
    if check_domain().is_err() {
        return "null".to_string();
    }

    let data: Vec<f64> = match serde_json::from_str(&data_json) {
        Ok(d) => d,
        Err(_) => return "null".to_string(),
    };

    serde_json::to_string(&stats::quantile(&data, q)).unwrap_or("null".to_string())
}

#[wasm_bindgen]
/// Computes the correlation coefficient between two JSON arrays of f64 values.
///
/// # Arguments
///
/// * `x_json` - JSON string representing the first array of f64 values
/// * `y_json` - JSON string representing the second array of f64 values
///
/// # Returns
///
/// JSON string with the correlation coefficient, or "null" if input is invalid
pub fn stats_correlation(x_json: String, y_json: String) -> String {
    if check_domain().is_err() {
        return "null".to_string();
    }

    let x: Vec<f64> = match serde_json::from_str(&x_json) {
        Ok(d) => d,
        Err(_) => return "null".to_string(),
    };

    let y: Vec<f64> = match serde_json::from_str(&y_json) {
        Ok(d) => d,
        Err(_) => return "null".to_string(),
    };

    serde_json::to_string(&stats::correlation(&x, &y)).unwrap_or("null".to_string())
}

// ============================================================================
// Domain Check (WASM-only)
// ============================================================================
//
// By default, all domains are allowed. To enable domain restriction, set the
// LINREG_DOMAIN_RESTRICT environment variable at build time:
//
//   LINREG_DOMAIN_RESTRICT=example.com,yoursite.com wasm-pack build
//
// Example for jesse-anderson.net:
//   LINREG_DOMAIN_RESTRICT=jesse-anderson.net,tools.jesse-anderson.net,localhost,127.0.0.1 wasm-pack build
//
// This allows downstream users to use the library without modification while
// still providing domain restriction as an opt-in security feature.

fn check_domain() -> Result<()> {
    // Read allowed domains from build-time environment variable
    let allowed_domains = option_env!("LINREG_DOMAIN_RESTRICT");

    match allowed_domains {
        Some(domains) if !domains.is_empty() => {
            // Domain restriction is enabled
            let window =
                web_sys::window().ok_or(Error::DomainCheck("No window found".to_string()))?;
            let location = window.location();
            let hostname = location
                .hostname()
                .map_err(|_| Error::DomainCheck("No hostname found".to_string()))?;

            let domain_list: Vec<&str> = domains.split(',').map(|s| s.trim()).collect();

            if domain_list.contains(&hostname.as_str()) {
                Ok(())
            } else {
                Err(Error::DomainCheck(format!(
                    "Unauthorized domain: {}. Allowed: {}",
                    hostname, domains
                )))
            }
        },
        _ => {
            // No restriction - allow all domains
            Ok(())
        },
    }
}

// ============================================================================
// Test Functions (WASM-only)
// ============================================================================

#[wasm_bindgen]
/// Simple test function to verify WASM is working.
///
/// Returns a success message confirming the WASM module loaded correctly.
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
pub fn test() -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }
    "Rust WASM is working!".to_string()
}

#[wasm_bindgen]
/// Returns the current version of the library.
///
/// Returns the Cargo package version as a string (e.g., "0.1.0").
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
pub fn get_version() -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen]
/// Test function for t-critical value computation.
///
/// Returns JSON with the computed t-critical value for the given parameters.
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
pub fn test_t_critical(df: f64, alpha: f64) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }
    let t_crit = core::t_critical_quantile(df, alpha);
    format!(
        r#"{{"df": {}, "alpha": {}, "t_critical": {}}}"#,
        df, alpha, t_crit
    )
}

#[wasm_bindgen]
/// Test function for confidence interval computation.
///
/// Returns JSON with the computed confidence interval for a coefficient.
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
pub fn test_ci(coef: f64, se: f64, df: f64, alpha: f64) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }
    let t_crit = core::t_critical_quantile(df, alpha);
    format!(
        r#"{{"lower": {}, "upper": {}}}"#,
        coef - t_crit * se,
        coef + t_crit * se
    )
}

#[wasm_bindgen]
/// Test function for R accuracy validation.
///
/// Returns JSON comparing our statistical functions against R reference values.
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
pub fn test_r_accuracy() -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }
    format!(
        r#"{{"two_tail_p": {}, "qt_975": {}}}"#,
        core::two_tailed_p_value(1.6717, 21.0),
        core::t_critical_quantile(21.0, 0.05)
    )
}

#[wasm_bindgen]
/// Test function for regression validation against R reference values.
///
/// Runs a regression on a housing dataset and compares results against R's lm() output.
/// Returns JSON with status "PASS" or "FAIL" with details.
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
pub fn test_housing_regression() -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    match test_housing_regression_native() {
        Ok(result) => result,
        Err(e) => serde_json::json!({ "status": "ERROR", "error": e.to_string() }).to_string(),
    }
}

// Native Rust test function (works without WASM feature)
#[cfg(any(test, feature = "wasm"))]
fn test_housing_regression_native() -> Result<String> {
    let y = vec![
        245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1, 445.8, 167.9, 367.4,
        289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9, 223.4, 312.5, 156.8, 423.7, 267.9,
    ];

    let square_feet = vec![
        1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0, 2200.0,
        900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0, 1250.0, 1700.0,
        850.0, 2350.0, 1400.0,
    ];
    let bedrooms = vec![
        3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0, 4.0, 2.0, 4.0, 3.0, 2.0, 5.0, 3.0, 4.0,
        2.0, 4.0, 3.0, 3.0, 2.0, 4.0, 3.0,
    ];
    let age = vec![
        15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0, 3.0, 30.0, 6.0, 14.0, 22.0, 1.0,
        16.0, 9.0, 28.0, 4.0, 19.0, 11.0, 35.0, 3.0, 13.0,
    ];

    let x_vars = vec![square_feet, bedrooms, age];
    let names = vec![
        "Intercept".to_string(),
        "Square_Feet".to_string(),
        "Bedrooms".to_string(),
        "Age".to_string(),
    ];

    let result = core::ols_regression(&y, &x_vars, &names)?;

    // Check against R results
    let expected_coeffs = [52.1271333, 0.1613877, 0.9545492, -1.1811815];
    let expected_std_errs = [31.18201809, 0.01875072, 10.44400198, 0.73219949];

    let tolerance = 1e-4;
    let mut mismatches = vec![];

    for i in 0..4 {
        if (result.coefficients[i] - expected_coeffs[i]).abs() > tolerance {
            mismatches.push(format!(
                "coeff[{}] differs: got {}, expected {}",
                i, result.coefficients[i], expected_coeffs[i]
            ));
        }
        if (result.std_errors[i] - expected_std_errs[i]).abs() > tolerance {
            mismatches.push(format!(
                "std_err[{}] differs: got {}, expected {}",
                i, result.std_errors[i], expected_std_errs[i]
            ));
        }
    }

    if mismatches.is_empty() {
        Ok(serde_json::json!({ "status": "PASS" }).to_string())
    } else {
        Ok(serde_json::json!({ "status": "FAIL", "mismatches": mismatches }).to_string())
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_housing_regression_integrity() {
        let result = test_housing_regression_native();
        if let Err(e) = result {
            panic!("Regression test failed: {}", e);
        }
    }

    /// Test that test_housing_regression_native produces valid JSON
    #[test]
    fn test_housing_regression_json_output() {
        let result = test_housing_regression_native().unwrap();
        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        // Should have status field
        assert!(parsed.get("status").is_some());
        // Status should be PASS (we control the test data)
        assert_eq!(parsed["status"], "PASS");
    }

    /// Test housing regression with actual R reference values
    #[test]
    fn test_housing_regression_coefficients() {
        let y = vec![
            245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1, 445.8, 167.9,
            367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9, 223.4, 312.5, 156.8, 423.7,
            267.9,
        ];

        let square_feet = vec![
            1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0,
            2200.0, 900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0,
            1250.0, 1700.0, 850.0, 2350.0, 1400.0,
        ];
        let bedrooms = vec![
            3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0, 4.0, 2.0, 4.0, 3.0, 2.0, 5.0,
            3.0, 4.0, 2.0, 4.0, 3.0, 3.0, 2.0, 4.0, 3.0,
        ];
        let age = vec![
            15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0, 3.0, 30.0, 6.0, 14.0,
            22.0, 1.0, 16.0, 9.0, 28.0, 4.0, 19.0, 11.0, 35.0, 3.0, 13.0,
        ];

        let x_vars = vec![square_feet, bedrooms, age];
        let names = vec![
            "Intercept".to_string(),
            "Square_Feet".to_string(),
            "Bedrooms".to_string(),
            "Age".to_string(),
        ];

        let result = core::ols_regression(&y, &x_vars, &names).unwrap();

        // Check against R results
        let expected_coeffs = [52.1271333, 0.1613877, 0.9545492, -1.1811815];
        let expected_std_errs = [31.18201809, 0.01875072, 10.44400198, 0.73219949];

        let tolerance = 1e-4;
        for i in 0..4 {
            assert!(
                (result.coefficients[i] - expected_coeffs[i]).abs() < tolerance,
                "coeff[{}] differs: got {}, expected {}",
                i,
                result.coefficients[i],
                expected_coeffs[i]
            );
            assert!(
                (result.std_errors[i] - expected_std_errs[i]).abs() < tolerance,
                "std_err[{}] differs: got {}, expected {}",
                i,
                result.std_errors[i],
                expected_std_errs[i]
            );
        }
    }

    /// Test R-squared calculation in housing regression
    #[test]
    fn test_housing_regression_r_squared() {
        let result = test_housing_regression_native().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

        // If status is PASS, R² should be reasonable (between 0 and 1)
        assert_eq!(parsed["status"], "PASS");
    }

    /// Test that housing regression handles all expected output fields
    #[test]
    fn test_housing_regression_comprehensive() {
        let y = vec![
            245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1,
        ];
        let x1 = vec![1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0];
        let x2 = vec![3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0];

        let result = core::ols_regression(&y, &[x1, x2], &["Intercept".into(), "X1".into(), "X2".into()])
            .unwrap();

        // Verify expected output fields exist
        assert!(!result.coefficients.is_empty());
        assert!(!result.std_errors.is_empty());
        assert!(!result.t_stats.is_empty());
        assert!(!result.p_values.is_empty());
        assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
        assert!(result.residuals.len() == y.len());
    }

    /// Test error handling when insufficient data is provided
    #[test]
    fn test_housing_regression_insufficient_data() {
        let y = vec![245.5, 312.8]; // Only 2 observations
        let x1 = vec![1200.0, 1800.0];
        let x2 = vec![3.0, 4.0];

        let result = core::ols_regression(&y, &[x1, x2], &["Intercept".into(), "X1".into(), "X2".into()]);
        assert!(result.is_err());
    }

    /// Test housing regression precision with tolerance check
    #[test]
    fn test_housing_regression_tolerance_check() {
        let y = vec![
            245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1, 445.8, 167.9,
            367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9, 223.4, 312.5, 156.8, 423.7,
            267.9,
        ];

        let square_feet = vec![
            1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0,
            2200.0, 900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0,
            1250.0, 1700.0, 850.0, 2350.0, 1400.0,
        ];
        let bedrooms = vec![
            3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0, 4.0, 2.0, 4.0, 3.0, 2.0, 5.0,
            3.0, 4.0, 2.0, 4.0, 3.0, 3.0, 2.0, 4.0, 3.0,
        ];
        let age = vec![
            15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0, 3.0, 30.0, 6.0, 14.0,
            22.0, 1.0, 16.0, 9.0, 28.0, 4.0, 19.0, 11.0, 35.0, 3.0, 13.0,
        ];

        let x_vars = vec![square_feet, bedrooms, age];
        let names = vec![
            "Intercept".to_string(),
            "Square_Feet".to_string(),
            "Bedrooms".to_string(),
            "Age".to_string(),
        ];

        let result = core::ols_regression(&y, &x_vars, &names).unwrap();

        // Verify all coefficient values are finite
        for coef in &result.coefficients {
            assert!(coef.is_finite(), "Coefficient should be finite");
        }
        // Verify all standard errors are positive and finite
        for se in &result.std_errors {
            assert!(se.is_finite(), "Standard error should be finite");
            if *se <= 0.0 {
                panic!("Standard error should be positive, got {}", se);
            }
        }
    }
}
