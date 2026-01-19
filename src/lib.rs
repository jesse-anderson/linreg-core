//! Linear Regression Library with WASM bindings.
//!
//! This library provides Ordinary Least Squares (OLS) regression and
//! statistical diagnostic tests. When the `wasm` feature is enabled,
//! it exposes JavaScript bindings for use in web browsers.
//!
//! # Features
//!
//! - OLS regression with QR decomposition for numerical stability
//! - Comprehensive diagnostic tests (Rainbow, Harvey-Collier, Breusch-Pagan, White, Jarque-Bera, Durbin-Watson, Shapiro-Wilk, Anderson-Darling)
//! - Variance Inflation Factor (VIF) analysis
//! - Custom implementations of all linear algebra and statistical functions
//!
//! # Module Structure
//!
//! - [`core`] - OLS regression implementation
//! - [`diagnostics`] - Statistical diagnostic tests
//! - [`distributions`] - Statistical distributions (t, F, chi-squared, normal)
//! - [`linalg`] - Matrix operations and QR decomposition
//! - [`error`] - Error types
//!
//! # WASM API
//!
//! When the `wasm` feature is enabled, the library exposes JavaScript bindings that
//! accept and return JSON strings for easy integration. By default, all domains are
//! allowed. To restrict usage to specific domains, set the `LINREG_DOMAIN_RESTRICT`
//! environment variable at build time:
//!
//! ```bash
//! LINREG_DOMAIN_RESTRICT=example.com,mysite.com wasm-pack build
//! ```
//!
//! # Example (Native Rust)
//!
//! ```
//! use linreg_core::core::ols_regression;
//! use linreg_core::Error;
//!
//! let y = vec![2.5, 3.7, 4.2, 5.1, 6.3, 7.0];
//! let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0, 2.0];
//! let names = vec![
//!     "Intercept".to_string(),
//!     "Temperature".to_string(),
//!     "Pressure".to_string(),
//! ];
//!
//! let result = ols_regression(&y, &[x1, x2], &names)?;
//! println!("R²: {}", result.r_squared);
//! # Ok::<(), Error>(())
//! ```

// Import core modules (always available)
pub mod core;
pub mod diagnostics;
pub mod distributions;
pub mod linalg;
pub mod error;

// Unit tests are now in tests/unit/ directory
// - error_tests.rs -> tests/unit/error_tests.rs
// - core_tests.rs -> tests/unit/core_tests.rs
// - linalg_tests.rs -> tests/unit/linalg_tests.rs
// - validation_tests.rs -> tests/validation/main.rs
// - diagnostics_tests.rs: disabled (references unimplemented functions)

// Re-export public API (always available)
pub use core::{RegressionOutput, VifResult};
pub use diagnostics::{
    DiagnosticTestResult,
    RainbowTestOutput, RainbowSingleResult, RainbowMethod,
    WhiteTestOutput, WhiteSingleResult, WhiteMethod,
    CooksDistanceResult,
};

// Re-export core test functions with different names to avoid WASM conflicts
pub use diagnostics::rainbow_test as rainbow_test_core;
pub use diagnostics::white_test as white_test_core;

pub use error::{Error, Result, error_json, error_to_json};

// ============================================================================
// WASM-specific code (only compiled when "wasm" feature is enabled)
// ============================================================================

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use std::collections::HashSet;

#[cfg(feature = "wasm")]
use serde::Serialize;

#[cfg(feature = "wasm")]
use crate::distributions::{student_t_cdf, normal_inverse_cdf};

// ============================================================================
// CSV Parsing (WASM-only)
// ============================================================================

#[cfg(feature = "wasm")]
#[derive(Serialize)]
struct ParsedCsv {
    headers: Vec<String>,
    data: Vec<serde_json::Map<String, serde_json::Value>>,
    numeric_columns: Vec<String>,
}

#[cfg(feature = "wasm")]
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
            row_map.insert(header.clone(), serde_json::Value::String(val_trimmed.to_string()));
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

#[cfg(feature = "wasm")]
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
pub fn ols_regression(
    y_json: &str,
    x_vars_json: &str,
    variable_names: &str,
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

#[cfg(feature = "wasm")]
#[wasm_bindgen]
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
pub fn rainbow_test(
    y_json: &str,
    x_vars_json: &str,
    fraction: f64,
    method: &str,
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

    // Parse method parameter (default to "r" for R)
    let method = match method.to_lowercase().as_str() {
        "python" => diagnostics::RainbowMethod::Python,
        "both" => diagnostics::RainbowMethod::Both,
        _ => diagnostics::RainbowMethod::R,  // Default to R
    };

    match diagnostics::rainbow_test(&y, &x_vars, fraction, method) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Rainbow test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
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
pub fn harvey_collier_test(
    y_json: &str,
    x_vars_json: &str,
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

    match diagnostics::harvey_collier_test(&y, &x_vars) {
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
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn breusch_pagan_test(
    y_json: &str,
    x_vars_json: &str,
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
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn white_test(
    y_json: &str,
    x_vars_json: &str,
    method: &str,
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

    // Parse method parameter (default to "r" for R)
    let method = match method.to_lowercase().as_str() {
        "python" => diagnostics::WhiteMethod::Python,
        "both" => diagnostics::WhiteMethod::Both,
        _ => diagnostics::WhiteMethod::R,  // Default to R
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
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn r_white_test(
    y_json: &str,
    x_vars_json: &str,
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
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn python_white_test(
    y_json: &str,
    x_vars_json: &str,
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
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn jarque_bera_test(
    y_json: &str,
    x_vars_json: &str,
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

    match diagnostics::jarque_bera_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Jarque-Bera test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

// ============================================================================
// Durbin-Watson Test (WASM wrapper)
// ============================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
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
pub fn durbin_watson_test(
    y_json: &str,
    x_vars_json: &str,
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
/// The Shapiro-Wilk test is a powerful tests for normality,
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
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn shapiro_wilk_test(
    y_json: &str,
    x_vars_json: &str,
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

    match diagnostics::shapiro_wilk_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Shapiro-Wilk test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
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
pub fn anderson_darling_test(
    y_json: &str,
    x_vars_json: &str,
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

    match diagnostics::anderson_darling_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Anderson-Darling test result")),
        Err(e) => error_json(&e.to_string()),
    }
}

// ============================================================================
// Cook's Distance (WASM wrapper)
// ============================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
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
pub fn cooks_distance_test(
    y_json: &str,
    x_vars_json: &str,
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

    match diagnostics::cooks_distance_test(&y, &x_vars) {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize Cook's distance result")),
        Err(e) => error_json(&e.to_string()),
    }
}

// ============================================================================
// Statistical Utility Functions (WASM wrappers)
// ============================================================================

#[cfg(feature = "wasm")]
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

#[cfg(feature = "wasm")]
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

#[cfg(feature = "wasm")]
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

#[cfg(feature = "wasm")]
fn check_domain() -> Result<()> {
    // Read allowed domains from build-time environment variable
    let allowed_domains = option_env!("LINREG_DOMAIN_RESTRICT");

    match allowed_domains {
        Some(domains) if !domains.is_empty() => {
            // Domain restriction is enabled
            let window = web_sys::window().ok_or(Error::DomainCheck("No window found".to_string()))?;
            let location = window.location();
            let hostname = location.hostname().map_err(|_| Error::DomainCheck("No hostname found".to_string()))?;

            let domain_list: Vec<&str> = domains.split(',').map(|s| s.trim()).collect();

            if domain_list.contains(&hostname.as_str()) {
                Ok(())
            } else {
                Err(Error::DomainCheck(format!(
                    "Unauthorized domain: {}. Allowed: {}",
                    hostname, domains
                )))
            }
        }
        _ => {
            // No restriction - allow all domains
            Ok(())
        }
    }
}

// ============================================================================
// Test Functions (WASM-only)
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
}

#[cfg(feature = "wasm")]
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

#[cfg(feature = "wasm")]
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

#[cfg(feature = "wasm")]
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
    format!(r#"{{"df": {}, "alpha": {}, "t_critical": {}}}"#, df, alpha, t_crit)
}

#[cfg(feature = "wasm")]
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
    format!(r#"{{"lower": {}, "upper": {}}}"#, coef - t_crit * se, coef + t_crit * se)
}

#[cfg(feature = "wasm")]
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

#[cfg(feature = "wasm")]
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
        Err(e) => serde_json::json!({ "status": "ERROR", "error": e.to_string() }).to_string()
    }
}

// Native Rust test function (works without WASM feature)
fn test_housing_regression_native() -> Result<String> {
    let y = vec![
        245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1,
        445.8, 167.9, 367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9,
        223.4, 312.5, 156.8, 423.7, 267.9
    ];

    let square_feet = vec![
        1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0,
        2200.0, 900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0,
        1250.0, 1700.0, 850.0, 2350.0, 1400.0
    ];
    let bedrooms = vec![
        3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0,
        4.0, 2.0, 4.0, 3.0, 2.0, 5.0, 3.0, 4.0, 2.0, 4.0,
        3.0, 3.0, 2.0, 4.0, 3.0
    ];
    let age = vec![
        15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0,
        3.0, 30.0, 6.0, 14.0, 22.0, 1.0, 16.0, 9.0, 28.0, 4.0,
        19.0, 11.0, 35.0, 3.0, 13.0
    ];

    let x_vars = vec![square_feet, bedrooms, age];
    let names = vec!["Intercept".to_string(), "Square_Feet".to_string(), "Bedrooms".to_string(), "Age".to_string()];

    let result = core::ols_regression(&y, &x_vars, &names)?;

    // Check against R results
    let expected_coeffs = [52.1271333, 0.1613877, 0.9545492, -1.1811815];
    let expected_std_errs = [31.18201809, 0.01875072, 10.44400198, 0.73219949];

    let tolerance = 1e-4;
    let mut mismatches = vec![];

    for i in 0..4 {
        if (result.coefficients[i] - expected_coeffs[i]).abs() > tolerance {
            mismatches.push(format!("coeff[{}] differs: got {}, expected {}", i, result.coefficients[i], expected_coeffs[i]));
        }
        if (result.std_errors[i] - expected_std_errs[i]).abs() > tolerance {
            mismatches.push(format!("std_err[{}] differs: got {}, expected {}", i, result.std_errors[i], expected_std_errs[i]));
        }
    }

    if mismatches.is_empty() {
        Ok(serde_json::json!({ "status": "PASS" }).to_string())
    } else {
        Ok(serde_json::json!({ "status": "FAIL", "mismatches": mismatches }).to_string())
    }
}