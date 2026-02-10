//! Diagnostic tests for WASM
//!
//! Provides WASM bindings for all statistical diagnostic tests.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use super::domain::check_domain;
use crate::diagnostics;
use crate::error::{error_json, error_to_json};

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
