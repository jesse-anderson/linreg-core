// ============================================================================
// Diagnostic Test Result Types
// ============================================================================
//
// Shared types for all diagnostic tests. These structs are serializable for
// JSON output via WASM bindings.

//! Diagnostic test result types.
//!
//! This module defines the common result types used by all diagnostic tests.
//! All types are serializable for JSON output via WASM bindings.

use serde::Serialize;

/// Result of a diagnostic test.
///
/// Contains the test statistic, p-value, pass/fail indication, and
/// human-readable interpretation and guidance for the test result.
///
/// # Fields
///
/// * `test_name` - Name of the diagnostic test
/// * `statistic` - Test statistic value
/// * `p_value` - P-value for the test
/// * `passed` - Whether the null hypothesis was not rejected (assumption met)
/// * `interpretation` - Human-readable explanation of the result
/// * `guidance` - Recommendations based on the test result
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticTestResult {
    pub test_name: String,
    pub statistic: f64,
    pub p_value: f64,
    #[serde(rename = "is_passed")]
    pub passed: bool, // true = null hypothesis NOT rejected (assumption met)
    pub interpretation: String,
    pub guidance: String,
}

/// Result of the Rainbow test for a single method (R or Python).
///
/// Contains the test statistic, p-value, and pass/fail indication
/// for one specific implementation of the Rainbow test.
///
/// # Fields
///
/// * `method` - Name of the method used ("r" or "python")
/// * `statistic` - F-statistic value
/// * `p_value` - P-value for the test
/// * `passed` - Whether linearity assumption was not rejected
#[derive(Debug, Clone, Serialize)]
pub struct RainbowSingleResult {
    pub method: String,
    pub statistic: f64,
    pub p_value: f64,
    #[serde(rename = "is_passed")]
    pub passed: bool,
}

/// Result of the Rainbow test supporting both R and Python variants.
///
/// The Rainbow test checks for linearity by comparing the fit of the full model
/// to a subset of central observations. This struct can contain results from
/// both R's lmtest::raintest and Python's statsmodels implementations.
///
/// # Fields
///
/// * `test_name` - Name of the test
/// * `r_result` - Result from R's lmtest::raintest (if computed)
/// * `python_result` - Result from Python's statsmodels (if computed)
/// * `interpretation` - Human-readable explanation of the result
/// * `guidance` - Recommendations based on the test result
#[derive(Debug, Clone, Serialize)]
pub struct RainbowTestOutput {
    pub test_name: String,
    pub r_result: Option<RainbowSingleResult>,
    pub python_result: Option<RainbowSingleResult>,
    pub interpretation: String,
    pub guidance: String,
}

/// Rainbow test implementation method.
///
/// Specifies which algorithm to use for computing the Rainbow test.
/// Different statistical packages use slightly different algorithms
/// for selecting the central subset of observations.
///
/// # Variants
///
/// * `R` - R's lmtest::raintest (Type 7 quantile with interpolation)
/// * `Python` - Python's statsmodels (direct formula using ceiling)
/// * `Both` - Compute both R and Python results
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RainbowMethod {
    /// R's lmtest::raintest (Type 7 quantile)
    R,
    /// Python's statsmodels (direct formula)
    Python,
    /// Both R and Python results
    Both,
}

/// Result of the White test for a single method (R or Python).
///
/// Contains the test statistic, p-value, and pass/fail indication
/// for one specific implementation of the White test.
///
/// # Fields
///
/// * `method` - Name of the method used ("r" or "python")
/// * `statistic` - LM test statistic value
/// * `p_value` - P-value for the test
/// * `passed` - Whether homoscedasticity assumption was not rejected
#[derive(Debug, Clone, Serialize)]
pub struct WhiteSingleResult {
    pub method: String,
    pub statistic: f64,
    pub p_value: f64,
    #[serde(rename = "is_passed")]
    pub passed: bool,
}

/// Result of the White test supporting both R and Python variants.
///
/// The White test is a general test for heteroscedasticity that does not
/// assume a specific form. R and Python use different auxiliary regression
/// specifications.
///
/// # Fields
///
/// * `test_name` - Name of the test
/// * `r_result` - Result from R's skedastic::white (if computed)
/// * `python_result` - Result from Python's statsmodels (if computed)
/// * `interpretation` - Human-readable explanation of the result
/// * `guidance` - Recommendations based on the test result
#[derive(Debug, Clone, Serialize)]
pub struct WhiteTestOutput {
    pub test_name: String,
    pub r_result: Option<WhiteSingleResult>,
    pub python_result: Option<WhiteSingleResult>,
    pub interpretation: String,
    pub guidance: String,
}

/// White test implementation method.
///
/// Specifies which algorithm to use for computing the White test.
/// The R and Python implementations differ in their auxiliary regression
/// specification.
///
/// # Variants
///
/// * `R` - R's skedastic::white (original variables and squares only)
/// * `Python` - Python's statsmodels (squares and cross-products)
/// * `Both` - Compute both R and Python results
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WhiteMethod {
    /// R's skedastic::white (original variables only)
    R,
    /// Python's statsmodels (squares + interactions)
    Python,
    /// Both R and Python results
    Both,
}

/// Result of Cook's distance analysis.
///
/// Cook's distance measures how much each observation influences the regression model.
/// Unlike hypothesis tests, this is an influence measure - not a test with p-values.
#[derive(Debug, Clone, Serialize)]
pub struct CooksDistanceResult {
    /// Name of the test
    pub test_name: String,
    /// Cook's distance for each observation (one per observation)
    pub distances: Vec<f64>,
    /// Number of parameters (including intercept)
    pub p: usize,
    /// Mean squared error of the model
    pub mse: f64,
    /// Common threshold: 4/n (observations above this are potentially influential)
    pub threshold_4_over_n: f64,
    /// Conservative threshold: 4/(n-p-1)
    pub threshold_4_over_df: f64,
    /// Absolute threshold: D_i > 1 indicates high influence
    pub threshold_1: f64,
    /// Indices of observations exceeding 4/n threshold
    pub influential_4_over_n: Vec<usize>,
    /// Indices of observations exceeding conservative threshold
    pub influential_4_over_df: Vec<usize>,
    /// Indices of observations exceeding D_i > 1 threshold
    pub influential_1: Vec<usize>,
    /// Interpretation of results
    pub interpretation: String,
    /// Guidance for handling influential observations
    pub guidance: String,
}
