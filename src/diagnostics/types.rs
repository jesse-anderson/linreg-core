// ============================================================================
// Diagnostic Test Result Types
// ============================================================================
//
// Shared types for all diagnostic tests. These structs are serializable for
// JSON output via WASM bindings.

use serde::Serialize;

/// Result of a diagnostic test
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticTestResult {
    pub test_name: String,
    pub statistic: f64,
    pub p_value: f64,
    #[serde(rename = "is_passed")]
    pub passed: bool,  // true = null hypothesis NOT rejected (assumption met)
    pub interpretation: String,
    pub guidance: String,
}

/// Result of Rainbow test with single method
#[derive(Debug, Clone, Serialize)]
pub struct RainbowSingleResult {
    pub method: String,
    pub statistic: f64,
    pub p_value: f64,
    #[serde(rename = "is_passed")]
    pub passed: bool,
}

/// Result of Rainbow test supporting both R and Python variants
#[derive(Debug, Clone, Serialize)]
pub struct RainbowTestOutput {
    pub test_name: String,
    pub r_result: Option<RainbowSingleResult>,
    pub python_result: Option<RainbowSingleResult>,
    pub interpretation: String,
    pub guidance: String,
}

/// Rainbow test implementation method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RainbowMethod {
    /// R's lmtest::raintest (Type 7 quantile)
    R,
    /// Python's statsmodels (direct formula)
    Python,
    /// Both R and Python results
    Both,
}

/// Result of White test with single method
#[derive(Debug, Clone, Serialize)]
pub struct WhiteSingleResult {
    pub method: String,
    pub statistic: f64,
    pub p_value: f64,
    #[serde(rename = "is_passed")]
    pub passed: bool,
}

/// Result of White test supporting both R and Python variants
#[derive(Debug, Clone, Serialize)]
pub struct WhiteTestOutput {
    pub test_name: String,
    pub r_result: Option<WhiteSingleResult>,
    pub python_result: Option<WhiteSingleResult>,
    pub interpretation: String,
    pub guidance: String,
}

/// White test implementation method
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
