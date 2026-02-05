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
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::DiagnosticTestResult;
///
/// let result = DiagnosticTestResult {
///     test_name: "Breusch-Pagan".to_string(),
///     statistic: 3.45,
///     p_value: 0.063,
///     passed: true,
///     interpretation: "Failed to reject null hypothesis of homoscedasticity".to_string(),
///     guidance: "No action needed - residuals appear to have constant variance.".to_string(),
/// };
///
/// assert_eq!(result.test_name, "Breusch-Pagan");
/// assert!(result.passed);
/// assert!(result.p_value > 0.05);
/// ```
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
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::RainbowSingleResult;
///
/// let result = RainbowSingleResult {
///     method: "r".to_string(),
///     statistic: 1.23,
///     p_value: 0.32,
///     passed: true,
/// };
///
/// assert_eq!(result.method, "r");
/// assert!(result.passed);
/// assert!(result.p_value > 0.05);
/// ```
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
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::{RainbowTestOutput, RainbowSingleResult};
///
/// let r_result = Some(RainbowSingleResult {
///     method: "r".to_string(),
///     statistic: 1.23,
///     p_value: 0.32,
///     passed: true,
/// });
///
/// let output = RainbowTestOutput {
///     test_name: "Rainbow Test".to_string(),
///     r_result,
///     python_result: None,
///     interpretation: "Linearity assumption not rejected".to_string(),
///     guidance: "Model appears to have linear relationship.".to_string(),
/// };
///
/// assert!(output.r_result.is_some());
/// assert!(output.python_result.is_none());
/// ```
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
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::RainbowMethod;
///
/// let method_r = RainbowMethod::R;
/// let method_python = RainbowMethod::Python;
/// let method_both = RainbowMethod::Both;
///
/// // Enum variants can be compared
/// assert_eq!(method_r, RainbowMethod::R);
/// assert_ne!(method_r, method_python);
/// ```
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
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::WhiteSingleResult;
///
/// let result = WhiteSingleResult {
///     method: "python".to_string(),
///     statistic: 5.67,
///     p_value: 0.128,
///     passed: true,
/// };
///
/// assert_eq!(result.method, "python");
/// assert!(result.passed);
/// assert!(result.statistic > 0.0);
/// ```
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
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::{WhiteTestOutput, WhiteSingleResult};
///
/// let py_result = Some(WhiteSingleResult {
///     method: "python".to_string(),
///     statistic: 5.67,
///     p_value: 0.128,
///     passed: true,
/// });
///
/// let output = WhiteTestOutput {
///     test_name: "White Test".to_string(),
///     r_result: None,
///     python_result: py_result,
///     interpretation: "Homoscedasticity not rejected".to_string(),
///     guidance: "Residuals appear to have constant variance.".to_string(),
/// };
///
/// assert!(output.python_result.is_some());
/// assert!(output.r_result.is_none());
/// ```
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
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::WhiteMethod;
///
/// let method_r = WhiteMethod::R;
/// let method_python = WhiteMethod::Python;
/// let method_both = WhiteMethod::Both;
///
/// // Enum variants can be compared
/// assert_eq!(method_r, WhiteMethod::R);
/// assert_ne!(method_r, method_python);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WhiteMethod {
    /// R's skedastic::white (original variables only)
    R,
    /// Python's statsmodels (squares + interactions)
    Python,
    /// Both R and Python results
    Both,
}

/// Method for computing the Harvey-Collier test.
///
/// Different statistical packages implement the Harvey-Collier test differently.
/// R's `lmtest::harvtest` uses all recursive residuals, while Python's
/// `statsmodels.stats.diagnostic.linear_harvey_collier` skips the first 3
/// elements of the standardized recursive residuals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HarveyCollierMethod {
    /// R's lmtest::harvtest (uses all recursive residuals)
    R,
    /// Python's statsmodels (skips first 3 recursive residuals)
    Python,
}

/// Result of Cook's distance analysis.
///
/// Cook's distance measures how much each observation influences the regression model.
/// Unlike hypothesis tests, this is an influence measure - not a test with p-values.
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::CooksDistanceResult;
///
/// let result = CooksDistanceResult {
///     test_name: "Cook's Distance".to_string(),
///     distances: vec![0.01, 0.05, 0.8, 0.02, 0.03],
///     p: 2,
///     mse: 1.5,
///     threshold_4_over_n: 0.8,
///     threshold_4_over_df: 1.33,
///     threshold_1: 1.0,
///     influential_4_over_n: vec![2],
///     influential_4_over_df: vec![],
///     influential_1: vec![],
///     interpretation: "Observation 2 shows elevated influence.".to_string(),
///     guidance: "Check if observation 2 is a valid data point.".to_string(),
/// };
///
/// assert_eq!(result.distances.len(), 5);
/// assert_eq!(result.influential_4_over_n, vec![2]);
/// ```
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

/// Result of DFBETAS analysis.
///
/// DFBETAS measures the influence of each observation on each regression coefficient.
/// For each observation i and each coefficient j, it measures the number of standard
/// errors that coefficient changes when observation i is omitted.
///
/// # Fields
///
/// * `test_name` - Name of the diagnostic
/// * `dfbetas` - Matrix of DFBETAS values (n x p), where dfbetas\[i\]\[j\] is the DFBETAS
///   for observation i on coefficient j (j=0 is intercept, j=1 is first predictor, etc.)
/// * `n` - Number of observations
/// * `p` - Number of parameters (including intercept)
/// * `threshold` - Common threshold (2/√n) for identifying influential observations
/// * `influential_observations` - Map of coefficients to list of influential observation indices
///   (1-based indexing for output compatibility with R/Python)
/// * `interpretation` - Human-readable explanation of results
/// * `guidance` - Recommendations based on results
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::DfbetasResult;
/// use std::collections::HashMap;
///
/// let result = DfbetasResult {
///     test_name: "DFBETAS".to_string(),
///     dfbetas: vec![vec![0.1, 0.05], vec![0.8, 0.3]],
///     n: 5,
///     p: 2,
///     threshold: 0.894,
///     influential_observations: HashMap::new(),
///     interpretation: "No highly influential observations.".to_string(),
///     guidance: "Model appears stable.".to_string(),
/// };
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct DfbetasResult {
    /// Name of the test
    pub test_name: String,
    /// DFBETAS matrix: n rows (observations) x p columns (parameters)
    /// dfbetas\[i\]\[j\] = standardized change in coefficient j when observation i is omitted
    pub dfbetas: Vec<Vec<f64>>,
    /// Number of observations
    pub n: usize,
    /// Number of parameters (including intercept)
    pub p: usize,
    /// Common threshold: 2/√n
    pub threshold: f64,
    /// Map of coefficient indices (1-based) to list of influential observation indices (1-based)
    /// Key: coefficient index (1=intercept, 2=first predictor, etc.)
    /// Value: vector of observation indices that exceed |DFBETAS| > threshold
    pub influential_observations: std::collections::HashMap<usize, Vec<usize>>,
    /// Interpretation of results
    pub interpretation: String,
    /// Guidance for handling influential observations
    pub guidance: String,
}

/// Result of DFFITS analysis.
///
/// DFFITS measures the influence of each observation on its own fitted value.
/// It is the number of standard errors that the fitted value changes when
/// observation i is omitted.
///
/// # Fields
///
/// * `test_name` - Name of the diagnostic
/// * `dffits` - Vector of DFFITS values (one per observation)
/// * `n` - Number of observations
/// * `p` - Number of parameters (including intercept)
/// * `threshold` - Common threshold (2*√(p/n)) for identifying influential observations
/// * `influential_observations` - Indices of observations exceeding the threshold (1-based)
/// * `interpretation` - Human-readable explanation of results
/// * `guidance` - Recommendations based on results
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::DffitsResult;
///
/// let result = DffitsResult {
///     test_name: "DFFITS".to_string(),
///     dffits: vec![0.1, 0.3, 1.2, 0.05, 0.2],
///     n: 5,
///     p: 2,
///     threshold: 1.26,
///     influential_observations: vec![],
///     interpretation: "No highly influential observations.".to_string(),
///     guidance: "Model appears stable.".to_string(),
/// };
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct DffitsResult {
    /// Name of the test
    pub test_name: String,
    /// DFFITS value for each observation
    pub dffits: Vec<f64>,
    /// Number of observations
    pub n: usize,
    /// Number of parameters (including intercept)
    pub p: usize,
    /// Common threshold: 2*√(p/n)
    pub threshold: f64,
    /// Indices of observations exceeding |DFFITS| > threshold (1-based indexing)
    pub influential_observations: Vec<usize>,
    /// Interpretation of results
    pub interpretation: String,
    /// Guidance for handling influential observations
    pub guidance: String,
}

/// Detailed VIF result for a single predictor variable.
///
/// Contains the Variance Inflation Factor and associated statistics for one predictor.
///
/// # Fields
///
/// * `variable` - Name of the predictor variable
/// * `vif` - Variance Inflation Factor (VIF > 10 indicates high multicollinearity)
/// * `rsquared` - R-squared from regressing this predictor on all others
/// * `interpretation` - Human-readable interpretation of this VIF value
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::VifDetail;
///
/// let detail = VifDetail {
///     variable: "x1".to_string(),
///     vif: 1.2,
///     rsquared: 0.17,
///     interpretation: "Low multicollinearity".to_string(),
/// };
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct VifDetail {
    pub variable: String,
    pub vif: f64,
    pub rsquared: f64,
    pub interpretation: String,
}

/// Result of VIF diagnostic analysis.
///
/// Unlike hypothesis tests, VIF is a diagnostic measure - not a test with p-values.
/// Each predictor gets its own VIF score measuring its correlation with other predictors.
///
/// # Fields
///
/// * `test_name` - Name of the diagnostic
/// * `max_vif` - Maximum VIF value across all predictors
/// * `vif_results` - Detailed VIF results for each predictor
/// * `interpretation` - Human-readable explanation of results
/// * `guidance` - Recommendations based on results
///
/// # Example
///
/// ```
/// # use linreg_core::diagnostics::VifDiagnosticResult;
/// let result = VifDiagnosticResult {
///     test_name: "Variance Inflation Factor (VIF)".to_string(),
///     max_vif: 1.5,
///     vif_results: vec![],
///     interpretation: "All VIF values are within acceptable range.".to_string(),
///     guidance: "No concerning multicollinearity detected.".to_string(),
/// };
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct VifDiagnosticResult {
    pub test_name: String,
    pub max_vif: f64,
    pub vif_results: Vec<VifDetail>,
    pub interpretation: String,
    pub guidance: String,
}
