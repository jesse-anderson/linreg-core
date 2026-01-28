// ============================================================================
// Validation Test Helpers
// ============================================================================
//
// Shared constants, data structures, and helper functions for all validation tests.
// This module provides:
//
// - JSON deserialization structs for R and Python reference files
// - Dataset loading utilities with categorical encoding support
// - Tolerance constants for statistical comparisons
// - Helper functions for assertions and output formatting

use serde::Deserialize;
use std::fs;
use std::path::Path;

// ============================================================================
// Tolerance Constants
// ============================================================================

/// Standard tolerance for statistical comparisons
pub const STAT_TOLERANCE: f64 = 0.001;

/// Tight tolerance for coefficient and model fit statistics (OLS)
/// OLS validation shows actual precision of 1e-11 to 1e-16 vs R/Python
pub const TIGHT_TOLERANCE: f64 = 1e-9;

/// Harvey-Collier test uses more lenient tolerance due to known numerical issues
/// with high multicollinearity data
pub const HARVEY_COLLIER_TOLERANCE: f64 = 0.1;

/// Durbin-Watson test uses more lenient tolerance due to different QR algorithms
/// (R/Python use LAPACK Householder, we use Gram-Schmidt)
pub const DURBIN_WATSON_TOLERANCE: f64 = 0.01;

/// Cook's Distance tolerance (more lenient due to numerical precision)
pub const COOKS_TOLERANCE: f64 = 1e-6;

/// RESET test tolerance (tight tolerance for F-statistic and p-value)
pub const RESET_TOLERANCE: f64 = 1e-9;

/// Ridge regression tolerance (allows for numerical differences in path construction)
pub const RIDGE_TOLERANCE: f64 = 1e-4;
pub const RIDGE_TOLERANCE_LOOSE: f64 = 1e-3;

/// Lasso regression tolerance (relative, for coordinate descent convergence)
/// Uses 1% for normal values, but small coefficients (<0.5) get 10% tolerance
/// since numerical differences dominate near the sparsity threshold
pub const LASSO_TOLERANCE: f64 = 1e-2;  // 1% for normal values
pub const LASSO_TOLERANCE_SMALL: f64 = 1e-1;  // 10% for |expected| < 0.5
pub const LASSO_TOLERANCE_LOOSE: f64 = 2e-2;  // 2% loose for normal values
pub const LASSO_TOLERANCE_SMALL_LOOSE: f64 = 2e-1;  // 20% loose for |expected| < 0.5

// ============================================================================
// Core Validation Data Structures
// ============================================================================

/// Wrapper for the main validation JSON files (R_results.json, Python_results.json)
#[derive(Debug, Deserialize)]
pub struct ValidationWrapper {
    pub housing_regression: RegressionResult,
}

/// Complete regression result from R/Python validation files
#[derive(Debug, Deserialize)]
pub struct RegressionResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub f_statistic: f64,
    #[allow(dead_code)]
    pub f_p_value: f64,
    #[allow(dead_code)]
    pub mse: f64,
    #[allow(dead_code)]
    pub std_error: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    #[allow(dead_code)]
    pub conf_int_lower: Vec<f64>,
    #[allow(dead_code)]
    pub conf_int_upper: Vec<f64>,
    #[allow(dead_code)]
    pub residuals: Vec<f64>,
    #[allow(dead_code)]
    pub standardized_residuals: Vec<f64>,
    pub vif: Vec<VifEntry>,
    pub rainbow: Option<DiagnosticResultJson>,
    pub harvey_collier: Option<DiagnosticResultJson>,
    pub breusch_pagan: Option<DiagnosticResultJson>,
    pub white: Option<DiagnosticResultJson>,
    pub jarque_bera: Option<DiagnosticResultJson>,
    pub durbin_watson: Option<DiagnosticResultJson>,
    pub anderson_darling: Option<DiagnosticResultJson>,
    pub shapiro_wilk: Option<DiagnosticResultJson>,
}

/// VIF entry from validation files
#[derive(Debug, Deserialize)]
pub struct VifEntry {
    pub variable: String,
    pub vif: f64,
    #[allow(dead_code)]
    pub rsquared: f64,
}

/// Generic diagnostic test result from validation files
#[derive(Debug, Deserialize)]
pub struct DiagnosticResultJson {
    pub statistic: f64,
    pub p_value: f64,
    #[allow(dead_code)]
    pub passed: bool,
}

// ============================================================================
// Generic R/Python Diagnostic Result Structures
// ============================================================================

/// Generic R diagnostic result format (uses arrays)
#[derive(Debug, Deserialize)]
pub struct RDiagnosticResult {
    #[allow(dead_code)]
    pub test_name: Vec<String>,
    #[allow(dead_code)]
    pub dataset: Vec<String>,
    #[allow(dead_code)]
    pub formula: Vec<String>,
    pub statistic: Vec<f64>,
    pub p_value: Vec<f64>,
    #[allow(dead_code)]
    pub passed: Vec<bool>,
    #[allow(dead_code)]
    pub description: Vec<String>,
}

/// Generic Python diagnostic result format (uses plain values)
#[derive(Debug, Deserialize)]
pub struct PythonDiagnosticResult {
    #[allow(dead_code)]
    pub test_name: String,
    #[allow(dead_code)]
    pub dataset: String,
    #[allow(dead_code)]
    pub formula: String,
    pub statistic: f64,
    pub p_value: f64,
    #[allow(dead_code)]
    pub passed: bool,
    #[allow(dead_code)]
    pub description: String,
    /// Optional f_statistic field (some tests include this)
    #[serde(default)]
    #[allow(dead_code)]
    pub f_statistic: Option<f64>,
    /// Optional f_p_value field (some tests include this)
    #[serde(default)]
    #[allow(dead_code)]
    pub f_p_value: Option<f64>,
    /// Indicates if the test was skipped (e.g., due to multicollinearity)
    #[serde(default)]
    #[allow(dead_code)]
    pub skipped: bool,
}

// ============================================================================
// Breusch-Pagan Specific Structures
// ============================================================================

/// R Breusch-Pagan result format
#[derive(Debug, Deserialize)]
pub struct RBreuschPaganResult {
    #[allow(dead_code)]
    pub test_name: Vec<String>,
    #[allow(dead_code)]
    pub dataset: Vec<String>,
    #[allow(dead_code)]
    pub formula: Vec<String>,
    pub statistic: Vec<f64>,
    pub p_value: Vec<f64>,
    #[allow(dead_code)]
    pub passed: Vec<bool>,
    #[allow(dead_code)]
    pub description: Vec<String>,
}

/// Python Breusch-Pagan result format
#[derive(Debug, Deserialize)]
pub struct PythonBreuschPaganResult {
    #[allow(dead_code)]
    pub test_name: String,
    #[allow(dead_code)]
    pub dataset: String,
    #[allow(dead_code)]
    pub formula: String,
    pub statistic: f64,
    pub p_value: f64,
    #[allow(dead_code)]
    pub passed: bool,
    #[allow(dead_code)]
    pub f_statistic: Option<f64>,
    #[allow(dead_code)]
    pub f_p_value: Option<f64>,
    #[allow(dead_code)]
    pub description: String,
}

// ============================================================================
// Shapiro-Wilk Specific Structures
// ============================================================================

/// R Shapiro-Wilk result format
#[derive(Debug, Deserialize)]
pub struct RShapiroWilkResult {
    #[allow(dead_code)]
    pub test_name: Vec<String>,
    pub statistic: Vec<f64>,
    pub p_value: Vec<f64>,
    #[allow(dead_code)]
    pub passed: Vec<bool>,
    #[allow(dead_code)]
    pub interpretation: Vec<String>,
    #[allow(dead_code)]
    pub guidance: Vec<String>,
}

/// Python Shapiro-Wilk result format
#[derive(Debug, Deserialize)]
pub struct PythonShapiroWilkResult {
    #[allow(dead_code)]
    pub test_name: String,
    pub statistic: f64,
    pub p_value: f64,
    #[allow(dead_code)]
    pub is_passed: bool,
    #[allow(dead_code)]
    pub interpretation: String,
    #[allow(dead_code)]
    pub guidance: String,
}

// ============================================================================
// Cook's Distance Specific Structures
// ============================================================================

/// R Cook's Distance result format
/// Note: R JSON format wraps single values in arrays (e.g., "p": [4])
/// but distances and influential_* fields are flat arrays, not nested
#[derive(Debug, Deserialize)]
pub struct RCooksDistanceResult {
    #[allow(dead_code)]
    pub test_name: Vec<String>,
    #[allow(dead_code)]
    pub dataset: Vec<String>,
    #[allow(dead_code)]
    pub formula: Vec<String>,
    pub distances: Vec<f64>,
    #[allow(dead_code)]
    pub p: Vec<usize>,
    #[allow(dead_code)]
    pub mse: Vec<f64>,
    #[allow(dead_code)]
    pub threshold_4_over_n: Vec<f64>,
    #[allow(dead_code)]
    pub threshold_4_over_df: Vec<f64>,
    #[allow(dead_code)]
    pub threshold_1: Vec<f64>,
    #[allow(dead_code)]
    pub influential_4_over_n: Vec<usize>,
    #[allow(dead_code)]
    pub influential_4_over_df: Vec<usize>,
    #[allow(dead_code)]
    pub influential_1: Vec<usize>,
    pub max_distance: Vec<f64>,
    pub max_index: Vec<usize>,
    #[allow(dead_code)]
    pub description: Vec<String>,
}

/// Python Cook's Distance result format
#[derive(Debug, Deserialize)]
pub struct PythonCooksDistanceResult {
    #[allow(dead_code)]
    pub test_name: String,
    #[allow(dead_code)]
    pub dataset: String,
    #[allow(dead_code)]
    pub formula: String,
    pub distances: Vec<f64>,
    #[allow(dead_code)]
    pub p: usize,
    #[allow(dead_code)]
    pub mse: f64,
    #[allow(dead_code)]
    pub threshold_4_over_n: f64,
    #[allow(dead_code)]
    pub threshold_4_over_df: f64,
    #[allow(dead_code)]
    pub threshold_1: f64,
    #[allow(dead_code)]
    pub influential_4_over_n: Vec<usize>,
    #[allow(dead_code)]
    pub influential_4_over_df: Vec<usize>,
    #[allow(dead_code)]
    pub influential_1: Vec<usize>,
    pub max_distance: f64,
    pub max_index: usize,
    #[allow(dead_code)]
    pub description: String,
}

// ============================================================================
// Breusch-Godfrey Specific Structures
// ============================================================================

/// R Breusch-Godfrey result format
/// Note: R JSON format wraps single values in arrays
#[derive(Debug, Deserialize)]
pub struct RBreuschGodfreyResult {
    #[allow(dead_code)]
    pub test_name: Vec<String>,
    #[allow(dead_code)]
    pub dataset: Vec<String>,
    #[allow(dead_code)]
    pub formula: Vec<String>,
    pub order: Vec<f64>,
    pub statistic: Vec<f64>,
    pub p_value: Vec<f64>,
    #[allow(dead_code)]
    pub test_type: Vec<String>,
    #[allow(dead_code)]
    pub df: Vec<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub f_statistic: Option<Vec<f64>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub f_p_value: Option<Vec<f64>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub f_df: Option<Vec<f64>>,
    #[allow(dead_code)]
    pub passed: Vec<bool>,
    #[allow(dead_code)]
    pub interpretation: Vec<String>,
    #[allow(dead_code)]
    pub description: Vec<String>,
}

/// Python Breusch-Godfrey result format
#[derive(Debug, Deserialize)]
pub struct PythonBreuschGodfreyResult {
    #[allow(dead_code)]
    pub test_name: String,
    #[allow(dead_code)]
    pub dataset: String,
    #[allow(dead_code)]
    pub formula: String,
    pub order: i64,
    pub statistic: f64,
    pub p_value: f64,
    #[allow(dead_code)]
    pub test_type: String,
    #[allow(dead_code)]
    pub df: Vec<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub f_statistic: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub f_p_value: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub f_df: Option<Vec<f64>>,
    #[allow(dead_code)]
    pub passed: bool,
    #[allow(dead_code)]
    pub interpretation: String,
    #[allow(dead_code)]
    pub description: String,
}

// ============================================================================
// Ridge & Lasso Specific Structures
// ============================================================================

/// Ridge regression result from glmnet (R format)
#[derive(Debug, Deserialize)]
pub struct RRidgeResult {
    #[allow(dead_code)]
    pub test: String,
    #[allow(dead_code)]
    pub method: String,
    #[allow(dead_code)]
    pub alpha: f64,
    pub n: usize,
    pub p: usize,
    pub lambda_sequence: Vec<f64>,
    pub coefficients: Vec<Vec<f64>>,
    #[allow(dead_code)]
    pub degrees_of_freedom: Vec<f64>,
    #[allow(dead_code)]
    pub test_lambdas: Vec<f64>,
    pub test_predictions: Vec<Vec<f64>>,
    #[allow(dead_code)]
    pub fitted_values: Vec<f64>,
    #[allow(dead_code)]
    pub residuals: Vec<f64>,
    pub glmnet_version: String,
}

/// Lasso regression result from glmnet (R format)
#[derive(Debug, Deserialize)]
pub struct RLassoResult {
    #[allow(dead_code)]
    pub test: String,
    #[allow(dead_code)]
    pub method: String,
    #[allow(dead_code)]
    pub alpha: f64,
    pub n: usize,
    pub p: usize,
    pub lambda_sequence: Vec<f64>,
    pub coefficients: Vec<Vec<f64>>,
    pub nonzero_counts: Vec<usize>,
    #[allow(dead_code)]
    pub degrees_of_freedom: Vec<f64>,
    #[allow(dead_code)]
    pub test_lambdas: Vec<f64>,
    pub test_predictions: Vec<Vec<f64>>,
    #[allow(dead_code)]
    pub fitted_values: Vec<f64>,
    #[allow(dead_code)]
    pub residuals: Vec<f64>,
    pub glmnet_version: String,
}

// ============================================================================
// Dataset Structure
// ============================================================================

/// Dataset loaded from CSV
pub struct Dataset {
    #[allow(dead_code)]
    pub name: String,
    pub y: Vec<f64>,
    pub x_vars: Vec<Vec<f64>>,
    #[allow(dead_code)]
    pub variable_names: Vec<String>,
}

// ============================================================================
// Data Loading Functions
// ============================================================================

/// Housing regression data (same as used in R/Python validation scripts)
pub fn get_housing_data() -> (Vec<f64>, Vec<Vec<f64>>) {
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
    (y, vec![square_feet, bedrooms, age])
}

/// Load validation results from JSON file
///
/// # Panics
///
/// Panics with a helpful error message if the file is not found or cannot be parsed,
/// including instructions on how to generate the missing reference file.
pub fn load_validation_results(json_path: &Path) -> RegressionResult {
    // Check file existence first to provide a better error message
    if !json_path.exists() {
        let script_name = if json_path.to_string_lossy().contains("/r/") {
            "verification/scripts/runners/run_all_diagnostics_r.R"
        } else {
            "verification/scripts/runners/run_all_diagnostics_python.py"
        };
        panic!(
            "Validation file not found: {}\n\
             \n\
             To generate this file, run:\n\
               {}",
            json_path.display(),
            script_name
        );
    }

    let json_content = fs::read_to_string(json_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read validation file {:?}: {}\n\
             \n\
             To generate this file, run the appropriate validation script from verification/scripts/runners/",
            json_path, e
        )
    });

    let wrapper: ValidationWrapper = serde_json::from_str(&json_content).unwrap_or_else(|e| {
        panic!(
            "Failed to parse JSON from {:?}: {}\n\
             \n\
             The file may be corrupted. Try regenerating it by running the appropriate validation script from verification/scripts/runners/",
            json_path, e
        )
    });

    wrapper.housing_regression
}

/// Categorical encoding scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategoricalEncoding {
    /// 0-based encoding: first category = 0, second = 1, etc. (Python-like)
    ZeroBased,
    /// 1-based encoding: first category = 1, second = 2, etc. (R-like)
    OneBased,
}

impl CategoricalEncoding {
    /// Get the offset to add to the index
    fn offset(self) -> f64 {
        match self {
            CategoricalEncoding::ZeroBased => 0.0,
            CategoricalEncoding::OneBased => 1.0,
        }
    }
}

/// Load a dataset from a CSV file with categorical encoding support
/// Similar to Python's pd.factorize() or R's factor() for categorical variables
/// Uses 0-based encoding by default (Python-like)
pub fn load_dataset(csv_path: &Path) -> Result<Dataset, Box<dyn std::error::Error>> {
    load_dataset_with_encoding(csv_path, CategoricalEncoding::ZeroBased)
}

/// Custom error for dataset loading failures
#[derive(Debug)]
pub struct DatasetLoadError {
    pub path: String,
    pub reason: String,
}

impl std::fmt::Display for DatasetLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Failed to load dataset from {}: {}\n\
             \n\
             Dataset files should be in verification/datasets/csv/\n\
             \n\
             Available datasets: bodyfat, cars_stopping, faithful, iris, lh, longley,\n\
                                 mtcars, prostate, ToothGrowth, synthetic_*.csv",
            self.path, self.reason
        )
    }
}

impl std::error::Error for DatasetLoadError {}

/// Load a dataset from a CSV file with specified categorical encoding
pub fn load_dataset_with_encoding(
    csv_path: &Path,
    encoding: CategoricalEncoding,
) -> Result<Dataset, Box<dyn std::error::Error>> {
    let dataset_name = csv_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // Check file existence first for a better error message
    if !csv_path.exists() {
        return Err(Box::new(DatasetLoadError {
            path: csv_path.display().to_string(),
            reason: "file not found".to_string(),
        }));
    }

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path).map_err(|e| {
            Box::new(DatasetLoadError {
                path: csv_path.display().to_string(),
                reason: format!("IO error: {}", e),
            }) as Box<dyn std::error::Error>
        })?;

    let headers = rdr.headers()?.clone();

    // First column is y (dependent variable), rest are x_vars
    let x_names: Vec<String> = headers.iter().skip(1).map(|s| s.to_string()).collect();

    // First pass: collect all raw string values for each column
    let mut raw_y_values: Vec<String> = Vec::new();
    let mut raw_x_values: Vec<Vec<String>> = vec![Vec::new(); x_names.len()];

    for result in rdr.records() {
        let record = result?;
        if record.len() < headers.len() {
            continue;
        }

        // Collect y value
        if let Some(y_str) = record.get(0) {
            raw_y_values.push(y_str.to_string());
        }

        // Collect x values
        for (i, x_val_str) in record.iter().skip(1).enumerate() {
            if i < raw_x_values.len() {
                raw_x_values[i].push(x_val_str.to_string());
            }
        }
    }

    // Build encoding maps for categorical columns
    let mut y_encoding: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut x_encodings: Vec<std::collections::HashMap<String, f64>> =
        vec![std::collections::HashMap::new(); x_names.len()];

    let offset = encoding.offset();
    let start_label = if offset == 0.0 { "0, 1, 2, ..." } else { "1, 2, 3, ..." };

    // Build y encoding (if needed)
    let y_needs_encoding = raw_y_values.iter().any(|v| v.parse::<f64>().is_err());
    if y_needs_encoding {
        let mut unique_vals: Vec<String> = raw_y_values.iter().map(|s| s.clone()).collect();
        unique_vals.sort();
        unique_vals.dedup();
        for (idx, val) in unique_vals.iter().enumerate() {
            y_encoding.insert(val.clone(), idx as f64 + offset);
        }
        eprintln!(
            "    INFO: y column is categorical, {} categories encoded as {}",
            unique_vals.len(),
            start_label
        );
    }

    // Build x encodings (if needed)
    for (col_idx, col_values) in raw_x_values.iter().enumerate() {
        let needs_encoding = col_values.iter().any(|v| v.parse::<f64>().is_err());
        if needs_encoding {
            let mut unique_vals: Vec<String> = col_values.iter().map(|s| s.clone()).collect();
            unique_vals.sort();
            unique_vals.dedup();
            for (idx, val) in unique_vals.iter().enumerate() {
                x_encodings[col_idx].insert(val.clone(), idx as f64 + offset);
            }
            eprintln!(
                "    INFO: {} is categorical, {} categories encoded as {}",
                x_names[col_idx],
                unique_vals.len(),
                start_label
            );
        }
    }

    // Second pass: convert using encodings
    let mut y_data = Vec::new();
    let mut x_data: Vec<Vec<f64>> = vec![Vec::new(); x_names.len()];

    for (row_idx, y_str) in raw_y_values.iter().enumerate() {
        // Convert y value
        let y_val = if let Some(&encoded) = y_encoding.get(y_str) {
            encoded
        } else {
            y_str.parse::<f64>().unwrap_or(0.0)
        };
        y_data.push(y_val);

        // Convert x values
        for (col_idx, x_str) in raw_x_values.iter().enumerate() {
            if let Some(x_val_str) = x_str.get(row_idx) {
                let x_val = if let Some(&encoded) = x_encodings[col_idx].get(x_val_str) {
                    encoded
                } else {
                    x_val_str.parse::<f64>().unwrap_or(0.0)
                };
                x_data[col_idx].push(x_val);
            }
        }
    }

    // Variable names: intercept + all predictors
    let mut variable_names = vec!["Intercept".to_string()];
    variable_names.extend(x_names);

    Ok(Dataset {
        name: dataset_name,
        y: y_data,
        x_vars: x_data,
        variable_names,
    })
}

// ============================================================================
// Result Loaders
// ============================================================================

/// Generic R diagnostic result loader
pub fn load_r_diagnostic_result(json_path: &Path) -> Option<RDiagnosticResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Generic Python diagnostic result loader
pub fn load_python_diagnostic_result(json_path: &Path) -> Option<PythonDiagnosticResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Check if a Python result file was marked as skipped (e.g., due to multicolllinearity)
/// Returns Some(true) if skipped, Some(false) if not skipped, None if file doesn't exist or can't be parsed
pub fn check_python_result_skipped(json_path: &Path) -> Option<bool> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    // Parse just the "skipped" field to avoid issues with null values
    #[derive(Deserialize)]
    struct SkippedOnly {
        #[serde(default)]
        skipped: bool,
    }
    let parsed: Result<SkippedOnly, _> = serde_json::from_str(&content);
    match parsed {
        Ok(s) => Some(s.skipped),
        Err(_) => None, // File exists but can't parse - treat as not skipped
    }
}

/// Load R Breusch-Pagan result from JSON
pub fn load_r_bp_result(json_path: &Path) -> Option<RBreuschPaganResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Load Python Breusch-Pagan result from JSON
pub fn load_python_bp_result(json_path: &Path) -> Option<PythonBreuschPaganResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Load R Shapiro-Wilk result from JSON
pub fn load_r_shapiro_wilk_result(json_path: &Path) -> Option<RShapiroWilkResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Load Python Shapiro-Wilk result from JSON
pub fn load_python_shapiro_wilk_result(json_path: &Path) -> Option<PythonShapiroWilkResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Load R Cook's Distance result from JSON
pub fn load_r_cooks_result(json_path: &Path) -> Option<RCooksDistanceResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Load Python Cook's Distance result from JSON
pub fn load_python_cooks_result(json_path: &Path) -> Option<PythonCooksDistanceResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Load R Breusch-Godfrey result from JSON
pub fn load_r_bg_result(json_path: &Path) -> Option<RBreuschGodfreyResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Load Python Breusch-Godfrey result from JSON
pub fn load_python_bg_result(json_path: &Path) -> Option<PythonBreuschGodfreyResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Load ridge result from JSON
pub fn load_ridge_result(json_path: &Path) -> Option<RRidgeResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Load lasso result from JSON
pub fn load_lasso_result(json_path: &Path) -> Option<RLassoResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

// ============================================================================
// Per-Dataset OLS Result Structures
// ============================================================================

/// OLS result from per-dataset R/Python validation
#[derive(Debug, Deserialize)]
pub struct OlsByDatasetResult {
    #[allow(dead_code)]
    pub test: String,
    #[allow(dead_code)]
    pub method: String,
    #[allow(dead_code)]
    pub dataset: String,
    #[allow(dead_code)]
    pub formula: String,
    pub n: usize,
    pub k: usize,
    pub df_residual: usize,
    pub variable_names: Vec<String>,
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub f_statistic: f64,
    pub f_p_value: f64,
    pub mse: f64,
    pub std_error: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub conf_int_lower: Vec<f64>,
    pub conf_int_upper: Vec<f64>,
    #[allow(dead_code)]
    pub residuals: Vec<f64>,
    pub vif: Vec<OlsVifEntry>,
}

#[derive(Debug, Deserialize)]
pub struct OlsVifEntry {
    pub variable: String,
    pub vif: f64,
    #[allow(dead_code)]
    pub rsquared: f64,
}

/// Load OLS result from per-dataset JSON file
pub fn load_ols_by_dataset_result(json_path: &Path) -> Option<OlsByDatasetResult> {
    if !json_path.exists() {
        return None;
    }
    let content = fs::read_to_string(json_path).ok()?;
    serde_json::from_str(&content).ok()
}

// ============================================================================
// Loud Failure Helpers (panic with instructions)
// ============================================================================

/// Load R diagnostic result, panicking loudly with instructions if file not found
pub fn expect_r_diagnostic_result(json_path: &Path, test_name: &str) -> RDiagnosticResult {
    if !json_path.exists() {
        panic!(
            "R diagnostic result file not found: {}\n\
             \n\
             Test: {}\n\
             \n\
             To generate this file, run:\n\
               cd verification/scripts/runners && Rscript run_all_diagnostics_r.R\n\
             \n\
             Or generate individual test results from verification/scripts/r/diagnostics/",
            json_path.display(),
            test_name
        );
    }
    load_r_diagnostic_result(json_path).unwrap_or_else(|| {
        panic!(
            "Failed to parse R diagnostic result file: {}\n\
             \n\
             The file may be corrupted. Try regenerating it by running:\n\
               cd verification/scripts/runners && Rscript run_all_diagnostics_r.R",
            json_path.display()
        )
    })
}

/// Load Python diagnostic result, panicking loudly with instructions if file not found
pub fn expect_python_diagnostic_result(json_path: &Path, test_name: &str) -> PythonDiagnosticResult {
    if !json_path.exists() {
        panic!(
            "Python diagnostic result file not found: {}\n\
             \n\
             Test: {}\n\
             \n\
             To generate this file, run:\n\
               cd verification/scripts/runners && python run_all_diagnostics_python.py\n\
             \n\
             Or generate individual test results from verification/scripts/python/diagnostics/",
            json_path.display(),
            test_name
        );
    }
    load_python_diagnostic_result(json_path).unwrap_or_else(|| {
        panic!(
            "Failed to parse Python diagnostic result file: {}\n\
             \n\
             The file may be corrupted. Try regenerating it by running:\n\
               cd verification/scripts/runners && python run_all_diagnostics_python.py",
            json_path.display()
        )
    })
}

/// Load ridge result, panicking loudly with instructions if file not found
pub fn expect_ridge_result(json_path: &Path) -> RRidgeResult {
    if !json_path.exists() {
        panic!(
            "Ridge result file not found: {}\n\
             \n\
             To generate this file, run:\n\
               cd verification/scripts/r/regularized && Rscript test_ridge.R\n\
             \n\
             Or generate all regularized results:\n\
               cd verification/scripts/r/regularized && Rscript generate_all_references.R",
            json_path.display()
        );
    }
    load_ridge_result(json_path).unwrap_or_else(|| {
        panic!(
            "Failed to parse ridge result file: {}\n\
             \n\
             The file may be corrupted. Try regenerating it.",
            json_path.display()
        )
    })
}

/// Load lasso result, panicking loudly with instructions if file not found
pub fn expect_lasso_result(json_path: &Path) -> RLassoResult {
    if !json_path.exists() {
        panic!(
            "Lasso result file not found: {}\n\
             \n\
             To generate this file, run:\n\
               cd verification/scripts/r/regularized && Rscript test_lasso.R\n\
             \n\
             Or generate all regularized results:\n\
               cd verification/scripts/r/regularized && Rscript generate_all_references.R",
            json_path.display()
        );
    }
    load_lasso_result(json_path).unwrap_or_else(|| {
        panic!(
            "Failed to parse lasso result file: {}\n\
             \n\
             The file may be corrupted. Try regenerating it.",
            json_path.display()
        )
    })
}

/// Load OLS by dataset result, panicking loudly with instructions if file not found
pub fn expect_ols_by_dataset_result(json_path: &Path) -> OlsByDatasetResult {
    if !json_path.exists() {
        panic!(
            "OLS result file not found: {}\n\
             \n\
             To generate this file, run:\n\
               cd verification/scripts/runners && Rscript run_all_diagnostics_r.R",
            json_path.display()
        );
    }
    load_ols_by_dataset_result(json_path).unwrap_or_else(|| {
        panic!(
            "Failed to parse OLS result file: {}\n\
             \n\
             The file may be corrupted. Try regenerating it.",
            json_path.display()
        )
    })
}

// ============================================================================
// Assertion Helpers
// ============================================================================

/// Helper function to assert two values are close within tolerance
///
/// Tolerance strategy:
/// 1. If |expected| < 1e-3: use absolute tolerance 1e-4 (near-zero check)
/// 2. Else if lasso and |expected| < 0.5: use relaxed relative tolerance (10%/20% for small coefficients)
/// 3. Else: use standard relative tolerance (1%/2%)
///
/// This handles:
/// - Near-zero values where relative error would blow up (use fixed absolute tolerance)
/// - Small lasso coefficients where numerical differences dominate (use relaxed relative)
/// - Normal values with standard relative tolerance
pub fn assert_close_to(actual: f64, expected: f64, tolerance: f64, context: &str) {
    let diff = (actual - expected).abs();

    // Determine appropriate tolerance and comparison method
    let (effective_tolerance, use_absolute) = if expected.abs() < 1e-3 {
        // Near-zero: use small fixed absolute tolerance (relative would blow up)
        (1e-4, true)  // 0.0001 absolute tolerance for near-zero values
    } else if context.contains("lasso") && expected.abs() < 0.5 {
        // Small lasso coefficient: use relaxed relative tolerance
        let relaxed_tol = if tolerance == LASSO_TOLERANCE {
            LASSO_TOLERANCE_SMALL
        } else if tolerance == LASSO_TOLERANCE_LOOSE {
            LASSO_TOLERANCE_SMALL_LOOSE
        } else {
            tolerance
        };
        (relaxed_tol, false)
    } else {
        // Normal case: use standard relative tolerance
        (tolerance, false)
    };

    let check_value = if use_absolute {
        diff
    } else {
        diff / expected.abs()
    };

    if check_value > effective_tolerance {
        panic!(
            "{} mismatch: actual = {:.6}, expected = {:.6}, diff = {:.6} (check = {:.6}, tolerance = {:.6})",
            context, actual, expected, diff, check_value, effective_tolerance
        );
    }
}

/// Helper function to print a comparison between Rust and R
pub fn print_comparison_r(label: &str, rust_val: f64, r_val: f64, indent: &str) {
    let diff = (rust_val - r_val).abs();
    println!("{}{}", indent, label);
    println!("{}  Rust:     {:.15}", indent, rust_val);
    println!("{}  R:        {:.15}", indent, r_val);
    println!("{}  Diff:     {:.2e}", indent, diff);
    println!();
}

/// Helper function to print a comparison between Rust and Python
pub fn print_comparison_python(label: &str, rust_val: f64, py_val: f64, indent: &str) {
    let diff = (rust_val - py_val).abs();
    println!("{}{}", indent, label);
    println!("{}  Rust:     {:.15}", indent, rust_val);
    println!("{}  Python:   {:.15}", indent, py_val);
    println!("{}  Diff:     {:.2e}", indent, diff);
    println!();
}

// ============================================================================
// Dataset Lists
// ============================================================================

/// All datasets available for validation
pub const ALL_DATASETS: &[&str] = &[
    "bodyfat",
    "cars_stopping",
    "faithful",
    "iris",
    "lh",
    "longley",
    "mtcars",
    "prostate",
    "synthetic_simple_linear",
    "synthetic_multiple",
    "synthetic_collinear",
    "synthetic_heteroscedastic",
    "synthetic_nonlinear",
    "synthetic_nonnormal",
    "synthetic_autocorrelated",
    "synthetic_high_vif",
    "synthetic_outliers",
    "synthetic_small",
    "synthetic_interaction",
    "ToothGrowth",
];

/// Datasets for Shapiro-Wilk validation (excludes synthetic due to column ordering)
pub const SHAPIRO_WILK_DATASETS: &[&str] = &["bodyfat", "longley", "mtcars", "prostate"];
