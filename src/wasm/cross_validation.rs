//! Cross-validation methods for WASM
//!
//! Provides WASM bindings for K-Fold Cross Validation with OLS, Ridge, Lasso,
//! and Elastic Net regression.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use super::domain::check_domain;
use crate::cross_validation::{kfold_cv_elastic_net as native_cv_enet, kfold_cv_lasso as native_cv_lasso, kfold_cv_ols as native_cv_ols, kfold_cv_ridge as native_cv_ridge, KFoldOptions};
use crate::error::{error_json, error_to_json};
use crate::Error;

/// Parses a JSON string to a boolean value.
///
/// Accepts "true"/"false" (case-insensitive) or "1"/"0".
fn parse_bool(json: &str) -> Result<bool, String> {
    let trimmed = json.trim().to_lowercase();
    match trimmed.as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        "1" => Ok(true),
        "0" => Ok(false),
        _ => Err(format!("Invalid boolean value: {}", json)),
    }
}

/// Parses a JSON string to an optional u64 seed value.
///
/// Accepts a number as a string, or "null" for None.
fn parse_seed(json: &str) -> Result<Option<u64>, String> {
    let trimmed = json.trim();
    if trimmed.eq_ignore_ascii_case("null") || trimmed.is_empty() {
        return Ok(None);
    }
    trimmed
        .parse::<u64>()
        .map(Some)
        .map_err(|e| format!("Invalid seed value: {}", e))
}

/// Serializes a CV result to JSON, or returns an error JSON.
fn serialize_cv_result<T: serde::Serialize>(result: Result<T, Error>) -> String {
    match result {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| error_json("Failed to serialize CV result")),
        Err(e) => error_json(&e.to_string()),
    }
}

/// Performs K-Fold Cross Validation for OLS regression via WASM.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `variable_names_json` - JSON array of variable names
/// * `n_folds` - Number of folds (must be >= 2)
/// * `shuffle_json` - JSON boolean: whether to shuffle data before splitting
/// * `seed_json` - JSON string with seed number or "null" for no seed
///
/// # Returns
///
/// JSON string containing CV results:
/// - `n_folds` - Number of folds used
/// - `n_samples` - Total number of observations
/// - `mean_mse`, `std_mse` - Mean and std of MSE across folds
/// - `mean_rmse`, `std_rmse` - Mean and std of RMSE across folds
/// - `mean_mae`, `std_mae` - Mean and std of MAE across folds
/// - `mean_r_squared`, `std_r_squared` - Mean and std of R² across folds
/// - `fold_results` - Array of individual fold results
/// - `fold_coefficients` - Array of coefficient arrays from each fold
///
/// # Errors
///
/// Returns a JSON error object if parsing fails, parameters are invalid,
/// or domain check fails.
#[wasm_bindgen]
pub fn kfold_cv_ols(
    y_json: &str,
    x_vars_json: &str,
    variable_names_json: &str,
    n_folds: usize,
    shuffle_json: &str,
    seed_json: &str,
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

    let variable_names: Vec<String> = match serde_json::from_str(variable_names_json) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("Failed to parse variable_names: {}", e)),
    };

    let shuffle = match parse_bool(shuffle_json) {
        Ok(b) => b,
        Err(e) => return error_json(&e),
    };

    let seed = match parse_seed(seed_json) {
        Ok(s) => s,
        Err(e) => return error_json(&e),
    };

    // Configure CV options
    let options = KFoldOptions {
        n_folds,
        shuffle,
        seed,
    };

    serialize_cv_result(native_cv_ols(&y, &x_vars, &variable_names, &options))
}

/// Performs K-Fold Cross Validation for Ridge regression via WASM.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `lambda` - Regularization strength (>= 0)
/// * `standardize` - Whether to standardize predictors
/// * `n_folds` - Number of folds (must be >= 2)
/// * `shuffle_json` - JSON boolean: whether to shuffle data before splitting
/// * `seed_json` - JSON string with seed number or "null" for no seed
///
/// # Returns
///
/// JSON string containing CV results (same structure as OLS).
///
/// # Errors
///
/// Returns a JSON error object if parsing fails, parameters are invalid,
/// or domain check fails.
#[wasm_bindgen]
pub fn kfold_cv_ridge(
    y_json: &str,
    x_vars_json: &str,
    lambda: f64,
    standardize: bool,
    n_folds: usize,
    shuffle_json: &str,
    seed_json: &str,
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

    let shuffle = match parse_bool(shuffle_json) {
        Ok(b) => b,
        Err(e) => return error_json(&e),
    };

    let seed = match parse_seed(seed_json) {
        Ok(s) => s,
        Err(e) => return error_json(&e),
    };

    let options = KFoldOptions {
        n_folds,
        shuffle,
        seed,
    };

    serialize_cv_result(native_cv_ridge(&x_vars, &y, lambda, standardize, &options))
}

/// Performs K-Fold Cross Validation for Lasso regression via WASM.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `lambda` - Regularization strength (>= 0)
/// * `standardize` - Whether to standardize predictors
/// * `n_folds` - Number of folds (must be >= 2)
/// * `shuffle_json` - JSON boolean: whether to shuffle data before splitting
/// * `seed_json` - JSON string with seed number or "null" for no seed
///
/// # Returns
///
/// JSON string containing CV results (same structure as OLS).
///
/// # Errors
///
/// Returns a JSON error object if parsing fails, parameters are invalid,
/// or domain check fails.
#[wasm_bindgen]
pub fn kfold_cv_lasso(
    y_json: &str,
    x_vars_json: &str,
    lambda: f64,
    standardize: bool,
    n_folds: usize,
    shuffle_json: &str,
    seed_json: &str,
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

    let shuffle = match parse_bool(shuffle_json) {
        Ok(b) => b,
        Err(e) => return error_json(&e),
    };

    let seed = match parse_seed(seed_json) {
        Ok(s) => s,
        Err(e) => return error_json(&e),
    };

    let options = KFoldOptions {
        n_folds,
        shuffle,
        seed,
    };

    serialize_cv_result(native_cv_lasso(&x_vars, &y, lambda, standardize, &options))
}

/// Performs K-Fold Cross Validation for Elastic Net regression via WASM.
///
/// # Arguments
///
/// * `y_json` - JSON array of response variable values
/// * `x_vars_json` - JSON array of predictor arrays
/// * `lambda` - Regularization strength (>= 0)
/// * `alpha` - Mixing parameter (0 = Ridge, 1 = Lasso)
/// * `standardize` - Whether to standardize predictors
/// * `n_folds` - Number of folds (must be >= 2)
/// * `shuffle_json` - JSON boolean: whether to shuffle data before splitting
/// * `seed_json` - JSON string with seed number or "null" for no seed
///
/// # Returns
///
/// JSON string containing CV results (same structure as OLS).
///
/// # Errors
///
/// Returns a JSON error object if parsing fails, parameters are invalid,
/// or domain check fails.
#[wasm_bindgen]
pub fn kfold_cv_elastic_net(
    y_json: &str,
    x_vars_json: &str,
    lambda: f64,
    alpha: f64,
    standardize: bool,
    n_folds: usize,
    shuffle_json: &str,
    seed_json: &str,
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

    let shuffle = match parse_bool(shuffle_json) {
        Ok(b) => b,
        Err(e) => return error_json(&e),
    };

    let seed = match parse_seed(seed_json) {
        Ok(s) => s,
        Err(e) => return error_json(&e),
    };

    let options = KFoldOptions {
        n_folds,
        shuffle,
        seed,
    };

    serialize_cv_result(native_cv_enet(&x_vars, &y, lambda, alpha, standardize, &options))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bool_valid() {
        assert!(parse_bool("true").unwrap());
        assert!(parse_bool("True").unwrap());
        assert!(parse_bool("TRUE").unwrap());
        assert!(!parse_bool("false").unwrap());
        assert!(!parse_bool("False").unwrap());
        assert!(parse_bool("1").unwrap());
        assert!(!parse_bool("0").unwrap());
    }

    #[test]
    fn test_parse_bool_invalid() {
        assert!(parse_bool("invalid").is_err());
        assert!(parse_bool("2").is_err());
    }

    #[test]
    fn test_parse_seed_valid() {
        assert_eq!(parse_seed("42").unwrap(), Some(42));
        assert_eq!(parse_seed("0").unwrap(), Some(0));
        assert_eq!(parse_seed("null").unwrap(), None);
        assert_eq!(parse_seed("NULL").unwrap(), None);
        assert_eq!(parse_seed("").unwrap(), None);
    }

    #[test]
    fn test_parse_seed_invalid() {
        assert!(parse_seed("-1").is_err());
        assert!(parse_seed("invalid").is_err());
    }
}
