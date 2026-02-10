//! LOESS regression for WASM
//!
//! Provides WASM bindings for LOESS (Locally Estimated Scatterplot Smoothing) regression.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use super::domain::check_domain;
use crate::error::{error_json, error_to_json};
use crate::loess;

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
