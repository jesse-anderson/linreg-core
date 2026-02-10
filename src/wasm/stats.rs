//! Statistical utility functions for WASM
//!
//! Provides WASM bindings for common statistical functions.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use super::domain::check_domain;
use crate::core;
use crate::distributions::{normal_inverse_cdf, student_t_cdf};
use crate::stats;

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
#[wasm_bindgen]
pub fn get_t_cdf(t: f64, df: f64) -> f64 {
    if check_domain().is_err() {
        return f64::NAN;
    }

    student_t_cdf(t, df)
}

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
#[wasm_bindgen]
pub fn get_t_critical(alpha: f64, df: f64) -> f64 {
    if check_domain().is_err() {
        return f64::NAN;
    }

    core::t_critical_quantile(df, alpha)
}

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
#[wasm_bindgen]
pub fn get_normal_inverse(p: f64) -> f64 {
    if check_domain().is_err() {
        return f64::NAN;
    }

    normal_inverse_cdf(p)
}

/// Computes the arithmetic mean of a JSON array of f64 values.
///
/// # Arguments
///
/// * `data_json` - JSON string representing an array of f64 values
///
/// # Returns
///
/// JSON string with the mean, or "null" if input is invalid/empty
#[wasm_bindgen]
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
#[wasm_bindgen]
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
#[wasm_bindgen]
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

/// Computes the median of a JSON array of f64 values.
///
/// # Arguments
///
/// * `data_json` - JSON string representing an array of f64 values
///
/// # Returns
///
/// JSON string with the median, or "null" if input is invalid/empty
#[wasm_bindgen]
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
#[wasm_bindgen]
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
#[wasm_bindgen]
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
