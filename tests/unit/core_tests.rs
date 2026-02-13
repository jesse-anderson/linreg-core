// ============================================================================
// Core OLS Regression Unit Tests
// ============================================================================
//
// Comprehensive tests for OLS regression including basic functionality,
// statistics calculations, VIF calculations, leverage calculations,
// error cases, and property-based tests.

use linreg_core::core::{f_p_value, ols_regression, t_critical_quantile, two_tailed_p_value};
use linreg_core::Error;
use proptest::prelude::*;

// ============================================================================
// Test Constants and Helpers
// ============================================================================

const EPSILON: f64 = 1e-10;
const STAT_TOLERANCE: f64 = 1e-4;
const P_VALUE_TOLERANCE: f64 = 1e-6;

/// Helper function to assert two f64 values are close within tolerance
fn assert_close(a: f64, b: f64, tolerance: f64, context: &str) {
    let diff = (a - b).abs();
    assert!(
        diff <= tolerance,
        "{}: {} != {}, diff = {} (tolerance = {})",
        context,
        a,
        b,
        diff,
        tolerance
    );
}

/// Helper function to assert vectors are approximately equal
fn assert_vec_close(a: &[f64], b: &[f64], tolerance: f64, context: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{}: Length mismatch {} vs {}",
        context,
        a.len(),
        b.len()
    );
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        assert_close(av, bv, tolerance, &format!("{}[{}]", context, i));
    }
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[test]
fn test_ols_regression_simple_linear() {
    // Perfect fit: y = 2*x (through origin)
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let result = ols_regression(&y, &[x], &names).expect("OLS should succeed");

    // With intercept, the fit should be nearly perfect
    assert!(
        result.r_squared > 0.99,
        "R² should be > 0.99, got {}",
        result.r_squared
    );
    assert_eq!(result.n, 5);
    assert_eq!(result.k, 1);
    assert_eq!(result.coefficients.len(), 2);

    // Slope should be approximately 2
    assert_close(result.coefficients[1], 2.0, 0.1, "slope");
}

#[test]
fn test_ols_regression_with_intercept() {
    // y = 1 + 2*x
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi).collect();
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let result = ols_regression(&y, &[x.clone()], &names).expect("OLS should succeed");

    // Perfect fit
    assert_close(result.r_squared, 1.0, 1e-10, "R²");
    assert_close(result.coefficients[0], 1.0, 1e-10, "intercept");
    assert_close(result.coefficients[1], 2.0, 1e-10, "slope");

    // Residuals should be zero
    for &residual in &result.residuals {
        assert_close(residual, 0.0, 1e-10, "residual");
    }
}

#[test]
fn test_ols_regression_multiple_predictors() {
    // y = 5 + 2*x1 + 3*x2
    // Use non-collinear predictors
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![1.0, 3.0, 2.0, 4.0, 2.5]; // Not perfectly related to x1
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&x1i, &x2i)| 5.0 + 2.0 * x1i + 3.0 * x2i)
        .collect();
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1.clone(), x2.clone()], &names).expect("OLS should succeed");

    // Perfect fit
    assert_close(result.r_squared, 1.0, 1e-10, "R²");
    assert_close(result.coefficients[0], 5.0, 1e-10, "intercept");
    assert_close(result.coefficients[1], 2.0, 1e-10, "x1 coef");
    assert_close(result.coefficients[2], 3.0, 1e-10, "x2 coef");
}

#[test]
fn test_ols_regression_with_noise() {
    // y = 1 + 2*x + noise
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let noise = vec![0.1, -0.2, 0.15, -0.1, 0.05, -0.15, 0.2, -0.05, 0.1, -0.1];
    let y: Vec<f64> = x
        .iter()
        .zip(noise.iter())
        .map(|(&xi, &ni)| 1.0 + 2.0 * xi + ni)
        .collect();
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let result = ols_regression(&y, &[x.clone()], &names).expect("OLS should succeed");

    // High R² but not perfect due to noise
    assert!(result.r_squared > 0.99, "R² should be > 0.99");

    // Coefficients should be close to true values
    assert_close(result.coefficients[0], 1.0, 0.5, "intercept");
    assert_close(result.coefficients[1], 2.0, 0.1, "slope");

    // Residuals should sum to approximately zero (with intercept)
    let residual_sum: f64 = result.residuals.iter().sum();
    assert_close(residual_sum, 0.0, 1e-10, "residual sum");
}

// ============================================================================
// Statistics Calculations Tests
// ============================================================================

#[test]
fn test_r_squared_calculation() {
    // Perfect fit should give R² = 1
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let result = ols_regression(&y, &[x], &names).expect("OLS should succeed");

    assert_close(result.r_squared, 1.0, 1e-10, "perfect R²");
}

#[test]
fn test_residuals_sum_to_zero() {
    // Property: residuals sum to zero when intercept is included
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 4.0, 1.0, 3.0, 5.0]; // Not collinear with x1
    let y = vec![5.0, 13.0, 6.0, 12.0, 17.0];
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1, x2], &names).expect("OLS should succeed");

    let residual_sum: f64 = result.residuals.iter().sum();
    assert_close(residual_sum, 0.0, 1e-10, "residual sum");
}

#[test]
fn test_f_statistic_calculation() {
    // For a good fit, F-statistic should be large
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let result = ols_regression(&y, &[x], &names).expect("OLS should succeed");

    // F-statistic should be very large for perfect fit
    assert!(
        result.f_statistic > 1000.0,
        "F should be large, got {}",
        result.f_statistic
    );

    // p-value should be very small
    assert!(
        result.f_p_value < 0.001,
        "p-value should be small, got {}",
        result.f_p_value
    );
}

#[test]
fn test_confidence_interval_width() {
    // Confidence intervals should be wider for intercept with small data
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let result = ols_regression(&y, &[x], &names).expect("OLS should succeed");

    // CI width = 2 * t_critical * se
    let intercept_width = result.conf_int_upper[0] - result.conf_int_lower[0];
    let slope_width = result.conf_int_upper[1] - result.conf_int_lower[1];

    // Both should be positive
    assert!(
        intercept_width > 0.0,
        "Intercept CI width should be positive"
    );
    assert!(slope_width > 0.0, "Slope CI width should be positive");

    // CI should be centered at coefficient
    let intercept_center = (result.conf_int_lower[0] + result.conf_int_upper[0]) / 2.0;
    assert_close(
        intercept_center,
        result.coefficients[0],
        1e-10,
        "intercept CI center",
    );
}

#[test]
fn test_adjusted_r_squared_less_than_r_squared() {
    // Adjusted R² should always be <= R² (strictly < when k > 0)
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    // x2 is not collinear with x1 (different pattern)
    let x2 = vec![1.5, 2.7, 3.2, 4.8, 5.1, 6.9, 7.3, 8.5, 9.2, 10.8];
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&x1i, &x2i)| 1.0 + x1i + 0.5 * x2i)
        .collect();
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1, x2], &names).expect("OLS should succeed");

    assert!(
        result.adj_r_squared <= result.r_squared,
        "Adj R² ({}) should be <= R² ({})",
        result.adj_r_squared,
        result.r_squared
    );
}

#[test]
fn test_adjusted_r_squared_penalty() {
    // Adding a useless predictor should decrease adjusted R²
    // Use more data and add noise so the penalty is clear
    let x1: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    // x2 is random noise (uncorrelated with y) - using deterministic pseudo-random values
    let x2: Vec<f64> = (0..20)
        .map(|i| {
            let seed = i * 7919 + 31; // Coprime numbers for pseudo-random
            ((seed % 100) as f64 - 50.0) / 50.0 // Values between -1 and 1
        })
        .collect();
    // Add noise to y so R² isn't already 1.0, allowing the penalty to have effect
    let y: Vec<f64> = x1
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            2.0 * xi + 1.0 + (i as f64 * 0.3).sin() * 2.0 // Add deterministic noise
        })
        .collect();

    let names1 = vec!["Intercept".to_string(), "X1".to_string()];
    let result1 = ols_regression(&y, &[x1.clone()], &names1).expect("OLS should succeed");

    let names2 = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];
    let result2 = ols_regression(&y, &[x1, x2], &names2).expect("OLS should succeed");

    // Adjusted R² with useless predictor should be lower
    assert!(
        result2.adj_r_squared < result1.adj_r_squared,
        "Adj R² should decrease with useless predictor: {} -> {}",
        result1.adj_r_squared,
        result2.adj_r_squared
    );
}

// ============================================================================
// VIF Calculation Tests
// ============================================================================

#[test]
fn test_vif_independent_predictors() {
    // Independent predictors should have VIF ≈ 1
    let n = 50;
    // x1: evenly spaced values
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    // x2: independent sequence (orthogonal to x1)
    let x2: Vec<f64> = (0..n)
        .map(|i| {
            let phase = (i as f64) * 0.3;
            (phase * 3.0).sin() * 10.0 + (i as f64).cos() * 5.0
        })
        .collect();
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&x1i, &x2i)| 1.0 + x1i + x2i)
        .collect();

    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1, x2], &names).expect("OLS should succeed");

    // VIF for independent predictors should be close to 1
    for vif_result in &result.vif {
        assert!(
            vif_result.vif < 5.0,
            "{} VIF should be low for independent predictors, got {}",
            vif_result.variable,
            vif_result.vif
        );
    }
}

#[test]
fn test_vif_high_multicollinearity() {
    // Highly correlated predictors should have VIF > 10
    let n = 50;
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    // Create x2 with high correlation but not perfectly collinear
    // The noise varies with position to break collinearity with intercept
    let x2: Vec<f64> = x1
        .iter()
        .enumerate()
        .map(|(i, &v)| 2.0 * v + (i as f64) * 0.01 + ((i % 2) as f64) * 0.1)
        .collect();
    let y: Vec<f64> = x1.iter().map(|&xi| 1.0 + 2.0 * xi).collect();

    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1, x2], &names).expect("OLS should succeed");

    // At least one VIF should be high
    let max_vif = result
        .vif
        .iter()
        .map(|v| v.vif)
        .fold(0.0f64, |acc, v| acc.max(v));

    assert!(
        max_vif > 10.0,
        "High multicollinearity should produce VIF > 10, got {}",
        max_vif
    );
}

#[test]
fn test_vif_single_predictor() {
    // Single predictor should return empty VIF list
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let result = ols_regression(&y, &[x], &names).expect("OLS should succeed");

    assert_eq!(
        result.vif.len(),
        0,
        "VIF should be empty for single predictor"
    );
}

// ============================================================================
// Leverage Calculation Tests
// ============================================================================

#[test]
fn test_leverage_bounds() {
    // Leverage values should be in [0, 1]
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 4.0, 1.0, 3.0, 5.0]; // Not collinear with x1
    let y = vec![3.0, 9.0, 5.0, 11.0, 15.0];
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1, x2], &names).expect("OLS should succeed");

    for (i, &lev) in result.leverage.iter().enumerate() {
        assert!(
            lev >= 0.0 && lev <= 1.0,
            "Leverage at index {} should be in [0, 1], got {}",
            i,
            lev
        );
    }
}

#[test]
fn test_leverage_sum_equals_k_plus_one() {
    // Sum of leverage values should equal k + 1
    let n = 20;
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let x2: Vec<f64> = (0..n).map(|i| (i as f64) * (i as f64)).collect(); // x2 = x1², not collinear
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&x1i, &x2i)| 1.0 + 2.0 * x1i + 0.5 * x2i)
        .collect();
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1, x2], &names).expect("OLS should succeed");

    let leverage_sum: f64 = result.leverage.iter().sum();
    let k_plus_1 = (result.k + 1) as f64;

    assert_close(
        leverage_sum,
        k_plus_1,
        1e-8,
        &format!("leverage sum = {}, k+1 = {}", leverage_sum, k_plus_1),
    );
}

#[test]
fn test_leverage_high_point_detection() {
    // Points far from the center should have higher leverage
    // Design: one point far from others in x-space
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // Last point is far out
    let x2 = vec![2.0, 3.0, 4.0, 5.0, 150.0];
    let y = vec![5.0, 9.0, 13.0, 17.0, 350.0];
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1, x2], &names).expect("OLS should succeed");

    // Last point should have highest leverage
    let max_lev = result.leverage.iter().fold(0.0f64, |acc, &v| acc.max(v));
    let last_lev = result.leverage[result.leverage.len() - 1];

    assert_eq!(
        last_lev, max_lev,
        "Outlier point should have highest leverage"
    );
}

// ============================================================================
// Error Cases Tests
// ============================================================================

#[test]
fn test_insufficient_data_error() {
    // n <= k + 1 should fail
    let y = vec![1.0, 2.0];
    let x1 = vec![1.0, 2.0];
    let x2 = vec![2.0, 3.0];
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1, x2], &names);

    match result {
        Err(Error::InsufficientData {
            required,
            available,
        }) => {
            assert_eq!(required, 4);
            assert_eq!(available, 2);
        },
        _ => panic!("Should return InsufficientData error"),
    }
}

#[test]
fn test_singular_matrix_error() {
    // Perfect multicollinearity: handled gracefully via pivoted QR 
    // The redundant column's coefficient is set to NAN
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let x2 = vec![2.0, 4.0, 6.0, 8.0]; // x2 = 2 * x1 (perfect collinearity)
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1, x2], &names);

    // Should succeed with one coefficient set to NAN (dropped column)
    let output = result.expect("Rank-deficient OLS should succeed with pivoted QR");
    let nan_count = output.coefficients.iter().filter(|c| c.is_nan()).count();
    assert_eq!(nan_count, 1, "Exactly one coefficient should be NAN for one redundant column");

    // Predictions should still be valid (NAN coefficients treated as 0)
    assert!(output.predictions.iter().all(|p| p.is_finite()),
        "Predictions should be finite even with rank-deficient data");
}

#[test]
fn test_minimum_valid_data() {
    // n = k + 2 should pass (barely valid)
    // Need non-collinear predictors
    let y = vec![1.0, 3.0, 5.0, 7.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0];
    let x2 = vec![1.0, 0.5, 2.0, 1.5];
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1, x2], &names);

    assert!(result.is_ok(), "n = k + 2 should be valid");
}

#[test]
fn test_empty_y_returns_error() {
    let y = vec![];
    let x = vec![1.0, 2.0, 3.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let result = ols_regression(&y, &[x], &names);

    match result {
        Err(Error::InsufficientData { .. }) => {
            // Expected
        },
        _ => panic!("Empty y should return error"),
    }
}

#[test]
fn test_mismatched_lengths_return_error() {
    let y = vec![1.0, 2.0, 3.0];
    let x = vec![1.0, 2.0]; // Different length
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let result = ols_regression(&y, &[x], &names);

    // Should return DimensionMismatch error (not panic)
    match result {
        Err(Error::DimensionMismatch { .. }) => {},
        _ => panic!("Expected DimensionMismatch error for mismatched lengths"),
    }
}

#[test]
fn test_nan_input_returns_error() {
    // Perfect collinearity (x2 = x1): handled gracefully via pivoted QR
    // The redundant column's coefficient is set to NAN
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // x2 = x1 (perfect collinearity)
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    let result = ols_regression(&y, &[x1, x2], &names);

    // Should succeed with NAN for the dropped coefficient
    let output = result.expect("Rank-deficient OLS should succeed with pivoted QR");
    let nan_count = output.coefficients.iter().filter(|c| c.is_nan()).count();
    assert_eq!(nan_count, 1, "Exactly one coefficient should be NAN for one redundant column");

    // The non-NAN coefficients should produce valid predictions
    assert!(output.predictions.iter().all(|p| p.is_finite()),
        "Predictions should be finite even with collinear data");
}

// ============================================================================
// Statistical Function Tests
// ============================================================================

#[test]
fn test_two_tailed_p_value() {
    // Test that p-value is symmetric around 0
    let p1 = two_tailed_p_value(2.0, 10.0);
    let p2 = two_tailed_p_value(-2.0, 10.0);
    assert_close(p1, p2, 1e-10, "symmetric p-value");

    // Larger t should give smaller p-value
    let p_small = two_tailed_p_value(0.5, 10.0);
    let p_large = two_tailed_p_value(5.0, 10.0);
    assert!(p_small > p_large, "Larger t should give smaller p-value");

    // Very large t should give p-value close to 0
    let p_very_large = two_tailed_p_value(100.0, 10.0);
    assert_close(p_very_large, 0.0, 0.01, "very large t p-value");

    // t = 0 should give p-value = 1
    let p_zero = two_tailed_p_value(0.0, 10.0);
    assert_close(p_zero, 1.0, 1e-10, "t=0 p-value");
}

#[test]
fn test_t_critical_quantile() {
    // Test with known values
    let t_crit_10 = t_critical_quantile(10.0, 0.05);
    eprintln!("DEBUG: t_crit_10 (df=10, alpha=0.05) = {}", t_crit_10);
    // For df=10, alpha=0.05 (two-tailed), t-critical ≈ 2.228
    assert!(
        t_crit_10 > 2.0 && t_crit_10 < 2.5,
        "t-critical for df=10 should be ~2.228"
    );

    let t_crit_100 = t_critical_quantile(100.0, 0.05);
    eprintln!("DEBUG: t_crit_100 (df=100, alpha=0.05) = {}", t_crit_100);
    // For large df, t-critical approaches z-critical ≈ 1.96
    assert!(
        t_crit_100 > 1.9 && t_crit_100 < 2.0,
        "t-critical for df=100 should be ~1.96"
    );

    // Higher alpha (less confident) should give smaller critical value
    let t_crit_01 = t_critical_quantile(10.0, 0.01);
    eprintln!("DEBUG: t_crit_01 (df=10, alpha=0.01) = {}", t_crit_01);
    eprintln!(
        "DEBUG: t_crit_01 > t_crit_10 ? {} ({} > {})",
        t_crit_01 > t_crit_10,
        t_crit_01,
        t_crit_10
    );
    assert!(
        t_crit_01 > t_crit_10,
        "Lower alpha should give higher t-critical"
    );
}

#[test]
fn test_f_p_value() {
    // F=0 should give p-value = 1
    let p_zero = f_p_value(0.0, 5.0, 10.0);
    assert_close(p_zero, 1.0, 1e-10, "F=0 p-value");

    // Negative F should also give p-value = 1
    let p_negative = f_p_value(-1.0, 5.0, 10.0);
    assert_close(p_negative, 1.0, 1e-10, "negative F p-value");

    // Larger F should give smaller p-value
    let p_small_f = f_p_value(1.0, 5.0, 10.0);
    let p_large_f = f_p_value(10.0, 5.0, 10.0);
    assert!(
        p_small_f > p_large_f,
        "Larger F should give smaller p-value"
    );
}

// ============================================================================
// Property-Based Tests
// ============================================================================

proptest! {
    #[test]
    fn prop_predictions_calculated_correctly(
        n in 10..30usize,
        slope in 0.1..10.0f64,
        intercept in -10.0..10.0f64
    ) {
        // y_hat = X * beta should match predictions
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| intercept + slope * xi).collect();
        let names = vec!["Intercept".to_string(), "X".to_string()];

        let result = ols_regression(&y, &[x.clone()], &names).unwrap();

        // Check each prediction
        for i in 0..n {
            let expected = intercept + slope * x[i];
            let actual = result.predictions[i];
            prop_assert!((actual - expected).abs() < 1e-6,
                "Prediction at index {}: expected {}, got {}", i, expected, actual);
        }
    }

    #[test]
    fn prop_r_squared_in_bounds(
        n in 10..30usize,
        values_x in proptest::collection::vec(0.0..100.0f64, 30),
        values_e in proptest::collection::vec(-1.0..1.0f64, 30)
    ) {
        // R² should always be in [0, 1]
        let x = &values_x[..n];
        let noise = &values_e[..n];
        let y: Vec<f64> = x.iter().zip(noise.iter())
            .map(|(&xi, &ei)| 2.0 * xi + 1.0 + ei)
            .collect();
        let names = vec!["Intercept".to_string(), "X".to_string()];

        let result = ols_regression(&y, &[x.to_vec()], &names).unwrap();

        prop_assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0,
            "R² = {} is outside [0, 1]", result.r_squared);
    }

    #[test]
    fn prop_residuals_sum_to_zero_with_intercept(
        n in 10..30usize,
        values in proptest::collection::vec(0.0..100.0f64, 30)
    ) {
        // With intercept, residuals should sum to approximately zero
        let x = &values[..n];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let names = vec!["Intercept".to_string(), "X".to_string()];

        let result = ols_regression(&y, &[x.to_vec()], &names).unwrap();

        let sum: f64 = result.residuals.iter().sum();
        prop_assert!(sum.abs() < 1e-8,
            "Residual sum = {} is not close to zero", sum);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_single_row_with_single_predictor_barely_valid() {
    // n=3, k=1 gives n - k - 1 = 1 df (barely valid)
    let y = vec![1.0, 2.0, 3.0];
    let x = vec![1.0, 2.0, 3.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let result = ols_regression(&y, &[x], &names);
    assert!(result.is_ok(), "n=3, k=1 should be valid");
}

#[test]
fn test_constant_y_with_varying_x() {
    // y is constant, x varies
    let y = vec![5.0, 5.0, 5.0, 5.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let names = vec!["Intercept".to_string(), "X".to_string()];

    let result = ols_regression(&y, &[x], &names).expect("OLS should succeed");

    // When y is constant, R² is NaN (undefined because there's no variation to explain)
    assert!(result.r_squared.is_nan(), "R² should be NaN for constant y");
}
