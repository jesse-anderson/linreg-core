// ============================================================================
// Ridge Regression Verification Tests
// ============================================================================
//
// These tests verify the ridge regression implementation by comparing
// against known R glmnet values. 

use linreg_core::regularized::{ridge_fit, RidgeFitOptions};
use linreg_core::linalg::Matrix;

// ============================================================================
// Known Values from R glmnet
// ============================================================================

/// Test against known R glmnet values for simple dataset
#[test]
fn test_ridge_standardization_verification() {
    // Data: y = [1,2,3,4,5], x1 = [1,2,3,4,5], x2 = [1,-1,1,-1,1]
    // R glmnet with lambda=1.0 gives: intercept≈1.242641, beta[0]≈0.585786
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1_data = y.clone();
    let x2_data = vec![1.0, -1.0, 1.0, -1.0, 1.0];

    let n = 5;
    let p = 2;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1_data[i];
        x_data[i * (p + 1) + 2] = x2_data[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let lambda = 1.0;
    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda,
        standardize: true,
        intercept: true,
        weights: None,
    };

    let result = ridge_fit(&x, &y, &options).unwrap();

    // Known values from R glmnet (lambda=1.0)
    let expected_intercept = 1.242641;
    let expected_beta0 = 0.5857864;
    let expected_beta1 = 0.0;  // x2 is uncorrelated

    println!("Standardization verification (vs R glmnet):");
    println!("  R intercept: {}, Rust intercept: {}", expected_intercept, result.intercept);
    println!("  R beta[0]: {}, Rust beta[0]: {}", expected_beta0, result.coefficients[0]);
    println!("  R beta[1]: {}, Rust beta[1]: {}", expected_beta1, result.coefficients[1]);

    // Verify results match R glmnet (within numerical tolerance)
    assert!((result.intercept - expected_intercept).abs() < 1e-5, "Intercept mismatch");
    assert!((result.coefficients[0] - expected_beta0).abs() < 1e-5, "Beta[0] mismatch");
    assert!((result.coefficients[1] - expected_beta1).abs() < 1e-5, "Beta[1] mismatch");
}

// ============================================================================
// Augmented System Verification
// ============================================================================

/// Test to verify the augmented system using known R values
#[test]
fn test_ridge_verify_augmented_system() {
    // Same test data as above
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1_data = y.clone();
    let x2_data = vec![1.0, -1.0, 1.0, -1.0, 1.0];
    let n = 5;

    let p = 2;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1_data[i];
        x_data[i * (p + 1) + 2] = x2_data[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let lambda = 1.0;
    let options = RidgeFitOptions {
        lambda,
        standardize: true,
        intercept: true,
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        weights: None,
    };

    let result = ridge_fit(&x, &y, &options).unwrap();

    // Known values from R glmnet
    let expected_intercept = 1.242641;
    let expected_beta0 = 0.5857864;
    let expected_beta1 = 0.0;

    println!("Augmented system verification (vs R glmnet):");
    println!("  Expected intercept: {}, Rust: {}", expected_intercept, result.intercept);
    println!("  Expected beta[0]: {}, Rust: {}", expected_beta0, result.coefficients[0]);
    println!("  Expected beta[1]: {}, Rust: {}", expected_beta1, result.coefficients[1]);

    assert!((result.intercept - expected_intercept).abs() < 1e-5, "Intercept mismatch");
    assert!((result.coefficients[0] - expected_beta0).abs() < 1e-5, "Beta[0] mismatch");
    assert!((result.coefficients[1] - expected_beta1).abs() < 1e-5, "Beta[1] mismatch");
}

// ============================================================================
// Direct Calculation Verification
// ============================================================================

/// Verify ridge solution matches direct calculation for non-collinear data
#[test]
fn test_ridge_direct_calculation_verification() {
    // Non-collinear data
    let y = vec![1.0, 2.0, 4.0, 3.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 1.0, 4.0, 3.0, 5.0];

    let n = 5;
    let p = 2;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
        x_data[i * (p + 1) + 2] = x2[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let lambda = 1.0;
    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda,
        standardize: true,
        intercept: true,
        weights: None,
    };

    let result = ridge_fit(&x, &y, &options).unwrap();

    // Compute expected using glmnet
    let v = 1.0 / (n as f64).sqrt();
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let x1_mean: f64 = x1.iter().sum::<f64>() / n as f64;
    let x2_mean: f64 = x2.iter().sum::<f64>() / n as f64;

    // Center and v-transform
    let y_c: Vec<f64> = y.iter().map(|yi| yi - y_mean).collect();
    let x1_c: Vec<f64> = x1.iter().map(|xi| xi - x1_mean).collect();
    let x2_c: Vec<f64> = x2.iter().map(|xi| xi - x2_mean).collect();

    let y_v: Vec<f64> = y_c.iter().map(|yi| v * yi).collect();
    let x1_v: Vec<f64> = x1_c.iter().map(|xi| v * xi).collect();
    let x2_v: Vec<f64> = x2_c.iter().map(|xi| v * xi).collect();

    let y_scale = y_v.iter().map(|yi| yi * yi).sum::<f64>().sqrt();
    let x1_scale = x1_v.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
    let x2_scale = x2_v.iter().map(|xi| xi * xi).sum::<f64>().sqrt();

    // Standardize to unit norm
    let y_std: Vec<f64> = y_v.iter().map(|yi| yi / y_scale).collect();
    let x1_std: Vec<f64> = x1_v.iter().map(|xi| xi / x1_scale).collect();
    let x2_std: Vec<f64> = x2_v.iter().map(|xi| xi / x2_scale).collect();

    // Compute X'X and X'y
    let x1x1: f64 = x1_std.iter().map(|xi| xi * xi).sum();
    let x1x2: f64 = x1_std.iter().zip(x2_std.iter()).map(|(xi, xj)| xi * xj).sum();
    let x2x2: f64 = x2_std.iter().map(|xi| xi * xi).sum();
    let x1y: f64 = x1_std.iter().zip(y_std.iter()).map(|(xi, yi)| xi * yi).sum();
    let x2y: f64 = x2_std.iter().zip(y_std.iter()).map(|(xi, yi)| xi * yi).sum();

    // Ridge normal equations: (X'X + lambda_eff*I) * beta_std = X'y
    // where lambda_eff = user_lambda / y_scale
    let lambda_eff = lambda / y_scale;
    let a00 = x1x1 + lambda_eff;
    let a01 = x1x2;
    let a10 = x1x2;
    let a11 = x2x2 + lambda_eff;

    let det = a00 * a11 - a01 * a10;
    let beta1_std = (a11 * x1y - a01 * x2y) / det;
    let beta2_std = (a00 * x2y - a10 * x1y) / det;

    // Unstandardize
    let beta1_orig = (y_scale / x1_scale) * beta1_std;
    let beta2_orig = (y_scale / x2_scale) * beta2_std;
    let intercept_expected = y_mean - (x1_mean * beta1_orig + x2_mean * beta2_orig);

    println!("Direct calculation verification (glmnet formula):");
    println!("  Rust intercept: {}", result.intercept);
    println!("  Direct intercept: {} (diff: {:.2e})", intercept_expected, result.intercept - intercept_expected);
    println!("  Rust beta[0]: {}", result.coefficients[0]);
    println!("  Direct beta[0]: {} (diff: {:.2e})", beta1_orig, result.coefficients[0] - beta1_orig);
    println!("  Rust beta[1]: {}", result.coefficients[1]);
    println!("  Direct beta[1]: {} (diff: {:.2e})", beta2_orig, result.coefficients[1] - beta2_orig);

    // Verify results match (within numerical tolerance)
    assert!((result.intercept - intercept_expected).abs() < 1e-5, "Intercept mismatch");
    assert!((result.coefficients[0] - beta1_orig).abs() < 1e-5, "Beta[0] mismatch");
    assert!((result.coefficients[1] - beta2_orig).abs() < 1e-5, "Beta[1] mismatch");
}
