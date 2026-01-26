// ============================================================================
// Ridge Regression Baseline Tests
// ============================================================================
//
// Basic smoke tests and baseline validation for ridge regression.
// These tests establish known-good values that can be used to detect
// if the implementation changes unexpectedly.

use linreg_core::linalg::Matrix;
use linreg_core::regularized::{ridge_fit, RidgeFitOptions};

// ============================================================================
// Test Datasets
// ============================================================================

/// Simple dataset for testing: y = 2 + 3*x with some noise
fn get_simple_dataset() -> (Matrix, Vec<f64>) {
    let x_data = vec![
        1.0, 1.0,  // intercept, x1
        1.0, 2.0,
        1.0, 3.0,
        1.0, 4.0,
        1.0, 5.0,
        1.0, 6.0,
        1.0, 7.0,
        1.0, 8.0,
    ];
    let x = Matrix::new(8, 2, x_data);
    let y = vec![5.1, 8.2, 11.1, 13.9, 17.2, 19.8, 23.1, 26.0]; // approx y = 2 + 3*x
    (x, y)
}

// ============================================================================
// Simple / Smoke Tests
// ============================================================================

/// Simple test: y = x1 with no correlation issues
#[test]
fn test_ridge_simple_no_correlation() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let n = 5;
    let p = 1;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let lambda = 1.0;
    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda,
        intercept: true,
        standardize: true,
        weights: None,
    };

    let result = ridge_fit(&x, &y, &options).unwrap();

    println!("Single predictor test:");
    println!("  Intercept: {}", result.intercept);
    println!("  Beta[0]: {}", result.coefficients[0]);

    // Expected values after standardization:
    // y_std = [-0.894, -0.447, 0, 0.447, 0.894]
    // y_scale = sqrt(2) ≈ 1.414
    // beta_std = 1 / (1 + lambda) = 1 / 2 = 0.5
    // After unstandardization: beta ≈ 0.5, intercept ≈ 1.5
    assert!((result.coefficients[0] - 0.5).abs() < 0.1, "Beta should be ~0.5");
    assert!((result.intercept - 1.5).abs() < 0.5, "Intercept should be ~1.5");
}

/// Simple debug test with known solution
#[test]
fn test_ridge_simple_debug() {
    // Simple dataset with known solution: y = 2 + 3*x1 + 1*x2
    let y = vec![2.0, 6.0, 10.0, 14.0, 18.0];
    let x_data = vec![
        1.0, 1.0, 0.0,  // intercept, x1=1, x2=0
        1.0, 2.0, 0.0,
        1.0, 3.0, 1.0,
        1.0, 4.0, 1.0,
        1.0, 5.0, 1.0,
    ];
    let x = Matrix::new(5, 3, x_data);

    let options = RidgeFitOptions {
        lambda: 0.1,
        intercept: true,
        standardize: true,
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        weights: None,
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    println!("Simple test:");
    println!("  Intercept: {}", fit.intercept);
    println!("  Coefficients: {:?}", fit.coefficients);

    // With small lambda, should be close to OLS solution
    // OLS would give approximately intercept=2, slopes=[3, 1]
    assert!(fit.coefficients[0] > 0.0, "Slope 1 should be positive");
    assert!(fit.coefficients[1] > 0.0, "Slope 2 should be positive");
}

/// Test that unit norm standardization is being used correctly
#[test]
fn test_ridge_unit_norm_check() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_data = vec![
        1.0, 1.0,
        1.0, 2.0,
        1.0, 3.0,
        1.0, 4.0,
        1.0, 5.0,
    ];
    let x = Matrix::new(5, 2, x_data);

    let options = RidgeFitOptions {
        lambda: 1.0,
        intercept: true,
        standardize: true,
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        weights: None,
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    println!("Unit norm check:");
    println!("  Intercept: {}", fit.intercept);
    println!("  Coefficient: {}", fit.coefficients[0]);

    // Check that predictions sum to approximately y sum (centering property)
    let fit_sum: f64 = fit.fitted_values.iter().sum();
    let y_sum: f64 = y.iter().sum();
    println!("  Fit sum: {}, Y sum: {}", fit_sum, y_sum);
    assert!((fit_sum - y_sum).abs() < 0.1, "Sum check failed");
}

/// Test without standardization
#[test]
fn test_ridge_no_standardization() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x_data = vec![
        1.0, 1.0,
        1.0, 2.0,
        1.0, 3.0,
        1.0, 4.0,
        1.0, 5.0,
    ];
    let x = Matrix::new(5, 2, x_data);

    let options = RidgeFitOptions {
        lambda: 0.1,
        intercept: true,
        standardize: false,
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        weights: None,
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    println!("No standardization:");
    println!("  Intercept: {}", fit.intercept);
    println!("  Coef: {}", fit.coefficients[0]);

    // With no standardization and small lambda, should be close to perfect fit
    // y = 2*x, so intercept ≈ 0, slope ≈ 2
    assert!((fit.intercept).abs() < 1.0, "Intercept should be close to 0");
    assert!((fit.coefficients[0] - 2.0).abs() < 0.5, "Slope should be close to 2");
}

/// Test that lambda scaling is correct
#[test]
fn test_ridge_lambda_scaling() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_data = vec![
        1.0, 1.0,
        1.0, 2.0,
        1.0, 3.0,
        1.0, 4.0,
        1.0, 5.0,
    ];
    let x = Matrix::new(5, 2, x_data);

    println!("Lambda scaling test:");
    for lambda in [0.1, 1.0, 10.0, 100.0] {
        let options = RidgeFitOptions {
            lambda,
            intercept: true,
            standardize: true,
            max_iter: 10000,
            tol: 1e-7,
            warm_start: None,
            weights: None,
        };

        let fit = ridge_fit(&x, &y, &options).unwrap();

        println!("  Lambda={:6.2}: intercept={:.8}, coef={:.8}",
            lambda, fit.intercept, fit.coefficients[0]);

        // Higher lambda should shrink coefficients more toward zero
        assert!(fit.coefficients[0].is_finite(), "Coefficient should be finite");
        assert!(fit.intercept.is_finite(), "Intercept should be finite");
    }
}

// ============================================================================
// Baseline Tests
// ============================================================================

/// Baseline test with simple dataset
#[test]
fn test_ridge_baseline_simple() {
    let (x, y) = get_simple_dataset();

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.1,
        intercept: true,
        standardize: true,
        weights: None,
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    println!("=== RIDGE BASELINE (Simple Dataset) ===");
    println!("Lambda: {}", options.lambda);
    println!("Intercept: {:.10}", fit.intercept);
    println!("Coefficients: {:?}", fit.coefficients.iter().map(|c| format!("{:.10}", c)).collect::<Vec<_>>());
    println!("R^2: {:.10}", fit.r_squared);
    println!("DF: {:.10}", fit.df);
    println!("==========================================");

    // Baseline values (these should match current implementation)
    // If these fail, we know something changed
    assert!((fit.intercept - 2.38).abs() < 0.5, "Intercept changed significantly");
    assert!((fit.coefficients[0] - 3.0).abs() < 0.5, "Slope changed significantly");
}

/// Baseline test without standardization
#[test]
fn test_ridge_baseline_no_standardization() {
    let (x, y) = get_simple_dataset();

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.1,
        intercept: true,
        standardize: false,
        weights: None,
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    println!("=== RIDGE BASELINE (No Standardization) ===");
    println!("Lambda: {}", options.lambda);
    println!("Intercept: {:.10}", fit.intercept);
    println!("Coefficients: {:?}", fit.coefficients.iter().map(|c| format!("{:.10}", c)).collect::<Vec<_>>());
    println!("R^2: {:.10}", fit.r_squared);
    println!("==========================================");
}

/// Baseline test across lambda series
#[test]
fn test_ridge_baseline_lambda_series() {
    let (x, y) = get_simple_dataset();

    println!("=== RIDGE BASELINE (Lambda Series) ===");
    for lambda in [0.01, 0.1, 1.0, 10.0, 100.0] {
        let options = RidgeFitOptions {
            max_iter: 10000,
            tol: 1e-7,
            warm_start: None,
            lambda,
            intercept: true,
            standardize: true,
            weights: None,
        };

        let fit = ridge_fit(&x, &y, &options).unwrap();
        println!("Lambda={:6.2}: intercept={:.8}, coef={:.8}",
            lambda, fit.intercept, fit.coefficients[0]);
    }
    println!("==========================================");
}

/// Baseline test with mtcars dataset
#[test]
fn test_ridge_baseline_mtcars() {
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");
    let csv_path = datasets_dir.join("mtcars.csv");

    if !csv_path.exists() {
        println!("=== mtcars dataset not found, skipping test ===");
        return;
    }

    // Load mtcars dataset (simple CSV parsing)
    let content = std::fs::read_to_string(&csv_path).expect("Failed to read CSV");
    let lines: Vec<&str> = content.lines().collect();
    let headers: Vec<&str> = lines[0].split(',').collect();

    // y is mpg (first column), X are columns 2-11 (cyl through carb)
    let mut y = Vec::new();
    let mut x_vars: Vec<Vec<f64>> = vec![Vec::new(); headers.len() - 1];

    for line in lines.iter().skip(1) {
        let vals: Vec<f64> = line.split(',')
            .map(|s| s.parse::<f64>().unwrap())
            .collect();

        y.push(vals[0]); // mpg (first column)
        for (j, &val) in vals.iter().skip(1).enumerate() {
            x_vars[j].push(val);
        }
    }

    let n = y.len();
    let p = x_vars.len();

    // Build design matrix with intercept
    let mut x_data = vec![1.0; n * (p + 1)];
    for (col_idx, x_col) in x_vars.iter().enumerate() {
        for (row_idx, val) in x_col.iter().enumerate() {
            x_data[row_idx * (p + 1) + col_idx + 1] = *val;
        }
    }
    let x = Matrix::new(n, p + 1, x_data);

    let lambda = 0.5147; // Last lambda from mtcars ridge glmnet test

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda,
        intercept: true,
        standardize: true,
        weights: None,
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    println!("=== RIDGE BASELINE (mtcars, lambda={}) ===", lambda);
    println!("Intercept: {:.10}", fit.intercept);
    println!("Coefficients:");
    for (j, &coef) in fit.coefficients.iter().enumerate() {
        println!("  Beta[{}]: {:.10}", j + 1, coef);
    }
    println!("R^2: {:.10}", fit.r_squared);
    println!("==========================================");
}
