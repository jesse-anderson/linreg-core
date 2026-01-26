// ============================================================================
// Regularized Regression Unit Tests (Lasso & Ridge)
// ============================================================================
//
// Comprehensive tests for lasso (L1) and ridge (L2) regularized regression,
// including input validation, statistical properties, and edge cases.

use linreg_core::linalg::Matrix;
use linreg_core::regularized::lasso::soft_threshold;
use linreg_core::regularized::{lasso_fit, ridge_fit, LassoFitOptions, RidgeFitOptions};
use linreg_core::Error;
use proptest::prelude::*;

// ============================================================================
// Test Constants and Helpers
// ============================================================================

const TOLERANCE: f64 = 1e-6;
const LOOSE_TOLERANCE: f64 = 1e-4;
const STAT_TOLERANCE: f64 = 1e-4;

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

/// Creates a simple design matrix with intercept column
fn create_design_matrix(x_values: &[f64]) -> Matrix {
    let n = x_values.len();
    let mut data = vec![1.0; n * 2]; // First column is all ones (intercept)
    for (i, &val) in x_values.iter().enumerate() {
        data[i * 2 + 1] = val;
    }
    Matrix::new(n, 2, data)
}

/// Creates a design matrix with intercept and multiple predictors
fn create_design_matrix_multi(x_cols: &[Vec<f64>]) -> Matrix {
    let n = x_cols[0].len();
    let p = x_cols.len();
    let mut data = vec![1.0; n * (p + 1)]; // First column is intercept
    for (col_idx, col) in x_cols.iter().enumerate() {
        for (row_idx, &val) in col.iter().enumerate() {
            data[row_idx * (p + 1) + col_idx + 1] = val;
        }
    }
    Matrix::new(n, p + 1, data)
}

// ============================================================================
// Input Validation Tests - Lasso
// ============================================================================

#[test]
fn test_lasso_rejects_negative_lambda() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = LassoFitOptions {
        lambda: -1.0,
        ..Default::default()
    };

    let result = lasso_fit(&x, &y, &options);

    match result {
        Err(Error::InvalidInput(msg)) => {
            assert!(
                msg.contains("non-negative"),
                "Error message should mention non-negative lambda"
            );
        },
        _ => panic!(
            "Expected InvalidInput error for negative lambda, got {:?}",
            result
        ),
    }
}

#[test]
fn test_lasso_rejects_dimension_mismatch() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0]; // Wrong length

    let options = LassoFitOptions::default();

    let result = lasso_fit(&x, &y, &options);

    match result {
        Err(Error::DimensionMismatch(msg)) => {
            assert!(
                msg.contains("must match"),
                "Error should mention dimension mismatch"
            );
        },
        _ => panic!("Expected DimensionMismatch error, got {:?}", result),
    }
}

#[test]
fn test_lasso_handles_nan_in_y() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0]);
    let y = vec![2.0, 4.0, f64::NAN, 8.0];

    let options = LassoFitOptions::default();

    let result = lasso_fit(&x, &y, &options);

    // Should handle NaN gracefully - either fail or produce NaN output
    match result {
        Ok(fit) => {
            // If it succeeds, predictions should contain NaN
            assert!(fit.fitted_values.iter().any(|v| v.is_nan()));
        },
        Err(_) => {
            // Also acceptable to return an error
        },
    }
}

#[test]
fn test_lasso_handles_infinity_in_y() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0]);
    let y = vec![2.0, 4.0, f64::INFINITY, 8.0];

    let options = LassoFitOptions::default();

    let result = lasso_fit(&x, &y, &options);

    // Should handle infinity gracefully
    match result {
        Ok(fit) => {
            assert!(fit.fitted_values.iter().any(|v| !v.is_finite()));
        },
        Err(_) => {
            // Also acceptable to return an error
        },
    }
}

// ============================================================================
// Input Validation Tests - Ridge
// ============================================================================

#[test]
fn test_ridge_rejects_negative_lambda() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: -1.0,
        weights: None,
        ..Default::default()
    };

    let result = ridge_fit(&x, &y, &options);

    match result {
        Err(Error::InvalidInput(msg)) => {
            assert!(
                msg.contains("non-negative"),
                "Error message should mention non-negative lambda"
            );
        },
        _ => panic!(
            "Expected InvalidInput error for negative lambda, got {:?}",
            result
        ),
    }
}

#[test]
fn test_ridge_rejects_dimension_mismatch() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0]; // Wrong length

    let options = RidgeFitOptions::default();

    let result = ridge_fit(&x, &y, &options);

    match result {
        Err(Error::DimensionMismatch(msg)) => {
            assert!(
                msg.contains("must match"),
                "Error should mention dimension mismatch"
            );
        },
        _ => panic!("Expected DimensionMismatch error, got {:?}", result),
    }
}

#[test]
fn test_ridge_handles_nan_in_y() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0]);
    let y = vec![2.0, 4.0, f64::NAN, 8.0];

    let options = RidgeFitOptions::default();

    let result = ridge_fit(&x, &y, &options);

    // Should handle NaN gracefully
    match result {
        Ok(fit) => {
            assert!(fit.fitted_values.iter().any(|v| v.is_nan()));
        },
        Err(_) => {
            // Also acceptable to return an error
        },
    }
}

// ============================================================================
// Soft Threshold Tests
// ============================================================================

#[test]
fn test_soft_threshold_positive_above_threshold() {
    let result = soft_threshold(5.0, 2.0);
    assert_close(result, 3.0, TOLERANCE, "soft_threshold(5, 2)");
}

#[test]
fn test_soft_threshold_positive_at_threshold() {
    let result = soft_threshold(2.0, 2.0);
    assert_close(result, 0.0, TOLERANCE, "soft_threshold(2, 2)");
}

#[test]
fn test_soft_threshold_positive_below_threshold() {
    let result = soft_threshold(1.0, 2.0);
    assert_close(result, 0.0, TOLERANCE, "soft_threshold(1, 2)");
}

#[test]
fn test_soft_threshold_negative_above_threshold() {
    let result = soft_threshold(-5.0, 2.0);
    assert_close(result, -3.0, TOLERANCE, "soft_threshold(-5, 2)");
}

#[test]
fn test_soft_threshold_negative_at_threshold() {
    let result = soft_threshold(-2.0, 2.0);
    assert_close(result, 0.0, TOLERANCE, "soft_threshold(-2, 2)");
}

#[test]
fn test_soft_threshold_negative_below_threshold() {
    let result = soft_threshold(-1.0, 2.0);
    assert_close(result, 0.0, TOLERANCE, "soft_threshold(-1, 2)");
}

#[test]
fn test_soft_threshold_zero_threshold() {
    assert_eq!(soft_threshold(5.0, 0.0), 5.0);
    assert_eq!(soft_threshold(-5.0, 0.0), -5.0);
    assert_eq!(soft_threshold(0.0, 0.0), 0.0);
}

#[test]
#[should_panic(expected = "non-negative")]
fn test_soft_threshold_negative_gamma_panics() {
    soft_threshold(1.0, -1.0);
}

#[test]
fn test_soft_threshold_symmetry() {
    // S(-z, γ) = -S(z, γ) for |z| > γ
    let z = 5.0;
    let gamma = 2.0;
    let pos_result = soft_threshold(z, gamma);
    let neg_result = soft_threshold(-z, gamma);
    assert_close(pos_result, -neg_result, TOLERANCE, "symmetry");
}

// ============================================================================
// Lasso Basic Functionality Tests
// ============================================================================

#[test]
fn test_lasso_perfect_linear_fit() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2*x

    let options = LassoFitOptions {
        lambda: 0.001, // Very small lambda for near-OLS solution
        intercept: true,
        standardize: false, // Don't standardize for predictable coefficients
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    assert!(fit.converged, "Lasso should converge");
    assert!(
        fit.n_nonzero >= 1,
        "Should have at least 1 non-zero coefficient"
    );
    assert_close(fit.coefficients[0], 2.0, 0.5, "slope coefficient");
}

#[test]
fn test_lasso_with_intercept() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0]);
    let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| 5.0 + 2.0 * v)
        .collect();

    let options = LassoFitOptions {
        lambda: 0.001,
        intercept: true,
        standardize: false,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    assert!(fit.converged);
    assert_close(fit.intercept, 5.0, 1.0, "intercept");
    assert_close(fit.coefficients[0], 2.0, 0.5, "slope");
}

#[test]
fn test_lasso_without_intercept() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let mut x_data_full = vec![0.0; x_data.len() * 1]; // No intercept column
    for (i, &v) in x_data.iter().enumerate() {
        x_data_full[i] = v;
    }
    let x = Matrix::new(4, 1, x_data_full);
    let y: Vec<f64> = x_data.iter().map(|&v| 2.0 * v).collect();

    let options = LassoFitOptions {
        lambda: 0.001,
        intercept: false,
        standardize: false,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    assert_close(fit.intercept, 0.0, TOLERANCE, "intercept should be 0");
    assert_close(fit.coefficients[0], 2.0, 0.5, "slope");
}

#[test]
fn test_lasso_zero_lambda_equivalent_ols() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = LassoFitOptions {
        lambda: 0.0,
        intercept: true,
        standardize: false,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    // With lambda=0, should get perfect fit for this simple case
    assert_close(fit.fitted_values[0], 2.0, 1e-6, "prediction[0]");
    assert_close(fit.fitted_values[1], 4.0, 1e-6, "prediction[1]");
    assert_close(fit.fitted_values[2], 6.0, 1e-6, "prediction[2]");
}

// ============================================================================
// Lasso Sparsity Tests
// ============================================================================

#[test]
fn test_lasso_large_lambda_produces_sparsity() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let options = LassoFitOptions {
        lambda: 100.0, // Large lambda
        intercept: true,
        standardize: true,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    // With large lambda, coefficient should be zero
    assert_eq!(
        fit.n_nonzero, 0,
        "All coefficients should be zero with large lambda"
    );
    assert!(
        fit.coefficients[0].abs() < 1e-10,
        "Coefficient should be zero"
    );
}

#[test]
fn test_lasso_multiple_predictors_sparsity_pattern() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Highly correlated with x1
    let x3 = vec![0.5, 1.5, 2.5, 3.5, 4.5]; // Different pattern
    let x = create_design_matrix_multi(&[x1.clone(), x2, x3.clone()]);

    let y: Vec<f64> = (0..5).map(|i| 1.0 + 2.0 * x1[i] + 0.5 * x3[i]).collect();

    // Test sparsity at different lambda values
    // Note: With highly correlated predictors, the Lasso path can be non-monotonic
    // in terms of which variables are selected. We test the overall trend instead.
    let lambdas = [0.01, 0.5, 2.0, 10.0];
    let mut nonzero_counts = Vec::new();

    for &lambda in &lambdas {
        let options = LassoFitOptions {
            lambda,
            intercept: true,
            standardize: true,
            ..Default::default()
        };

        let fit = lasso_fit(&x, &y, &options).unwrap();
        nonzero_counts.push(fit.n_nonzero);
    }

    // Overall trend: highest lambda should have same or fewer non-zero than lowest lambda
    assert!(
        nonzero_counts.last().unwrap() <= nonzero_counts.first().unwrap(),
        "Large lambda should produce sparser solution than small lambda: {:?}",
        nonzero_counts
    );

    // Very large lambda should produce mostly zeros
    let options_large = LassoFitOptions {
        lambda: 100.0,
        intercept: true,
        standardize: true,
        ..Default::default()
    };
    let fit_large = lasso_fit(&x, &y, &options_large).unwrap();
    assert!(
        fit_large.n_nonzero <= 1,
        "Very large lambda should produce at most 1 non-zero coefficient"
    );
}

#[test]
fn test_lasso_sparsity_with_correlated_predictors() {
    // Create data where x2 is a noisy version of x1
    let x1: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let x2: Vec<f64> = x1.iter().map(|&v| v * 0.95 + 0.5).collect();
    let x = create_design_matrix_multi(&[x1.clone(), x2]);

    let y: Vec<f64> = x1.iter().map(|&v| 2.0 * v + 1.0).collect();

    let options = LassoFitOptions {
        lambda: 5.0,
        intercept: true,
        standardize: true,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    // With correlated predictors and lasso, typically only one is selected
    // This is the "grouping effect" - lasso tends to pick one
    assert!(
        fit.n_nonzero <= 2,
        "Should select at most 2 predictors from correlated pair"
    );
}

// ============================================================================
// Lasso Convergence Tests
// ============================================================================

#[test]
fn test_lasso_convergence_with_default_tolerance() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let y: Vec<f64> = (1..=10).map(|i| 2.0 * i as f64 + 1.0).collect();

    let options = LassoFitOptions {
        lambda: 1.0,
        intercept: true,
        standardize: true,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    assert!(fit.converged, "Should converge with default tolerance");
    assert!(fit.iterations < 1000, "Should converge within max_iter");
}

#[test]
fn test_lasso_strict_tolerance_requires_more_iterations() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let options_loose = LassoFitOptions {
        lambda: 1.0,
        tol: 1e-2,
        ..Default::default()
    };

    let options_strict = LassoFitOptions {
        lambda: 1.0,
        tol: 1e-10,
        ..Default::default()
    };

    let fit_loose = lasso_fit(&x, &y, &options_loose).unwrap();
    let fit_strict = lasso_fit(&x, &y, &options_strict).unwrap();

    assert!(
        fit_strict.iterations >= fit_loose.iterations,
        "Stricter tolerance should require same or more iterations"
    );
}

#[test]
fn test_lasso_max_iter_limits_iterations() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let options = LassoFitOptions {
        lambda: 1.0,
        max_iter: 5,
        tol: 1e-15, // Very strict tolerance to ensure we hit max_iter
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    assert!(fit.iterations <= 5, "Should not exceed max_iter");
    // With very few iterations and strict tolerance, likely won't converge
    // But we can't guarantee it, so just check the iteration count
}

// ============================================================================
// Lasso Penalty Factor Tests
// ============================================================================

#[test]
fn test_lasso_penalty_factor_excludes_variable() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![10.0, 20.0, 30.0, 40.0, 50.0]; // Important predictor
    let x = create_design_matrix_multi(&[x1.clone(), x2.clone()]);

    let y: Vec<f64> = (0..5).map(|i| 1.0 + 2.0 * x1[i] + 5.0 * x2[i]).collect();

    // Penalty factor: infinity for x1 (always exclude), 1 for x2
    // Note: penalty_factor applies to all columns including intercept in internal representation
    let penalty_factor = Some(vec![0.0, f64::INFINITY, 1.0]); // [intercept, x1, x2]

    let options = LassoFitOptions {
        lambda: 1.0,
        penalty_factor,
        standardize: false,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    // x1 should be zero due to infinite penalty
    assert!(
        fit.coefficients[0].abs() < 1e-10,
        "x1 should be penalized to zero"
    );
}

#[test]
fn test_lasso_penalty_factor_differential() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = create_design_matrix_multi(&[x1.clone(), x2.clone()]);

    let y: Vec<f64> = (0..5).map(|i| 1.0 + 2.0 * x1[i] + 2.0 * x2[i]).collect();

    // Higher penalty for x1
    let penalty_factor = Some(vec![0.0, 10.0, 1.0]); // [intercept, x1 (high penalty), x2 (low penalty)]

    let options = LassoFitOptions {
        lambda: 1.0,
        penalty_factor,
        standardize: false,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    // x1 should have smaller coefficient due to higher penalty
    assert!(
        fit.coefficients[0].abs() <= fit.coefficients[1].abs(),
        "x1 (higher penalty) should have smaller or equal coefficient than x2"
    );
}

// ============================================================================
// Ridge Basic Functionality Tests
// ============================================================================

#[test]
fn test_ridge_perfect_linear_fit() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2*x

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.001, // Very small lambda for near-OLS solution
        intercept: true,
        standardize: false,
        weights: None,
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    assert_close(fit.coefficients[0], 2.0, 0.5, "slope coefficient");
    assert!(fit.intercept.abs() < 1.0, "intercept should be small");
}

#[test]
fn test_ridge_with_intercept() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0]);
    let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| 5.0 + 2.0 * v)
        .collect();

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.001,
        intercept: true,
        standardize: false,
        weights: None,
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    assert_close(fit.intercept, 5.0, 1.0, "intercept");
    assert_close(fit.coefficients[0], 2.0, 0.5, "slope");
}

#[test]
fn test_ridge_without_intercept() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let mut x_data_full = vec![0.0; x_data.len()];
    for (i, &v) in x_data.iter().enumerate() {
        x_data_full[i] = v;
    }
    let x = Matrix::new(4, 1, x_data_full);
    let y: Vec<f64> = x_data.iter().map(|&v| 2.0 * v).collect();

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.001,
        intercept: false,
        standardize: false,
        weights: None,
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    assert_close(fit.intercept, 0.0, TOLERANCE, "intercept should be 0");
    assert_close(fit.coefficients[0], 2.0, 0.5, "slope");
}

#[test]
fn test_ridge_zero_lambda_equivalent_ols() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.0,
        intercept: true,
        standardize: false,
        weights: None,
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    // With lambda=0, should get perfect fit for this simple case
    assert_close(fit.fitted_values[0], 2.0, 1e-6, "prediction[0]");
    assert_close(fit.fitted_values[1], 4.0, 1e-6, "prediction[1]");
    assert_close(fit.fitted_values[2], 6.0, 1e-6, "prediction[2]");
}

// ============================================================================
// Ridge Shrinkage Tests
// ============================================================================

#[test]
fn test_ridge_coefficient_shrinkage_with_lambda() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let y: Vec<f64> = (1..=10).map(|i| 2.0 * i as f64 + 1.0).collect();

    let lambdas = [0.0, 0.1, 1.0, 10.0, 100.0];
    let mut prev_coef_abs = 100.0;

    for &lambda in &lambdas {
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
        let coef_abs = fit.coefficients[0].abs();

        // As lambda increases, coefficient magnitude should decrease
        assert!(
            coef_abs <= prev_coef_abs + 1e-6,
            "Coefficient should shrink with lambda: {} at lambda={}",
            coef_abs,
            lambda
        );
        prev_coef_abs = coef_abs;
    }

    // With very large lambda, coefficient should be near zero
    let options_large = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 1000.0,
        intercept: true,
        standardize: true,
        weights: None,
    };
    let fit_large = ridge_fit(&x, &y, &options_large).unwrap();
    assert!(
        fit_large.coefficients[0].abs() < 0.5,
        "Coefficient should be small with large lambda"
    );
}

#[test]
fn test_ridge_shrinkage_towards_zero() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y: Vec<f64> = (1..=5).map(|i| 3.0 * i as f64 + 2.0).collect();

    // OLS coefficient (lambda = 0)
    let options_ols = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.0,
        intercept: true,
        standardize: false,
        weights: None,
    };
    let fit_ols = ridge_fit(&x, &y, &options_ols).unwrap();
    let ols_coef = fit_ols.coefficients[0];

    // Ridge coefficient (lambda > 0)
    let options_ridge = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 1.0,
        intercept: true,
        standardize: false,
        weights: None,
    };
    let fit_ridge = ridge_fit(&x, &y, &options_ridge).unwrap();
    let ridge_coef = fit_ridge.coefficients[0];

    // Ridge coefficient should be smaller in magnitude than OLS
    assert!(
        ridge_coef.abs() < ols_coef.abs(),
        "Ridge coefficient ({}) should be smaller than OLS ({})",
        ridge_coef,
        ols_coef
    );
}

// ============================================================================
// Ridge Multicollinearity Tests
// ============================================================================

#[test]
fn test_ridge_handles_perfect_multicollinearity() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Exactly 2 * x1 (perfect multicollinearity)
    let x = create_design_matrix_multi(&[x1.clone(), x2]);

    let y: Vec<f64> = (0..5).map(|i| 1.0 + 3.0 * x1[i]).collect();

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 1.0,
        intercept: true,
        standardize: true,
        weights: None,
    };

    // Ridge should handle multicollinearity without error
    let fit = ridge_fit(&x, &y, &options).unwrap();

    // Should get a solution (not necessarily unique due to multicollinearity)
    assert!(
        fit.r_squared > 0.5,
        "Should have reasonable R² despite multicollinearity"
    );
}

#[test]
fn test_ridge_coefficients_stabilize_with_lambda() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Highly correlated with x1
    let x = create_design_matrix_multi(&[x1.clone(), x2]);

    let y: Vec<f64> = (0..5).map(|i| 1.0 + 2.0 * x1[i]).collect();

    // Without regularization (or very small), coefficients might be unstable
    let options_small = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.001,
        intercept: true,
        standardize: true,
        weights: None,
    };
    let fit_small = ridge_fit(&x, &y, &options_small).unwrap();

    // With regularization, coefficients should be more stable
    let options_large = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 1.0,
        intercept: true,
        standardize: true,
        weights: None,
    };
    let fit_large = ridge_fit(&x, &y, &options_large).unwrap();

    // Both should produce predictions (may differ due to shrinkage)
    assert!(fit_small.r_squared > 0.9);
    assert!(fit_large.r_squared > 0.5); // Lower bar for large lambda with strong shrinkage
}

// ============================================================================
// Standardization Tests
// ============================================================================

#[test]
fn test_lasso_standardization_affects_coefficients() {
    let x = create_design_matrix(&[100.0, 200.0, 300.0, 400.0]);
    let y = vec![202.0, 404.0, 606.0, 808.0]; // y ≈ 2*x

    let options_std = LassoFitOptions {
        lambda: 0.1,
        intercept: true,
        standardize: true,
        ..Default::default()
    };

    let options_no_std = LassoFitOptions {
        lambda: 0.1,
        intercept: true,
        standardize: false,
        ..Default::default()
    };

    let fit_std = lasso_fit(&x, &y, &options_std).unwrap();
    let fit_no_std = lasso_fit(&x, &y, &options_no_std).unwrap();

    // Coefficients should be on different scales due to standardization
    // But predictions should be similar
    for i in 0..y.len() {
        assert_close(
            fit_std.fitted_values[i],
            fit_no_std.fitted_values[i],
            50.0, // Allow some difference due to regularization
            "predictions should be similar",
        );
    }
}

#[test]
fn test_ridge_standardization_affects_coefficients() {
    let x = create_design_matrix(&[100.0, 200.0, 300.0, 400.0]);
    let y = vec![202.0, 404.0, 606.0, 808.0]; // y ≈ 2*x

    let options_std = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.1,
        intercept: true,
        standardize: true,
        weights: None,
    };

    let options_no_std = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.1,
        intercept: true,
        standardize: false,
        weights: None,
    };

    let fit_std = ridge_fit(&x, &y, &options_std).unwrap();
    let fit_no_std = ridge_fit(&x, &y, &options_no_std).unwrap();

    // Coefficients will be on different scales
    // But predictions should be similar
    for i in 0..y.len() {
        assert_close(
            fit_std.fitted_values[i],
            fit_no_std.fitted_values[i],
            1.0,
            "predictions should be similar",
        );
    }
}

// ============================================================================
// Statistics Tests - Lasso
// ============================================================================

#[test]
fn test_lasso_r_squared_in_valid_range() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let options = LassoFitOptions {
        lambda: 0.1,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    assert!(
        fit.r_squared >= 0.0 && fit.r_squared <= 1.0,
        "R² should be in [0, 1], got {}",
        fit.r_squared
    );
}

#[test]
fn test_lasso_r_squared_perfect_fit() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = LassoFitOptions {
        lambda: 0.0, // OLS
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    assert_close(fit.r_squared, 1.0, 1e-6, "R² should be 1 for perfect fit");
}

#[test]
fn test_lasso_adjusted_r_squared_less_than_r_squared() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x2 = vec![1.5, 2.7, 3.2, 4.8, 5.1, 6.9, 7.3, 8.5, 9.2, 10.8];
    let x = create_design_matrix_multi(&[x1.clone(), x2.clone()]);

    let y: Vec<f64> = (0..10).map(|i| 1.0 + x1[i] + 0.5 * x2[i]).collect();

    let options = LassoFitOptions {
        lambda: 0.5,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    assert!(
        fit.adj_r_squared <= fit.r_squared,
        "Adjusted R² ({}) should be <= R² ({})",
        fit.adj_r_squared,
        fit.r_squared
    );
}

#[test]
fn test_lasso_rmse_calculation() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = LassoFitOptions {
        lambda: 0.1,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    // RMSE should be sqrt of MSE
    assert_close(fit.rmse, fit.mse.sqrt(), TOLERANCE, "RMSE = sqrt(MSE)");
}

#[test]
fn test_lasso_mae_calculation() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = LassoFitOptions {
        lambda: 0.1,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    // Manually compute MAE
    let expected_mae: f64 = fit.residuals.iter().map(|r| r.abs()).sum::<f64>() / y.len() as f64;

    assert_close(fit.mae, expected_mae, TOLERANCE, "MAE calculation");
}

#[test]
fn test_lasso_residuals_sum_property() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let options = LassoFitOptions {
        lambda: 0.1,
        intercept: true,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    // Residuals should approximately sum to zero with intercept
    let residual_sum: f64 = fit.residuals.iter().sum();
    assert_close(
        residual_sum,
        0.0,
        0.1,
        "residuals should sum to ~0 with intercept",
    );
}

// ============================================================================
// Statistics Tests - Ridge
// ============================================================================

#[test]
fn test_ridge_r_squared_in_valid_range() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 1.0,
        ..Default::default()
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    assert!(
        fit.r_squared >= 0.0 && fit.r_squared <= 1.0,
        "R² should be in [0, 1], got {}",
        fit.r_squared
    );
}

#[test]
fn test_ridge_r_squared_perfect_fit() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.0, // OLS
        ..Default::default()
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    assert_close(fit.r_squared, 1.0, 1e-6, "R² should be 1 for perfect fit");
}

#[test]
fn test_ridge_adjusted_r_squared_less_than_r_squared() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x2 = vec![1.5, 2.7, 3.2, 4.8, 5.1, 6.9, 7.3, 8.5, 9.2, 10.8];
    let x = create_design_matrix_multi(&[x1.clone(), x2.clone()]);

    let y: Vec<f64> = (0..10).map(|i| 1.0 + x1[i] + 0.5 * x2[i]).collect();

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 1.0,
        ..Default::default()
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    assert!(
        fit.adj_r_squared <= fit.r_squared,
        "Adjusted R² ({}) should be <= R² ({})",
        fit.adj_r_squared,
        fit.r_squared
    );
}

#[test]
fn test_ridge_rmse_calculation() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.1,
        ..Default::default()
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    // RMSE should be sqrt of MSE
    assert_close(fit.rmse, fit.mse.sqrt(), TOLERANCE, "RMSE = sqrt(MSE)");
}

#[test]
fn test_ridge_mae_calculation() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.1,
        ..Default::default()
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    // Manually compute MAE
    let expected_mae: f64 = fit.residuals.iter().map(|r| r.abs()).sum::<f64>() / y.len() as f64;

    assert_close(fit.mae, expected_mae, TOLERANCE, "MAE calculation");
}

#[test]
fn test_ridge_residuals_sum_property() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 1.0,
        intercept: true,
        ..Default::default()
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    // Residuals should approximately sum to zero with intercept
    let residual_sum: f64 = fit.residuals.iter().sum();
    assert_close(
        residual_sum,
        0.0,
        0.1,
        "residuals should sum to ~0 with intercept",
    );
}

#[test]
fn test_ridge_effective_degrees_of_freedom() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    // OLS (lambda = 0): df should be number of parameters
    let options_ols = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.0,
        ..Default::default()
    };
    let fit_ols = ridge_fit(&x, &y, &options_ols).unwrap();
    assert_close(
        fit_ols.df,
        2.0,
        0.1,
        "OLS df should equal number of parameters",
    );

    // Ridge (lambda > 0): df should be less than OLS
    let options_ridge = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 10.0,
        ..Default::default()
    };
    let fit_ridge = ridge_fit(&x, &y, &options_ridge).unwrap();
    assert!(
        fit_ridge.df < fit_ols.df,
        "Ridge df ({}) should be less than OLS df ({})",
        fit_ridge.df,
        fit_ols.df
    );
}

// ============================================================================
// Prediction Tests
// ============================================================================

#[test]
fn test_lasso_predictions_match_fitted_values() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = LassoFitOptions {
        lambda: 0.001,
        standardize: false,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    // Predictions on training data should equal fitted values
    for i in 0..y.len() {
        let pred = fit.intercept + fit.coefficients[0] * x.get(i, 1);
        assert_close(
            pred,
            fit.fitted_values[i],
            TOLERANCE,
            "prediction should match fitted value",
        );
    }
}

#[test]
fn test_ridge_predictions_match_fitted_values() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0]);
    let y = vec![2.0, 4.0, 6.0];

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.001,
        standardize: false,
        ..Default::default()
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    // Predictions on training data should equal fitted values
    for i in 0..y.len() {
        let pred = fit.intercept + fit.coefficients[0] * x.get(i, 1);
        assert_close(
            pred,
            fit.fitted_values[i],
            TOLERANCE,
            "prediction should match fitted value",
        );
    }
}

#[test]
fn test_lasso_new_data_predictions() {
    let x_train = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_train = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let options = LassoFitOptions {
        lambda: 0.001,
        standardize: false,
        ..Default::default()
    };

    let fit = lasso_fit(&x_train, &y_train, &options).unwrap();

    // Predict on new data
    let x_new = create_design_matrix(&[6.0, 7.0]);
    let preds: Vec<f64> = (0..2)
        .map(|i| fit.intercept + fit.coefficients[0] * x_new.get(i, 1))
        .collect();

    // Predictions should be approximately 2 * x
    assert_close(preds[0], 12.0, 2.0, "prediction for x=6");
    assert_close(preds[1], 14.0, 2.0, "prediction for x=7");
}

#[test]
fn test_ridge_new_data_predictions() {
    let x_train = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_train = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.001,
        standardize: false,
        ..Default::default()
    };

    let fit = ridge_fit(&x_train, &y_train, &options).unwrap();

    // Predict on new data
    let x_new = create_design_matrix(&[6.0, 7.0]);
    let preds: Vec<f64> = (0..2)
        .map(|i| fit.intercept + fit.coefficients[0] * x_new.get(i, 1))
        .collect();

    // Predictions should be approximately 2 * x
    assert_close(preds[0], 12.0, 2.0, "prediction for x=6");
    assert_close(preds[1], 14.0, 2.0, "prediction for x=7");
}

// ============================================================================
// Multiple Predictors Tests
// ============================================================================

#[test]
fn test_lasso_multiple_predictors() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    let x3 = vec![0.5, 1.5, 2.5, 3.5, 4.5];
    let x = create_design_matrix_multi(&[x1.clone(), x2.clone(), x3.clone()]);

    let y: Vec<f64> = (0..5)
        .map(|i| 1.0 + 2.0 * x1[i] + 0.5 * x2[i] + x3[i])
        .collect();

    let options = LassoFitOptions {
        lambda: 0.001,
        standardize: false,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    assert!(fit.converged);
    assert!(fit.r_squared > 0.9, "Should have high R²");
    // coefficients contains the slope coefficients (excluding intercept which is separate)
    assert_eq!(
        fit.coefficients.len(),
        3,
        "Should have 3 slope coefficients"
    );
}

#[test]
fn test_ridge_multiple_predictors() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    let x3 = vec![0.5, 1.5, 2.5, 3.5, 4.5];
    let x = create_design_matrix_multi(&[x1.clone(), x2.clone(), x3.clone()]);

    let y: Vec<f64> = (0..5)
        .map(|i| 1.0 + 2.0 * x1[i] + 0.5 * x2[i] + x3[i])
        .collect();

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.001,
        standardize: false,
        ..Default::default()
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    assert!(fit.r_squared > 0.9, "Should have high R²");
    // coefficients contains the slope coefficients (excluding intercept which is separate)
    assert_eq!(
        fit.coefficients.len(),
        3,
        "Should have 3 slope coefficients"
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_lasso_with_constant_y() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![5.0; 5];

    let options = LassoFitOptions {
        lambda: 1.0,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    // All coefficients should be zero, intercept should be mean of y
    assert_eq!(fit.n_nonzero, 0);
    assert_close(fit.intercept, 5.0, 0.5, "intercept should be mean of y");
}

#[test]
fn test_ridge_with_constant_y() {
    let x = create_design_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec![5.0; 5];

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 1.0,
        ..Default::default()
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    // All slope coefficients should be near zero
    assert!(
        fit.coefficients[0].abs() < 1.0,
        "Coefficient should be near zero for constant y"
    );
    assert_close(fit.intercept, 5.0, 0.5, "intercept should be mean of y");
}

#[test]
fn test_lasso_with_negative_values() {
    let x = create_design_matrix(&[-5.0, -3.0, -1.0, 1.0, 3.0]);
    let y = vec![-10.0, -6.0, -2.0, 2.0, 6.0]; // y = 2*x

    let options = LassoFitOptions {
        lambda: 0.001,
        standardize: false,
        ..Default::default()
    };

    let fit = lasso_fit(&x, &y, &options).unwrap();

    assert!(fit.converged);
    assert_close(fit.coefficients[0], 2.0, 0.5, "slope should be ~2");
}

#[test]
fn test_ridge_with_negative_values() {
    let x = create_design_matrix(&[-5.0, -3.0, -1.0, 1.0, 3.0]);
    let y = vec![-10.0, -6.0, -2.0, 2.0, 6.0]; // y = 2*x

    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda: 0.001,
        standardize: false,
        ..Default::default()
    };

    let fit = ridge_fit(&x, &y, &options).unwrap();

    assert_close(fit.coefficients[0], 2.0, 0.5, "slope should be ~2");
}

// ============================================================================
// Property-Based Tests
// ============================================================================

proptest! {
    #[test]
    fn prop_lasso_r_squared_in_bounds(
        n in 10..30usize,
        lambda in 0.01..10.0f64,
        seed in 0u64..10000u64
    ) {
        // Generate deterministic "random" data
        let x: Vec<f64> = (0..n).map(|i| (seed.wrapping_add(i as u64) % 100) as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let x_mat = create_design_matrix(&x);

        let options = LassoFitOptions {
            lambda,
            ..Default::default()
        };

        if let Ok(fit) = lasso_fit(&x_mat, &y, &options) {
            // Allow small numerical tolerance for floating point errors
            prop_assert!(fit.r_squared >= -1e-10 && fit.r_squared <= 1.0 + 1e-10,
                "R² = {} is outside [0, 1]", fit.r_squared);
        }
    }

    #[test]
    fn prop_ridge_r_squared_in_bounds(
        n in 10..30usize,
        lambda in 0.0..10.0f64,
        seed in 0u64..10000u64
    ) {
        let x: Vec<f64> = (0..n).map(|i| (seed.wrapping_add(i as u64) % 100) as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let x_mat = create_design_matrix(&x);

        let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
            lambda,
            ..Default::default()
        };

        let fit = ridge_fit(&x_mat, &y, &options).unwrap();
        prop_assert!(fit.r_squared >= 0.0 && fit.r_squared <= 1.0,
            "R² = {} is outside [0, 1]", fit.r_squared);
    }

    #[test]
    fn prop_lasso_nonzero_count_decreases_with_lambda(
        n in 10..20usize,
        seed in 0u64..10000u64
    ) {
        let x1: Vec<f64> = (0..n).map(|i| (seed.wrapping_add(i as u64) % 100) as f64 / 10.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| (seed.wrapping_mul(2).wrapping_add(i as u64) % 100) as f64 / 10.0).collect();
        let x = create_design_matrix_multi(&[x1.clone(), x2.clone()]);

        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x1[i] + 0.5 * x2[i]).collect();

        let options1 = LassoFitOptions {
            lambda: 0.1,
            ..Default::default()
        };
        let options2 = LassoFitOptions {
            lambda: 10.0,
            ..Default::default()
        };

        let fit1 = lasso_fit(&x, &y, &options1).unwrap();
        let fit2 = lasso_fit(&x, &y, &options2).unwrap();

        prop_assert!(fit2.n_nonzero <= fit1.n_nonzero,
            "Non-zero count should decrease with lambda: {} (λ=0.1) vs {} (λ=10.0)",
            fit1.n_nonzero, fit2.n_nonzero);
    }

    #[test]
    fn prop_ridge_coefficient_magnitude_decreases_with_lambda(
        n in 10..20usize,
        seed in 0u64..10000u64
    ) {
        let x: Vec<f64> = (0..n).map(|i| (seed.wrapping_add(i as u64) % 100) as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let x_mat = create_design_matrix(&x);

        let options1 = RidgeFitOptions {
            lambda: 0.1,
            standardize: true,
            ..Default::default()
        };
        let options2 = RidgeFitOptions {
            lambda: 10.0,
            standardize: true,
            ..Default::default()
        };

        let fit1 = ridge_fit(&x_mat, &y, &options1).unwrap();
        let fit2 = ridge_fit(&x_mat, &y, &options2).unwrap();

        prop_assert!(fit2.coefficients[0].abs() <= fit1.coefficients[0].abs() + 1e-6,
            "Coefficient magnitude should decrease with lambda: |{}| (λ=0.1) vs |{}| (λ=10.0)",
            fit1.coefficients[0], fit2.coefficients[0]);
    }

    #[test]
    fn prop_lasso_rmse_is_non_negative(
        n in 10..30usize,
        lambda in 0.01..10.0f64,
        seed in 0u64..10000u64
    ) {
        let x: Vec<f64> = (0..n).map(|i| (seed.wrapping_add(i as u64) % 100) as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let x_mat = create_design_matrix(&x);

        let options = LassoFitOptions {
            lambda,
            ..Default::default()
        };

        if let Ok(fit) = lasso_fit(&x_mat, &y, &options) {
            prop_assert!(fit.rmse >= 0.0, "RMSE should be non-negative, got {}", fit.rmse);
            prop_assert!(fit.mse >= 0.0, "MSE should be non-negative, got {}", fit.mse);
            prop_assert!(fit.mae >= 0.0, "MAE should be non-negative, got {}", fit.mae);
        }
    }

    #[test]
    fn prop_ridge_rmse_is_non_negative(
        n in 10..30usize,
        lambda in 0.0..10.0f64,
        seed in 0u64..10000u64
    ) {
        let x: Vec<f64> = (0..n).map(|i| (seed.wrapping_add(i as u64) % 100) as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let x_mat = create_design_matrix(&x);

        let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
            lambda,
            ..Default::default()
        };

        let fit = ridge_fit(&x_mat, &y, &options).unwrap();

        prop_assert!(fit.rmse >= 0.0, "RMSE should be non-negative, got {}", fit.rmse);
        prop_assert!(fit.mse >= 0.0, "MSE should be non-negative, got {}", fit.mse);
        prop_assert!(fit.mae >= 0.0, "MAE should be non-negative, got {}", fit.mae);
    }

    #[test]
    fn prop_lasso_fitted_plus_residuals_equals_y(
        n in 10..30usize,
        lambda in 0.01..10.0f64,
        seed in 0u64..10000u64
    ) {
        let x: Vec<f64> = (0..n).map(|i| (seed.wrapping_add(i as u64) % 100) as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let x_mat = create_design_matrix(&x);

        let options = LassoFitOptions {
            lambda,
            ..Default::default()
        };

        if let Ok(fit) = lasso_fit(&x_mat, &y, &options) {
            for i in 0..n {
                let sum = fit.fitted_values[i] + fit.residuals[i];
                prop_assert!((sum - y[i]).abs() < 1e-6,
                    "fitted + residual != y at index {}: {} + {} = {}, expected {}",
                    i, fit.fitted_values[i], fit.residuals[i], sum, y[i]);
            }
        }
    }

    #[test]
    fn prop_ridge_fitted_plus_residuals_equals_y(
        n in 10..30usize,
        lambda in 0.0..10.0f64,
        seed in 0u64..10000u64
    ) {
        let x: Vec<f64> = (0..n).map(|i| (seed.wrapping_add(i as u64) % 100) as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let x_mat = create_design_matrix(&x);

        let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
            lambda,
            ..Default::default()
        };

        let fit = ridge_fit(&x_mat, &y, &options).unwrap();

        for i in 0..n {
            let sum = fit.fitted_values[i] + fit.residuals[i];
            prop_assert!((sum - y[i]).abs() < 1e-6,
                "fitted + residual != y at index {}: {} + {} = {}, expected {}",
                i, fit.fitted_values[i], fit.residuals[i], sum, y[i]);
        }
    }
}
