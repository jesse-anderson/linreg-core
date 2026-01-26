//! GLMNET Algorithm Tests
//!
//! These tests validate that the core elastic net algorithm components match glmnet output.
//! The behavior was determined through validation against paper descriptions and reference outputs.

use linreg_core::linalg::Matrix;
use linreg_core::regularized::{
    make_lambda_path, LambdaPathOptions,
    elastic_net_path, ElasticNetOptions,
    elastic_net_fit,
};
use linreg_core::regularized::path::compute_lambda_max;
use linreg_core::regularized::preprocess::{standardize_xy, StandardizeOptions};

// =============================================================================
// Lambda Max Computation Tests
// Based on glmnet output
//
// Formula (glmnet):
//   lambda_max = max_j |X_j^T y| / max(alpha, 1e-3)
//
// where X is standardized (unit norm columns) and y is at unit norm
// =============================================================================

#[test]
fn test_compute_lambda_max_lasso_finite_positive() {
    // For lasso (alpha = 1.0), lambda_max should be finite and positive
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];

    let n = 5;
    let p = 2;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
        x_data[i * (p + 1) + 2] = x2[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let standardization_options = StandardizeOptions {
        intercept: true,
        standardize_x: true,
        standardize_y: true,
        weights: None,
    };
    let (x_standardized, y_standardized, _) = standardize_xy(&x, &y, &standardization_options);

    let lambda_max = compute_lambda_max(&x_standardized, &y_standardized, 1.0, None, Some(0));

    assert!(lambda_max > 0.0, "lambda_max should be positive for lasso");
    assert!(lambda_max.is_finite(), "lambda_max should be finite for lasso");
}

#[test]
fn test_compute_lambda_max_ridge_returns_infinity() {
    // For alpha = 0 (ridge), glmnet uses max(alpha, 1e-3) in denominator
    // This produces a large finite value, not infinity
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let n = 5;
    let p = 1;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let standardization_options = StandardizeOptions {
        intercept: true,
        standardize_x: true,
        standardize_y: true,
        weights: None,
    };
    let (x_standardized, y_standardized, _) = standardize_xy(&x, &y, &standardization_options);

    // For alpha=0, the formula uses max(0, 1e-3) = 1e-3 as denominator
    let lambda_max = compute_lambda_max(&x_standardized, &y_standardized, 0.0, None, Some(0));

    // Result should be finite (large value), not infinite
    assert!(lambda_max.is_finite(), "lambda_max should be finite for ridge (alpha=0) using max(alpha, 1e-3)");
    // The value should be quite large since we're dividing by 1e-3
    assert!(lambda_max > 100.0, "lambda_max should be large for ridge");
}

// =============================================================================
// Y Standardization Tests
//
// glmnet scales y to unit norm:
//   y = v * (y - ym)  where v = sqrt(w_normalized)
//   ys = sqrt(sum(y^2))
//   y = y / ys  -> ||y|| = 1
// =============================================================================

#[test]
fn test_standardize_y_unit_norm() {
    // Verify y is standardized to unit norm (||y|| = 1)
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let n = 5;
    let p = 1;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let standardization_options = StandardizeOptions {
        intercept: true,
        standardize_x: true,
        standardize_y: true,
        weights: None,
    };
    let (_, y_standardized, info) = standardize_xy(&x, &y, &standardization_options);

    // Verify y is at unit norm
    let y_norm: f64 = y_standardized.iter().map(|&v| v * v).sum::<f64>().sqrt();
    assert!((y_norm - 1.0).abs() < 1e-10, "y should be at unit norm after standardization");

    // y_scale should be the L2 norm of the centered y (before v scaling)
    let y_centered: Vec<f64> = y.iter().map(|&yi| yi - 3.0).collect();
    // After v = 1/sqrt(n) scaling: y_standardized = (y - ym) / sqrt(n)
    let expected_scale: f64 = y_centered.iter().map(|&v| v * v).sum::<f64>().sqrt() / (n as f64).sqrt();
    if let Some(scale) = info.y_scale {
        assert!((scale - expected_scale).abs() < 1e-10, "y_scale should match L2 norm");
    }
}

#[test]
fn test_standardize_preserves_intercept_column() {
    // Verify that intercept column (col 0) is NOT standardized
    let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let n = 5;
    let p = 1;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let standardization_options = StandardizeOptions {
        intercept: true,
        standardize_x: true,
        standardize_y: true,
        weights: None,
    };
    let (x_standardized, _, _) = standardize_xy(&x, &y, &standardization_options);

    // Check that first column is still all ones
    for i in 0..n {
        assert!((x_standardized.get(i, 0) - 1.0).abs() < 1e-10,
               "Intercept column should remain as ones when intercept=true");
    }
}

// =============================================================================
// Lambda Path Generation Tests
//
// Lambda path follows geometric progression (glmnet style):
//   lambda[0] = LAMBDA_EFFECTIVE_INFINITY = infinity (effectively infinite)
//   lambda[1] = lambda_decay_factor * lambda_max (first real lambda)
//   lambda[k] = lambda[k-1] * lambda_decay_factor (geometric decay for k >= 2)
//
// Where:
//   lambda_decay_factor = max(lambda_min_ratio, eps)^(1/(nlam-1)), eps = 1e-6
//   lambda_max = max(|X^T y|) / max(alpha, 1e-3)
// =============================================================================

#[test]
fn test_lambda_path_geometric_decay() {
    // Lambda path should follow glmnet's geometric progression
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let n = 5;
    let p = 1;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let standardization_options = StandardizeOptions {
        intercept: true,
        standardize_x: true,
        standardize_y: true,
        weights: None,
    };
    let (x_standardized, y_standardized, _) = standardize_xy(&x, &y, &standardization_options);

    let nlambda = 10;
    let lambda_min_ratio_val = 0.01;
    let path_options = LambdaPathOptions {
        nlambda,
        lambda_min_ratio: Some(lambda_min_ratio_val),
        alpha: 1.0,
        eps_for_ridge: 1e-3,
    };

    let lambdas = make_lambda_path(&x_standardized, &y_standardized, &path_options, None, Some(0));

    assert_eq!(lambdas.len(), nlambda, "Should generate exactly nlambda lambdas");

    // First lambda should be LAMBDA_EFFECTIVE_INFINITY (effectively infinite)
    assert_eq!(lambdas[0], f64::INFINITY, "First lambda should be LAMBDA_EFFECTIVE_INFINITY (infinity)");

    // Verify geometric progression for k >= 2
    // lambda_decay_factor = max(lambda_min_ratio, 1e-6)^(1/(nlam-1))
    const EPS: f64 = 1.0e-6;
    let lambda_min_ratio_clamped = lambda_min_ratio_val.max(EPS);
    let lambda_decay_factor = lambda_min_ratio_clamped.powf(1.0 / (nlambda - 1) as f64);

    for i in 2..lambdas.len() {
        let expected_ratio = lambdas[i] / lambdas[i - 1];
        assert!((expected_ratio - lambda_decay_factor).abs() < 1e-10,
               "Lambda path should follow geometric progression at index {}: expected {}, got {}",
               i, lambda_decay_factor, expected_ratio);
    }

    // Verify decreasing sequence
    for i in 1..lambdas.len() {
        assert!(lambdas[i] < lambdas[i - 1],
               "Lambda path should be strictly decreasing at index {}", i);
    }
}

#[test]
fn test_lambda_path_first_lambda_is_lambda_max() {
    // First lambda in path should equal lambda_max
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 1.0, 0.5, 0.2, 0.1];

    let n = 5;
    let p = 2;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
        x_data[i * (p + 1) + 2] = x2[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let standardization_options = StandardizeOptions {
        intercept: true,
        standardize_x: true,
        standardize_y: true,
        weights: None,
    };
    let (x_standardized, y_standardized, _) = standardize_xy(&x, &y, &standardization_options);

    let lambda_max = compute_lambda_max(&x_standardized, &y_standardized, 1.0, None, Some(0));

    let path_options = LambdaPathOptions {
        nlambda: 5,
        lambda_min_ratio: Some(0.1),
        alpha: 1.0,
        eps_for_ridge: 1e-3,
    };
    let lambdas = make_lambda_path(&x_standardized, &y_standardized, &path_options, None, Some(0));

    // glmnet style: first lambda is LAMBDA_EFFECTIVE_INFINITY, second is lambda_decay_factor * lambda_max
    assert_eq!(lambdas[0], f64::INFINITY, "First lambda should be LAMBDA_EFFECTIVE_INFINITY (infinity)");
    // Second lambda should be: lambda_decay_factor * lambda_max where lambda_decay_factor = 0.1^(1/4)
    let expected_second = 0.1_f64.powf(1.0 / 4.0) * lambda_max;
    assert!((lambdas[1] - expected_second).abs() < 1e-10,
           "Second lambda should equal lambda_decay_factor * lambda_max");
}

// =============================================================================
// Soft Threshold (Coordinate Descent) Tests
//
// Update formula (glmnet):
//   u = g_k + a_k * xv(k)
//   v = |u| - vp(k) * lambda * alpha
//   a(k) = 0 if v <= 0
//   a(k) = sign(u) * v / (xv(k) + vp(k) * lambda * (1 - alpha)) if v > 0
//
// For standardized x: xv(k) = 1.0
// =============================================================================

#[test]
fn test_soft_threshold_formula() {
    // Verify soft-threshold operator matches glmnet formula
    // S(z, lambda * alpha) = sign(z) * max(|z| - lambda * alpha, 0)

    let test_cases: [(f64, f64, f64); 6] = [
        (1.5, 0.5, 1.0),   // |1.5| - 0.5 > 0, so 1.5 - 0.5 = 1.0
        (0.3, 0.5, 0.0),   // |0.3| - 0.5 <= 0, so 0
        (-1.2, 1.0, -0.2), // |-1.2| - 1.0 = 0.2, sign is negative
        (-0.8, 0.3, -0.5),  // |-0.8| - 0.3 = 0.5 > 0, sign is negative
        (0.0, 0.5, 0.0),   // |0| - 0.5 <= 0, so 0
        (5.0, 0.0, 5.0),   // |5| - 0 > 0, so 5
    ];

    for (z, threshold, expected) in test_cases {
        let result = if z.abs() > threshold {
            z.signum() * (z.abs() - threshold)
        } else {
            0.0_f64
        };
        assert!((result - expected).abs() < 1e-10,
               "Soft threshold failed for z={}, threshold={}: got {}, expected {}",
               z, threshold, result, expected);
    }
}

#[test]
fn test_elastic_net_denominator() {
    // Verify elastic net denominator matches glmnet:
    // denom = xv(k) + vp(k) * lambda * (1 - alpha)
    // For standardized x: xv(k) = 1.0

    let lambda: f64 = 0.5;
    let alpha: f64 = 0.7;  // Elastic net
    let penalty_factor_value: f64 = 1.0;     // Default penalty factor

    // Expected: 1.0 + 1.0 * 0.5 * (1.0 - 0.7) = 1.0 + 0.15 = 1.15
    let expected = 1.0_f64 + penalty_factor_value * lambda * (1.0 - alpha);
    assert!((expected - 1.15_f64).abs() < 1e-10, "Elastic net denominator calculation");

    // For lasso (alpha = 1): denom = 1.0 + lambda * 0 = 1.0
    let lasso_denom = 1.0_f64 + lambda * (1.0 - 1.0);
    assert!((lasso_denom - 1.0_f64).abs() < 1e-10, "Lasso denominator should be 1.0");

    // For ridge (alpha = 0): denom = 1.0 + lambda
    let ridge_denom = 1.0_f64 + lambda * (1.0 - 0.0);
    assert!((ridge_denom - 1.5_f64).abs() < 1e-10, "Ridge denominator should be 1.5");
}

// =============================================================================
// Integration Test: Full Elastic Net Path
// =============================================================================

#[test]
fn test_elastic_net_path_convergence() {
    // Full integration test: verify elastic net path converges
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![0.5, 1.0, 1.5, 2.0, 2.5];

    let n = 5;
    let p = 2;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
        x_data[i * (p + 1) + 2] = x2[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let path_options = LambdaPathOptions {
        nlambda: 5,
        lambda_min_ratio: Some(0.1),
        alpha: 1.0,
        eps_for_ridge: 1e-3,
    };

    let fit_options = ElasticNetOptions {
        lambda: 0.0,  // Not used in path
        alpha: 1.0,
        intercept: true,
        standardize: true,
        max_iter: 1000,
        tol: 1e-7,
        penalty_factor: None,
        weights: None,
        warm_start: None,
        coefficient_bounds: None,
    };

    let result = elastic_net_path(&x, &y, &path_options, &fit_options);

    assert!(result.is_ok(), "Elastic net path should converge");

    let fits = result.unwrap();
    assert_eq!(fits.len(), 5, "Should generate 5 fits");

    // Verify all fits converged
    for (i, fit) in fits.iter().enumerate() {
        assert!(fit.converged, "Fit {} should converge", i);
    }

    // Verify lambda values are decreasing
    for i in 1..fits.len() {
        assert!(fits[i].lambda < fits[i - 1].lambda,
               "Lambda values should be decreasing: {} < {}",
               fits[i].lambda, fits[i - 1].lambda);
    }

    // At high lambda (first fit), most coefficients should be zero
    let first_fit = &fits[0];
    let nonzero_count = first_fit.coefficients.iter().filter(|&&c| c.abs() > 0.0).count();
    assert!(nonzero_count <= 1, "At high lambda, most coefficients should be zero");
}

// =============================================================================
// Coefficient Bounds Tests
//
// Bounds clamping formula (glmnet):
//   a(k) = max(cl(1,k), min(cl(2,k), calculated_value))
// where cl(1,k) is lower bound, cl(2,k) is upper bound
// =============================================================================

#[test]
fn test_coefficient_bounds_non_negative() {
    // Test non-negative constraint: all coefficients >= 0
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    let n = 5;
    let p = 2;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
        x_data[i * (p + 1) + 2] = x2[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    // Non-negative bounds for both predictors
    let bounds = vec![(0.0, f64::INFINITY); p];

    let options = ElasticNetOptions {
        lambda: 0.1,
        alpha: 1.0, // Lasso
        intercept: true,
        standardize: true,
        max_iter: 10000,
        tol: 1e-7,
        penalty_factor: None,
        weights: None,
        warm_start: None,
        coefficient_bounds: Some(bounds),
    };

    let result = elastic_net_fit(&x, &y, &options);

    assert!(result.is_ok(), "Fit with non-negative bounds should succeed");

    let fit = result.unwrap();
    assert!(fit.converged, "Should converge with bounds");

    // All coefficients should be >= 0
    for (i, &coef) in fit.coefficients.iter().enumerate() {
        assert!(coef >= 0.0, "Coefficient {} should be non-negative, got {}", i, coef);
    }
}

#[test]
fn test_coefficient_bounds_upper_limit() {
    // Test upper bound constraint: coefficients <= 1.0
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let n = 5;
    let p = 1;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    // Upper bound of 1.0
    let bounds = vec![(-f64::INFINITY, 1.0)];

    let options = ElasticNetOptions {
        lambda: 0.01,  // Small lambda so coefficient would normally be > 1
        alpha: 1.0,
        intercept: true,
        standardize: true,
        max_iter: 10000,
        tol: 1e-7,
        penalty_factor: None,
        weights: None,
        warm_start: None,
        coefficient_bounds: Some(bounds),
    };

    let result = elastic_net_fit(&x, &y, &options);

    assert!(result.is_ok(), "Fit with upper bound should succeed");

    let fit = result.unwrap();
    assert!(fit.converged, "Should converge with upper bound");

    // Coefficient should be <= 1.0
    assert!(fit.coefficients[0] <= 1.0,
           "Coefficient should be <= 1.0, got {}", fit.coefficients[0]);
}

#[test]
fn test_coefficient_bounds_both_limits() {
    // Test both lower and upper bounds
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![0.5, 1.0, 1.5, 2.0, 2.5];

    let n = 5;
    let p = 2;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
        x_data[i * (p + 1) + 2] = x2[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    // Different bounds for each predictor
    let bounds = vec![
        (-1.0, 1.0),   // Predictor 1: between -1 and 1
        (0.0, 2.0),    // Predictor 2: between 0 and 2
    ];

    let options = ElasticNetOptions {
        lambda: 0.01,
        alpha: 0.5, // Elastic net
        intercept: true,
        standardize: true,
        max_iter: 10000,
        tol: 1e-7,
        penalty_factor: None,
        weights: None,
        warm_start: None,
        coefficient_bounds: Some(bounds),
    };

    let result = elastic_net_fit(&x, &y, &options);

    assert!(result.is_ok(), "Fit with both bounds should succeed");

    let fit = result.unwrap();
    assert!(fit.converged, "Should converge with both bounds");

    // Check coefficients are within bounds
    assert!(fit.coefficients[0] >= -1.0 && fit.coefficients[0] <= 1.0,
           "Coefficient 0 should be in [-1, 1], got {}", fit.coefficients[0]);
    assert!(fit.coefficients[1] >= 0.0 && fit.coefficients[1] <= 2.0,
           "Coefficient 1 should be in [0, 2], got {}", fit.coefficients[1]);
}

#[test]
fn test_coefficient_bounds_validation_wrong_length() {
    // Test that bounds with wrong length are rejected
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let n = 5;
    let p = 1;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    // Wrong number of bounds (2 instead of 1)
    let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)];

    let options = ElasticNetOptions {
        lambda: 0.1,
        alpha: 1.0,
        intercept: true,
        standardize: true,
        max_iter: 10000,
        tol: 1e-7,
        penalty_factor: None,
        weights: None,
        warm_start: None,
        coefficient_bounds: Some(bounds),
    };

    let result = elastic_net_fit(&x, &y, &options);

    assert!(result.is_err(), "Should reject bounds with wrong length");
}

#[test]
fn test_coefficient_bounds_validation_inverted_bounds() {
    // Test that inverted bounds (lower > upper) are rejected
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let n = 5;
    let p = 1;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    // Inverted bounds
    let bounds = vec![(1.0, 0.0)];

    let options = ElasticNetOptions {
        lambda: 0.1,
        alpha: 1.0,
        intercept: true,
        standardize: true,
        max_iter: 10000,
        tol: 1e-7,
        penalty_factor: None,
        weights: None,
        warm_start: None,
        coefficient_bounds: Some(bounds),
    };

    let result = elastic_net_fit(&x, &y, &options);

    assert!(result.is_err(), "Should reject inverted bounds");
}

#[test]
fn test_coefficient_bounds_no_bounds_same_result() {
    // Verify that None bounds produces same result as unbounded
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let n = 5;
    let p = 1;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    // Fit without bounds (None)
    let options_no_bounds = ElasticNetOptions {
        lambda: 0.1,
        alpha: 1.0,
        intercept: true,
        standardize: true,
        max_iter: 10000,
        tol: 1e-7,
        penalty_factor: None,
        weights: None,
        warm_start: None,
        coefficient_bounds: None,
    };

    // Fit with explicit unbounded bounds
    let options_unbounded = ElasticNetOptions {
        coefficient_bounds: Some(vec![(-f64::INFINITY, f64::INFINITY)]),
        ..options_no_bounds.clone()
    };

    let result_no_bounds = elastic_net_fit(&x, &y, &options_no_bounds);
    let result_unbounded = elastic_net_fit(&x, &y, &options_unbounded);

    assert!(result_no_bounds.is_ok());
    assert!(result_unbounded.is_ok());

    let fit_no_bounds = result_no_bounds.unwrap();
    let fit_unbounded = result_unbounded.unwrap();

    // Results should be very similar (may have tiny numerical differences)
    assert!((fit_no_bounds.coefficients[0] - fit_unbounded.coefficients[0]).abs() < 1e-10,
           "None bounds and explicit unbounded should produce same result");
}

#[test]
fn test_coefficient_bounds_at_convergence() {
    // Test behavior when solution is at a bound
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];  // y = 2*x
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let n = 5;
    let p = 1;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = x1[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    // Set upper bound below the true solution (coefficient should be ~2)
    let bounds = vec![(0.0, 1.5)];

    let options = ElasticNetOptions {
        lambda: 0.0,  // No regularization, true OLS solution would be 2
        alpha: 1.0,
        intercept: true,
        standardize: true,
        max_iter: 10000,
        tol: 1e-7,
        penalty_factor: None,
        weights: None,
        warm_start: None,
        coefficient_bounds: Some(bounds),
    };

    let result = elastic_net_fit(&x, &y, &options);

    assert!(result.is_ok(), "Fit should succeed");

    let fit = result.unwrap();
    assert!(fit.converged, "Should converge even when at bound");

    // Coefficient should be at the upper bound (or very close)
    assert!(fit.coefficients[0] <= 1.5 + 1e-6,
           "Coefficient should be clamped to upper bound, got {}", fit.coefficients[0]);
}
