// ============================================================================
// Polynomial Regression Unit Tests
// ============================================================================

use linreg_core::polynomial::{polynomial_regression, predict, PolynomialOptions};
use linreg_core::error::Error;

// ============================================================================
// Test Helpers
// ============================================================================

fn assert_close(a: f64, b: f64, tol: f64, ctx: &str) {
    let diff = (a - b).abs();
    assert!(
        diff <= tol,
        "{}: {} != {}, diff = {} (tolerance = {})",
        ctx,
        a,
        b,
        diff,
        tol
    );
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[test]
fn test_polynomial_degree_1_is_linear() {
    // Degree 1 should match simple linear regression: y = 1 + 2x
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi).collect();

    let options = PolynomialOptions {
        degree: 1,
        center: false,
        standardize: false,
        intercept: true,
    };

    let fit = polynomial_regression(&y, &x, &options).unwrap();

    assert!(fit.ols_output.r_squared > 0.9999, "R² should be near 1");
    assert_close(fit.ols_output.coefficients[0], 1.0, 1e-6, "intercept");
    assert_close(fit.ols_output.coefficients[1], 2.0, 1e-6, "slope");
    assert_eq!(fit.degree, 1);
    assert_eq!(fit.n_features, 1);
    assert_eq!(fit.ols_output.coefficients.len(), 2);
}

#[test]
fn test_polynomial_degree_2_perfect_fit() {
    // Perfect quadratic: y = 1 + 2x + x²
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let options = PolynomialOptions::default(); // degree = 2
    let fit = polynomial_regression(&y, &x, &options).unwrap();

    assert_close(fit.ols_output.r_squared, 1.0, 1e-10, "R²");
    assert_close(fit.ols_output.coefficients[0], 1.0, 1e-6, "intercept");
    assert_close(fit.ols_output.coefficients[1], 2.0, 1e-6, "x coeff");
    assert_close(fit.ols_output.coefficients[2], 1.0, 1e-6, "x² coeff");
    assert_eq!(fit.ols_output.coefficients.len(), 3);
}

#[test]
fn test_polynomial_degree_3_cubic_fit() {
    // y = 5 + 3x - 2x² + 0.5x³
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 5.0 + 3.0 * xi - 2.0 * xi * xi + 0.5 * xi * xi * xi)
        .collect();

    let options = PolynomialOptions {
        degree: 3,
        ..Default::default()
    };
    let fit = polynomial_regression(&y, &x, &options).unwrap();

    assert_close(fit.ols_output.r_squared, 1.0, 1e-9, "R²");
    assert_eq!(fit.ols_output.coefficients.len(), 4, "4 coefficients");
    assert_eq!(fit.degree, 3);
    assert_eq!(fit.n_features, 3);
}

#[test]
fn test_polynomial_feature_names_uncentered() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = x.clone();

    let options = PolynomialOptions {
        degree: 3,
        center: false,
        ..Default::default()
    };
    let fit = polynomial_regression(&y, &x, &options).unwrap();

    // feature_names: Intercept, x, x^2, x^3
    assert_eq!(fit.feature_names[0], "Intercept");
    assert_eq!(fit.feature_names[1], "x");
    assert_eq!(fit.feature_names[2], "x^2");
    assert_eq!(fit.feature_names[3], "x^3");
}

#[test]
fn test_polynomial_feature_names_centered() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = x.clone();

    let options = PolynomialOptions {
        degree: 2,
        center: true,
        ..Default::default()
    };
    let fit = polynomial_regression(&y, &x, &options).unwrap();

    assert_eq!(fit.feature_names[0], "Intercept");
    assert_eq!(fit.feature_names[1], "x_centered");
    assert_eq!(fit.feature_names[2], "x^2_centered");
}

#[test]
fn test_polynomial_centering_stores_x_mean() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = x.clone();

    let options = PolynomialOptions {
        degree: 2,
        center: true,
        ..Default::default()
    };
    let fit = polynomial_regression(&y, &x, &options).unwrap();

    assert!(fit.centered);
    assert_close(fit.x_mean, 3.0, 1e-10, "x_mean");
}

#[test]
fn test_polynomial_no_centering_stores_zero_mean() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = x.clone();

    let options = PolynomialOptions {
        degree: 2,
        center: false,
        ..Default::default()
    };
    let fit = polynomial_regression(&y, &x, &options).unwrap();

    assert!(!fit.centered);
    assert_close(fit.x_mean, 0.0, 1e-15, "x_mean should be 0.0");
}

#[test]
fn test_polynomial_standardize_stores_stats() {
    let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

    let options = PolynomialOptions {
        degree: 2,
        center: false,
        standardize: true,
        intercept: true,
    };
    let fit = polynomial_regression(&y, &x, &options).unwrap();

    assert!(fit.standardized);
    // Should have stored 2 means and 2 stds (one per polynomial term)
    assert_eq!(fit.feature_means.len(), 2);
    assert_eq!(fit.feature_stds.len(), 2);
    // All stds should be positive
    for std_val in &fit.feature_stds {
        assert!(*std_val > 0.0, "std must be positive");
    }
}

// ============================================================================
// Centering Reduces Multicollinearity
// ============================================================================

#[test]
fn test_polynomial_centering_reduces_max_vif() {
    // For degree 3 on a well-spaced x, centering should reduce VIF
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + xi + 0.5 * xi * xi).collect();

    let opts_no_center = PolynomialOptions {
        degree: 3,
        center: false,
        ..Default::default()
    };
    let fit_no_center = polynomial_regression(&y, &x, &opts_no_center).unwrap();

    let opts_centered = PolynomialOptions {
        degree: 3,
        center: true,
        ..Default::default()
    };
    let fit_centered = polynomial_regression(&y, &x, &opts_centered).unwrap();

    // Both should fit well
    assert!(fit_no_center.ols_output.r_squared > 0.95);
    assert!(fit_centered.ols_output.r_squared > 0.95);

    // Centered fit should have lower maximum VIF
    let max_vif_no_center = fit_no_center
        .ols_output
        .vif
        .iter()
        .map(|v| v.vif)
        .fold(0.0f64, f64::max);

    let max_vif_centered = fit_centered
        .ols_output
        .vif
        .iter()
        .map(|v| v.vif)
        .fold(0.0f64, f64::max);

    assert!(
        max_vif_centered < max_vif_no_center,
        "Centering should reduce VIF: centered={} vs uncentered={}",
        max_vif_centered,
        max_vif_no_center
    );
}

// ============================================================================
// Prediction Tests
// ============================================================================

#[test]
fn test_polynomial_predict_quadratic() {
    // y = x² → train on 1..5, predict at 6 and 7
    let x_train = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_train: Vec<f64> = x_train.iter().map(|&xi| xi * xi).collect();

    let options = PolynomialOptions {
        degree: 2,
        ..Default::default()
    };
    let fit = polynomial_regression(&y_train, &x_train, &options).unwrap();

    let preds = predict(&fit, &[6.0, 7.0]).unwrap();
    assert_eq!(preds.len(), 2);
    assert_close(preds[0], 36.0, 0.1, "pred at x=6");
    assert_close(preds[1], 49.0, 0.1, "pred at x=7");
}

#[test]
fn test_polynomial_predict_empty_x_returns_empty() {
    let x_train = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_train: Vec<f64> = x_train.iter().map(|&xi| xi * xi).collect();

    let options = PolynomialOptions::default();
    let fit = polynomial_regression(&y_train, &x_train, &options).unwrap();

    let preds = predict(&fit, &[]).unwrap();
    assert!(preds.is_empty());
}

#[test]
fn test_polynomial_predict_centered() {
    // Even with centering, prediction at training points should closely match y
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let options = PolynomialOptions {
        degree: 2,
        center: true,
        ..Default::default()
    };
    let fit = polynomial_regression(&y, &x, &options).unwrap();

    let preds = predict(&fit, &x).unwrap();
    for (i, (&actual, &pred)) in y.iter().zip(preds.iter()).enumerate() {
        assert_close(pred, actual, 1e-6, &format!("pred[{}]", i));
    }
}

#[test]
fn test_polynomial_predict_standardized() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

    let options = PolynomialOptions {
        degree: 2,
        center: false,
        standardize: true,
        intercept: true,
    };
    let fit = polynomial_regression(&y, &x, &options).unwrap();

    // Predictions at training points should still match
    let preds = predict(&fit, &x).unwrap();
    for (i, (&actual, &pred)) in y.iter().zip(preds.iter()).enumerate() {
        assert_close(pred, actual, 1e-6, &format!("pred[{}] standardized", i));
    }
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_polynomial_degree_zero_returns_error() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    let options = PolynomialOptions {
        degree: 0,
        ..Default::default()
    };
    let result = polynomial_regression(&y, &x, &options);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), Error::InvalidInput(_)));
}

#[test]
fn test_polynomial_mismatched_lengths_return_error() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0]; // Different length
    let options = PolynomialOptions::default();
    let result = polynomial_regression(&y, &x, &options);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), Error::DimensionMismatch(_)));
}

#[test]
fn test_polynomial_insufficient_data_returns_error() {
    let x = vec![1.0];
    let y = vec![2.0];
    let options = PolynomialOptions::default();
    let result = polynomial_regression(&y, &x, &options);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        Error::InsufficientData { .. }
    ));
}

#[test]
fn test_polynomial_nan_x_returns_error() {
    let x = vec![1.0, f64::NAN, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    let options = PolynomialOptions::default();
    let result = polynomial_regression(&y, &x, &options);
    assert!(result.is_err());
}

#[test]
fn test_polynomial_inf_x_returns_error() {
    let x = vec![1.0, f64::INFINITY, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    let options = PolynomialOptions::default();
    let result = polynomial_regression(&y, &x, &options);
    assert!(result.is_err());
}

// ============================================================================
// Statistics Tests
// ============================================================================

#[test]
fn test_polynomial_r_squared_improves_with_correct_degree() {
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    // True relationship is quadratic
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 + xi + 0.3 * xi * xi).collect();

    let fit1 = polynomial_regression(
        &y,
        &x,
        &PolynomialOptions {
            degree: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let fit2 = polynomial_regression(
        &y,
        &x,
        &PolynomialOptions {
            degree: 2,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(
        fit2.ols_output.r_squared > fit1.ols_output.r_squared,
        "Degree 2 should fit better for quadratic data"
    );
    assert_close(fit2.ols_output.r_squared, 1.0, 1e-9, "Degree 2 R²");
}

#[test]
fn test_polynomial_coefficient_count() {
    for degree in 1usize..=5 {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y = x.clone();

        let options = PolynomialOptions {
            degree,
            ..Default::default()
        };
        let fit = polynomial_regression(&y, &x, &options).unwrap();

        assert_eq!(
            fit.ols_output.coefficients.len(),
            degree + 1,
            "degree {} should produce {} coefficients",
            degree,
            degree + 1
        );
    }
}

#[test]
fn test_polynomial_fitted_values_match_y_for_perfect_fit() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let options = PolynomialOptions::default();
    let fit = polynomial_regression(&y, &x, &options).unwrap();

    for (i, (&actual, &fitted)) in y.iter().zip(fit.ols_output.predictions.iter()).enumerate() {
        assert_close(fitted, actual, 1e-8, &format!("fitted[{}]", i));
    }
}

// ============================================================================
// Regularized Polynomial Tests
// ============================================================================

#[test]
fn test_polynomial_ridge_basic() {
    use linreg_core::polynomial::polynomial_ridge;

    let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let fit = polynomial_ridge(&y, &x, 2, 0.1, false, true).unwrap();
    // Ridge shrinks but should still have positive R²
    assert!(fit.r_squared > 0.9, "Ridge polynomial R² = {}", fit.r_squared);
    // Should have 2 slope coefficients (x and x²)
    assert_eq!(fit.coefficients.len(), 2);
}

#[test]
fn test_polynomial_lasso_basic() {
    use linreg_core::polynomial::polynomial_lasso;

    let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let fit = polynomial_lasso(&y, &x, 2, 0.01, false, true).unwrap();
    assert!(fit.r_squared > 0.9, "Lasso polynomial R² = {}", fit.r_squared);
    assert_eq!(fit.coefficients.len(), 2);
}

#[test]
fn test_polynomial_elastic_net_basic() {
    use linreg_core::polynomial::polynomial_elastic_net;

    let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let fit = polynomial_elastic_net(&y, &x, 2, 0.05, 0.5, false, true).unwrap();
    assert!(fit.r_squared > 0.9, "Elastic net polynomial R² = {}", fit.r_squared);
    assert_eq!(fit.coefficients.len(), 2);
}

// ============================================================================
// Default Options Test
// ============================================================================

#[test]
fn test_polynomial_options_default() {
    let opts = PolynomialOptions::default();
    assert_eq!(opts.degree, 2);
    assert!(!opts.center);
    assert!(!opts.standardize);
    assert!(opts.intercept);
}
