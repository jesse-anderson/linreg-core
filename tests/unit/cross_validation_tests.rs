// ============================================================================
// Cross Validation Unit Tests
// ============================================================================
//
// Comprehensive tests for K-Fold Cross Validation including basic functionality,
// different regression types, edge cases, and validation.

use linreg_core::cross_validation::{
    kfold_cv_elastic_net, kfold_cv_lasso, kfold_cv_ols, kfold_cv_ridge, KFoldOptions,
};
use linreg_core::Error;

// ============================================================================
// Test Constants and Helpers
// ============================================================================

const CV_TOLERANCE: f64 = 1e-4;

/// Helper function to assert two f64 values are close within tolerance
fn assert_close(a: f64, b: f64, tolerance: f64, context: &str) {
    let diff = (a - b).abs();
    assert!(
        diff <= tolerance,
        "{}: {} != {}, diff = {} (tolerance = {})",
        context, a, b, diff, tolerance
    );
}

// ============================================================================
// KFoldOptions Tests
// ============================================================================

#[test]
fn test_kfold_options_default() {
    let options = KFoldOptions::default();
    assert_eq!(options.n_folds, 5);
    assert_eq!(options.shuffle, false);
    assert_eq!(options.seed, None);
}

#[test]
fn test_kfold_options_builder() {
    let options = KFoldOptions::new(10).with_shuffle(true).with_seed(42);
    assert_eq!(options.n_folds, 10);
    assert_eq!(options.shuffle, true);
    assert_eq!(options.seed, Some(42));
}

// ============================================================================
// OLS Cross Validation Tests
// ============================================================================

#[test]
fn test_kfold_cv_ols_basic() {
    // Simple linear relationship: y = 2*x + 1
    let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let names = vec!["Intercept".into(), "X1".into()];

    let options = KFoldOptions::new(5);
    let result = kfold_cv_ols(&y, &[x1], &names, &options);

    assert!(result.is_ok());
    let cv = result.unwrap();

    assert_eq!(cv.n_folds, 5);
    assert_eq!(cv.n_samples, 10);
    assert_eq!(cv.fold_results.len(), 5);
    assert_eq!(cv.fold_coefficients.len(), 5);

    // R² should be high (good fit)
    assert!(cv.mean_r_squared > 0.9);
}

#[test]
fn test_kfold_cv_ols_multiple_predictors() {
    // y = 1 + 2*x1 + 3*x2
    let y = vec![6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x2 = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
    let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

    let options = KFoldOptions::new(4);
    let result = kfold_cv_ols(&y, &[x1, x2], &names, &options);

    assert!(result.is_ok());
    let cv = result.unwrap();

    assert_eq!(cv.n_folds, 4);
    assert_eq!(cv.fold_results.len(), 4);

    // Check that each fold has correct structure
    for fold in &cv.fold_results {
        assert!(fold.mse >= 0.0);
        assert!(fold.rmse >= 0.0);
        assert!(fold.mae >= 0.0);
        assert!(fold.train_size > 0);
        assert!(fold.test_size > 0);
    }
}

#[test]
fn test_kfold_cv_ols_with_shuffle() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let names = vec!["Intercept".into(), "X1".into()];

    let options = KFoldOptions::new(4).with_shuffle(true).with_seed(42);
    let result = kfold_cv_ols(&y, &[x1], &names, &options);

    assert!(result.is_ok());
    let cv = result.unwrap();

    assert_eq!(cv.n_folds, 4);
    assert_eq!(cv.fold_results.len(), 4);
}

#[test]
fn test_kfold_cv_ols_reproducible_shuffle() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let names = vec!["Intercept".into(), "X1".into()];

    let options = KFoldOptions::new(4).with_shuffle(true).with_seed(42);

    let result1 = kfold_cv_ols(&y, &[x1.clone()], &names, &options).unwrap();
    let result2 = kfold_cv_ols(&y, &[x1], &names, &options).unwrap();

    // Same seed should give identical results
    assert_eq!(result1.mean_mse, result2.mean_mse);
    assert_eq!(result1.mean_rmse, result2.mean_rmse);
    assert_eq!(result1.mean_r_squared, result2.mean_r_squared);
}

#[test]
fn test_kfold_cv_ols_insufficient_samples() {
    let y = vec![1.0, 2.0];
    let x1 = vec![1.0, 2.0];
    let names = vec!["Intercept".into(), "X1".into()];

    let options = KFoldOptions::new(5); // More folds than samples
    let result = kfold_cv_ols(&y, &[x1], &names, &options);

    assert!(result.is_err());
    match result {
        Err(Error::InsufficientData { required, available }) => {
            assert_eq!(required, 5);
            assert_eq!(available, 2);
        }
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_kfold_cv_ols_invalid_folds() {
    let y = vec![1.0, 2.0, 3.0];
    let x1 = vec![1.0, 2.0, 3.0];
    let names = vec!["Intercept".into(), "X1".into()];

    let options = KFoldOptions::new(1); // n_folds must be >= 2
    let result = kfold_cv_ols(&y, &[x1], &names, &options);

    assert!(result.is_err());
}

#[test]
fn test_kfold_cv_ols_n_folds_equals_n_samples() {
    // Leave-one-out cross-validation
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let names = vec!["Intercept".into(), "X1".into()];

    let options = KFoldOptions::new(5);
    let result = kfold_cv_ols(&y, &[x1], &names, &options);

    assert!(result.is_ok());
    let cv = result.unwrap();

    assert_eq!(cv.n_folds, 5);
    // Each fold should have 1 test sample
    for fold in &cv.fold_results {
        assert_eq!(fold.test_size, 1);
        assert_eq!(fold.train_size, 4);
    }
}

#[test]
fn test_kfold_cv_ols_coefficient_tracking() {
    let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let names = vec!["Intercept".into(), "X1".into()];

    let options = KFoldOptions::new(5);
    let result = kfold_cv_ols(&y, &[x1], &names, &options).unwrap();

    // Each fold should have 2 coefficients (intercept + slope)
    for coeffs in &result.fold_coefficients {
        assert_eq!(coeffs.len(), 2);
    }

    // Coefficients should be relatively stable (low variance)
    // Intercept ~1, slope ~2
    let intercepts: Vec<f64> = result.fold_coefficients.iter().map(|c| c[0]).collect();
    let slopes: Vec<f64> = result.fold_coefficients.iter().map(|c| c[1]).collect();

    let mean_intercept = intercepts.iter().sum::<f64>() / intercepts.len() as f64;
    let mean_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;

    assert_close(mean_intercept, 1.0, 0.5, "Intercept should be near 1.0");
    assert_close(mean_slope, 2.0, 0.5, "Slope should be near 2.0");
}

// ============================================================================
// Ridge Cross Validation Tests
// ============================================================================

#[test]
fn test_kfold_cv_ridge_basic() {
    let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let options = KFoldOptions::new(4);
    let result = kfold_cv_ridge(&[x1, x2], &y, 0.1, true, &options);

    assert!(result.is_ok());
    let cv = result.unwrap();

    assert_eq!(cv.n_folds, 4);
    assert_eq!(cv.fold_results.len(), 4);
    assert!(cv.mean_r_squared > 0.0);
}

#[test]
fn test_kfold_cv_ridge_invalid_lambda() {
    let y = vec![1.0, 2.0, 3.0];
    let x1 = vec![1.0, 2.0, 3.0];

    let options = KFoldOptions::new(2);
    let result = kfold_cv_ridge(&[x1], &y, -1.0, true, &options);

    assert!(result.is_err());
}

// ============================================================================
// Lasso Cross Validation Tests
// ============================================================================

#[test]
fn test_kfold_cv_lasso_basic() {
    let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let options = KFoldOptions::new(4);
    let result = kfold_cv_lasso(&[x1, x2], &y, 0.1, true, &options);

    assert!(result.is_ok());
    let cv = result.unwrap();

    assert_eq!(cv.n_folds, 4);
    assert_eq!(cv.fold_results.len(), 4);
}

#[test]
fn test_kfold_cv_lasso_invalid_lambda() {
    let y = vec![1.0, 2.0, 3.0];
    let x1 = vec![1.0, 2.0, 3.0];

    let options = KFoldOptions::new(2);
    let result = kfold_cv_lasso(&[x1], &y, -1.0, true, &options);

    assert!(result.is_err());
}

// ============================================================================
// Elastic Net Cross Validation Tests
// ============================================================================

#[test]
fn test_kfold_cv_elastic_net_basic() {
    let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let options = KFoldOptions::new(4);
    let result = kfold_cv_elastic_net(&[x1, x2], &y, 0.1, 0.5, true, &options);

    assert!(result.is_ok());
    let cv = result.unwrap();

    assert_eq!(cv.n_folds, 4);
    assert_eq!(cv.fold_results.len(), 4);
}

#[test]
fn test_kfold_cv_elastic_net_invalid_alpha() {
    let y = vec![1.0, 2.0, 3.0];
    let x1 = vec![1.0, 2.0, 3.0];

    let options = KFoldOptions::new(2);
    let result = kfold_cv_elastic_net(&[x1], &y, 0.1, 1.5, true, &options);

    assert!(result.is_err());
}

#[test]
fn test_kfold_cv_elastic_net_invalid_lambda() {
    let y = vec![1.0, 2.0, 3.0];
    let x1 = vec![1.0, 2.0, 3.0];

    let options = KFoldOptions::new(2);
    let result = kfold_cv_elastic_net(&[x1], &y, -1.0, 0.5, true, &options);

    assert!(result.is_err());
}

// ============================================================================
// Edge Cases and Integration Tests
// ============================================================================

#[test]
fn test_kfold_cv_small_dataset() {
    // With 3 folds and 2 params (intercept + 1 predictor),
    // each training set needs at least 3 samples.
    // Using 6 samples gives 2 per fold, so train sets have 4 samples each.
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let names = vec!["Intercept".into(), "X1".into()];

    let options = KFoldOptions::new(3);
    let result = kfold_cv_ols(&y, &[x1], &names, &options);

    assert!(result.is_ok(), "kfold_cv_ols failed: {:?}", result);
    let cv = result.unwrap();

    assert_eq!(cv.n_folds, 3);
    for fold in &cv.fold_results {
        assert!(fold.train_size >= 3);
    }
}

#[test]
fn test_kfold_cv_all_methods_consistent() {
    // Test that all CV methods work with the same data
    let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // Use non-collinear predictor
    let x2 = vec![1.0, 3.0, 2.0, 4.0, 1.5, 3.5, 2.5, 4.5];
    let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

    let options = KFoldOptions::new(4);

    // OLS
    let ols_result = kfold_cv_ols(&y, &[x1.clone(), x2.clone()], &names, &options);
    assert!(ols_result.is_ok(), "OLS failed: {:?}", ols_result);

    // Ridge
    let ridge_result = kfold_cv_ridge(&[x1.clone(), x2.clone()], &y, 0.1, true, &options);
    assert!(ridge_result.is_ok());

    // Lasso
    let lasso_result = kfold_cv_lasso(&[x1.clone(), x2.clone()], &y, 0.1, true, &options);
    assert!(lasso_result.is_ok());

    // Elastic Net
    let enet_result =
        kfold_cv_elastic_net(&[x1, x2], &y, 0.1, 0.5, true, &options);
    assert!(enet_result.is_ok());

    // All should have the same number of folds
    let ols_cv = ols_result.unwrap();
    let ridge_cv = ridge_result.unwrap();
    let lasso_cv = lasso_result.unwrap();
    let enet_cv = enet_result.unwrap();

    assert_eq!(ols_cv.n_folds, ridge_cv.n_folds);
    assert_eq!(ols_cv.n_folds, lasso_cv.n_folds);
    assert_eq!(ols_cv.n_folds, enet_cv.n_folds);
}
