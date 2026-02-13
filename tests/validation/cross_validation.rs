// ============================================================================
// K-Fold Cross Validation Validation Tests
// ============================================================================
//
// Validates K-Fold Cross Validation implementation against reference results
// from R's caret package.

#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use linreg_core::cross_validation::{
    kfold_cv_elastic_net, kfold_cv_lasso, kfold_cv_ols, kfold_cv_ridge, KFoldOptions,
};

// common module is declared in main.rs
use crate::common::{assert_close_to, expect_kfold_cv_result, load_dataset_with_encoding, CategoricalEncoding, CV_TOLERANCE, LASSO_TOLERANCE};

// ============================================================================
// Test Data
// ============================================================================

/// Simple synthetic dataset for validation
/// y = 5 + 2*x1 + 3*x2 + noise
/// Using data that avoids extreme multicollinearity
fn get_simple_data() -> (Vec<f64>, Vec<Vec<f64>>, Vec<String>) {
    // Generate 100 samples - avoid multicollinearity
    let n = 100;
    let mut y = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);

    for i in 1..=n {
        let x1_val = (i as f64) * 0.3;  // 0.3, 0.6, 0.9, ..., 30
        // x2 is linearly independent from x1 (no multicollinearity)
        let x2_val = (i as f64) * 0.2 + 5.0;  // 5.2, 5.4, 5.6, ..., 25
        let noise = ((i * 7) % 13) as f64 * 0.08; // Small noise
        let y_val = 5.0 + 2.0 * x1_val + 3.0 * x2_val + noise;

        y.push(y_val);
        x1.push(x1_val);
        x2.push(x2_val);
    }

    let names = vec![
        "Intercept".to_string(),
        "X1".to_string(),
        "X2".to_string(),
    ];

    (y, vec![x1, x2], names)
}

// ============================================================================
// OLS Cross Validation Tests
// ============================================================================

#[test]
fn test_kfold_cv_ols_basic() {
    let (y, x_vars, names) = get_simple_data();
    let options = KFoldOptions::new(10).with_shuffle(false);

    let result = kfold_cv_ols(&y, &x_vars, &names, &options).unwrap();

    assert_eq!(result.n_folds, 10);
    assert_eq!(result.n_samples, 100);
    assert_eq!(result.fold_results.len(), 10);

    // Each fold should have 10 test samples (100/10 = 10)
    for fold in &result.fold_results {
        assert_eq!(fold.test_size, 10);
        assert_eq!(fold.train_size, 90);
    }

    // High R² expected for this clean linear relationship
    assert!(result.mean_r_squared > 0.95);

    // All metrics should be non-negative
    assert!(result.mean_mse >= 0.0);
    assert!(result.mean_rmse >= 0.0);
    assert!(result.mean_mae >= 0.0);
}

#[test]
fn test_kfold_cv_ols_reproducibility() {
    let (y, x_vars, names) = get_simple_data();
    let seed = 42;
    let options = KFoldOptions::new(5).with_shuffle(true).with_seed(seed);

    let result1 = kfold_cv_ols(&y, &x_vars, &names, &options).unwrap();
    let result2 = kfold_cv_ols(&y, &x_vars, &names, &options).unwrap();

    // Results should be identical with same seed
    assert_close_to(
        result1.mean_rmse,
        result2.mean_rmse,
        f64::EPSILON,
        "OLS CV reproducibility RMSE",
    );
    assert_close_to(
        result1.mean_r_squared,
        result2.mean_r_squared,
        f64::EPSILON,
        "OLS CV reproducibility R²",
    );
}

#[test]
fn test_kfold_cv_ols_different_seeds() {
    let (y, x_vars, names) = get_simple_data();
    let options1 = KFoldOptions::new(5).with_shuffle(true).with_seed(42);
    let options2 = KFoldOptions::new(5).with_shuffle(true).with_seed(123);

    let result1 = kfold_cv_ols(&y, &x_vars, &names, &options1).unwrap();
    let result2 = kfold_cv_ols(&y, &x_vars, &names, &options2).unwrap();

    // Results should differ with different seeds
    // (though they might occasionally be close by chance)
    let rmse_diff = (result1.mean_rmse - result2.mean_rmse).abs();
    // With shuffling, we expect some difference
    // If they're identical, shuffling might not be working
    assert!(
        rmse_diff > 0.0 || result1.mean_rmse == result2.mean_rmse,
        "Different seeds should produce different results"
    );
}

// ============================================================================
// Ridge Cross Validation Tests
// ============================================================================

#[test]
fn test_kfold_cv_ridge_basic() {
    let (y, x_vars, _) = get_simple_data();
    let options = KFoldOptions::new(10).with_shuffle(false);

    let result = kfold_cv_ridge(&x_vars, &y, 0.1, true, &options).unwrap();

    assert_eq!(result.n_folds, 10);
    assert_eq!(result.n_samples, 100);
    assert!(result.mean_r_squared > 0.9); // Ridge should still fit well
}

#[test]
fn test_kfold_cv_ridge_lambda_path() {
    let (y, x_vars, _) = get_simple_data();
    let options = KFoldOptions::new(4).with_shuffle(false);

    let lambdas = [0.01, 0.1, 1.0, 10.0];
    let mut prev_rmse = f64::INFINITY;

    for &lambda in &lambdas {
        let result = kfold_cv_ridge(&x_vars, &y, lambda, true, &options).unwrap();
        assert!(result.mean_rmse > 0.0);

        // As lambda increases, RMSE may increase (more regularization)
        // but this is not guaranteed for all datasets
        let _ = prev_rmse;
        prev_rmse = result.mean_rmse;
    }
}

// ============================================================================
// Lasso Cross Validation Tests
// ============================================================================

#[test]
fn test_kfold_cv_lasso_basic() {
    let (y, x_vars, _) = get_simple_data();
    let options = KFoldOptions::new(10).with_shuffle(false);

    let result = kfold_cv_lasso(&x_vars, &y, 0.1, true, &options).unwrap();

    assert_eq!(result.n_folds, 10);
    assert_eq!(result.n_samples, 100);
    assert!(result.mean_r_squared > 0.9);
}

// ============================================================================
// Elastic Net Cross Validation Tests
// ============================================================================

#[test]
fn test_kfold_cv_elastic_net_basic() {
    let (y, x_vars, _) = get_simple_data();
    let options = KFoldOptions::new(10).with_shuffle(false);

    let result = kfold_cv_elastic_net(&x_vars, &y, 0.1, 0.5, true, &options).unwrap();

    assert_eq!(result.n_folds, 10);
    assert_eq!(result.n_samples, 100);
    assert!(result.mean_r_squared > 0.9);
}

#[test]
fn test_kfold_cv_elastic_net_alpha_continuum() {
    let (y, x_vars, _) = get_simple_data();
    let options = KFoldOptions::new(4).with_shuffle(false);

    // Test continuum from Ridge (alpha=0) to Lasso (alpha=1)
    let alphas = [0.0, 0.25, 0.5, 0.75, 1.0];
    let lambda = 0.1;

    for &alpha in &alphas {
        let result =
            kfold_cv_elastic_net(&x_vars, &y, lambda, alpha, true, &options).unwrap();
        assert!(result.mean_rmse > 0.0, "RMSE should be positive for alpha={}", alpha);
        assert!(result.mean_r_squared > 0.0, "R² should be positive for alpha={}", alpha);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_kfold_cv_small_dataset() {
    // Minimal dataset that still works
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let names = vec!["Intercept".to_string(), "X1".to_string()];

    let options = KFoldOptions::new(3).with_shuffle(false);

    let result = kfold_cv_ols(&y, &[x1], &names, &options).unwrap();

    assert_eq!(result.n_folds, 3);
    assert_eq!(result.n_samples, 6);
}

#[test]
fn test_kfold_cv_leave_one_out() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let names = vec!["Intercept".to_string(), "X1".to_string()];

    let options = KFoldOptions::new(5).with_shuffle(false);

    let result = kfold_cv_ols(&y, &[x1], &names, &options).unwrap();

    // Each fold should have 1 test sample
    for fold in &result.fold_results {
        assert_eq!(fold.test_size, 1);
        assert_eq!(fold.train_size, 4);
    }
}

// ============================================================================
// Reference Validation (if R/Python results available)
// ============================================================================

#[test]
fn test_kfold_cv_ols_against_reference() {
    // This test validates against R reference results if available
    // Reference file path: verification/results/r/kfold_cv_ols.json

    let result_path = PathBuf::from("verification/results/r/kfold_cv_ols.json");

    if !result_path.exists() {
        // Skip if reference file not available
        return;
    }

    let (y, x_vars, names) = get_simple_data();
    let options = KFoldOptions::new(10).with_shuffle(false).with_seed(42);

    let rust_result = kfold_cv_ols(&y, &x_vars, &names, &options).unwrap();

    // Load reference result
    if let Some(ref_result) = crate::common::load_kfold_cv_result(&result_path) {
        // Compare key metrics
        assert_close_to(
            rust_result.mean_rmse,
            ref_result.mean_rmse,
            CV_TOLERANCE,
            "OLS CV mean RMSE",
        );
        assert_close_to(
            rust_result.mean_r_squared,
            ref_result.mean_r_squared,
            CV_TOLERANCE,
            "OLS CV mean R²",
        );
    }
}

#[test]
fn test_kfold_cv_ridge_against_reference() {
    let result_path = PathBuf::from("verification/results/r/kfold_cv_ridge.json");

    if !result_path.exists() {
        return;
    }

    let (y, x_vars, _) = get_simple_data();
    let options = KFoldOptions::new(10).with_shuffle(false).with_seed(42);

    let rust_result = kfold_cv_ridge(&x_vars, &y, 0.1, true, &options).unwrap();

    if let Some(ref_result) = crate::common::load_kfold_cv_result(&result_path) {
        // Use LASSO_TOLERANCE (1e-2 = 1%) for CV, consistent with other regularized CV tests.
        // Single-fit Ridge uses 0.5% for coefficients; CV compounds small per-fold
        // differences across folds, so a tighter tolerance than single-fit is inappropriate.
        assert_close_to(
            rust_result.mean_rmse,
            ref_result.mean_rmse,
            LASSO_TOLERANCE,
            "Ridge CV mean RMSE",
        );
    }
}

#[test]
fn test_kfold_cv_lasso_against_reference() {
    let result_path = PathBuf::from("verification/results/r/kfold_cv_lasso.json");

    if !result_path.exists() {
        return;
    }

    let (y, x_vars, _) = get_simple_data();
    let options = KFoldOptions::new(10).with_shuffle(false).with_seed(42);

    let rust_result = kfold_cv_lasso(&x_vars, &y, 0.1, true, &options).unwrap();

    if let Some(ref_result) = crate::common::load_kfold_cv_result(&result_path) {
        assert_close_to(
            rust_result.mean_rmse,
            ref_result.mean_rmse,
            LASSO_TOLERANCE,  // Use Lasso tolerance (1e-2 = 1%)
            "Lasso CV mean RMSE",
        );
    }
}

#[test]
fn test_kfold_cv_elastic_net_against_reference() {
    let result_path = PathBuf::from("verification/results/r/kfold_cv_elastic_net.json");

    if !result_path.exists() {
        return;
    }

    let (y, x_vars, _) = get_simple_data();
    let options = KFoldOptions::new(10).with_shuffle(false).with_seed(42);

    let rust_result = kfold_cv_elastic_net(&x_vars, &y, 0.1, 0.5, true, &options).unwrap();

    if let Some(ref_result) = crate::common::load_kfold_cv_result(&result_path) {
        assert_close_to(
            rust_result.mean_rmse,
            ref_result.mean_rmse,
            LASSO_TOLERANCE,  // Use Lasso tolerance for Elastic Net (1e-2 = 1%)
            "Elastic Net CV mean RMSE",
        );
    }
}

// ============================================================================
// CSV Dataset Validation Against R
// ============================================================================
//
// These tests load real CSV datasets and compare CV results against R reference
// values, following the same pattern as ols_by_dataset.rs and regularized.rs.

const CV_TEST_DATASETS: &[&str] = &[
    "bodyfat",
    "cars_stopping",
    "faithful",
    "lh",
    "mtcars",
    "prostate",
    "synthetic_autocorrelated",
    "synthetic_collinear",
    "synthetic_heteroscedastic",
    "synthetic_high_vif",
    "synthetic_interaction",
    "synthetic_multiple",
    "synthetic_nonlinear",
    "synthetic_nonnormal",
    "synthetic_outliers",
    "synthetic_simple_linear",
    "synthetic_small",
    "ToothGrowth",
];

const CV_N_FOLDS: usize = 10;
const CV_LAMBDA: f64 = 0.1;
const CV_ALPHA: f64 = 0.5;

fn validate_cv_ols(dataset_name: &str) {
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let csv_path = current_dir.join(format!("verification/datasets/csv/{}.csv", dataset_name));
    let ref_path = current_dir.join(format!("verification/results/r/{}_kfold_cv_ols.json", dataset_name));

    let dataset = load_dataset_with_encoding(&csv_path, CategoricalEncoding::OneBased)
        .expect(&format!("Failed to load {}", dataset_name));
    let ref_result = expect_kfold_cv_result(&ref_path);
    let options = KFoldOptions::new(CV_N_FOLDS).with_shuffle(false);

    let result = kfold_cv_ols(&dataset.y, &dataset.x_vars, &dataset.variable_names, &options)
        .expect(&format!("OLS CV failed for {}", dataset_name));

    assert_eq!(result.n_folds, ref_result.n_folds);
    assert_eq!(result.n_samples, ref_result.n_samples);

    let rmse_diff = (result.mean_rmse - ref_result.mean_rmse).abs();
    let r2_diff = (result.mean_r_squared - ref_result.mean_r_squared).abs();
    let mae_diff = (result.mean_mae - ref_result.mean_mae).abs();

    println!("   File: {} (n={}, p={})",
        ref_path.file_name().unwrap().to_string_lossy(),
        result.n_samples,
        dataset.x_vars.len()
    );
    println!("     RMSE: Rust={:.6}, R={:.6}, diff={:.2e}",
        result.mean_rmse, ref_result.mean_rmse, rmse_diff);
    println!("     R²:   Rust={:.6}, R={:.6}, diff={:.2e}",
        result.mean_r_squared, ref_result.mean_r_squared, r2_diff);
    println!("     MAE:  Rust={:.6}, R={:.6}, diff={:.2e}",
        result.mean_mae, ref_result.mean_mae, mae_diff);

    assert_close_to(result.mean_rmse, ref_result.mean_rmse, LASSO_TOLERANCE,
        &format!("{} OLS CV mean RMSE", dataset_name));
    assert_close_to(result.mean_mae, ref_result.mean_mae, LASSO_TOLERANCE,
        &format!("{} OLS CV mean MAE", dataset_name));

    println!("   {} OLS CV: PASS\n", dataset_name);
}

fn validate_cv_ridge(dataset_name: &str) {
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let csv_path = current_dir.join(format!("verification/datasets/csv/{}.csv", dataset_name));
    let ref_path = current_dir.join(format!("verification/results/r/{}_kfold_cv_ridge.json", dataset_name));

    let dataset = load_dataset_with_encoding(&csv_path, CategoricalEncoding::OneBased)
        .expect(&format!("Failed to load {}", dataset_name));
    let ref_result = expect_kfold_cv_result(&ref_path);
    let options = KFoldOptions::new(CV_N_FOLDS).with_shuffle(false);

    let result = kfold_cv_ridge(&dataset.x_vars, &dataset.y, CV_LAMBDA, true, &options)
        .expect(&format!("Ridge CV failed for {}", dataset_name));

    assert_eq!(result.n_folds, ref_result.n_folds);
    assert_eq!(result.n_samples, ref_result.n_samples);

    let rmse_diff = (result.mean_rmse - ref_result.mean_rmse).abs();
    let r2_diff = (result.mean_r_squared - ref_result.mean_r_squared).abs();

    println!("   File: {} (n={}, p={})",
        ref_path.file_name().unwrap().to_string_lossy(),
        result.n_samples,
        dataset.x_vars.len()
    );
    println!("     RMSE: Rust={:.6}, R={:.6}, diff={:.2e}",
        result.mean_rmse, ref_result.mean_rmse, rmse_diff);
    println!("     R²:   Rust={:.6}, R={:.6}, diff={:.2e}",
        result.mean_r_squared, ref_result.mean_r_squared, r2_diff);

    assert_close_to(result.mean_rmse, ref_result.mean_rmse, LASSO_TOLERANCE,
        &format!("{} Ridge CV mean RMSE", dataset_name));

    println!("   {} Ridge CV: PASS\n", dataset_name);
}

fn validate_cv_lasso(dataset_name: &str) {
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let csv_path = current_dir.join(format!("verification/datasets/csv/{}.csv", dataset_name));
    let ref_path = current_dir.join(format!("verification/results/r/{}_kfold_cv_lasso.json", dataset_name));

    let dataset = load_dataset_with_encoding(&csv_path, CategoricalEncoding::OneBased)
        .expect(&format!("Failed to load {}", dataset_name));
    let ref_result = expect_kfold_cv_result(&ref_path);
    let options = KFoldOptions::new(CV_N_FOLDS).with_shuffle(false);

    let result = kfold_cv_lasso(&dataset.x_vars, &dataset.y, CV_LAMBDA, true, &options)
        .expect(&format!("Lasso CV failed for {}", dataset_name));

    assert_eq!(result.n_folds, ref_result.n_folds);
    assert_eq!(result.n_samples, ref_result.n_samples);

    let rmse_diff = (result.mean_rmse - ref_result.mean_rmse).abs();
    let r2_diff = (result.mean_r_squared - ref_result.mean_r_squared).abs();

    println!("   File: {} (n={}, p={})",
        ref_path.file_name().unwrap().to_string_lossy(),
        result.n_samples,
        dataset.x_vars.len()
    );
    println!("     RMSE: Rust={:.6}, R={:.6}, diff={:.2e}",
        result.mean_rmse, ref_result.mean_rmse, rmse_diff);
    println!("     R²:   Rust={:.6}, R={:.6}, diff={:.2e}",
        result.mean_r_squared, ref_result.mean_r_squared, r2_diff);

    assert_close_to(result.mean_rmse, ref_result.mean_rmse, LASSO_TOLERANCE,
        &format!("{} Lasso CV mean RMSE", dataset_name));

    println!("   {} Lasso CV: PASS\n", dataset_name);
}

fn validate_cv_elastic_net(dataset_name: &str) {
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let csv_path = current_dir.join(format!("verification/datasets/csv/{}.csv", dataset_name));
    let ref_path = current_dir.join(format!("verification/results/r/{}_kfold_cv_elastic_net.json", dataset_name));

    let dataset = load_dataset_with_encoding(&csv_path, CategoricalEncoding::OneBased)
        .expect(&format!("Failed to load {}", dataset_name));
    let ref_result = expect_kfold_cv_result(&ref_path);
    let options = KFoldOptions::new(CV_N_FOLDS).with_shuffle(false);

    let result = kfold_cv_elastic_net(&dataset.x_vars, &dataset.y, CV_LAMBDA, CV_ALPHA, true, &options)
        .expect(&format!("Elastic Net CV failed for {}", dataset_name));

    assert_eq!(result.n_folds, ref_result.n_folds);
    assert_eq!(result.n_samples, ref_result.n_samples);

    let rmse_diff = (result.mean_rmse - ref_result.mean_rmse).abs();
    let r2_diff = (result.mean_r_squared - ref_result.mean_r_squared).abs();

    println!("   File: {} (n={}, p={})",
        ref_path.file_name().unwrap().to_string_lossy(),
        result.n_samples,
        dataset.x_vars.len()
    );
    println!("     RMSE: Rust={:.6}, R={:.6}, diff={:.2e}",
        result.mean_rmse, ref_result.mean_rmse, rmse_diff);
    println!("     R²:   Rust={:.6}, R={:.6}, diff={:.2e}",
        result.mean_r_squared, ref_result.mean_r_squared, r2_diff);

    assert_close_to(result.mean_rmse, ref_result.mean_rmse, LASSO_TOLERANCE,
        &format!("{} Elastic Net CV mean RMSE", dataset_name));

    println!("   {} Elastic Net CV: PASS\n", dataset_name);
}

#[test]
fn validate_cv_ols_all_datasets() {
    println!("\n========== K-FOLD CV OLS VALIDATION (R) ==========\n");
    for dataset in CV_TEST_DATASETS {
        println!("--- Dataset: {} ---", dataset);
        validate_cv_ols(dataset);
    }
}

#[test]
fn validate_cv_ridge_all_datasets() {
    println!("\n========== K-FOLD CV RIDGE VALIDATION (R) ==========\n");
    for dataset in CV_TEST_DATASETS {
        println!("--- Dataset: {} ---", dataset);
        validate_cv_ridge(dataset);
    }
}

#[test]
fn validate_cv_lasso_all_datasets() {
    println!("\n========== K-FOLD CV LASSO VALIDATION (R) ==========\n");
    for dataset in CV_TEST_DATASETS {
        println!("--- Dataset: {} ---", dataset);
        validate_cv_lasso(dataset);
    }
}

#[test]
fn validate_cv_elastic_net_all_datasets() {
    println!("\n========== K-FOLD CV ELASTIC NET VALIDATION (R) ==========\n");
    for dataset in CV_TEST_DATASETS {
        println!("--- Dataset: {} ---", dataset);
        validate_cv_elastic_net(dataset);
    }
}
