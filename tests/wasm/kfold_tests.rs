// ============================================================================
// K-Fold Cross Validation WASM Tests
// ============================================================================

#![cfg(target_arch = "wasm32")]

use crate::wasm::fixtures::*;
use wasm_bindgen_test::*;
use linreg_core::wasm::*;

// ============================================================================
// Helpers
// ============================================================================

fn check_cv_result_fields(result: &serde_json::Value) {
    for field in &["n_folds", "n_samples", "mean_mse", "std_mse",
                   "mean_rmse", "std_rmse", "mean_mae", "std_mae",
                   "mean_r_squared", "std_r_squared", "fold_results"] {
        assert!(result.get(field).is_some(), "Missing CV field: {}", field);
    }
    assert!(result["mean_mse"].as_f64().unwrap() >= 0.0, "mean_mse should be non-negative");
    assert!(result["mean_rmse"].as_f64().unwrap() >= 0.0, "mean_rmse should be non-negative");
    assert!(result["mean_mae"].as_f64().unwrap() >= 0.0, "mean_mae should be non-negative");
}

// ============================================================================
// OLS K-Fold CV
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_kfold_cv_ols_basic() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let result_json = kfold_cv_ols(&y_json, &x_vars_json, &names_json, 5, "false", "null");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "OLS k-fold CV should not error");
    check_cv_result_fields(&result);
    assert_eq!(result["n_folds"].as_u64().unwrap(), 5);
    assert_eq!(result["n_samples"].as_u64().unwrap(), 25);
    assert_eq!(result["fold_results"].as_array().unwrap().len(), 5);
}

#[wasm_bindgen_test]
fn test_wasm_kfold_cv_ols_with_seed() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    // Same seed should produce identical results
    let r1 = serde_json::from_str::<serde_json::Value>(
        &kfold_cv_ols(&y_json, &x_vars_json, &names_json, 5, "true", "42")
    ).unwrap();
    let r2 = serde_json::from_str::<serde_json::Value>(
        &kfold_cv_ols(&y_json, &x_vars_json, &names_json, 5, "true", "42")
    ).unwrap();

    assert!((r1["mean_mse"].as_f64().unwrap() - r2["mean_mse"].as_f64().unwrap()).abs() < 1e-10,
        "Same seed should give identical mean_mse");
}

#[wasm_bindgen_test]
fn test_wasm_kfold_cv_ols_fold_count() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    for n_folds in [3usize, 5, 10] {
        let result: serde_json::Value = serde_json::from_str(
            &kfold_cv_ols(&y_json, &x_vars_json, &names_json, n_folds, "false", "null")
        ).unwrap();
        assert!(result.get("error").is_none(), "k={} should not error", n_folds);
        assert_eq!(result["n_folds"].as_u64().unwrap(), n_folds as u64);
        assert_eq!(result["fold_results"].as_array().unwrap().len(), n_folds);
    }
}

// ============================================================================
// Ridge K-Fold CV
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_kfold_cv_ridge_basic() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = kfold_cv_ridge(&y_json, &x_vars_json, 1.0, true, 5, "false", "null");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "Ridge k-fold CV should not error");
    check_cv_result_fields(&result);
    assert_eq!(result["n_folds"].as_u64().unwrap(), 5);
}

#[wasm_bindgen_test]
fn test_wasm_kfold_cv_ridge_lambda_effect() {
    // Higher lambda -> more regularization -> potentially higher CV error
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let r_small: serde_json::Value = serde_json::from_str(
        &kfold_cv_ridge(&y_json, &x_vars_json, 0.001, true, 5, "false", "1")
    ).unwrap();
    let r_large: serde_json::Value = serde_json::from_str(
        &kfold_cv_ridge(&y_json, &x_vars_json, 1000.0, true, 5, "false", "1")
    ).unwrap();

    assert!(r_small.get("error").is_none());
    assert!(r_large.get("error").is_none());
    // Both should produce valid MSE values
    assert!(r_small["mean_mse"].as_f64().unwrap() >= 0.0);
    assert!(r_large["mean_mse"].as_f64().unwrap() >= 0.0);
}

// ============================================================================
// Lasso K-Fold CV
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_kfold_cv_lasso_basic() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = kfold_cv_lasso(&y_json, &x_vars_json, 0.1, true, 5, "false", "null");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "Lasso k-fold CV should not error");
    check_cv_result_fields(&result);
    assert_eq!(result["n_folds"].as_u64().unwrap(), 5);
}

// ============================================================================
// Elastic Net K-Fold CV
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_kfold_cv_elastic_net_basic() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = kfold_cv_elastic_net(&y_json, &x_vars_json, 0.1, 0.5, true, 5, "false", "null");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "Elastic Net k-fold CV should not error");
    check_cv_result_fields(&result);
    assert_eq!(result["n_folds"].as_u64().unwrap(), 5);
}

#[wasm_bindgen_test]
fn test_wasm_kfold_cv_elastic_net_ridge_variant() {
    // alpha=0 is pure Ridge; results should match kfold_cv_ridge closely
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let enet: serde_json::Value = serde_json::from_str(
        &kfold_cv_elastic_net(&y_json, &x_vars_json, 1.0, 0.0, true, 5, "false", "42")
    ).unwrap();
    let ridge: serde_json::Value = serde_json::from_str(
        &kfold_cv_ridge(&y_json, &x_vars_json, 1.0, true, 5, "false", "42")
    ).unwrap();

    assert!(enet.get("error").is_none());
    assert!(ridge.get("error").is_none());

    let enet_mse = enet["mean_mse"].as_f64().unwrap();
    let ridge_mse = ridge["mean_mse"].as_f64().unwrap();
    assert!((enet_mse - ridge_mse).abs() / ridge_mse.max(1e-10) < 0.01,
        "Elastic net alpha=0 CV MSE should be close to Ridge: enet={}, ridge={}", enet_mse, ridge_mse);
}

// ============================================================================
// Error cases
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_kfold_cv_too_few_folds() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let result: serde_json::Value = serde_json::from_str(
        &kfold_cv_ols(&y_json, &x_vars_json, &names_json, 1, "false", "null")
    ).unwrap();
    assert!(result.get("error").is_some(), "n_folds=1 should return an error");
}
