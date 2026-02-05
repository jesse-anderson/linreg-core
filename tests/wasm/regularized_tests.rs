// ============================================================================
// Regularized Regression WASM Tests
// ============================================================================
//
// Tests for Ridge, Lasso, and Elastic Net regression.

#![cfg(target_arch = "wasm32")]

use crate::wasm::fixtures::*;
use wasm_bindgen_test::*;
use linreg_core::wasm::*;

// ============================================================================
// Ridge Regression Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_ridge_regression() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let result_json = ridge_regression(&y_json, &x_vars_json, &names_json, 1.0, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have lambda
    let lambda = result
        .get("lambda")
        .expect("Should have lambda")
        .as_f64()
        .unwrap();

    assert_eq!(lambda, 1.0, "Lambda should be 1.0");

    // Should have intercept
    assert!(result.get("intercept").is_some(), "Should have intercept");

    // Should have coefficients array
    let coefficients = result
        .get("coefficients")
        .expect("Should have coefficients")
        .as_array()
        .unwrap();

    assert_eq!(coefficients.len(), 3, "Should have 3 coefficients");

    // Should have residuals array
    let residuals = result
        .get("residuals")
        .expect("Should have residuals")
        .as_array()
        .unwrap();

    assert_eq!(residuals.len(), 25, "Should have 25 residuals");

    // Should have df (effective degrees of freedom)
    assert!(result.get("df").is_some(), "Should have df");
}

#[wasm_bindgen_test]
fn test_wasm_ridge_regression_no_standardization() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let names_json = serde_json::to_string(&vec!["Intercept", "X"]).unwrap();

    let result_json = ridge_regression(&y_json, &x_vars_json, &names_json, 0.1, false);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("intercept").is_some(), "Should have intercept");
    assert!(result.get("coefficients").is_some(), "Should have coefficients");
}

// ============================================================================
// Lasso Regression Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_lasso_regression() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let result_json = lasso_regression(&y_json, &x_vars_json, &names_json, 0.1, true, 1000, 1e-7);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have lambda
    let lambda = result
        .get("lambda")
        .expect("Should have lambda")
        .as_f64()
        .unwrap();

    assert_eq!(lambda, 0.1, "Lambda should be 0.1");

    // Should have intercept
    assert!(result.get("intercept").is_some(), "Should have intercept");

    // Should have coefficients array
    let coefficients = result
        .get("coefficients")
        .expect("Should have coefficients")
        .as_array()
        .unwrap();

    assert_eq!(coefficients.len(), 3, "Should have 3 coefficients");

    // Should have n_nonzero
    let n_nonzero = result
        .get("n_nonzero")
        .expect("Should have n_nonzero")
        .as_u64()
        .unwrap();

    assert!(n_nonzero <= 3, "n_nonzero should be <= 3");

    // Should have converged flag
    let converged = result
        .get("converged")
        .expect("Should have converged")
        .as_bool()
        .unwrap();

    assert!(converged, "Lasso should have converged");
}

#[wasm_bindgen_test]
fn test_wasm_lasso_regression_strong_penalty() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let names_json = serde_json::to_string(&vec!["Intercept", "X"]).unwrap();

    // Strong penalty should zero out coefficients
    let result_json = lasso_regression(&y_json, &x_vars_json, &names_json, 10.0, true, 1000, 1e-7);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    let n_nonzero = result
        .get("n_nonzero")
        .expect("Should have n_nonzero")
        .as_u64()
        .unwrap();

    // With strong penalty, may have zero non-zero coefficients
    assert!(n_nonzero <= 1, "Strong penalty should zero out most coefficients");
}

// ============================================================================
// Elastic Net Regression Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_elastic_net_regression() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    // alpha = 0.5 means equal mix of L1 and L2
    let result_json =
        elastic_net_regression(&y_json, &x_vars_json, &names_json, 0.1, 0.5, true, 1000, 1e-7);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have lambda
    let lambda = result
        .get("lambda")
        .expect("Should have lambda")
        .as_f64()
        .unwrap();

    assert_eq!(lambda, 0.1, "Lambda should be 0.1");

    // Should have intercept
    assert!(result.get("intercept").is_some(), "Should have intercept");

    // Should have coefficients array
    let coefficients = result
        .get("coefficients")
        .expect("Should have coefficients")
        .as_array()
        .unwrap();

    assert_eq!(coefficients.len(), 3, "Should have 3 coefficients");

    // Should have n_nonzero
    assert!(result.get("n_nonzero").is_some(), "Should have n_nonzero");

    // Should have converged flag
    let converged = result
        .get("converged")
        .expect("Should have converged")
        .as_bool()
        .unwrap();

    assert!(converged, "Elastic Net should have converged");
}

#[wasm_bindgen_test]
fn test_wasm_elastic_net_ridge_variant() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let names_json = serde_json::to_string(&vec!["Intercept", "X"]).unwrap();

    // alpha = 0 means pure Ridge (L2 only)
    let result_json =
        elastic_net_regression(&y_json, &x_vars_json, &names_json, 0.1, 0.0, true, 1000, 1e-7);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("intercept").is_some(), "Should have intercept");
    assert!(result.get("coefficients").is_some(), "Should have coefficients");
}

#[wasm_bindgen_test]
fn test_wasm_elastic_net_lasso_variant() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let names_json = serde_json::to_string(&vec!["Intercept", "X"]).unwrap();

    // alpha = 1 means pure Lasso (L1 only)
    let result_json =
        elastic_net_regression(&y_json, &x_vars_json, &names_json, 0.1, 1.0, true, 1000, 1e-7);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("intercept").is_some(), "Should have intercept");
    assert!(result.get("coefficients").is_some(), "Should have coefficients");
    assert!(result.get("n_nonzero").is_some(), "Should have n_nonzero");
}

// ============================================================================
// Lambda Path Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_make_lambda_path() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = make_lambda_path(&y_json, &x_vars_json, 100, 0.01);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have lambda_path array
    let lambda_path = result
        .get("lambda_path")
        .expect("Should have lambda_path")
        .as_array()
        .unwrap();

    assert_eq!(lambda_path.len(), 100, "Should have 100 lambda values");

    // Note: First lambda is infinity (serialized as null in JSON)
    // For testing, skip the first element and check the second instead
    let second = lambda_path
        .get(1)
        .unwrap()
        .as_f64()
        .expect("Second value should be a number");
    let last = lambda_path
        .last()
        .unwrap()
        .as_f64()
        .expect("Last value should be a number");

    assert!(second > last, "Lambda path should be in decreasing order");

    // Should have lambda_max and lambda_min
    // Note: lambda_max is infinity, which serializes as null in JSON
    let lambda_max_val = result.get("lambda_max").expect("Should have lambda_max");
    assert!(lambda_max_val.is_null() || lambda_max_val.as_f64().map_or(false, |v| v.is_infinite()),
        "lambda_max should be infinity (serialized as null)");

    let lambda_min = result
        .get("lambda_min")
        .expect("Should have lambda_min")
        .as_f64()
        .unwrap();

    assert_eq!(lambda_min, last, "lambda_min should equal last value");

    // Should have n_lambda
    let n_lambda = result
        .get("n_lambda")
        .expect("Should have n_lambda")
        .as_u64()
        .unwrap();

    assert_eq!(n_lambda, 100, "n_lambda should be 100");
}

#[wasm_bindgen_test]
fn test_wasm_make_lambda_path_small() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();

    let result_json = make_lambda_path(&y_json, &x_vars_json, 10, 0.001);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    let lambda_path = result
        .get("lambda_path")
        .expect("Should have lambda_path")
        .as_array()
        .unwrap();

    assert_eq!(lambda_path.len(), 10, "Should have 10 lambda values");
}
