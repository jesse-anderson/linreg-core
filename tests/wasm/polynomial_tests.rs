// ============================================================================
// Polynomial Regression WASM Tests
// ============================================================================
//
// Tests for polynomial regression via WASM bindings, including OLS polynomial
// regression, prediction, and regularized variants (Ridge, Lasso, Elastic Net).

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use linreg_core::wasm::*;

// ============================================================================
// OLS Polynomial Regression Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_polynomial_regression_quadratic() {
    // Quadratic relationship: y = 1 + 2x + x^2
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result_json = polynomial_regression_wasm(&y_json, &x_json, 2, true, false);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have high R² for perfect quadratic relationship
    let ols_output = result.get("ols_output").unwrap();
    let r_squared = ols_output.get("r_squared").unwrap().as_f64().unwrap();
    assert!(r_squared > 0.999, "Quadratic fit should have R² > 0.999, got {}", r_squared);

    // Should have degree metadata
    assert_eq!(result.get("degree").unwrap().as_u64().unwrap(), 2);
    assert!(result.get("centered").unwrap().as_bool().unwrap());
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_regression_cubic() {
    // Cubic relationship: y = 1 + x + x^2 + 0.1*x^3
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + xi + xi * xi + 0.1 * xi.powi(3)).collect();

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result_json = polynomial_regression_wasm(&y_json, &x_json, 3, true, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have high R²
    let ols_output = result.get("ols_output").unwrap();
    let r_squared = ols_output.get("r_squared").unwrap().as_f64().unwrap();
    assert!(r_squared > 0.999, "Cubic fit should have high R²");

    // Should have standardized metadata
    assert!(result.get("standardized").unwrap().as_bool().unwrap());

    // Should have feature names
    let feature_names = result.get("feature_names").unwrap().as_array().unwrap();
    assert_eq!(feature_names.len(), 4); // intercept + x + x² + x³
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_predict() {
    // Fit a quadratic model and make predictions
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let fit_json = polynomial_regression_wasm(&y_json, &x_json, 2, true, false);

    // Predict at new points
    let x_new = vec![6.0, 7.0];
    let x_new_json = serde_json::to_string(&x_new).unwrap();

    let pred_json = polynomial_predict_wasm(&fit_json, &x_new_json);
    let predictions: Vec<f64> = serde_json::from_str(&pred_json).unwrap();

    assert_eq!(predictions.len(), 2);

    // Expected values: y = 1 + 2x + x²
    // x=6: 1 + 12 + 36 = 49
    // x=7: 1 + 14 + 49 = 64
    assert!((predictions[0] - 49.0).abs() < 0.1, "Prediction for x=6 should be ~49, got {}", predictions[0]);
    assert!((predictions[1] - 64.0).abs() < 0.1, "Prediction for x=7 should be ~64, got {}", predictions[1]);
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_regression_centered() {
    // Test that centering is properly applied
    let x = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    // With centering
    let result_json_centered = polynomial_regression_wasm(&y_json, &x_json, 2, true, false);
    let result_centered: serde_json::Value = serde_json::from_str(&result_json_centered).unwrap();

    assert!(result_centered.get("x_mean").unwrap().as_f64().is_some());
    assert!(result_centered.get("centered").unwrap().as_bool().unwrap());

    // Without centering
    let result_json_uncentered = polynomial_regression_wasm(&y_json, &x_json, 2, false, false);
    let result_uncentered: serde_json::Value = serde_json::from_str(&result_json_uncentered).unwrap();

    assert!(!result_uncentered.get("centered").unwrap().as_bool().unwrap());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_polynomial_error_invalid_json() {
    let invalid_json = "{not valid json";
    let x_json = serde_json::to_string(&vec![1.0, 2.0, 3.0]).unwrap();

    let result_json = polynomial_regression_wasm(invalid_json, &x_json, 2, true, false);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_some(), "Should return error for invalid JSON");
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_error_dimension_mismatch() {
    let y = vec![1.0, 2.0, 3.0];
    let x = vec![1.0, 2.0]; // Different length

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result_json = polynomial_regression_wasm(&y_json, &x_json, 2, true, false);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_some(), "Should return error for dimension mismatch");
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_error_zero_degree() {
    let y = vec![1.0, 2.0, 3.0];
    let x = vec![1.0, 2.0, 3.0];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result_json = polynomial_regression_wasm(&y_json, &x_json, 0, true, false);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_some(), "Should return error for degree 0");
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_error_insufficient_data() {
    let y = vec![1.0, 2.0];
    let x = vec![1.0, 2.0];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    // Degree 3 requires at least 4 data points
    let result_json = polynomial_regression_wasm(&y_json, &x_json, 3, true, false);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_some(), "Should return error for insufficient data");
}

// ============================================================================
// Ridge Polynomial Regression Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_polynomial_ridge() {
    // Quadratic with some noise
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result_json = polynomial_ridge_wasm(&y_json, &x_json, 2, 0.1, true, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have coefficients
    let coefficients = result.get("coefficients").unwrap().as_array().unwrap();
    assert!(coefficients.len() > 0);

    // Should have high R²
    let r_squared = result.get("r_squared").unwrap().as_f64().unwrap();
    assert!(r_squared > 0.99, "Ridge polynomial should have high R²");

    // Should have df (degrees of freedom)
    assert!(result.get("df").unwrap().as_f64().is_some());
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_ridge_high_lambda() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    // High lambda should shrink coefficients toward zero
    let result_json = polynomial_ridge_wasm(&y_json, &x_json, 2, 10.0, true, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should still work but with smaller coefficients
    assert!(result.get("coefficients").is_some());
}

// ============================================================================
// Lasso Polynomial Regression Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_polynomial_lasso() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result_json = polynomial_lasso_wasm(&y_json, &x_json, 2, 0.1, true, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have coefficients
    let coefficients = result.get("coefficients").unwrap().as_array().unwrap();
    assert!(coefficients.len() > 0);

    // Should have high R²
    let r_squared = result.get("r_squared").unwrap().as_f64().unwrap();
    assert!(r_squared > 0.99, "Lasso polynomial should have high R²");

    // Should have convergence info
    assert!(result.get("converged").unwrap().as_bool().unwrap());
    assert!(result.get("n_nonzero").unwrap().as_u64().is_some());
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_lasso_variable_selection() {
    // Test with higher degree and moderate lambda for variable selection
    let x: Vec<f64> = (1..=15).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    // Fit degree 5 with moderate lambda
    let result_json = polynomial_lasso_wasm(&y_json, &x_json, 5, 0.5, true, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Lasso should zero out some higher-order terms
    let coefficients = result.get("coefficients").unwrap().as_array().unwrap();
    let n_nonzero = result.get("n_nonzero").unwrap().as_u64().unwrap();

    assert!(n_nonzero <= coefficients.len() as u64, "Non-zero count should not exceed total coefficients");
}

// ============================================================================
// Elastic Net Polynomial Regression Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_polynomial_elastic_net() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    // alpha=0.5 is equal mix of L1 and L2
    let result_json = polynomial_elastic_net_wasm(&y_json, &x_json, 2, 0.1, 0.5, true, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have coefficients
    let coefficients = result.get("coefficients").unwrap().as_array().unwrap();
    assert!(coefficients.len() > 0);

    // Should have high R²
    let r_squared = result.get("r_squared").unwrap().as_f64().unwrap();
    assert!(r_squared > 0.99, "Elastic Net polynomial should have high R²");

    // Should have convergence info
    assert!(result.get("converged").unwrap().as_bool().unwrap());
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_elastic_net_ridge_like() {
    // alpha=0 should behave like pure Ridge
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result_json = polynomial_elastic_net_wasm(&y_json, &x_json, 2, 0.1, 0.0, true, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("coefficients").is_some());
    assert!(result.get("r_squared").is_some());
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_elastic_net_lasso_like() {
    // alpha=1 should behave like pure Lasso
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result_json = polynomial_elastic_net_wasm(&y_json, &x_json, 2, 0.1, 1.0, true, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("coefficients").is_some());
    assert!(result.get("n_nonzero").is_some());
}

// ============================================================================
// Regularized Error Handling Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_polynomial_ridge_error_invalid_json() {
    let invalid_json = "{bad json";
    let x_json = serde_json::to_string(&vec![1.0, 2.0]).unwrap();

    let result_json = polynomial_ridge_wasm(invalid_json, &x_json, 2, 1.0, true, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_some());
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_lasso_error_dimension_mismatch() {
    let y = vec![1.0, 2.0];
    let x = vec![1.0]; // Different lengths

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result_json = polynomial_lasso_wasm(&y_json, &x_json, 2, 0.1, true, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_some());
}

#[wasm_bindgen_test]
fn test_wasm_polynomial_elastic_net_error_zero_degree() {
    let y = vec![1.0, 2.0];
    let x = vec![1.0, 2.0];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result_json = polynomial_elastic_net_wasm(&y_json, &x_json, 0, 0.1, 0.5, true, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_some());
}
