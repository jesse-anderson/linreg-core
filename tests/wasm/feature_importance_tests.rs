// ============================================================================
// Feature Importance WASM Tests
// ============================================================================
//
// Tests for feature importance metrics including standardized coefficients,
// SHAP values, permutation importance, and VIF ranking.

#![cfg(target_arch = "wasm32")]

use crate::wasm::fixtures::*;
use wasm_bindgen_test::*;
use linreg_core::wasm::*;

// ============================================================================
// Standardized Coefficients Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_standardized_coefficients() {
    // Simple test data with known coefficients
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];  // Perfectly correlated with x1

    // Use known coefficients: intercept=1.0, x1=0.5, x2=0.0 (x2 should be dropped)
    let coefficients = vec![1.0, 0.5, 0.0];
    let y_std = 2.7386;

    let coeffs_json = serde_json::to_string(&coefficients).unwrap();
    let x_vars_json = serde_json::to_string(&vec![x1, x2]).unwrap();
    let var_names_json = serde_json::to_string(&vec!["X1", "X2"]).unwrap();

    let result_json = standardized_coefficients(&x_vars_json, &coeffs_json, &var_names_json, y_std);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Check for error
    if let Some(error) = result.get("error") {
        panic!("standardized_coefficients failed: {:?}", error);
    }

    // Should have standardized_coefficients array
    let std_coefs = result
        .get("standardized_coefficients")
        .expect("Should have standardized_coefficients")
        .as_array()
        .expect("standardized_coefficients should be an array");

    assert_eq!(std_coefs.len(), 2, "Should have 2 standardized coefficients");

    // Values should be finite
    for coef in std_coefs {
        assert!(
            coef.as_f64().map(|v| v.is_finite()).unwrap_or(false),
            "Standardized coefficients should be finite"
        );
    }
}

// ============================================================================
// SHAP Values Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_shap_values_linear() {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    let coefficients = vec![1.0, 0.5, -0.3]; // Intercept, X1, X2

    let x_vars_json = serde_json::to_string(&vec![x1, x2]).unwrap();
    let coeffs_json = serde_json::to_string(&coefficients).unwrap();
    let var_names_json = serde_json::to_string(&vec!["X1", "X2"]).unwrap();

    let result_json = shap_values_linear(&x_vars_json, &coeffs_json, &var_names_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have mean_abs_shap array
    let mean_abs_shap = result
        .get("mean_abs_shap")
        .expect("Should have mean_abs_shap")
        .as_array()
        .expect("mean_abs_shap should be an array");

    assert_eq!(mean_abs_shap.len(), 2, "Should have 2 SHAP values");

    // Should have shap_values matrix
    let shap_values = result
        .get("shap_values")
        .expect("Should have shap_values")
        .as_array()
        .expect("shap_values should be an array");

    assert_eq!(shap_values.len(), 5, "Should have SHAP values for 5 observations");

    // Should have base_value (intercept)
    let base_value = result
        .get("base_value")
        .expect("Should have base_value")
        .as_f64()
        .expect("base_value should be a number");

    assert!((base_value - 1.0).abs() < 0.01, "base_value should be close to intercept");
}

// ============================================================================
// VIF Ranking Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_vif_ranking() {
    // Create some VIF results matching VifResult struct from core module
    // which has: variable, vif, rsquared, interpretation
    let vif_data = serde_json::json!([
        {"variable": "X1", "vif": 1.5, "rsquared": 0.33, "interpretation": "Low multicollinearity"},
        {"variable": "X2", "vif": 5.2, "rsquared": 0.81, "interpretation": "Moderate multicollinearity"},
        {"variable": "X3", "vif": 2.8, "rsquared": 0.64, "interpretation": "Low multicollinearity"}
    ]);

    let vif_json = serde_json::to_string(&vif_data).unwrap();
    let result_json = vif_ranking(&vif_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Check for error
    if let Some(error) = result.get("error") {
        panic!("vif_ranking failed: {:?}", error);
    }

    // Should have vif_values array
    let vif_values = result
        .get("vif_values")
        .expect("Should have vif_values")
        .as_array()
        .expect("vif_values should be an array");

    assert_eq!(vif_values.len(), 3, "Should have 3 VIF values");

    // Should have variable_names array
    let var_names = result
        .get("variable_names")
        .expect("Should have variable_names")
        .as_array()
        .expect("variable_names should be an array");

    assert_eq!(var_names.len(), 3, "Should have 3 variable names");
}

// ============================================================================
// Permutation Importance Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_permutation_importance_ols() {
    // Use single predictor to avoid multicollinearity issues
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    // Fit OLS model first
    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&vec![x1]).unwrap();
    let names_json = serde_json::to_string(&vec!["Intercept", "X1"]).unwrap();

    let fit_json = ols_regression(&y_json, &x_vars_json, &names_json);
    let fit: serde_json::Value = serde_json::from_str(&fit_json).unwrap();

    // Check if OLS succeeded
    if let Some(error) = fit.get("error") {
        panic!("OLS regression failed: {:?}", error);
    }

    // Compute permutation importance with small number of permutations for speed
    let result_json = permutation_importance_ols(&y_json, &x_vars_json, &fit_json, 10, 42);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Check for error
    if let Some(error) = result.get("error") {
        panic!("permutation_importance_ols failed: {:?}", error);
    }

    // Should have importance array
    let importance = result
        .get("importance")
        .expect("Should have importance")
        .as_array()
        .expect("importance should be an array");

    assert_eq!(importance.len(), 1, "Should have 1 importance score");

    // All values should be finite
    for imp in importance {
        assert!(
            imp.as_f64().map(|v| v.is_finite()).unwrap_or(false),
            "Importance values should be finite"
        );
    }

    // Should have baseline_score
    let baseline = result
        .get("baseline_score")
        .expect("Should have baseline_score")
        .as_f64()
        .expect("baseline_score should be a number");

    assert!(baseline >= 0.0 && baseline <= 1.0, "Baseline R² should be in [0, 1]");
}

// ============================================================================
// Combined Feature Importance Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_feature_importance_ols() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&vec![x1, x2]).unwrap();
    let var_names_json = serde_json::to_string(&vec!["X1", "X2"]).unwrap();
    let y_std = 2.7386;

    let result_json = feature_importance_ols(&y_json, &x_vars_json, &var_names_json, y_std, 10, 42);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should contain all four metrics
    assert!(
        result.get("standardized_coefficients").is_some(),
        "Should have standardized_coefficients"
    );
    assert!(result.get("shap").is_some(), "Should have shap");
    assert!(
        result.get("permutation_importance").is_some(),
        "Should have permutation_importance"
    );
    assert!(result.get("vif_ranking").is_some(), "Should have vif_ranking");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_standardized_coefficients_error() {
    let invalid_json = "{not valid";
    let coeffs_json = serde_json::to_string(&vec![1.0, 0.5]).unwrap();
    let var_names_json = serde_json::to_string(&vec!["X1"]).unwrap();

    let result_json = standardized_coefficients(invalid_json, &coeffs_json, &var_names_json, 1.0);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_some(), "Should return error for invalid JSON");
}

#[wasm_bindgen_test]
fn test_wasm_shap_values_error() {
    let invalid_json = "{not valid";
    let coeffs_json = serde_json::to_string(&vec![1.0, 0.5]).unwrap();
    let var_names_json = serde_json::to_string(&vec!["X1"]).unwrap();

    let result_json = shap_values_linear(invalid_json, &coeffs_json, &var_names_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_some(), "Should return error for invalid JSON");
}

#[wasm_bindgen_test]
fn test_wasm_vif_ranking_error() {
    let invalid_json = "{not valid";

    let result_json = vif_ranking(invalid_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_some(), "Should return error for invalid JSON");
}
