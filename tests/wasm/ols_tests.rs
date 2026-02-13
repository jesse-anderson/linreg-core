// ============================================================================
// OLS Regression WASM Tests
// ============================================================================
//
// Tests for OLS regression, error handling, and CSV parsing.

#![cfg(target_arch = "wasm32")]

use crate::wasm::fixtures::*;
use wasm_bindgen_test::*;
use linreg_core::wasm::*;

// ============================================================================
// OLS Regression Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_ols_regression() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let result_json = ols_regression(&y_json, &x_vars_json, &names_json);

    // Should be valid JSON
    let result: serde_json::Value =
        serde_json::from_str(&result_json).expect("Result should be valid JSON");

    // Check structure
    assert!(result.is_object(), "Result should be an object");

    // Check coefficients array exists
    let coefficients = result
        .get("coefficients")
        .expect("Should have coefficients")
        .as_array()
        .expect("coefficients should be an array");

    assert_eq!(coefficients.len(), 4, "Should have 4 coefficients");

    // Check R-squared exists and is in valid range
    let r_squared = result
        .get("r_squared")
        .expect("Should have r_squared")
        .as_f64()
        .expect("r_squared should be a number");

    assert!(
        r_squared > 0.0 && r_squared <= 1.0,
        "R² should be in [0, 1]"
    );
    assert!(r_squared > 0.9, "Housing data should have high R²");
}

#[wasm_bindgen_test]
fn test_wasm_ols_regression_simple() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let names_json = serde_json::to_string(&vec!["Intercept", "X"]).unwrap();

    let result_json = ols_regression(&y_json, &x_vars_json, &names_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have high R² for perfect linear relationship
    let r_squared = result.get("r_squared").unwrap().as_f64().unwrap();
    assert!(r_squared > 0.99, "Simple linear should have R² > 0.99");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_error_handling_invalid_json() {
    let invalid_json = "{not valid json";
    let x_vars_json = serde_json::to_string(&vec![vec![1.0, 2.0]]).unwrap();
    let names_json = serde_json::to_string(&vec!["Intercept", "X"]).unwrap();

    let result_json = ols_regression(invalid_json, &x_vars_json, &names_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have error field
    assert!(result.get("error").is_some(), "Should return error object");
}

#[wasm_bindgen_test]
fn test_wasm_error_handling_insufficient_data() {
    let y = vec![1.0, 2.0];
    let x = vec![vec![1.0, 2.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let names_json = serde_json::to_string(&vec!["Intercept", "X"]).unwrap();

    let result_json = ols_regression(&y_json, &x_vars_json, &names_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have error field
    assert!(
        result.get("error").is_some(),
        "Should return error for insufficient data"
    );

    let error_msg = result.get("error").unwrap().as_str().unwrap();
    assert!(
        error_msg.to_lowercase().contains("insufficient")
            || error_msg.to_lowercase().contains("data"),
        "Error should mention insufficient data"
    );
}

#[wasm_bindgen_test]
fn test_wasm_error_handling_singular_matrix() {
    // Perfect multicollinearity: x2 = 2 * x1
    // LINPACK QR with column pivoting handles this gracefully by dropping
    // the redundant column, rather than erroring.
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // x2 = 2 * x1

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&vec![x1, x2]).unwrap();
    let names_json = serde_json::to_string(&vec!["Intercept", "X1", "X2"]).unwrap();

    let result_json = ols_regression(&y_json, &x_vars_json, &names_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should succeed (not error) - LINPACK QR drops redundant columns gracefully
    assert!(
        result.get("error").is_none(),
        "LINPACK QR should handle singular matrix gracefully, got error: {:?}",
        result.get("error")
    );

    // The result should have coefficients, with the dropped column having NaN stats
    let coefficients = result.get("coefficients").unwrap().as_array().unwrap();
    assert_eq!(coefficients.len(), 3, "Should have 3 coefficients (Intercept + X1 + X2)");

    // At least one standard error should be null (the dropped column)
    let std_errors = result.get("std_errors").unwrap().as_array().unwrap();
    let has_dropped = std_errors.iter().any(|se| se.is_null());
    assert!(
        has_dropped,
        "At least one std_error should be null due to collinearity"
    );
}

// ============================================================================
// CSV Parsing Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_csv_parsing() {
    let csv_content = "Name,Age,Score\nAlice,25,85.5\nBob,30,92.0\nCharlie,28,78.5";

    let result_json = parse_csv(csv_content);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have headers
    let headers = result
        .get("headers")
        .expect("Should have headers")
        .as_array()
        .unwrap();

    assert_eq!(headers.len(), 3, "Should have 3 headers");

    // Should have data
    let data = result
        .get("data")
        .expect("Should have data")
        .as_array()
        .unwrap();

    assert_eq!(data.len(), 3, "Should have 3 data rows");

    // Should have numeric_columns
    let numeric = result
        .get("numeric_columns")
        .expect("Should have numeric_columns")
        .as_array()
        .unwrap();

    assert!(
        numeric.len() >= 2,
        "Should have at least 2 numeric columns (Age, Score)"
    );
}

#[wasm_bindgen_test]
fn test_wasm_csv_parsing_invalid() {
    let invalid_csv = "";

    let result_json = parse_csv(invalid_csv);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Empty CSV should still return valid structure with empty arrays
    let headers = result
        .get("headers")
        .expect("Should have headers field")
        .as_array()
        .unwrap();

    assert_eq!(headers.len(), 0, "Empty CSV should have no headers");
}
