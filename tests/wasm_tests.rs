// ============================================================================
// WASM Integration Tests
// ============================================================================
//
// Browser-based WASM tests using wasm-bindgen-test framework.
// Tests cover OLS regression, diagnostic tests, error handling, CSV parsing,
// and domain checking.
//
// Run with: wasm-pack test --chrome --headless

#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;
use wasm_bindgen_test::*;
use linear_regression_wasm::*;

// ============================================================================
// Test Fixtures
// ============================================================================

fn get_housing_y() -> String {
    serde_json::to_string(&vec![
        245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1,
        445.8, 167.9, 367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9,
        223.4, 312.5, 156.8, 423.7, 267.9
    ]).unwrap()
}

fn get_housing_x_vars() -> String {
    let square_feet = vec![
        1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0,
        2200.0, 900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0,
        1250.0, 1700.0, 850.0, 2350.0, 1400.0
    ];
    let bedrooms = vec![
        3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0,
        4.0, 2.0, 4.0, 3.0, 2.0, 5.0, 3.0, 4.0, 2.0, 4.0,
        3.0, 3.0, 2.0, 4.0, 3.0
    ];
    let age = vec![
        15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0,
        3.0, 30.0, 6.0, 14.0, 22.0, 1.0, 16.0, 9.0, 28.0, 4.0,
        19.0, 11.0, 35.0, 3.0, 13.0
    ];
    serde_json::to_string(&vec![square_feet, bedrooms, age]).unwrap()
}

fn get_variable_names() -> String {
    serde_json::to_string(&vec!["Intercept", "Square_Feet", "Bedrooms", "Age"]).unwrap()
}

// ============================================================================
// OLS Regression WASM Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_ols_regression() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let result_json = ols_regression(&y_json, &x_vars_json, &names_json);

    // Should be valid JSON
    let result: serde_json::Value = serde_json::from_str(&result_json)
        .expect("Result should be valid JSON");

    // Check structure
    assert!(result.is_object(), "Result should be an object");

    // Check coefficients array exists
    let coefficients = result.get("coefficients")
        .expect("Should have coefficients")
        .as_array()
        .expect("coefficients should be an array");

    assert_eq!(coefficients.len(), 4, "Should have 4 coefficients");

    // Check R-squared exists and is in valid range
    let r_squared = result.get("r_squared")
        .expect("Should have r_squared")
        .as_f64()
        .expect("r_squared should be a number");

    assert!(r_squared > 0.0 && r_squared <= 1.0, "R² should be in [0, 1]");
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
// Diagnostic Tests WASM Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_rainbow_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = rainbow_test(&y_json, &x_vars_json, 0.5, "r");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result.get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(test_name.contains("Rainbow"), "Test name should mention Rainbow");

    // Should have R result
    let r_result = result.get("r_result")
        .expect("Should have r_result for 'r' method");

    assert!(r_result.is_object(), "r_result should be an object");

    let statistic = r_result.get("statistic")
        .expect("Should have statistic")
        .as_f64()
        .unwrap();

    assert!(statistic >= 0.0, "Statistic should be non-negative");
}

#[wasm_bindgen_test]
fn test_wasm_rainbow_test_python_method() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = rainbow_test(&y_json, &x_vars_json, 0.5, "python");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have python_result, not r_result
    assert!(result.get("python_result").is_some(), "Should have python_result");
    assert!(result.get("r_result").is_none(), "Should not have r_result for python method");
}

#[wasm_bindgen_test]
fn test_wasm_rainbow_test_both_methods() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = rainbow_test(&y_json, &x_vars_json, 0.5, "both");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have both results
    assert!(result.get("r_result").is_some(), "Should have r_result");
    assert!(result.get("python_result").is_some(), "Should have python_result");
}

#[wasm_bindgen_test]
fn test_wasm_harvey_collier_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = harvey_collier_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result.get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(test_name.contains("Harvey-Collier") || test_name.contains("Harvey"),
        "Test name should mention Harvey-Collier");

    // Should have statistic
    let statistic = result.get("statistic")
        .expect("Should have statistic")
        .as_f64()
        .unwrap();

    assert!(statistic >= 0.0, "Statistic should be non-negative");
}

#[wasm_bindgen_test]
fn test_wasm_breusch_pagan_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = breusch_pagan_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result.get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(test_name.contains("Breusch-Pagan") || test_name.contains("Breusch"),
        "Test name should mention Breusch-Pagan");

    // Should have statistic and p_value
    assert!(result.get("statistic").is_some(), "Should have statistic");
    assert!(result.get("p_value").is_some(), "Should have p_value");
}

#[wasm_bindgen_test]
fn test_wasm_white_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = white_test(&y_json, &x_vars_json, "r");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result.get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(test_name.contains("White"), "Test name should mention White");

    // Should have R result
    let r_result = result.get("r_result")
        .expect("Should have r_result for 'r' method");

    assert!(r_result.is_object(), "r_result should be an object");
}

#[wasm_bindgen_test]
fn test_wasm_jarque_bera_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = jarque_bera_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result.get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(test_name.contains("Jarque-Bera") || test_name.contains("Jarque"),
        "Test name should mention Jarque-Bera");

    // Should have statistic
    let statistic = result.get("statistic")
        .expect("Should have statistic")
        .as_f64()
        .unwrap();

    assert!(statistic >= 0.0, "Statistic should be non-negative");
}

#[wasm_bindgen_test]
fn test_wasm_durbin_watson_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = durbin_watson_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result.get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(test_name.contains("Durbin-Watson") || test_name.contains("Durbin"),
        "Test name should mention Durbin-Watson");

    // DW statistic should be in [0, 4]
    let statistic = result.get("statistic")
        .expect("Should have statistic")
        .as_f64()
        .unwrap();

    assert!(statistic >= 0.0 && statistic <= 4.0,
        "DW statistic should be in [0, 4], got {}", statistic);
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
    assert!(result.get("error").is_some(), "Should return error for insufficient data");

    let error_msg = result.get("error").unwrap().as_str().unwrap();
    assert!(error_msg.to_lowercase().contains("insufficient") ||
            error_msg.to_lowercase().contains("data"),
        "Error should mention insufficient data");
}

#[wasm_bindgen_test]
fn test_wasm_error_handling_singular_matrix() {
    // Perfect multicollinearity: x2 = 2 * x1
    let y = vec![1.0, 2.0, 3.0];
    let x1 = vec![1.0, 2.0, 3.0];
    let x2 = vec![2.0, 4.0, 6.0]; // x2 = 2 * x1

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&vec![x1, x2]).unwrap();
    let names_json = serde_json::to_string(&vec!["Intercept", "X1", "X2"]).unwrap();

    let result_json = ols_regression(&y_json, &x_vars_json, &names_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have error field
    assert!(result.get("error").is_some(), "Should return error for singular matrix");

    let error_msg = result.get("error").unwrap().as_str().unwrap();
    assert!(error_msg.to_lowercase().contains("singular") ||
            error_msg.to_lowercase().contains("multicollinearity"),
        "Error should mention singular or multicollinearity");
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
    let headers = result.get("headers")
        .expect("Should have headers")
        .as_array()
        .unwrap();

    assert_eq!(headers.len(), 3, "Should have 3 headers");

    // Should have data
    let data = result.get("data")
        .expect("Should have data")
        .as_array()
        .unwrap();

    assert_eq!(data.len(), 3, "Should have 3 data rows");

    // Should have numeric_columns
    let numeric = result.get("numeric_columns")
        .expect("Should have numeric_columns")
        .as_array()
        .unwrap();

    assert!(numeric.len() >= 2, "Should have at least 2 numeric columns (Age, Score)");
}

#[wasm_bindgen_test]
fn test_wasm_csv_parsing_invalid() {
    let invalid_csv = "";

    let result_json = parse_csv(invalid_csv);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Empty CSV should still return valid structure with empty arrays
    let headers = result.get("headers")
        .expect("Should have headers field")
        .as_array()
        .unwrap();

    assert_eq!(headers.len(), 0, "Empty CSV should have no headers");
}

// ============================================================================
// Utility Function Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_get_t_cdf() {
    // t = 0 should give CDF = 0.5
    let cdf = get_t_cdf(0.0, 10.0);
    assert!((cdf - 0.5).abs() < 0.01, "t=0 should give CDF ≈ 0.5");

    // Large positive t should give CDF close to 1
    let cdf_high = get_t_cdf(10.0, 10.0);
    assert!(cdf_high > 0.99, "Large t should give CDF close to 1");
}

#[wasm_bindgen_test]
fn test_wasm_get_t_critical() {
    let t_crit = get_t_critical(0.05, 10.0);

    // For df=10, alpha=0.05, t-critical ≈ 2.23
    assert!(t_crit > 2.0 && t_crit < 2.5,
        "t-critical for df=10, alpha=0.05 should be ~2.23, got {}", t_crit);
}

#[wasm_bindgen_test]
fn test_wasm_get_normal_inverse() {
    // p = 0.5 should give z = 0
    let z = get_normal_inverse(0.5);
    assert!((z - 0.0).abs() < 0.01, "p=0.5 should give z ≈ 0");

    // p = 0.975 should give z ≈ 1.96
    let z_975 = get_normal_inverse(0.975);
    assert!(z_975 > 1.9 && z_975 < 2.0,
        "p=0.975 should give z ≈ 1.96, got {}", z_975);
}

#[wasm_bindgen_test]
fn test_wasm_get_version() {
    let version = get_version();

    assert!(!version.is_empty(), "Version should not be empty");
    assert!(version.contains('.'), "Version should contain dots");
}

// ============================================================================
// Integration Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_full_diagnostic_workflow() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    // Run regression
    let regression_result = ols_regression(&y_json, &x_vars_json, &names_json);
    let reg: serde_json::Value = serde_json::from_str(&regression_result).unwrap();
    assert!(reg.get("coefficients").is_some(), "Regression should succeed");

    // Run all 6 diagnostic tests
    let rainbow = rainbow_test(&y_json, &x_vars_json, 0.5, "r");
    assert!(serde_json::from_str::<serde_json::Value>(&rainbow).is_ok(),
        "Rainbow test should return valid JSON");

    let hc = harvey_collier_test(&y_json, &x_vars_json);
    assert!(serde_json::from_str::<serde_json::Value>(&hc).is_ok(),
        "Harvey-Collier test should return valid JSON");

    let bp = breusch_pagan_test(&y_json, &x_vars_json);
    assert!(serde_json::from_str::<serde_json::Value>(&bp).is_ok(),
        "Breusch-Pagan test should return valid JSON");

    let white = white_test(&y_json, &x_vars_json, "r");
    assert!(serde_json::from_str::<serde_json::Value>(&white).is_ok(),
        "White test should return valid JSON");

    let jb = jarque_bera_test(&y_json, &x_vars_json);
    assert!(serde_json::from_str::<serde_json::Value>(&jb).is_ok(),
        "Jarque-Bera test should return valid JSON");

    let dw = durbin_watson_test(&y_json, &x_vars_json);
    assert!(serde_json::from_str::<serde_json::Value>(&dw).is_ok(),
        "Durbin-Watson test should return valid JSON");
}
