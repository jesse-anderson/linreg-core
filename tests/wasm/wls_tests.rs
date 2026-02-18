// ============================================================================
// WLS Regression WASM Tests
// ============================================================================

#![cfg(target_arch = "wasm32")]

use crate::wasm::fixtures::*;
use wasm_bindgen_test::*;
use linreg_core::wasm::*;

#[wasm_bindgen_test]
fn test_wasm_wls_equal_weights_matches_ols() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let names = vec!["Intercept", "X"];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();
    let w_json = serde_json::to_string(&weights).unwrap();
    let n_json = serde_json::to_string(&names).unwrap();

    let wls_json = wls_regression(&y_json, &x_json, &w_json);
    let ols_json = ols_regression(&y_json, &x_json, &n_json);

    let wls: serde_json::Value = serde_json::from_str(&wls_json).unwrap();
    let ols: serde_json::Value = serde_json::from_str(&ols_json).unwrap();

    assert!(wls.get("error").is_none(), "WLS should not error");

    // With equal weights, WLS intercept (coefficients[0]) should match OLS
    let wls_intercept = wls["coefficients"][0].as_f64().unwrap();
    let ols_intercept = ols["coefficients"][0].as_f64().unwrap();
    assert!((wls_intercept - ols_intercept).abs() < 1e-8,
        "WLS with equal weights should match OLS intercept");
}

#[wasm_bindgen_test]
fn test_wasm_wls_result_fields() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();
    let w_json = serde_json::to_string(&weights).unwrap();

    let result: serde_json::Value = serde_json::from_str(&wls_regression(&y_json, &x_json, &w_json)).unwrap();

    for field in &["coefficients", "standard_errors", "t_statistics", "p_values",
                   "fitted_values", "residuals", "r_squared", "adj_r_squared",
                   "f_statistic", "f_p_value", "residual_std_error", "df_residuals"] {
        assert!(result.get(field).is_some(), "Missing field: {}", field);
    }
}

#[wasm_bindgen_test]
fn test_wasm_wls_outlier_downweighting() {
    // Low weight on outlier: fit should ignore it
    let y = vec![2.0, 4.0, 6.0, 8.0, 100.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let low_w = vec![1.0, 1.0, 1.0, 1.0, 0.01];
    let high_w = vec![1.0, 1.0, 1.0, 1.0, 10.0];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let r_low: serde_json::Value = serde_json::from_str(
        &wls_regression(&y_json, &x_json, &serde_json::to_string(&low_w).unwrap())
    ).unwrap();
    let r_high: serde_json::Value = serde_json::from_str(
        &wls_regression(&y_json, &x_json, &serde_json::to_string(&high_w).unwrap())
    ).unwrap();

    // coefficients[0] = intercept, coefficients[1] = slope
    let slope_low = r_low["coefficients"][1].as_f64().unwrap();
    let slope_high = r_high["coefficients"][1].as_f64().unwrap();

    // Low weight on outlier -> smaller slope; high weight -> larger slope
    assert!(slope_low < slope_high, "High weight on outlier should produce larger slope");
}

#[wasm_bindgen_test]
fn test_wasm_wls_negative_weight_error() {
    let y = vec![1.0, 2.0, 3.0];
    let x = vec![vec![1.0, 2.0, 3.0]];
    let weights = vec![1.0, -1.0, 1.0];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();
    let w_json = serde_json::to_string(&weights).unwrap();

    let result: serde_json::Value = serde_json::from_str(&wls_regression(&y_json, &x_json, &w_json)).unwrap();
    assert!(result.get("error").is_some(), "Negative weights should return an error");
}

#[wasm_bindgen_test]
fn test_wasm_wls_housing_data() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    // Equal weights
    let weights: Vec<f64> = vec![1.0; 25];
    let w_json = serde_json::to_string(&weights).unwrap();

    let result: serde_json::Value = serde_json::from_str(&wls_regression(&y_json, &x_vars_json, &w_json)).unwrap();

    assert!(result.get("error").is_none(), "WLS on housing data should not error");
    let r2 = result["r_squared"].as_f64().unwrap();
    assert!(r2 > 0.0 && r2 <= 1.0, "R² should be in [0, 1]");
}
