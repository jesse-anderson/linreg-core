// ============================================================================
// Prediction Interval WASM Tests
// ============================================================================
//
// Tests for OLS, Ridge, Lasso, and Elastic Net prediction intervals.

#![cfg(target_arch = "wasm32")]

use crate::wasm::fixtures::*;
use wasm_bindgen_test::*;
use linreg_core::wasm::*;

// ============================================================================
// OLS Prediction Intervals
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_ols_prediction_intervals_basic() {
    let y = vec![3.1, 4.9, 7.2, 8.8, 11.1];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let new_x = vec![vec![6.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let result_json = ols_prediction_intervals(&y_json, &x_vars_json, &new_x_json, 0.05);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "Should not return an error");

    let predicted = result.get("predicted").unwrap().as_array().unwrap();
    let lower = result.get("lower_bound").unwrap().as_array().unwrap();
    let upper = result.get("upper_bound").unwrap().as_array().unwrap();
    let se_pred = result.get("se_pred").unwrap().as_array().unwrap();

    assert_eq!(predicted.len(), 1);
    assert_eq!(lower.len(), 1);
    assert_eq!(upper.len(), 1);
    assert_eq!(se_pred.len(), 1);

    let p = predicted[0].as_f64().unwrap();
    let lo = lower[0].as_f64().unwrap();
    let hi = upper[0].as_f64().unwrap();

    assert!(lo < p, "lower_bound should be below predicted");
    assert!(hi > p, "upper_bound should be above predicted");
    assert!(se_pred[0].as_f64().unwrap() > 0.0, "se_pred should be positive");
}

#[wasm_bindgen_test]
fn test_wasm_ols_prediction_intervals_fields() {
    let y = vec![3.1, 4.9, 7.2, 8.8, 11.1];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let new_x = vec![vec![6.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let result_json = ols_prediction_intervals(&y_json, &x_vars_json, &new_x_json, 0.05);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    for field in &["predicted", "lower_bound", "upper_bound", "se_pred", "leverage", "alpha", "df_residuals"] {
        assert!(result.get(field).is_some(), "Missing field: {}", field);
    }

    let alpha = result.get("alpha").unwrap().as_f64().unwrap();
    assert!((alpha - 0.05).abs() < 1e-10, "alpha should be 0.05");

    let df = result.get("df_residuals").unwrap().as_f64().unwrap();
    assert!(df > 0.0, "df_residuals should be positive");
}

#[wasm_bindgen_test]
fn test_wasm_ols_prediction_intervals_multiple_observations() {
    let y = vec![3.1, 4.9, 7.2, 8.8, 11.1];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let new_x = vec![vec![6.0, 7.0, 3.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let result_json = ols_prediction_intervals(&y_json, &x_vars_json, &new_x_json, 0.05);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    let predicted = result.get("predicted").unwrap().as_array().unwrap();
    assert_eq!(predicted.len(), 3, "Should have 3 predictions");
}

#[wasm_bindgen_test]
fn test_wasm_ols_prediction_intervals_housing() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    // Predict for a new house: 2000 sqft, 4 bed, 10 years old
    let new_x = vec![vec![2000.0], vec![4.0], vec![10.0]];
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let result_json = ols_prediction_intervals(&y_json, &x_vars_json, &new_x_json, 0.05);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "Should not return an error");
    let predicted = result.get("predicted").unwrap().as_array().unwrap();
    assert_eq!(predicted.len(), 1);
}

#[wasm_bindgen_test]
fn test_wasm_ols_prediction_intervals_wider_at_99pct() {
    let y = vec![3.1, 4.9, 7.2, 8.8, 11.1];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let new_x = vec![vec![3.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let r95 = serde_json::from_str::<serde_json::Value>(
        &ols_prediction_intervals(&y_json, &x_vars_json, &new_x_json, 0.05)
    ).unwrap();
    let r99 = serde_json::from_str::<serde_json::Value>(
        &ols_prediction_intervals(&y_json, &x_vars_json, &new_x_json, 0.01)
    ).unwrap();

    let w95 = r95["upper_bound"][0].as_f64().unwrap() - r95["lower_bound"][0].as_f64().unwrap();
    let w99 = r99["upper_bound"][0].as_f64().unwrap() - r99["lower_bound"][0].as_f64().unwrap();

    assert!(w99 > w95, "99% PI must be wider than 95% PI");
}

#[wasm_bindgen_test]
fn test_wasm_ols_prediction_intervals_invalid_alpha() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let new_x = vec![vec![6.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let result_json = ols_prediction_intervals(&y_json, &x_vars_json, &new_x_json, 0.0);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();
    assert!(result.get("error").is_some(), "alpha=0 should return an error");
}

// ============================================================================
// Ridge Prediction Intervals
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_ridge_prediction_intervals_basic() {
    let y = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]];
    let new_x = vec![vec![8.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let result_json = wasm_ridge_pi(&y_json, &x_vars_json, &new_x_json, 0.05, 0.1, true);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "Should not return an error");

    let predicted = result["predicted"][0].as_f64().unwrap();
    let lower = result["lower_bound"][0].as_f64().unwrap();
    let upper = result["upper_bound"][0].as_f64().unwrap();

    assert!(lower < predicted, "lower_bound should be below predicted");
    assert!(upper > predicted, "upper_bound should be above predicted");

    // y ≈ 2x + 1, so at x=8 expect ~17
    assert!((predicted - 17.0).abs() < 2.0, "Prediction at x=8 should be ~17, got {}", predicted);
}

#[wasm_bindgen_test]
fn test_wasm_ridge_prediction_intervals_wider_at_99pct() {
    let y = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]];
    let new_x = vec![vec![8.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let r95 = serde_json::from_str::<serde_json::Value>(
        &wasm_ridge_pi(&y_json, &x_vars_json, &new_x_json, 0.05, 0.1, true)
    ).unwrap();
    let r99 = serde_json::from_str::<serde_json::Value>(
        &wasm_ridge_pi(&y_json, &x_vars_json, &new_x_json, 0.01, 0.1, true)
    ).unwrap();

    let w95 = r95["upper_bound"][0].as_f64().unwrap() - r95["lower_bound"][0].as_f64().unwrap();
    let w99 = r99["upper_bound"][0].as_f64().unwrap() - r99["lower_bound"][0].as_f64().unwrap();

    assert!(w99 > w95, "99% PI must be wider than 95% PI");
}

// ============================================================================
// Lasso Prediction Intervals
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_lasso_prediction_intervals_basic() {
    let y = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]];
    let new_x = vec![vec![8.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let result_json = wasm_lasso_pi(&y_json, &x_vars_json, &new_x_json, 0.05, 0.01, true, 1000, 1e-7);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "Should not return an error");

    let predicted = result["predicted"][0].as_f64().unwrap();
    let lower = result["lower_bound"][0].as_f64().unwrap();
    let upper = result["upper_bound"][0].as_f64().unwrap();

    assert!(lower < predicted);
    assert!(upper > predicted);
    assert!(result["se_pred"][0].as_f64().unwrap() > 0.0);
}

// ============================================================================
// Elastic Net Prediction Intervals
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_elastic_net_prediction_intervals_basic() {
    let y = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]];
    let new_x = vec![vec![8.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let result_json = wasm_enet_pi(&y_json, &x_vars_json, &new_x_json, 0.05, 0.01, 0.5, true, 1000, 1e-7);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "Should not return an error");

    let predicted = result["predicted"][0].as_f64().unwrap();
    let lower = result["lower_bound"][0].as_f64().unwrap();
    let upper = result["upper_bound"][0].as_f64().unwrap();

    assert!(lower < predicted);
    assert!(upper > predicted);
}

#[wasm_bindgen_test]
fn test_wasm_elastic_net_prediction_intervals_ridge_variant() {
    // alpha=0 is pure Ridge; results should be close to wasm_ridge_pi
    let y = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]];
    let new_x = vec![vec![8.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_vars_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let enet = serde_json::from_str::<serde_json::Value>(
        &wasm_enet_pi(&y_json, &x_vars_json, &new_x_json, 0.05, 0.1, 0.0, true, 1000, 1e-7)
    ).unwrap();
    let ridge = serde_json::from_str::<serde_json::Value>(
        &wasm_ridge_pi(&y_json, &x_vars_json, &new_x_json, 0.05, 0.1, true)
    ).unwrap();

    let enet_pred = enet["predicted"][0].as_f64().unwrap();
    let ridge_pred = ridge["predicted"][0].as_f64().unwrap();

    assert!((enet_pred - ridge_pred).abs() < 0.5, "alpha=0 enet should be close to ridge");
}
