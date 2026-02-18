// ============================================================================
// LOESS Regression WASM Tests
// ============================================================================

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use linreg_core::wasm::*;

#[wasm_bindgen_test]
fn test_wasm_loess_fit_basic() {
    let y = vec![2.0, 4.1, 5.9, 8.2, 10.0, 11.8, 14.1];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result_json = loess_fit(&y_json, &x_json, 0.75, 1, 0, "direct");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "LOESS fit should not error");

    let fitted = result.get("fitted").expect("Should have fitted values").as_array().unwrap();
    assert_eq!(fitted.len(), 7, "Should have 7 fitted values");
    for v in fitted {
        assert!(v.as_f64().unwrap().is_finite(), "Fitted values should be finite");
    }
}

#[wasm_bindgen_test]
fn test_wasm_loess_fit_result_fields() {
    let y = vec![2.0, 4.1, 5.9, 8.2, 10.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result: serde_json::Value = serde_json::from_str(
        &loess_fit(&y_json, &x_json, 0.75, 1, 0, "direct")
    ).unwrap();

    for field in &["fitted", "span", "degree", "robust_iterations", "surface"] {
        assert!(result.get(field).is_some(), "Missing field: {}", field);
    }

    assert!((result["span"].as_f64().unwrap() - 0.75).abs() < 1e-10);
    assert_eq!(result["degree"].as_u64().unwrap(), 1);
    assert_eq!(result["robust_iterations"].as_u64().unwrap(), 0);
}

#[wasm_bindgen_test]
fn test_wasm_loess_fit_degree_zero() {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result: serde_json::Value = serde_json::from_str(
        &loess_fit(&y_json, &x_json, 0.8, 0, 0, "direct")
    ).unwrap();

    assert!(result.get("error").is_none(), "degree=0 LOESS should not error");
    let fitted = result["fitted"].as_array().unwrap();
    assert_eq!(fitted.len(), 5);
}

#[wasm_bindgen_test]
fn test_wasm_loess_fit_quadratic() {
    let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result: serde_json::Value = serde_json::from_str(
        &loess_fit(&y_json, &x_json, 1.0, 2, 0, "direct")
    ).unwrap();

    assert!(result.get("error").is_none(), "degree=2 LOESS should not error");
}

#[wasm_bindgen_test]
fn test_wasm_loess_fit_robust() {
    let y = vec![2.0, 4.0, 6.0, 100.0, 10.0]; // outlier at index 3
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();

    let result: serde_json::Value = serde_json::from_str(
        &loess_fit(&y_json, &x_json, 1.0, 1, 3, "direct")
    ).unwrap();

    assert!(result.get("error").is_none(), "Robust LOESS should not error");
    assert_eq!(result["robust_iterations"].as_u64().unwrap(), 3);
}

#[wasm_bindgen_test]
fn test_wasm_loess_predict_basic() {
    let y = vec![2.0, 4.1, 5.9, 8.2, 10.0, 11.8, 14.1];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]];
    let new_x = vec![vec![3.5, 5.5]]; // predict at new points

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    // loess_predict arg order: new_x, original_x, original_y, span, degree, robust_iterations, surface
    let result_json = loess_predict(&new_x_json, &x_json, &y_json, 0.75, 1, 0, "direct");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "LOESS predict should not error");
    let predictions = result.get("predictions").expect("Should have predictions").as_array().unwrap();
    assert_eq!(predictions.len(), 2, "Should predict 2 values");
    for v in predictions {
        assert!(v.as_f64().unwrap().is_finite(), "Predicted values should be finite");
    }
}

#[wasm_bindgen_test]
fn test_wasm_loess_predict_interpolation() {
    // For linear data, LOESS should interpolate well at interior points
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let new_x = vec![vec![2.5]]; // midpoint

    let y_json = serde_json::to_string(&y).unwrap();
    let x_json = serde_json::to_string(&x).unwrap();
    let new_x_json = serde_json::to_string(&new_x).unwrap();

    let result: serde_json::Value = serde_json::from_str(
        &loess_predict(&new_x_json, &x_json, &y_json, 1.0, 1, 0, "direct")
    ).unwrap();

    assert!(result.get("error").is_none());
    let pred = result["predictions"][0].as_f64().unwrap();
    // For linear y=2x, at x=2.5 expect ~5
    assert!((pred - 5.0).abs() < 1.0, "LOESS prediction at x=2.5 should be ~5, got {}", pred);
}
