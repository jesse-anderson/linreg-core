// ============================================================================
// Diagnostic Tests WASM Tests
// ============================================================================
//
// Tests for all statistical diagnostic tests.

#![cfg(target_arch = "wasm32")]

use crate::wasm::fixtures::*;
use wasm_bindgen_test::*;
use linreg_core::wasm::*;

// ============================================================================
// Linearity Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_rainbow_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = rainbow_test(&y_json, &x_vars_json, 0.5, "r");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("Rainbow"),
        "Test name should mention Rainbow"
    );

    // Should have R result
    let r_result = result
        .get("r_result")
        .expect("Should have r_result for 'r' method");

    assert!(r_result.is_object(), "r_result should be an object");

    let statistic = r_result
        .get("statistic")
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

    // Check for error first
    if let Some(_err) = result.get("error") {
        panic!("rainbow_test returned error: {}", _err);
    }

    // Should have python_result, not r_result (None serializes to null)
    let python_result = result.get("python_result").expect("Should have python_result");
    assert!(
        python_result.is_object() || python_result.is_null(),
        "python_result should be an object or null"
    );

    let r_result = result.get("r_result").expect("r_result key should exist");
    assert!(
        r_result.is_null(),
        "r_result should be null for python method, got: {:?}", r_result
    );
}

#[wasm_bindgen_test]
fn test_wasm_rainbow_test_both_methods() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = rainbow_test(&y_json, &x_vars_json, 0.5, "both");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have both results
    assert!(result.get("r_result").is_some(), "Should have r_result");
    assert!(
        result.get("python_result").is_some(),
        "Should have python_result"
    );
}

#[wasm_bindgen_test]
fn test_wasm_harvey_collier_test() {
    // Use simpler data to avoid multicollinearity issues with housing data
    let y_json = serde_json::to_string(&vec![
        2.0, 4.0, 5.0, 4.0, 11.0, 14.0, 13.0, 10.0, 8.0, 15.0,
    ]).unwrap();
    // x_vars_json should be an array of arrays: [[x values]]
    let x_vars = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
    let x_vars_json = serde_json::to_string(&x_vars).unwrap();

    let result_json = harvey_collier_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Check for error first
    if let Some(err) = result.get("error") {
        panic!("harvey_collier_test returned error: {}", err);
    }

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("Harvey-Collier") || test_name.contains("Harvey"),
        "Test name should mention Harvey-Collier"
    );

    // Should have statistic
    let statistic = result
        .get("statistic")
        .expect("Should have statistic")
        .as_f64()
        .unwrap();

    assert!(statistic >= 0.0, "Statistic should be non-negative");
}

#[wasm_bindgen_test]
fn test_wasm_reset_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let powers_json = serde_json::to_string(&vec![2usize, 3]).unwrap();

    // Test with "fitted" type (default)
    let result_json = reset_test(&y_json, &x_vars_json, &powers_json, "fitted");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("RESET") || test_name.contains("Reset"),
        "Test name should mention RESET"
    );

    // Should have F-statistic (returned as generic "statistic")
    let f_statistic = result
        .get("statistic")
        .expect("Should have statistic")
        .as_f64()
        .unwrap();

    assert!(f_statistic >= 0.0, "F-statistic should be non-negative");

    // Should have p_value
    assert!(result.get("p_value").is_some(), "Should have p_value");
}

#[wasm_bindgen_test]
fn test_wasm_reset_test_regressor_type() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let powers_json = serde_json::to_string(&vec![2usize]).unwrap();

    // Test with "regressor" type
    let result_json = reset_test(&y_json, &x_vars_json, &powers_json, "regressor");
    eprintln!("DEBUG reset regressor: {}", result_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(
        result.get("statistic").is_some(),
        "Regressor type RESET should have statistic, got: {:?}",
        result
    );
}

// ============================================================================
// Heteroscedasticity Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_breusch_pagan_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = breusch_pagan_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("Breusch-Pagan") || test_name.contains("Breusch"),
        "Test name should mention Breusch-Pagan"
    );

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
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("White"),
        "Test name should mention White"
    );

    // Should have R result
    let r_result = result
        .get("r_result")
        .expect("Should have r_result for 'r' method");

    assert!(r_result.is_object(), "r_result should be an object");
}

#[wasm_bindgen_test]
fn test_wasm_r_white_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = r_white_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Check for error first
    if let Some(err) = result.get("error") {
        panic!("r_white_test returned error: {}", err);
    }

    // WhiteSingleResult has method, statistic, p_value, passed (not test_name)
    let method = result
        .get("method")
        .expect("Should have method")
        .as_str()
        .unwrap();

    assert!(method.contains("R"), "Method should contain 'R'");

    // Should have statistic and p_value
    assert!(result.get("statistic").is_some(), "Should have statistic");
    assert!(result.get("p_value").is_some(), "Should have p_value");
}

#[wasm_bindgen_test]
fn test_wasm_python_white_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = python_white_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Check for error first
    if let Some(err) = result.get("error") {
        panic!("python_white_test returned error: {}", err);
    }

    // WhiteSingleResult has method, statistic, p_value, passed (not test_name)
    let method = result
        .get("method")
        .expect("Should have method")
        .as_str()
        .unwrap();

    assert!(method.contains("Python"), "Method should contain 'Python'");

    // Should have statistic and p_value
    assert!(result.get("statistic").is_some(), "Should have statistic");
    assert!(result.get("p_value").is_some(), "Should have p_value");
}

// ============================================================================
// Normality Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_jarque_bera_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = jarque_bera_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("Jarque-Bera") || test_name.contains("Jarque"),
        "Test name should mention Jarque-Bera"
    );

    // Should have statistic
    let statistic = result
        .get("statistic")
        .expect("Should have statistic")
        .as_f64()
        .unwrap();

    assert!(statistic >= 0.0, "Statistic should be non-negative");
}

#[wasm_bindgen_test]
fn test_wasm_shapiro_wilk_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = shapiro_wilk_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("Shapiro-Wilk") || test_name.contains("Shapiro"),
        "Test name should mention Shapiro-Wilk"
    );

    // W statistic should be in [0, 1]
    let statistic = result
        .get("statistic")
        .expect("Should have statistic (W)")
        .as_f64()
        .unwrap();

    assert!(
        statistic >= 0.0 && statistic <= 1.0,
        "W statistic should be in [0, 1], got {}",
        statistic
    );

    // Should have p_value
    assert!(result.get("p_value").is_some(), "Should have p_value");
}

#[wasm_bindgen_test]
fn test_wasm_anderson_darling_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = anderson_darling_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("Anderson-Darling") || test_name.contains("Anderson"),
        "Test name should mention Anderson-Darling"
    );

    // Should have statistic (A²)
    let statistic = result
        .get("statistic")
        .expect("Should have statistic")
        .as_f64()
        .unwrap();

    assert!(statistic >= 0.0, "A² statistic should be non-negative");
}

// ============================================================================
// Autocorrelation Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_durbin_watson_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = durbin_watson_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Check for error first
    if let Some(err) = result.get("error") {
        panic!("durbin_watson_test returned error: {}", err);
    }

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("Durbin-Watson") || test_name.contains("Durbin"),
        "Test name should mention Durbin-Watson"
    );

    // DW statistic should be in [0, 4]
    let statistic = result
        .get("statistic")
        .expect("Should have statistic")
        .as_f64()
        .unwrap();

    assert!(
        statistic >= 0.0 && statistic <= 4.0,
        "DW statistic should be in [0, 4], got {}",
        statistic
    );
}

#[wasm_bindgen_test]
fn test_wasm_breusch_godfrey_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    // Test with order=2 (test for second-order autocorrelation)
    let result_json = breusch_godfrey_test(&y_json, &x_vars_json, 2, "chisq");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("Breusch-Godfrey") || test_name.contains("Breusch"),
        "Test name should mention Breusch-Godfrey"
    );

    // Should have statistic
    let statistic = result
        .get("statistic")
        .expect("Should have statistic")
        .as_f64()
        .unwrap();

    assert!(statistic >= 0.0, "Statistic should be non-negative");

    // Should have p_value
    assert!(result.get("p_value").is_some(), "Should have p_value");

    // Should have order
    let order = result
        .get("order")
        .expect("Should have order")
        .as_u64()
        .unwrap();

    assert_eq!(order, 2, "Order should be 2");
}

#[wasm_bindgen_test]
fn test_wasm_breusch_godfrey_test_f_type() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    // Test with "f" test type
    let result_json = breusch_godfrey_test(&y_json, &x_vars_json, 1, "f");
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(
        result.get("statistic").is_some(),
        "F test type should have statistic"
    );

    let test_type = result
        .get("test_type")
        .expect("Should have test_type")
        .as_str()
        .unwrap();

    assert_eq!(test_type, "F", "Test type should be F");
}

// ============================================================================
// Influential Observations Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_cooks_distance_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = cooks_distance_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Check for error first
    if let Some(err) = result.get("error") {
        panic!("cooks_distance_test returned error: {}", err);
    }

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("Cook"),
        "Test name should mention Cook's Distance"
    );

    // Should have distances array
    let distances = result
        .get("distances")
        .expect("Should have distances array")
        .as_array()
        .unwrap();

    assert_eq!(
        distances.len(),
        25,
        "Should have 25 Cook's distances (one per observation)"
    );

    // All distances should be non-negative
    for (i, dist) in distances.iter().enumerate() {
        let d = dist.as_f64().unwrap();
        assert!(
            d >= 0.0,
            "Cook's distance at index {} should be non-negative, got {}",
            i,
            d
        );
    }

    // Should have influential_1 array (D_i > 1 threshold)
    let influential = result
        .get("influential_1")
        .expect("Should have influential_1 array")
        .as_array()
        .unwrap();

    // Check we got some array (may be empty if no influential points)
    // Array type already confirmed by as_array()
}

#[wasm_bindgen_test]
fn test_wasm_dfbetas_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = dfbetas_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Check for error first
    if let Some(err) = result.get("error") {
        panic!("dfbetas_test returned error: {}", err);
    }

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("DFBETAS") || test_name.contains("DFBETAS"),
        "Test name should mention DFBETAS"
    );

    // Should have dfbetas matrix (n x p)
    let dfbetas = result
        .get("dfbetas")
        .expect("Should have dfbetas matrix")
        .as_array()
        .unwrap();

    assert_eq!(
        dfbetas.len(),
        25,
        "Should have 25 rows (one per observation)"
    );

    // Each row should have p=4 values (intercept + 3 predictors)
    for (i, row) in dfbetas.iter().enumerate() {
        let row_array = row.as_array().expect("Row should be an array");
        assert_eq!(
            row_array.len(),
            4,
            "Row {} should have 4 DFBETAS values (one per coefficient)",
            i
        );
    }

    // Should have threshold
    let threshold = result
        .get("threshold")
        .expect("Should have threshold")
        .as_f64()
        .unwrap();

    assert!(threshold > 0.0, "Threshold should be positive");

    // Should have n and p
    let n = result
        .get("n")
        .expect("Should have n")
        .as_u64()
        .unwrap();

    let p = result
        .get("p")
        .expect("Should have p")
        .as_u64()
        .unwrap();

    assert_eq!(n, 25, "n should be 25");
    assert_eq!(p, 4, "p should be 4 (intercept + 3 predictors)");

    // Should have influential_observations (may be empty object)
    assert!(
        result.get("influential_observations").is_some(),
        "Should have influential_observations"
    );
}

#[wasm_bindgen_test]
fn test_wasm_dffits_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = dffits_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    // Check for error first
    if let Some(err) = result.get("error") {
        panic!("dffits_test returned error: {}", err);
    }

    // Should have test name
    let test_name = result
        .get("test_name")
        .expect("Should have test_name")
        .as_str()
        .unwrap();

    assert!(
        test_name.contains("DFFITS") || test_name.contains("DFFITS"),
        "Test name should mention DFFITS"
    );

    // Should have dffits array (one per observation)
    let dffits = result
        .get("dffits")
        .expect("Should have dffits array")
        .as_array()
        .unwrap();

    assert_eq!(
        dffits.len(),
        25,
        "Should have 25 DFFITS values (one per observation)"
    );

    // All DFFITS should be finite (can be positive or negative)
    for (i, d) in dffits.iter().enumerate() {
        let val = d.as_f64().unwrap();
        assert!(
            val.is_finite(),
            "DFFITS at index {} should be finite, got {}",
            i,
            val
        );
    }

    // Should have threshold (2*sqrt(p/n))
    let threshold = result
        .get("threshold")
        .expect("Should have threshold")
        .as_f64()
        .unwrap();

    assert!(threshold > 0.0, "Threshold should be positive");

    // Should have n and p
    let n = result
        .get("n")
        .expect("Should have n")
        .as_u64()
        .unwrap();

    let p = result
        .get("p")
        .expect("Should have p")
        .as_u64()
        .unwrap();

    assert_eq!(n, 25, "n should be 25");
    assert_eq!(p, 4, "p should be 4 (intercept + 3 predictors)");

    // Should have influential_observations array (may be empty)
    let influential = result
        .get("influential_observations")
        .expect("Should have influential_observations array")
        .as_array()
        .unwrap();

    // Check we got some array (may be empty if no influential points)
    // Array type already confirmed by as_array()
}

// ============================================================================
// VIF Test
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_vif_test() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();

    let result_json = vif_test(&y_json, &x_vars_json);
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(result.get("error").is_none(), "vif_test should not error");

    let test_name = result["test_name"].as_str().unwrap();
    assert!(test_name.contains("VIF") || test_name.contains("Variance"), "Test name should mention VIF");

    let vif_results = result.get("vif_results").expect("Should have vif_results").as_array().unwrap();
    assert_eq!(vif_results.len(), 3, "Housing data has 3 predictors; should have 3 VIF values");

    for v in vif_results {
        let vif = v["vif"].as_f64().unwrap();
        assert!(vif >= 1.0, "VIF should be >= 1.0, got {}", vif);
    }
}
