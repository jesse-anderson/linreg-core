// ============================================================================
// Integration Tests
// ============================================================================
//
// End-to-end workflow tests that combine multiple WASM functions.

#![cfg(target_arch = "wasm32")]

use crate::wasm::fixtures::*;
use wasm_bindgen_test::*;
use linreg_core::wasm::*;

#[wasm_bindgen_test]
fn test_wasm_full_diagnostic_workflow() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    // Run regression
    let regression_result = ols_regression(&y_json, &x_vars_json, &names_json);
    let reg: serde_json::Value = serde_json::from_str(&regression_result).unwrap();
    assert!(
        reg.get("coefficients").is_some(),
        "Regression should succeed"
    );

    // Run all diagnostic tests
    let rainbow = rainbow_test(&y_json, &x_vars_json, 0.5, "r");
    assert!(
        serde_json::from_str::<serde_json::Value>(&rainbow).is_ok(),
        "Rainbow test should return valid JSON"
    );

    let hc = harvey_collier_test(&y_json, &x_vars_json);
    assert!(
        serde_json::from_str::<serde_json::Value>(&hc).is_ok(),
        "Harvey-Collier test should return valid JSON"
    );

    let bp = breusch_pagan_test(&y_json, &x_vars_json);
    assert!(
        serde_json::from_str::<serde_json::Value>(&bp).is_ok(),
        "Breusch-Pagan test should return valid JSON"
    );

    let white = white_test(&y_json, &x_vars_json, "r");
    assert!(
        serde_json::from_str::<serde_json::Value>(&white).is_ok(),
        "White test should return valid JSON"
    );

    let jb = jarque_bera_test(&y_json, &x_vars_json);
    assert!(
        serde_json::from_str::<serde_json::Value>(&jb).is_ok(),
        "Jarque-Bera test should return valid JSON"
    );

    let dw = durbin_watson_test(&y_json, &x_vars_json);
    assert!(
        serde_json::from_str::<serde_json::Value>(&dw).is_ok(),
        "Durbin-Watson test should return valid JSON"
    );

    let sw = shapiro_wilk_test(&y_json, &x_vars_json);
    assert!(
        serde_json::from_str::<serde_json::Value>(&sw).is_ok(),
        "Shapiro-Wilk test should return valid JSON"
    );

    let ad = anderson_darling_test(&y_json, &x_vars_json);
    assert!(
        serde_json::from_str::<serde_json::Value>(&ad).is_ok(),
        "Anderson-Darling test should return valid JSON"
    );

    let cd = cooks_distance_test(&y_json, &x_vars_json);
    assert!(
        serde_json::from_str::<serde_json::Value>(&cd).is_ok(),
        "Cook's distance test should return valid JSON"
    );

    let reset = reset_test(&y_json, &x_vars_json, &serde_json::to_string(&vec![2, 3]).unwrap(), "fitted");
    assert!(
        serde_json::from_str::<serde_json::Value>(&reset).is_ok(),
        "RESET test should return valid JSON"
    );

    let bg = breusch_godfrey_test(&y_json, &x_vars_json, 2, "chisq");
    assert!(
        serde_json::from_str::<serde_json::Value>(&bg).is_ok(),
        "Breusch-Godfrey test should return valid JSON"
    );
}

#[wasm_bindgen_test]
fn test_wasm_full_regularized_workflow() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    // Run all three regularized regression types
    let ridge = ridge_regression(&y_json, &x_vars_json, &names_json, 1.0, true);
    assert!(
        serde_json::from_str::<serde_json::Value>(&ridge).is_ok(),
        "Ridge regression should return valid JSON"
    );

    let lasso = lasso_regression(&y_json, &x_vars_json, &names_json, 0.1, true, 1000, 1e-7);
    assert!(
        serde_json::from_str::<serde_json::Value>(&lasso).is_ok(),
        "Lasso regression should return valid JSON"
    );

    let enet = elastic_net_regression(&y_json, &x_vars_json, &names_json, 0.1, 0.5, true, 1000, 1e-7);
    assert!(
        serde_json::from_str::<serde_json::Value>(&enet).is_ok(),
        "Elastic Net regression should return valid JSON"
    );

    // Lambda path generation
    let path = make_lambda_path(&y_json, &x_vars_json, 50, 0.01);
    assert!(
        serde_json::from_str::<serde_json::Value>(&path).is_ok(),
        "Lambda path should return valid JSON"
    );
}
