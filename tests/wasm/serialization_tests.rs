// ============================================================================
// Model Serialization WASM Tests
// ============================================================================

#![cfg(target_arch = "wasm32")]

use crate::wasm::fixtures::*;
use wasm_bindgen_test::*;
use linreg_core::wasm::*;

// ============================================================================
// serialize_model
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_serialize_ols_model() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let model_json = ols_regression(&y_json, &x_vars_json, &names_json);
    let serialized_json = serialize_model(&model_json, "OLS", Some("Housing Model".to_string()));
    let serialized: serde_json::Value = serde_json::from_str(&serialized_json).unwrap();

    assert!(serialized.get("error").is_none(), "serialize_model should not error");
    assert!(serialized.get("metadata").is_some(), "Should have metadata");
    assert!(serialized.get("data").is_some(), "Should have model data");

    let metadata = &serialized["metadata"];
    assert_eq!(metadata["model_type"].as_str().unwrap(), "OLS");
    assert_eq!(metadata["name"].as_str().unwrap(), "Housing Model");
    assert!(metadata.get("format_version").is_some());
    assert!(metadata.get("library_version").is_some());
    assert!(metadata.get("created_at").is_some());
}

#[wasm_bindgen_test]
fn test_wasm_serialize_ridge_model() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let model_json = ridge_regression(&y_json, &x_vars_json, &names_json, 1.0, true);
    let serialized_json = serialize_model(&model_json, "Ridge", None);
    let serialized: serde_json::Value = serde_json::from_str(&serialized_json).unwrap();

    assert!(serialized.get("error").is_none());
    assert_eq!(serialized["metadata"]["model_type"].as_str().unwrap(), "Ridge");
}

#[wasm_bindgen_test]
fn test_wasm_serialize_invalid_model_type() {
    let model_json = r#"{"intercept": 1.0}"#;
    let result: serde_json::Value = serde_json::from_str(
        &serialize_model(model_json, "InvalidType", None)
    ).unwrap();
    assert!(result.get("error").is_some(), "Invalid model type should return an error");
}

// ============================================================================
// deserialize_model
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_serialize_deserialize_roundtrip() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let model_json = ols_regression(&y_json, &x_vars_json, &names_json);
    let serialized_json = serialize_model(&model_json, "OLS", Some("Test".to_string()));

    let deserialized_json = deserialize_model(&serialized_json);
    let deserialized: serde_json::Value = serde_json::from_str(&deserialized_json).unwrap();

    assert!(deserialized.get("error").is_none(), "deserialize_model should not error");
    // The deserialized model should have the same coefficients as the original
    let original: serde_json::Value = serde_json::from_str(&model_json).unwrap();
    let orig_coefs = original["coefficients"].as_array().unwrap();
    let deser_coefs = deserialized["coefficients"].as_array().unwrap();
    assert_eq!(orig_coefs.len(), deser_coefs.len(), "Coefficient count should match after roundtrip");
}

#[wasm_bindgen_test]
fn test_wasm_deserialize_invalid_json() {
    let result: serde_json::Value = serde_json::from_str(
        &deserialize_model("not valid json")
    ).unwrap();
    assert!(result.get("error").is_some(), "Invalid JSON should return an error");
}

#[wasm_bindgen_test]
fn test_wasm_deserialize_missing_metadata() {
    let result: serde_json::Value = serde_json::from_str(
        &deserialize_model(r#"{"model": {"intercept": 1.0}}"#)
    ).unwrap();
    assert!(result.get("error").is_some(), "Missing metadata should return an error");
}

// ============================================================================
// get_model_metadata
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_get_model_metadata() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let model_json = ols_regression(&y_json, &x_vars_json, &names_json);
    let serialized_json = serialize_model(&model_json, "OLS", Some("My Model".to_string()));

    let metadata_json = get_model_metadata(&serialized_json);
    let metadata: serde_json::Value = serde_json::from_str(&metadata_json).unwrap();

    assert!(metadata.get("error").is_none(), "get_model_metadata should not error");
    assert_eq!(metadata["model_type"].as_str().unwrap(), "OLS");
    assert_eq!(metadata["name"].as_str().unwrap(), "My Model");
    assert!(metadata.get("format_version").is_some());
    assert!(metadata.get("library_version").is_some());
}

#[wasm_bindgen_test]
fn test_wasm_get_model_metadata_all_types() {
    let y_json = get_housing_y();
    let x_vars_json = get_housing_x_vars();
    let names_json = get_variable_names();

    let model_types = [
        (ridge_regression(&y_json, &x_vars_json, &names_json, 1.0, true), "Ridge"),
        (lasso_regression(&y_json, &x_vars_json, &names_json, 0.1, true, 1000, 1e-7), "Lasso"),
        (elastic_net_regression(&y_json, &x_vars_json, &names_json, 0.1, 0.5, true, 1000, 1e-7), "ElasticNet"),
    ];

    for (model_json, model_type) in &model_types {
        let serialized = serialize_model(model_json, model_type, None);
        let metadata: serde_json::Value = serde_json::from_str(&get_model_metadata(&serialized)).unwrap();
        assert!(metadata.get("error").is_none(), "get_model_metadata should not error for {}", model_type);
        assert_eq!(metadata["model_type"].as_str().unwrap(), *model_type);
    }
}
