//! Model serialization for WASM
//!
//! Provides WASM bindings for serializing and deserializing trained models.
//! Since WASM cannot directly access the file system, these functions work with
//! JSON strings that the browser can save to IndexedDB or trigger downloads.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use super::domain::check_domain;
use crate::error::{error_json, error_to_json};
use crate::serialization::types::{ModelMetadata, ModelType, SerializedModel};
use crate::serialization::FORMAT_VERSION;

/// Generate an ISO 8601 timestamp using JavaScript Date.
///
/// This is a WASM-specific version that uses the browser's Date object
/// instead of SystemTime, which isn't available in WASM.
fn iso_timestamp_wasm() -> String {
    // Get current date/time in ISO format from JavaScript
    let date = js_sys::Date::new_0();
    // Convert JsString to Rust String and format to ISO 8601
    date.to_iso_string().into()
}

/// Serialize a model by wrapping its JSON data with metadata.
///
/// This function takes a model's JSON representation (as returned by regression
/// functions), wraps it with version and type metadata, and returns a serialized
/// JSON string suitable for storage or download.
///
/// # Arguments
///
/// * `model_json` - JSON string of the model result (e.g., from ols_regression)
/// * `model_type` - Type of model: "OLS", "Ridge", "Lasso", "ElasticNet", "WLS", or "LOESS"
/// * `name` - Optional custom name for the model
///
/// # Returns
///
/// JSON string containing the serialized model with metadata, or a JSON error object
/// if the input is invalid or the domain check fails.
///
/// # Example
///
/// ```javascript
/// import { serialize_model, ols_regression } from './linreg_core.js';
///
/// // Train a model
/// const resultJson = ols_regression(yJson, xJson, namesJson);
///
/// // Serialize it
/// const serialized = serialize_model(resultJson, "OLS", "My Housing Model");
///
/// // Download (browser-side)
/// const blob = new Blob([serialized], { type: 'application/json' });
/// const url = URL.createObjectURL(blob);
/// const a = document.createElement('a');
/// a.href = url;
/// a.download = 'my_model.json';
/// a.click();
/// ```
#[wasm_bindgen]
pub fn serialize_model(model_json: &str, model_type: &str, name: Option<String>) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    // Parse the model JSON
    let model_data: serde_json::Value = match serde_json::from_str(model_json) {
        Ok(data) => data,
        Err(e) => return error_json(&format!("Failed to parse model JSON: {}", e)),
    };

    // Parse the model type
    let model_type_enum: ModelType = match model_type.parse() {
        Ok(t) => t,
        Err(_) => {
            return error_json(&format!(
                "Invalid model_type: '{}'. Must be one of: OLS, Ridge, Lasso, ElasticNet, WLS, LOESS",
                model_type
            ))
        }
    };

    // Create metadata with WASM-compatible timestamp
    let metadata = ModelMetadata {
        format_version: FORMAT_VERSION.to_string(),
        library_version: env!("CARGO_PKG_VERSION").to_string(),
        model_type: model_type_enum,
        created_at: iso_timestamp_wasm(),
        name,
    };

    // Create and serialize the model
    let serialized = SerializedModel::new(metadata, model_data);
    match serde_json::to_string(&serialized) {
        Ok(json) => json,
        Err(e) => error_json(&format!("Failed to serialize model: {}", e)),
    }
}

/// Deserialize a serialized model, extracting the inner model data.
///
/// This function takes a serialized model JSON (as created by serialize_model),
/// validates the format version, and returns the inner model data as JSON.
///
/// # Arguments
///
/// * `json_string` - JSON string of the serialized model (with metadata wrapper)
///
/// # Returns
///
/// JSON string of the inner model data (coefficients, statistics, etc.),
/// or a JSON error object if the input is invalid, the format version is
/// incompatible, or the domain check fails.
///
/// # Example
///
/// ```javascript
/// import { deserialize_model } from './linreg_core.js';
///
/// // Load from file (browser-side)
/// const response = await fetch('my_model.json');
/// const serialized = await response.text();
///
/// // Deserialize to get the model data
/// const modelJson = deserialize_model(serialized);
/// const model = JSON.parse(modelJson);
///
/// console.log(model.coefficients);
/// console.log(model.r_squared);
/// ```
#[wasm_bindgen]
pub fn deserialize_model(json_string: &str) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    // Parse the serialized model
    let serialized: SerializedModel = match serde_json::from_str(json_string) {
        Ok(model) => model,
        Err(e) => return error_json(&format!("Failed to parse serialized model: {}", e)),
    };

    // Validate format version
    let file_version = &serialized.metadata.format_version;
    let parts: Vec<&str> = file_version.split('.').collect();
    let supported_parts: Vec<&str> = FORMAT_VERSION.split('.').collect();

    if parts.len() >= 2 && supported_parts.len() >= 2 {
        // Check major version compatibility
        if let (Ok(file_major), Ok(supported_major)) = (
            parts[0].parse::<u32>(),
            supported_parts[0].parse::<u32>(),
        ) {
            if file_major > supported_major {
                return error_json(&format!(
                    "Incompatible format version: file is v{}, supported is v{}",
                    file_version, FORMAT_VERSION
                ));
            }
        }
    }

    // Return the inner model data
    match serde_json::to_string(&serialized.data) {
        Ok(json) => json,
        Err(e) => error_json(&format!("Failed to serialize model data: {}", e)),
    }
}

/// Extract metadata from a serialized model without deserializing the full model.
///
/// This function returns only the metadata portion of a serialized model,
/// which includes information like model type, library version, creation time,
/// and optional model name.
///
/// # Arguments
///
/// * `json_string` - JSON string of the serialized model
///
/// # Returns
///
/// JSON string containing the metadata object with fields:
/// - `format_version` - Format version (e.g., "1.0")
/// - `library_version` - linreg-core version used to create the model
/// - `model_type` - Type of model ("OLS", "Ridge", etc.)
/// - `created_at` - ISO 8601 timestamp of creation
/// - `name` - Optional custom model name
///
/// Returns a JSON error object if the input is invalid or the domain check fails.
///
/// # Example
///
/// ```javascript
/// import { get_model_metadata } from './linreg_core.js';
///
/// const response = await fetch('my_model.json');
/// const serialized = await response.text();
///
/// const metadataJson = get_model_metadata(serialized);
/// const metadata = JSON.parse(metadataJson);
///
/// console.log('Model type:', metadata.model_type);
/// console.log('Created:', metadata.created_at);
/// console.log('Name:', metadata.name || '(unnamed)');
/// ```
#[wasm_bindgen]
pub fn get_model_metadata(json_string: &str) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    // Parse the serialized model
    let serialized: SerializedModel = match serde_json::from_str(json_string) {
        Ok(model) => model,
        Err(e) => return error_json(&format!("Failed to parse serialized model: {}", e)),
    };

    // Return just the metadata
    match serde_json::to_string(&serialized.metadata) {
        Ok(json) => json,
        Err(e) => error_json(&format!("Failed to serialize metadata: {}", e)),
    }
}
