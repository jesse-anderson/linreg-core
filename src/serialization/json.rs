//! JSON file I/O for model serialization.
//!
//! Provides functions for reading and writing model files,
//! plus utility functions for timestamp generation and version validation.

use crate::error::Error;
use crate::serialization::types::SerializedModel;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

/// Generate an ISO 8601 timestamp from the current system time.
///
/// Returns a string in the format "YYYY-MM-DDTHH:MM:SSZ" using UTC.
/// This avoids adding a chrono dependency.
///
/// # Example
///
/// ```ignore
/// let ts = iso_timestamp();
/// // Returns something like "2026-02-10T15:30:00Z"
/// ```
pub fn iso_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = now.as_secs();

    // Convert Unix timestamp to UTC datetime components
    // Algorithm from: https://howardhinnant.github.io/date_algorithms.html
    let days_since_epoch = secs / 86400;
    let secs_in_day = secs % 86400;

    // Days from March 1, 0000 (proleptic Gregorian calendar)
    let days = days_since_epoch + 719468;

    // Extract year, month, day
    let era = days / 146097;
    let day_of_era = days % 146097;
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) / 365;
    let year = era * 400 + year_of_era;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let day = day_of_year - (153 * mp + 2) / 5 + 1;

    // Adjust year if month is January or February
    let adjusted_year = if month <= 2 { year - 1 } else { year };

    // Calculate time components
    let hours = secs_in_day / 3600;
    let mins = (secs_in_day % 3600) / 60;
    let secs = secs_in_day % 60;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        adjusted_year, month, day, hours, mins, secs
    )
}

/// Validate that a format version is compatible with the current version.
///
/// Compatibility rules:
/// - Major version must match exactly (breaking changes)
/// - Minor version can be higher (forward compatible)
/// - Minor version can be lower (backward compatible)
///
/// # Arguments
///
/// * `file_version` - Version string from the loaded file (e.g., "1.0")
///
/// # Returns
///
/// Returns `Ok(())` if compatible, `Error::IncompatibleFormatVersion` otherwise.
pub fn validate_format_version(file_version: &str) -> Result<(), Error> {
    let current_major = super::FORMAT_VERSION
        .split('.')
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1);

    let file_major = file_version
        .split('.')
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    if current_major != file_major {
        return Err(Error::IncompatibleFormatVersion {
            file_version: file_version.to_string(),
            supported: super::FORMAT_VERSION.to_string(),
        });
    }

    Ok(())
}

/// Save a serialized model to a file as formatted JSON.
///
/// The output is pretty-printed with 4-space indentation for human readability.
///
/// # Arguments
///
/// * `model` - The serialized model to save
/// * `path` - File path to write to
///
/// # Returns
///
/// Returns `Ok(())` on success, or an `Error` if serialization or file I/O fails.
pub fn save_to_file(model: &SerializedModel, path: &str) -> Result<(), Error> {
    // Validate format version before saving (sanity check)
    validate_format_version(&model.metadata.format_version)?;

    // Serialize to pretty JSON
    let json = serde_json::to_string_pretty(model).map_err(|e| {
        Error::SerializationError(format!("Failed to serialize model: {}", e))
    })?;

    // Write to file
    fs::write(path, json).map_err(|e| {
        Error::IoError(format!("Failed to write to file '{}': {}", path, e))
    })?;

    Ok(())
}

/// Load a serialized model from a file.
///
/// This validates the format version but does not validate the model type.
/// Type validation happens when converting to a specific model type.
///
/// # Arguments
///
/// * `path` - File path to read from
///
/// # Returns
///
/// Returns the `SerializedModel` on success, or an `Error` if reading fails.
pub fn load_from_file(path: &str) -> Result<SerializedModel, Error> {
    // Read file content
    let content = fs::read_to_string(path).map_err(|e| {
        Error::IoError(format!("Failed to read file '{}': {}", path, e))
    })?;

    // Parse JSON
    let model: SerializedModel = serde_json::from_str(&content).map_err(|e| {
        Error::DeserializationError(format!("Failed to parse JSON from '{}': {}", path, e))
    })?;

    // Validate format version
    validate_format_version(&model.metadata.format_version)?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialization::{ModelMetadata, ModelType};
    use serde_json::json;

    #[test]
    fn test_iso_timestamp_format() {
        let ts = iso_timestamp();
        // Should match ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        assert!(ts.len() == 20);
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
        // Check basic structure
        let parts: Vec<&str> = ts.split(&['T', '-', ':'][..]).collect();
        assert_eq!(parts.len(), 6);
    }

    #[test]
    fn test_validate_format_version_compatible() {
        // Same version
        assert!(validate_format_version("1.0").is_ok());

        // Compatible minor version
        assert!(validate_format_version("1.5").is_ok());
        assert!(validate_format_version("1.99").is_ok());
    }

    #[test]
    fn test_validate_format_version_incompatible() {
        // Different major version
        let result = validate_format_version("2.0");
        assert!(matches!(result, Err(Error::IncompatibleFormatVersion { .. })));

        let result = validate_format_version("0.9");
        assert!(matches!(result, Err(Error::IncompatibleFormatVersion { .. })));
    }

    #[test]
    fn test_validate_format_version_invalid() {
        // Invalid format - should not panic
        // Will default to major version 0, which is incompatible with 1.x
        assert!(validate_format_version("invalid").is_err());
        assert!(validate_format_version("").is_err());
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        // Create a test metadata
        let metadata = ModelMetadata {
            format_version: "1.0".to_string(),
            library_version: "0.6.0".to_string(),
            model_type: ModelType::OLS,
            created_at: "2026-02-10T15:30:00Z".to_string(),
            name: Some("Test Model".to_string()),
        };

        // Create a test data object
        let data = json!({
            "coefficients": [1.0, 2.0, 3.0],
            "r_squared": 0.95,
            "n_observations": 100
        });

        let model = SerializedModel::new(metadata.clone(), data);

        // Serialize to JSON
        let json_str = serde_json::to_string(&model).unwrap();
        let parsed: SerializedModel = serde_json::from_str(&json_str).unwrap();

        // Verify metadata
        assert_eq!(parsed.metadata.format_version, metadata.format_version);
        assert_eq!(parsed.metadata.library_version, metadata.library_version);
        assert_eq!(parsed.metadata.model_type, ModelType::OLS);
        assert_eq!(parsed.metadata.created_at, metadata.created_at);
        assert_eq!(parsed.metadata.name, metadata.name);

        // Verify data
        assert_eq!(parsed.data["coefficients"][0], 1.0);
        assert_eq!(parsed.data["r_squared"], 0.95);
        assert_eq!(parsed.data["n_observations"], 100);
    }

    #[test]
    fn test_serialized_model_json_structure() {
        let metadata = ModelMetadata::new(ModelType::Ridge, "0.6.0".to_string());
        let data = json!({"test": "value"});

        let model = SerializedModel::new(metadata, data);
        let json = serde_json::to_string_pretty(&model).unwrap();

        // Verify JSON structure
        assert!(json.contains("\"metadata\""));
        assert!(json.contains("\"data\""));
        assert!(json.contains("\"format_version\""));
        assert!(json.contains("\"model_type\""));
        assert!(json.contains("\"Ridge\""));
    }

    #[test]
    fn test_model_type() {
        let model = SerializedModel {
            metadata: ModelMetadata {
                format_version: "1.0".to_string(),
                library_version: "0.6.0".to_string(),
                model_type: ModelType::Lasso,
                created_at: "2026-02-10T00:00:00Z".to_string(),
                name: None,
            },
            data: json!({}),
        };

        assert_eq!(model.model_type(), &ModelType::Lasso);
    }
}
