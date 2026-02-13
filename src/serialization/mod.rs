//! Model serialization module for saving and loading regression models.
//!
//! This module provides a unified serialization framework that works across:
//! - Native Rust (direct file I/O)
//! - Python (PyO3 bindings)
//! - WASM (JSON string serialization)
//!
//! # Format Version
//!
//! The current serialization format version is `1.0`. This version:
//! - Wraps model data in metadata (format version, library version, model type, timestamp)
//! - Uses JSON for cross-platform compatibility
//! - Supports forward compatibility (unknown fields are ignored)
//!
//! # Module Structure
//!
//! - [`types`] — ModelType enum, ModelMetadata, SerializedModel
//! - [`traits`] — ModelSave and ModelLoad trait definitions
//! - [`json`] — File I/O and version validation

pub mod types;
pub mod traits;
pub mod json;

/// Current serialization format version
///
/// Major version changes are breaking. Minor version changes are additive.
pub const FORMAT_VERSION: &str = "1.0";

// Re-export core types for convenience
pub use types::{ModelMetadata, ModelType, SerializedModel};
pub use traits::{ModelLoad, ModelSave};

/// Macro to generate ModelSave and ModelLoad implementations for a model type.
///
/// This macro eliminates the repetitive boilerplate of implementing the
/// serialization traits. Each model type only needs to specify:
/// - The type name
/// - The ModelType variant
/// - The type name string (for error messages)
///
/// # Example
///
/// ```ignore
/// impl_serialization!(MyModel, ModelType::MyModel, "MyModel");
/// ```
///
/// This expands to full implementations of both `ModelSave` and `ModelLoad`.
#[macro_export]
macro_rules! impl_serialization {
    ($type_name:ty, $model_type:expr, $type_str:expr) => {
        impl $crate::serialization::ModelSave for $type_name {
            fn save_with_name(&self, path: &str, name: Option<String>) -> $crate::error::Result<()> {
                use $crate::serialization::{ModelMetadata, SerializedModel};
                use $crate::error::Error;

                // Convert model to JSON value
                let data = serde_json::to_value(self).map_err(|e| {
                    Error::SerializationError(format!("Failed to serialize {}: {}", $type_str, e))
                })?;

                // Create metadata
                let mut metadata = ModelMetadata::new($model_type, env!("CARGO_PKG_VERSION").to_string());
                if let Some(n) = name {
                    metadata = metadata.with_name(n);
                }

                // Create serialized model and save to file
                let model = SerializedModel::new(metadata, data);
                $crate::serialization::json::save_to_file(&model, path)
            }

            fn model_type() -> $crate::serialization::ModelType {
                $model_type
            }
        }

        impl $crate::serialization::ModelLoad for $type_name {
            fn load(path: &str) -> $crate::error::Result<Self> {
                let model = $crate::serialization::json::load_from_file(path)?;

                // Validate model type
                if model.metadata.model_type != $model_type {
                    return Err($crate::error::Error::ModelTypeMismatch {
                        expected: $type_str.to_string(),
                        found: model.metadata.model_type.to_string(),
                    });
                }

                Self::from_serialized(model)
            }

            fn from_serialized(model: $crate::serialization::SerializedModel) -> $crate::error::Result<Self> {
                use $crate::error::Error;
                serde_json::from_value(model.data).map_err(|e| {
                    Error::DeserializationError(format!("Failed to deserialize {}: {}", $type_str, e))
                })
            }

            fn model_type() -> $crate::serialization::ModelType {
                $model_type
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestModel {
        value: f64,
    }

    impl_serialization!(TestModel, ModelType::OLS, "TestModel");

    #[test]
    fn test_macro_generates_save() {
        let model = TestModel { value: 42.0 };
        assert_eq!(<TestModel as ModelSave>::model_type(), ModelType::OLS);
    }

    #[test]
    fn test_macro_generates_load() {
        assert_eq!(<TestModel as ModelLoad>::model_type(), ModelType::OLS);
    }
}
