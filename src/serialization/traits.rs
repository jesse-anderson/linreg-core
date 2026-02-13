//! Trait definitions for model serialization.
//!
//! These traits provide a unified API for saving and loading models
//! across all regression types.

use crate::error::Error;
use crate::serialization::types::{ModelType, SerializedModel};
use serde::Serialize;

/// Trait for saving models to disk.
///
/// This trait is implemented by all regression result types that support
/// serialization. Models are saved as JSON with a metadata wrapper.
///
/// # Example
///
/// ```ignore
/// # use linreg_core::core::ols_regression;
/// # use linreg_core::serialization::ModelSave;
/// let y = vec![2.0, 4.0, 6.0, 8.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0];
/// let names = vec!["Intercept".into(), "X1".into()];
///
/// let model = ols_regression(&y, &[x1], &names).unwrap();
/// model.save("my_model.json").unwrap();
/// ```
pub trait ModelSave: Serialize {
    /// Save the model to a file.
    ///
    /// The file will contain JSON with metadata (format version, model type,
    /// timestamp) and the model data.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save to (will be created/overwritten)
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an `Error` if serialization or file I/O fails.
    fn save(&self, path: &str) -> Result<(), Error> {
        self.save_with_name(path, None)
    }

    /// Save the model to a file with a custom name.
    ///
    /// The name is stored in the model metadata and can be used to identify
    /// the model later.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save to
    /// * `name` - Optional custom name for the model
    fn save_with_name(&self, path: &str, name: Option<String>) -> Result<(), Error>;

    /// Get the model type identifier.
    ///
    /// This is used when serializing to store the model type in metadata.
    fn model_type() -> ModelType;
}

/// Trait for loading models from disk.
///
/// This trait is implemented by all regression result types that support
/// deserialization. Loading validates the format version and model type.
///
/// # Example
///
/// ```ignore
/// # use linreg_core::core::RegressionOutput;
/// # use linreg_core::serialization::ModelLoad;
/// let model: RegressionOutput = RegressionOutput::load("my_model.json").unwrap();
/// println!("R²: {}", model.r_squared);
/// ```
pub trait ModelLoad: Sized {
    /// Load a model from a file.
    ///
    /// This validates that:
    /// - The file exists and contains valid JSON
    /// - The format version is compatible
    /// - The model type matches the expected type
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load from
    ///
    /// # Returns
    ///
    /// Returns the deserialized model on success, or an `Error` if loading fails.
    fn load(path: &str) -> Result<Self, Error>;

    /// Load a model from an already-deserialized wrapper.
    ///
    /// This is useful when you have a `SerializedModel` and want to convert
    /// it to a specific model type.
    ///
    /// # Arguments
    ///
    /// * `model` - The serialized model wrapper
    ///
    /// # Returns
    ///
    /// Returns the deserialized model on success, or an `Error` if conversion fails.
    fn from_serialized(model: SerializedModel) -> Result<Self, Error>;

    /// Get the model type identifier.
    ///
    /// This is used to validate that the loaded file contains the correct model type.
    fn model_type() -> ModelType;
}

#[cfg(test)]
mod tests {
    use crate::serialization::ModelType;

    // We'll implement the traits for actual model types in their respective modules
    // This module just defines the trait interface

    #[test]
    fn test_model_type_display() {
        // Verify ModelType works correctly
        assert_eq!(ModelType::OLS.to_string(), "OLS");
        assert_eq!(ModelType::Ridge.to_string(), "Ridge");
        assert_eq!(ModelType::Lasso.to_string(), "Lasso");
        assert_eq!(ModelType::ElasticNet.to_string(), "ElasticNet");
        assert_eq!(ModelType::WLS.to_string(), "WLS");
        assert_eq!(ModelType::LOESS.to_string(), "LOESS");
    }
}
