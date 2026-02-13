//! Core types for model serialization.
//!
//! Defines the model type enum and metadata structures used for
//! cross-platform model persistence.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported regression model types for serialization.
///
/// Each variant corresponds to a specific regression result type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    /// Ordinary Least Squares regression (`RegressionOutput`)
    #[serde(rename = "OLS")]
    OLS,
    /// Ridge regression (`RidgeFit`)
    #[serde(rename = "Ridge")]
    Ridge,
    /// Lasso regression (`LassoFit`)
    #[serde(rename = "Lasso")]
    Lasso,
    /// Elastic Net regression (`ElasticNetFit`)
    #[serde(rename = "ElasticNet")]
    ElasticNet,
    /// Weighted Least Squares regression (`WlsFit`)
    #[serde(rename = "WLS")]
    WLS,
    /// LOESS local regression (`LoessFit`)
    #[serde(rename = "LOESS")]
    LOESS,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelType::OLS => write!(f, "OLS"),
            ModelType::Ridge => write!(f, "Ridge"),
            ModelType::Lasso => write!(f, "Lasso"),
            ModelType::ElasticNet => write!(f, "ElasticNet"),
            ModelType::WLS => write!(f, "WLS"),
            ModelType::LOESS => write!(f, "LOESS"),
        }
    }
}

impl std::str::FromStr for ModelType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "OLS" => Ok(ModelType::OLS),
            "RIDGE" => Ok(ModelType::Ridge),
            "LASSO" => Ok(ModelType::Lasso),
            "ELASTICNET" => Ok(ModelType::ElasticNet),
            "WLS" => Ok(ModelType::WLS),
            "LOESS" => Ok(ModelType::LOESS),
            _ => Err(format!("Unknown model type: {}", s)),
        }
    }
}

/// Metadata attached to serialized models.
///
/// This wrapper provides provenance information and version compatibility
/// for saved models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Serialization format version (e.g., "1.0")
    pub format_version: String,

    /// Library version that created this model (e.g., "0.6.0")
    pub library_version: String,

    /// Type of regression model
    pub model_type: ModelType,

    /// ISO 8601 timestamp when model was saved
    pub created_at: String,

    /// Optional user-provided model name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ModelMetadata {
    /// Create new model metadata.
    pub fn new(model_type: ModelType, library_version: String) -> Self {
        Self {
            format_version: super::FORMAT_VERSION.to_string(),
            library_version,
            model_type,
            created_at: crate::serialization::json::iso_timestamp(),
            name: None,
        }
    }

    /// Create metadata with a custom name.
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
}

/// A serialized model with metadata and data.
///
/// This is the on-disk representation format. The `data` field contains
/// the raw model fields as a JSON value, which can be deserialized into
/// the specific model type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedModel {
    /// Model metadata (version, type, timestamp, etc.)
    pub metadata: ModelMetadata,

    /// The model data as raw JSON (to be deserialized into specific types)
    pub data: serde_json::Value,
}

impl SerializedModel {
    /// Create a new serialized model from metadata and data.
    pub fn new(metadata: ModelMetadata, data: serde_json::Value) -> Self {
        Self { metadata, data }
    }

    /// Get the model type from metadata.
    pub fn model_type(&self) -> &ModelType {
        &self.metadata.model_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_display() {
        assert_eq!(ModelType::OLS.to_string(), "OLS");
        assert_eq!(ModelType::Ridge.to_string(), "Ridge");
        assert_eq!(ModelType::Lasso.to_string(), "Lasso");
        assert_eq!(ModelType::ElasticNet.to_string(), "ElasticNet");
        assert_eq!(ModelType::WLS.to_string(), "WLS");
        assert_eq!(ModelType::LOESS.to_string(), "LOESS");
    }

    #[test]
    fn test_model_type_from_str() {
        assert_eq!("OLS".parse::<ModelType>().unwrap(), ModelType::OLS);
        assert_eq!("ols".parse::<ModelType>().unwrap(), ModelType::OLS);
        assert_eq!("Ridge".parse::<ModelType>().unwrap(), ModelType::Ridge);
        assert_eq!("Lasso".parse::<ModelType>().unwrap(), ModelType::Lasso);
        assert_eq!("ElasticNet".parse::<ModelType>().unwrap(), ModelType::ElasticNet);
        assert_eq!("elasticnet".parse::<ModelType>().unwrap(), ModelType::ElasticNet);
        assert_eq!("WLS".parse::<ModelType>().unwrap(), ModelType::WLS);
        assert_eq!("LOESS".parse::<ModelType>().unwrap(), ModelType::LOESS);
    }

    #[test]
    fn test_model_type_from_str_invalid() {
        assert!("Invalid".parse::<ModelType>().is_err());
        assert!("".parse::<ModelType>().is_err());
    }
}
