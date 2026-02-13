//! Error types for the linear regression library.
//!
//! This module provides a comprehensive error type for all failure modes in
//! linear regression operations, including matrix operations, statistical
//! computations, and data parsing.

use std::fmt;

/// Error types for linear regression operations
///
/// # Example
///
/// ```
/// # use linreg_core::Error;
/// let err = Error::InvalidInput("negative value".to_string());
/// assert!(err.to_string().contains("Invalid input"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Matrix is singular (perfect multicollinearity).
    ///
    /// This occurs when one or more predictor variables are linear combinations
    /// of others, making the matrix non-invertible. Remove redundant variables
    /// to resolve this error.
    SingularMatrix,

    /// Insufficient data points for the model.
    ///
    /// OLS regression requires more observations than predictor variables.
    InsufficientData {
        /// Minimum number of observations required
        required: usize,
        /// Actual number of observations available
        available: usize,
    },

    /// Invalid input parameter.
    ///
    /// Indicates that an input parameter has an invalid value (e.g., negative
    /// variance, empty data arrays, incompatible dimensions).
    InvalidInput(String),

    /// Dimension mismatch in matrix/vector operations.
    ///
    /// This occurs when the dimensions of matrices or vectors are incompatible
    /// for the requested operation.
    DimensionMismatch(String),

    /// Computation failed due to numerical issues.
    ///
    /// This occurs when a numerical computation fails due to issues like
    /// singularity, non-convergence, or overflow/underflow.
    ComputationFailed(String),

    /// Parse error for JSON/CSV data.
    ///
    /// Raised when input data cannot be parsed as JSON or CSV.
    ParseError(String),

    /// Domain check failed (for WASM with domain restriction enabled).
    ///
    /// By default, the WASM module allows all domains. This error is only returned
    /// when the `LINREG_DOMAIN_RESTRICT` environment variable is set at build time
    /// and the module is accessed from an unauthorized domain.
    ///
    /// To enable domain restriction:
    /// ```bash
    /// LINREG_DOMAIN_RESTRICT=example.com,yoursite.com wasm-pack build
    /// ```
    DomainCheck(String),

    /// File I/O error during model save/load operations.
    ///
    /// Raised when reading or writing model files fails due to permissions,
    /// missing files, or other I/O issues.
    IoError(String),

    /// Serialization error when converting model to JSON.
    ///
    /// Raised when a model cannot be serialized to JSON format.
    SerializationError(String),

    /// Deserialization error when parsing model from JSON.
    ///
    /// Raised when a JSON file cannot be parsed into a model structure.
    DeserializationError(String),

    /// Incompatible format version when loading a model.
    ///
    /// Raised when the format version of a saved model is not compatible
    /// with the current library version.
    IncompatibleFormatVersion {
        /// Version from the file
        file_version: String,
        /// Version supported by this library
        supported: String,
    },

    /// Model type mismatch when loading a model.
    ///
    /// Raised when attempting to load a model as the wrong type
    /// (e.g., loading an OLS model as Ridge).
    ModelTypeMismatch {
        /// Expected model type
        expected: String,
        /// Actual model type found in file
        found: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::SingularMatrix => {
                write!(
                    f,
                    "Matrix is singular (perfect multicollinearity). Remove redundant variables."
                )
            },
            Error::InsufficientData {
                required,
                available,
            } => {
                write!(
                    f,
                    "Insufficient data: need at least {} observations, have {}",
                    required, available
                )
            },
            Error::InvalidInput(msg) => {
                write!(f, "Invalid input: {}", msg)
            },
            Error::DimensionMismatch(msg) => {
                write!(f, "Dimension mismatch: {}", msg)
            },
            Error::ComputationFailed(msg) => {
                write!(f, "Computation failed: {}", msg)
            },
            Error::ParseError(msg) => {
                write!(f, "Parse error: {}", msg)
            },
            Error::DomainCheck(msg) => {
                write!(f, "Domain check failed: {}", msg)
            },
            Error::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            },
            Error::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            },
            Error::DeserializationError(msg) => {
                write!(f, "Deserialization error: {}", msg)
            },
            Error::IncompatibleFormatVersion { file_version, supported } => {
                write!(
                    f,
                    "Incompatible format version: file has version {}, supported version is {}",
                    file_version, supported
                )
            },
            Error::ModelTypeMismatch { expected, found } => {
                write!(
                    f,
                    "Model type mismatch: expected {}, found {}",
                    expected, found
                )
            },
        }
    }
}

impl std::error::Error for Error {}

/// Result type for linear regression operations.
///
/// Alias for `std::result::Result<T, Error>`.
///
/// # Example
///
/// ```
/// # use linreg_core::{Error, Result};
/// # fn falls_back() -> Result<f64> {
/// #     Ok(42.0)
/// # }
/// let result: Result<f64> = falls_back();
/// assert_eq!(result.unwrap(), 42.0);
/// ```
pub type Result<T> = std::result::Result<T, Error>;

// ============================================================================
// Helper Functions for WASM Integration
// ============================================================================
//
// These functions convert errors to JSON format for use in WASM bindings,
// enabling proper error reporting to JavaScript code.

/// Converts an error message to a JSON error string.
///
/// Creates a JSON object with a single "error" field containing the message.
/// Used in WASM bindings to return error information to JavaScript.
///
/// # Examples
///
/// ```
/// # use linreg_core::error_json;
/// let json = error_json("Invalid input");
/// assert_eq!(json, r#"{"error":"Invalid input"}"#);
/// ```
pub fn error_json(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}

/// Converts an [`Error`] to a JSON error string.
///
/// Convenience function that converts any error variant to its display
/// representation and wraps it in a JSON object.
///
/// # Examples
///
/// ```
/// # use linreg_core::Error;
/// # use linreg_core::error_to_json;
/// let err = Error::SingularMatrix;
/// let json = error_to_json(&err);
/// assert!(json.contains("singular"));
/// ```
pub fn error_to_json(err: &Error) -> String {
    error_json(&err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Error::SingularMatrix Display implementation
    #[test]
    fn test_singular_matrix_display() {
        let err = Error::SingularMatrix;
        let msg = err.to_string();
        assert!(msg.contains("singular"));
        assert!(msg.contains("multicollinearity"));
    }

    /// Test Error::InsufficientData Display implementation
    #[test]
    fn test_insufficient_data_display() {
        let err = Error::InsufficientData {
            required: 10,
            available: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("Insufficient data"));
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    /// Test Error::InvalidInput Display implementation
    #[test]
    fn test_invalid_input_display() {
        let err = Error::InvalidInput("negative value".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Invalid input"));
        assert!(msg.contains("negative value"));
    }

    /// Test Error::DimensionMismatch Display implementation
    ///
    /// Covers lines 95-96: DimensionMismatch Display impl
    #[test]
    fn test_dimension_mismatch_display() {
        let err = Error::DimensionMismatch("matrix 3x3 cannot multiply with 2x2".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Dimension mismatch"));
        assert!(msg.contains("matrix 3x3"));
    }

    /// Test Error::ComputationFailed Display implementation
    ///
    /// Covers lines 98-99: ComputationFailed Display impl
    #[test]
    fn test_computation_failed_display() {
        let err = Error::ComputationFailed("QR decomposition failed".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Computation failed"));
        assert!(msg.contains("QR decomposition"));
    }

    /// Test Error::ParseError Display implementation
    #[test]
    fn test_parse_error_display() {
        let err = Error::ParseError("invalid JSON syntax".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Parse error"));
        assert!(msg.contains("JSON"));
    }

    /// Test Error::DomainCheck Display implementation
    #[test]
    fn test_domain_check_display() {
        let err = Error::DomainCheck("unauthorized domain".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Domain check failed"));
        assert!(msg.contains("unauthorized"));
    }

    /// Test error_json function
    #[test]
    fn test_error_json() {
        let json = error_json("test error");
        assert_eq!(json, r#"{"error":"test error"}"#);
    }

    /// Test error_to_json function with SingularMatrix
    #[test]
    fn test_error_to_json_singular_matrix() {
        let err = Error::SingularMatrix;
        let json = error_to_json(&err);
        assert!(json.contains(r#""error":"#));
        assert!(json.contains("singular"));
    }

    /// Test error_to_json function with DimensionMismatch
    #[test]
    fn test_error_to_json_dimension_mismatch() {
        let err = Error::DimensionMismatch("incompatible dimensions".to_string());
        let json = error_to_json(&err);
        assert!(json.contains(r#""error":"#));
        assert!(json.contains("Dimension"));
    }

    /// Test error_to_json function with ComputationFailed
    #[test]
    fn test_error_to_json_computation_failed() {
        let err = Error::ComputationFailed("convergence failure".to_string());
        let json = error_to_json(&err);
        assert!(json.contains(r#""error":"#));
        assert!(json.contains("Computation"));
    }

    /// Test Error PartialEq implementation
    #[test]
    fn test_error_partial_eq() {
        let err1 = Error::SingularMatrix;
        let err2 = Error::SingularMatrix;
        let err3 = Error::InvalidInput("test".to_string());

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    /// Test Error Clone implementation
    #[test]
    fn test_error_clone() {
        let err1 = Error::InvalidInput("test".to_string());
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    /// Test Error Debug implementation
    #[test]
    fn test_error_debug() {
        let err = Error::ComputationFailed("test failure".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("ComputationFailed"));
    }

    /// Test Result type alias
    #[test]
    fn test_result_type_alias() {
        fn returns_ok() -> Result<f64> {
            Ok(42.0)
        }
        fn returns_err() -> Result<f64> {
            Err(Error::InvalidInput("test".to_string()))
        }

        assert_eq!(returns_ok().unwrap(), 42.0);
        assert!(returns_err().is_err());
    }

    /// Test Error::IoError Display implementation
    #[test]
    fn test_io_error_display() {
        let err = Error::IoError("Failed to open file".to_string());
        let msg = err.to_string();
        assert!(msg.contains("I/O error"));
        assert!(msg.contains("Failed to open file"));
    }

    /// Test Error::SerializationError Display implementation
    #[test]
    fn test_serialization_error_display() {
        let err = Error::SerializationError("Failed to serialize model".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Serialization error"));
        assert!(msg.contains("Failed to serialize"));
    }

    /// Test Error::DeserializationError Display implementation
    #[test]
    fn test_deserialization_error_display() {
        let err = Error::DeserializationError("Invalid JSON".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Deserialization error"));
        assert!(msg.contains("Invalid JSON"));
    }

    /// Test Error::IncompatibleFormatVersion Display implementation
    #[test]
    fn test_incompatible_format_version_display() {
        let err = Error::IncompatibleFormatVersion {
            file_version: "2.0".to_string(),
            supported: "1.0".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Incompatible format version"));
        assert!(msg.contains("2.0"));
        assert!(msg.contains("1.0"));
    }

    /// Test Error::ModelTypeMismatch Display implementation
    #[test]
    fn test_model_type_mismatch_display() {
        let err = Error::ModelTypeMismatch {
            expected: "OLS".to_string(),
            found: "Ridge".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Model type mismatch"));
        assert!(msg.contains("OLS"));
        assert!(msg.contains("Ridge"));
    }

    /// Test serialization errors work with error_to_json
    #[test]
    fn test_error_to_json_serialization() {
        let err = Error::SerializationError("test".to_string());
        let json = error_to_json(&err);
        assert!(json.contains(r#""error":"#));
        assert!(json.contains("Serialization"));
    }

    /// Test deserialization errors work with error_to_json
    #[test]
    fn test_error_to_json_deserialization() {
        let err = Error::DeserializationError("test".to_string());
        let json = error_to_json(&err);
        assert!(json.contains(r#""error":"#));
        assert!(json.contains("Deserialization"));
    }
}
