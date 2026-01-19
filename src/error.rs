//! Error types for the linear regression library.
//!
//! This module provides a comprehensive error type for all failure modes in
//! linear regression operations, including matrix operations, statistical
//! computations, and data parsing.

use std::fmt;

/// Error types for linear regression operations
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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::SingularMatrix => {
                write!(f, "Matrix is singular (perfect multicollinearity). Remove redundant variables.")
            }
            Error::InsufficientData { required, available } => {
                write!(f, "Insufficient data: need at least {} observations, have {}", required, available)
            }
            Error::InvalidInput(msg) => {
                write!(f, "Invalid input: {}", msg)
            }
            Error::ParseError(msg) => {
                write!(f, "Parse error: {}", msg)
            }
            Error::DomainCheck(msg) => {
                write!(f, "Domain check failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for Error {}

/// Result type for linear regression operations.
///
/// Alias for `std::result::Result<T, Error>`.
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
