// ============================================================================
// Custom Exception Types for Python Bindings
// ============================================================================
// These exception types provide Python-specific error handling with
// clear error messages and proper exception hierarchy.

#[cfg(feature = "python")]
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::fmt;

// ============================================================================
// PythonError - General error for Python bindings
// ============================================================================

/// General error for Python bindings.
/// This wraps the core library Error for use in Python context.
#[cfg(feature = "python")]
#[derive(Debug)]
pub struct PythonError {
    pub message: String,
}

#[cfg(feature = "python")]
impl fmt::Display for PythonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

#[cfg(feature = "python")]
impl From<crate::Error> for PythonError {
    fn from(err: crate::Error) -> Self {
        PythonError {
            message: err.to_string(),
        }
    }
}

#[cfg(feature = "python")]
impl std::convert::From<PythonError> for PyErr {
    fn from(err: PythonError) -> PyErr {
        PyRuntimeError::new_err(err.message)
    }
}

// ============================================================================
// LinregError - Base exception for linreg-core errors
// ============================================================================

/// Custom exception for linreg-core errors.
/// This is exposed to Python as `linreg_core.LinregError`.
#[cfg(feature = "python")]
#[pyclass(extends = PyRuntimeError)]
pub struct LinregError {
    #[pyo3(get, set)]
    pub message: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl LinregError {
    #[new]
    fn new(message: String) -> Self {
        Self { message }
    }

    fn __str__(&self) -> String {
        self.message.clone()
    }
}

// ============================================================================
// DataValidationError - Exception for input validation errors
// ============================================================================

/// Custom exception for data validation errors.
/// This is exposed to Python as `linreg_core.DataValidationError`.
#[cfg(feature = "python")]
#[pyclass(extends = PyRuntimeError)]
pub struct DataValidationError {
    #[pyo3(get, set)]
    pub message: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl DataValidationError {
    #[new]
    fn new(message: String) -> Self {
        Self { message }
    }

    fn __str__(&self) -> String {
        self.message.clone()
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Convert a core library Error to a LinregError PyErr.
#[cfg(feature = "python")]
pub fn error_to_pyerr(err: crate::Error) -> PyErr {
    PyErr::new::<LinregError, _>(err.to_string())
}

/// Create a DataValidationError PyErr from a message.
#[cfg(feature = "python")]
pub fn validation_error(message: String) -> PyErr {
    PyErr::new::<DataValidationError, _>(message)
}
