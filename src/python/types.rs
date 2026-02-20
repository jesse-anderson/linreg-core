// ============================================================================
// Type Conversion Utilities for Python Bindings
// ============================================================================
// This module provides utilities for converting between Python types
// (lists, tuples, numpy arrays) and Rust types (Vec<f64>, Vec<Vec<f64>>).

#[cfg(feature = "python")]
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::exceptions::{PyTypeError, PyValueError};

// ============================================================================
// Type extraction traits and functions
// ============================================================================

/// Extract a `Vec<f64>` from various Python types.
///
/// Supports:
/// - Python list of floats/ints
/// - Python tuple of floats/ints
/// - 1D numpy array of floats (when numpy feature is enabled)
///
/// This is also available as `extract_f64_sequence` for compatibility.
#[cfg(feature = "python")]
pub fn extract_f64_vec(obj: &Bound<PyAny>) -> PyResult<Vec<f64>> {
    // First try as a list
    if let Ok(list) = obj.downcast::<PyList>() {
        return list.iter().map(|item| item.extract::<f64>()).collect::<PyResult<Vec<_>>>();
    }

    // Try as a tuple
    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        return tuple.iter().map(|item| item.extract::<f64>()).collect::<PyResult<Vec<_>>>();
    }

    // Try as a numpy array (if numpy feature is enabled)
    #[cfg(feature = "numpy")]
    {
        use numpy::{PyArray1, PyArrayMethods};
        if let Ok(array) = obj.downcast::<PyArray1<f64>>() {
            return Ok(unsafe { array.as_slice()? }.to_vec());
        }
        // Also try to handle integer arrays (i32 for Windows/32-bit, i64 for 64-bit)
        if let Ok(array) = obj.downcast::<PyArray1<i32>>() {
            return Ok(unsafe { array.as_slice()?.iter().map(|&x| x as f64).collect() });
        }
        if let Ok(array) = obj.downcast::<PyArray1<i64>>() {
            return Ok(unsafe { array.as_slice()?.iter().map(|&x| x as f64).collect() });
        }
    }

    Err(PyTypeError::new_err(
        "Expected list[float], tuple[float], or np.ndarray",
    ))
}

/// Alias for `extract_f64_vec` - extracts a sequence (list or tuple) of f64 values.
#[cfg(feature = "python")]
pub use extract_f64_vec as extract_f64_sequence;

/// Extract a `Vec<Vec<f64>>` from various Python types.
///
/// Supports:
/// - List of lists of floats/ints
/// - List of tuples of floats/ints
/// - Tuple of tuples of floats/ints
/// - 2D numpy array of floats/ints (when numpy feature is enabled)
#[cfg(feature = "python")]
pub fn extract_f64_matrix(obj: &Bound<PyAny>) -> PyResult<Vec<Vec<f64>>> {
    // Try as a list of lists
    if let Ok(outer_list) = obj.downcast::<PyList>() {
        let mut result = Vec::new();
        for (i, item) in outer_list.iter().enumerate() {
            // Each inner item can be a list or tuple
            if let Ok(inner_list) = item.downcast::<PyList>() {
                let row: Vec<f64> = inner_list.iter().map(|v| v.extract::<f64>()).collect::<PyResult<Vec<_>>>()?;
                result.push(row);
            } else if let Ok(inner_tuple) = item.downcast::<PyTuple>() {
                let row: Vec<f64> = inner_tuple.iter().map(|v| v.extract::<f64>()).collect::<PyResult<Vec<_>>>()?;
                result.push(row);
            } else {
                // Try as 1D array
                #[cfg(feature = "numpy")]
                {
                    use numpy::{PyArray1, PyArrayMethods};
                    if let Ok(array) = item.downcast::<PyArray1<f64>>() {
                        let row = unsafe { array.as_slice()? }.to_vec();
                        result.push(row);
                        continue;
                    }
                }
                return Err(PyTypeError::new_err(format!(
                    "Expected list or tuple at index {}, got {}",
                    i,
                    item.get_type().name()?
                )));
            }
        }
        return Ok(result);
    }

    // Try as tuple of tuples
    if let Ok(outer_tuple) = obj.downcast::<PyTuple>() {
        let mut result = Vec::new();
        for (i, item) in outer_tuple.iter().enumerate() {
            if let Ok(inner_list) = item.downcast::<PyList>() {
                let row: Vec<f64> = inner_list.iter().map(|v| v.extract::<f64>()).collect::<PyResult<Vec<_>>>()?;
                result.push(row);
            } else if let Ok(inner_tuple) = item.downcast::<PyTuple>() {
                let row: Vec<f64> = inner_tuple.iter().map(|v| v.extract::<f64>()).collect::<PyResult<Vec<_>>>()?;
                result.push(row);
            } else {
                return Err(PyTypeError::new_err(format!(
                    "Expected list or tuple at index {}, got {}",
                    i,
                    item.get_type().name()?
                )));
            }
        }
        return Ok(result);
    }

    // Try as 2D numpy array
    #[cfg(feature = "numpy")]
    {
        use numpy::{PyArray2, PyArrayMethods};
        // Try float64 arrays
        if let Ok(array) = obj.downcast::<PyArray2<f64>>() {
            let slice = unsafe { array.as_slice()? };
            let dims = array.dims();
            let nrows = dims[0];
            let ncols = dims[1];
            let mut result = Vec::with_capacity(nrows);
            for row_idx in 0..nrows {
                let start = row_idx * ncols;
                let end = start + ncols;
                result.push(slice[start..end].to_vec());
            }
            return Ok(result);
        }
        // Try int32 arrays (Windows/32-bit default)
        if let Ok(array) = obj.downcast::<PyArray2<i32>>() {
            let slice = unsafe { array.as_slice()? };
            let dims = array.dims();
            let nrows = dims[0];
            let ncols = dims[1];
            let mut result = Vec::with_capacity(nrows);
            for row_idx in 0..nrows {
                let start = row_idx * ncols;
                let end = start + ncols;
                result.push(slice[start..end].iter().map(|&x| x as f64).collect());
            }
            return Ok(result);
        }
        // Try int64 arrays (64-bit default)
        if let Ok(array) = obj.downcast::<PyArray2<i64>>() {
            let slice = unsafe { array.as_slice()? };
            let dims = array.dims();
            let nrows = dims[0];
            let ncols = dims[1];
            let mut result = Vec::with_capacity(nrows);
            for row_idx in 0..nrows {
                let start = row_idx * ncols;
                let end = start + ncols;
                result.push(slice[start..end].iter().map(|&x| x as f64).collect());
            }
            return Ok(result);
        }
    }

    Err(PyTypeError::new_err(
        "Expected list[list[float]], tuple[tuple[float]], or 2D np.ndarray",
    ))
}

/// Extract a `Vec<String>` from Python list of strings.
#[cfg(feature = "python")]
pub fn extract_string_list(obj: &Bound<PyAny>) -> PyResult<Vec<String>> {
    if let Ok(list) = obj.downcast::<PyList>() {
        return list.iter().map(|item| item.extract::<String>()).collect::<PyResult<Vec<_>>>();
    }

    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        return tuple.iter().map(|item| item.extract::<String>()).collect::<PyResult<Vec<_>>>();
    }

    Err(PyTypeError::new_err("Expected list[str] or tuple[str]"))
}

/// Extract a `Vec<usize>` from Python list of ints.
#[cfg(feature = "python")]
pub fn extract_usize_list(obj: &Bound<PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(list) = obj.downcast::<PyList>() {
        return list.iter().map(|item| item.extract::<usize>()).collect::<PyResult<Vec<_>>>();
    }

    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        return tuple.iter().map(|item| item.extract::<usize>()).collect::<PyResult<Vec<_>>>();
    }

    Err(PyTypeError::new_err("Expected list[int] or tuple[int]"))
}

// ============================================================================
// Conversion from Rust to Python types
// ============================================================================

/// Convert a `Vec<f64>` to a Python list.
#[cfg(feature = "python")]
pub fn vec_f64_to_pylist<'a>(py: Python<'a>, data: &'a [f64]) -> PyResult<Bound<'a, PyList>> {
    Ok(PyList::new_bound(py, data.iter().copied()))
}

/// Convert a `Vec<Vec<f64>>` to a Python list of lists.
#[cfg(feature = "python")]
pub fn vec_vec_f64_to_pylist<'a>(py: Python<'a>, data: &'a [Vec<f64>]) -> PyResult<Bound<'a, PyList>> {
    let inner_lists: PyResult<Vec<Bound<'a, PyList>>> = data.iter().map(|inner| vec_f64_to_pylist(py, inner)).collect();
    let inner_lists = inner_lists?;
    Ok(PyList::new_bound(py, inner_lists))
}

/// Convert a `Vec<String>` to a Python list of strings.
#[cfg(feature = "python")]
pub fn vec_string_to_pylist<'a>(py: Python<'a>, data: &'a [String]) -> PyResult<Bound<'a, PyList>> {
    Ok(PyList::new_bound(py, data.iter()))
}

/// Convert a `Vec<usize>` to a Python list of ints.
#[cfg(feature = "python")]
pub fn vec_usize_to_pylist<'a>(py: Python<'a>, data: &'a [usize]) -> PyResult<Bound<'a, PyList>> {
    Ok(PyList::new_bound(py, data.iter()))
}

// ============================================================================
// Validation helpers
// ============================================================================

/// Validate that all vectors in x_vars have the same length as y.
#[cfg(feature = "python")]
pub fn validate_dimensions(y: &[f64], x_vars: &[Vec<f64>]) -> PyResult<()> {
    let n = y.len();
    for (i, x_var) in x_vars.iter().enumerate() {
        if x_var.len() != n {
            return Err(PyValueError::new_err(format!(
                "x_vars[{}] has {} elements, expected {} (same as y)",
                i,
                x_var.len(),
                n
            )));
        }
    }
    Ok(())
}

/// Check that we have sufficient data for the number of predictors.
#[cfg(feature = "python")]
pub fn check_sufficient_data(n: usize, p: usize) -> PyResult<()> {
    if n <= p + 1 {
        Err(PyValueError::new_err(format!(
            "Insufficient data: need at least {} observations for {} predictors, got {}",
            p + 2,
            p,
            n
        )))
    } else {
        Ok(())
    }
}
