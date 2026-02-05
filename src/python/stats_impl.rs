// ============================================================================
// Statistical Utilities (Native Python Types API)
// ============================================================================

// Use type utilities from shared types module
use crate::python::types::extract_f64_sequence as stats_extract_f64_sequence;

// ============================================================================
// Simple scalar utility functions
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
fn get_t_cdf(t: f64, df: f64) -> PyResult<f64> {
    Ok(crate::distributions::student_t_cdf(t, df))
}

#[cfg(feature = "python")]
#[pyfunction]
fn get_t_critical(alpha: f64, df: f64) -> PyResult<f64> {
    Ok(crate::core::t_critical_quantile(df, alpha))
}

#[cfg(feature = "python")]
#[pyfunction]
fn get_normal_inverse(p: f64) -> PyResult<f64> {
    Ok(crate::distributions::normal_inverse_cdf(p))
}

#[cfg(feature = "python")]
#[pyfunction]
fn get_version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}

// ============================================================================
// Descriptive Statistics - Native Python types
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
fn stats_mean(data: &Bound<PyAny>) -> PyResult<f64> {
    let vec = stats_extract_f64_sequence(data)?;
    if vec.is_empty() {
        return Err(crate::python::error::PythonError {
            message: "Cannot compute mean of empty data".to_string()
        }.into());
    }
    Ok(crate::stats::mean(&vec))
}

#[cfg(feature = "python")]
#[pyfunction]
fn stats_variance(data: &Bound<PyAny>) -> PyResult<f64> {
    let vec = stats_extract_f64_sequence(data)?;
    Ok(crate::stats::variance(&vec))
}

#[cfg(feature = "python")]
#[pyfunction]
fn stats_stddev(data: &Bound<PyAny>) -> PyResult<f64> {
    let vec = stats_extract_f64_sequence(data)?;
    Ok(crate::stats::stddev(&vec))
}

#[cfg(feature = "python")]
#[pyfunction]
fn stats_median(data: &Bound<PyAny>) -> PyResult<f64> {
    let vec = stats_extract_f64_sequence(data)?;
    Ok(crate::stats::median(&vec))
}

#[cfg(feature = "python")]
#[pyfunction]
fn stats_quantile(data: &Bound<PyAny>, q: f64) -> PyResult<f64> {
    let vec = stats_extract_f64_sequence(data)?;
    Ok(crate::stats::quantile(&vec, q))
}

#[cfg(feature = "python")]
#[pyfunction]
fn stats_correlation(x: &Bound<PyAny>, y: &Bound<PyAny>) -> PyResult<f64> {
    let x_vec = stats_extract_f64_sequence(x)?;
    let y_vec = stats_extract_f64_sequence(y)?;
    if x_vec.len() != y_vec.len() {
        return Err(crate::python::error::PythonError {
            message: format!(
                "Cannot compute correlation with mismatched lengths: x has {} elements, y has {} elements",
                x_vec.len(),
                y_vec.len()
            )
        }.into());
    }
    if x_vec.len() < 2 {
        return Err(crate::python::error::PythonError {
            message: "Cannot compute correlation with less than 2 observations".to_string()
        }.into());
    }
    let result = crate::stats::correlation(&x_vec, &y_vec);
    // Check for NaN or Inf which indicate numerical issues
    if result.is_nan() {
        return Err(crate::python::error::PythonError {
            message: "Cannot compute correlation: numerical underflow or invalid input resulted in NaN".to_string()
        }.into());
    }
    if result.is_infinite() {
        return Err(crate::python::error::PythonError {
            message: "Cannot compute correlation: numerical overflow resulted in infinity".to_string()
        }.into());
    }
    Ok(result)
}
