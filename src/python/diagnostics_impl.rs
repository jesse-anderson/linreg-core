// ============================================================================
// Diagnostic Tests (Native Python Types API)
// ============================================================================

#[cfg(feature = "python")]
use pyo3::prelude::*;
use crate::python::types::{extract_f64_sequence, extract_f64_matrix};

// ============================================================================
// Native Python types
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
fn rainbow_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>, fraction: f64, method: &str) -> PyResult<PyRainbowTestResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let method_enum = match method.to_lowercase().as_str() {
        "python" => crate::diagnostics::RainbowMethod::Python,
        "both" => crate::diagnostics::RainbowMethod::Both,
        _ => crate::diagnostics::RainbowMethod::R,
    };
    let result = crate::diagnostics::rainbow_test(&y_vec, &x_vars_vec, fraction, method_enum).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyRainbowTestResult::from_core_result(result))
}

#[cfg(feature = "python")]
#[pyfunction]
fn harvey_collier_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>) -> PyResult<PyDiagnosticResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let result = crate::diagnostics::harvey_collier_test(&y_vec, &x_vars_vec, crate::diagnostics::HarveyCollierMethod::R).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyDiagnosticResult { statistic: result.statistic, p_value: result.p_value, test_name: "Harvey-Collier".to_string() })
}

#[cfg(feature = "python")]
#[pyfunction]
fn breusch_pagan_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>) -> PyResult<PyDiagnosticResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let result = crate::diagnostics::breusch_pagan_test(&y_vec, &x_vars_vec).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyDiagnosticResult { statistic: result.statistic, p_value: result.p_value, test_name: "Breusch-Pagan".to_string() })
}

#[cfg(feature = "python")]
#[pyfunction]
fn white_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>, method: &str) -> PyResult<PyWhiteTestResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let method_enum = match method.to_lowercase().as_str() {
        "python" => crate::diagnostics::WhiteMethod::Python,
        "both" => crate::diagnostics::WhiteMethod::Both,
        _ => crate::diagnostics::WhiteMethod::R,
    };
    let result = crate::diagnostics::white_test(&y_vec, &x_vars_vec, method_enum).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyWhiteTestResult::from_core_result(result))
}

#[cfg(feature = "python")]
#[pyfunction]
fn r_white_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>) -> PyResult<PyDiagnosticResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let result = crate::diagnostics::r_white_method(&y_vec, &x_vars_vec).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyDiagnosticResult { statistic: result.statistic, p_value: result.p_value, test_name: "White (R)".to_string() })
}

#[cfg(feature = "python")]
#[pyfunction]
fn python_white_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>) -> PyResult<PyDiagnosticResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let result = crate::diagnostics::python_white_method(&y_vec, &x_vars_vec).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyDiagnosticResult { statistic: result.statistic, p_value: result.p_value, test_name: "White (Python)".to_string() })
}

#[cfg(feature = "python")]
#[pyfunction]
fn jarque_bera_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>) -> PyResult<PyDiagnosticResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let result = crate::diagnostics::jarque_bera_test(&y_vec, &x_vars_vec).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyDiagnosticResult { statistic: result.statistic, p_value: result.p_value, test_name: "Jarque-Bera".to_string() })
}

#[cfg(feature = "python")]
#[pyfunction]
fn durbin_watson_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>) -> PyResult<PyDurbinWatsonResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let result = crate::diagnostics::durbin_watson_test(&y_vec, &x_vars_vec).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyDurbinWatsonResult { statistic: result.statistic })
}

#[cfg(feature = "python")]
#[pyfunction]
fn shapiro_wilk_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>) -> PyResult<PyDiagnosticResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let result = crate::diagnostics::shapiro_wilk_test(&y_vec, &x_vars_vec).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyDiagnosticResult { statistic: result.statistic, p_value: result.p_value, test_name: "Shapiro-Wilk".to_string() })
}

#[cfg(feature = "python")]
#[pyfunction]
fn anderson_darling_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>) -> PyResult<PyDiagnosticResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let result = crate::diagnostics::anderson_darling_test(&y_vec, &x_vars_vec).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyDiagnosticResult { statistic: result.statistic, p_value: result.p_value, test_name: "Anderson-Darling".to_string() })
}

#[cfg(feature = "python")]
#[pyfunction]
fn cooks_distance_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>) -> PyResult<PyCooksDistanceResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let result = crate::diagnostics::cooks_distance_test(&y_vec, &x_vars_vec).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;

    Ok(PyCooksDistanceResult {
        distances: result.distances,
        p: result.p,
        mse: result.mse,
        threshold_4_over_n: result.threshold_4_over_n,
        threshold_4_over_df: result.threshold_4_over_df,
        threshold_1: result.threshold_1,
        influential_4_over_n: result.influential_4_over_n,
        influential_4_over_df: result.influential_4_over_df,
        influential_1: result.influential_1,
        interpretation: result.interpretation,
        guidance: result.guidance,
    })
}

#[cfg(feature = "python")]
#[pyfunction]
fn reset_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>, powers: &Bound<PyAny>, type_: &str) -> PyResult<PyDiagnosticResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let powers_vec = extract_f64_sequence(powers)?;
    let powers_usize: Vec<usize> = powers_vec.into_iter().map(|v| v as usize).collect();
    let reset_type = match type_.to_lowercase().as_str() {
        "regressor" => crate::diagnostics::ResetType::Regressor,
        "princomp" => crate::diagnostics::ResetType::PrincipalComponent,
        _ => crate::diagnostics::ResetType::Fitted,
    };
    let result = crate::diagnostics::reset_test(&y_vec, &x_vars_vec, &powers_usize, reset_type).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyDiagnosticResult { statistic: result.statistic, p_value: result.p_value, test_name: "RESET".to_string() })
}

#[cfg(feature = "python")]
#[pyfunction]
fn breusch_godfrey_test(y: &Bound<PyAny>, x_vars: &Bound<PyAny>, order: usize, test_type: &str) -> PyResult<PyBreuschGodfreyResult> {
    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;
    let bg_test_type = match test_type.to_lowercase().as_str() {
        "f" => crate::diagnostics::BGTestType::F,
        _ => crate::diagnostics::BGTestType::Chisq,
    };
    let result = crate::diagnostics::breusch_godfrey_test(&y_vec, &x_vars_vec, order, bg_test_type).map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;
    Ok(PyBreuschGodfreyResult {
        test_name: result.test_name,
        order: result.order,
        test_type: result.test_type,
        statistic: result.statistic,
        p_value: result.p_value,
        df: result.df,
        passed: result.passed,
        interpretation: result.interpretation,
        guidance: result.guidance,
    })
}
