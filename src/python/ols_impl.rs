// ============================================================================
// OLS Regression (Native Types API - Phase 4)
// ============================================================================

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyList;

// ============================================================================
// Native Python types API
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
fn ols_regression(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    variable_names: &Bound<PyAny>,
) -> PyResult<PyOLSResult> {
    // Extract y vector
    let y_vec = crate::python::extract_f64_sequence(y)?;

    // Extract x_vars matrix
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;

    // Extract variable names
    let names_vec = if let Ok(list) = variable_names.downcast::<PyList>() {
        list.iter().map(|item| item.extract::<String>()).collect::<PyResult<Vec<_>>>()?
    } else {
        // Try as a single string and return as single-element list
        let s: String = variable_names.extract()?;
        vec![s]
    };

    // Call core OLS regression
    let result = crate::core::ols_regression(&y_vec, &x_vars_vec, &names_vec)
        .map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;

    // Extract VIF values (VifResult structs contain vif field)
    let vif: Vec<f64> = result.vif.iter().map(|v| v.vif).collect();

    // Construct result directly (struct literal - not .new() due to PyO3 limitation)
    Ok(PyOLSResult {
        coefficients: result.coefficients,
        standard_errors: result.std_errors,
        t_statistics: result.t_stats,
        p_values: result.p_values,
        r_squared: result.r_squared,
        r_squared_adjusted: result.adj_r_squared,
        f_statistic: result.f_statistic,
        f_p_value: result.f_p_value,
        residuals: result.residuals,
        standardized_residuals: result.standardized_residuals,
        leverage: result.leverage,
        vif,
        n_observations: result.n,
        n_predictors: result.k,
        degrees_of_freedom: result.df,
        mse: result.mse,
        rmse: result.rmse,
    })
}
