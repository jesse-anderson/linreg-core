// ============================================================================
// WLS Regression (Native Types API)
// ============================================================================

// ============================================================================
// Native Python types API
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
fn wls_regression(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    weights: &Bound<PyAny>,
) -> PyResult<PyWlsResult> {
    // Extract y vector
    let y_vec = crate::python::extract_f64_sequence(y)?;

    // Extract x_vars matrix
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;

    // Extract weights
    let weights_vec = crate::python::extract_f64_sequence(weights)?;

    // Call WLS regression
    let result = crate::weighted_regression::wls_regression(&y_vec, &x_vars_vec, &weights_vec)
        .map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;

    // Construct result
    Ok(PyWlsResult {
        // Coefficient statistics
        coefficients: result.coefficients,
        standard_errors: result.standard_errors,
        t_statistics: result.t_statistics,
        p_values: result.p_values,
        conf_int_lower: result.conf_int_lower,
        conf_int_upper: result.conf_int_upper,

        // Model fit statistics
        r_squared: result.r_squared,
        r_squared_adjusted: result.adj_r_squared,
        f_statistic: result.f_statistic,
        f_p_value: result.f_p_value,
        residual_std_error: result.residual_std_error,
        df_residuals: result.df_residuals,
        df_model: result.df_model,

        // Predictions and diagnostics
        fitted_values: result.fitted_values,
        residuals: result.residuals,
        mse: result.mse,
        rmse: result.rmse,
        mae: result.mae,

        // Sample information
        n_observations: result.n,
        n_predictors: result.k,
    })
}
