// ============================================================================
// Prediction Intervals (Native Types API)
// ============================================================================

// ============================================================================
// OLS Prediction Intervals
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (y, x_vars, new_x, alpha=0.05))]
fn ols_prediction_intervals(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    new_x: &Bound<PyAny>,
    alpha: f64,
) -> PyResult<PyPredictionIntervalResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;
    let new_x_vec = crate::python::extract_f64_matrix(new_x)?;

    let new_x_refs: Vec<&[f64]> = new_x_vec.iter().map(|v| v.as_slice()).collect();

    let result = crate::prediction_intervals::prediction_intervals(
        &y_vec, &x_vars_vec, &new_x_refs, alpha,
    ).map_err(|e| pyo3::PyErr::from(PythonError::from(e)))?;

    Ok(PyPredictionIntervalResult {
        predicted: result.predicted,
        lower_bound: result.lower_bound,
        upper_bound: result.upper_bound,
        se_pred: result.se_pred,
        leverage: result.leverage,
        alpha: result.alpha,
        df_residuals: result.df_residuals,
    })
}

// ============================================================================
// Ridge Prediction Intervals
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (y, x_vars, new_x, alpha=0.05, lambda_val=1.0, standardize=true))]
fn ridge_prediction_intervals(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    new_x: &Bound<PyAny>,
    alpha: f64,
    lambda_val: f64,
    standardize: bool,
) -> PyResult<PyPredictionIntervalResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;
    let new_x_vec = crate::python::extract_f64_matrix(new_x)?;

    let n = y_vec.len();
    let p = x_vars_vec.len();
    let x = build_pi_matrix(n, p, &x_vars_vec)?;

    let options = crate::regularized::ridge::RidgeFitOptions {
        lambda: lambda_val,
        intercept: true,
        standardize,
        ..Default::default()
    };

    let fit = crate::regularized::ridge::ridge_fit(&x, &y_vec, &options)
        .map_err(|e| pyo3::PyErr::from(PythonError::from(e)))?;

    let new_x_refs: Vec<&[f64]> = new_x_vec.iter().map(|v| v.as_slice()).collect();

    let result = crate::prediction_intervals::ridge_prediction_intervals(
        &fit, &x_vars_vec, &new_x_refs, alpha,
    ).map_err(|e| pyo3::PyErr::from(PythonError::from(e)))?;

    Ok(PyPredictionIntervalResult {
        predicted: result.predicted,
        lower_bound: result.lower_bound,
        upper_bound: result.upper_bound,
        se_pred: result.se_pred,
        leverage: result.leverage,
        alpha: result.alpha,
        df_residuals: result.df_residuals,
    })
}

// ============================================================================
// Lasso Prediction Intervals
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (y, x_vars, new_x, alpha=0.05, lambda_val=1.0, standardize=true, max_iter=100000, tol=1e-7))]
#[allow(clippy::too_many_arguments)]
fn lasso_prediction_intervals(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    new_x: &Bound<PyAny>,
    alpha: f64,
    lambda_val: f64,
    standardize: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyPredictionIntervalResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;
    let new_x_vec = crate::python::extract_f64_matrix(new_x)?;

    let n = y_vec.len();
    let p = x_vars_vec.len();
    let x = build_pi_matrix(n, p, &x_vars_vec)?;

    let options = crate::regularized::lasso::LassoFitOptions {
        lambda: lambda_val,
        intercept: true,
        standardize,
        max_iter,
        tol,
        ..Default::default()
    };

    let fit = crate::regularized::lasso::lasso_fit(&x, &y_vec, &options)
        .map_err(|e| pyo3::PyErr::from(PythonError::from(e)))?;

    let new_x_refs: Vec<&[f64]> = new_x_vec.iter().map(|v| v.as_slice()).collect();

    let result = crate::prediction_intervals::lasso_prediction_intervals(
        &fit, &x_vars_vec, &new_x_refs, alpha,
    ).map_err(|e| pyo3::PyErr::from(PythonError::from(e)))?;

    Ok(PyPredictionIntervalResult {
        predicted: result.predicted,
        lower_bound: result.lower_bound,
        upper_bound: result.upper_bound,
        se_pred: result.se_pred,
        leverage: result.leverage,
        alpha: result.alpha,
        df_residuals: result.df_residuals,
    })
}

// ============================================================================
// Elastic Net Prediction Intervals
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (y, x_vars, new_x, alpha=0.05, lambda_val=1.0, enet_alpha=0.5, standardize=true, max_iter=100000, tol=1e-7))]
#[allow(clippy::too_many_arguments)]
fn elastic_net_prediction_intervals(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    new_x: &Bound<PyAny>,
    alpha: f64,
    lambda_val: f64,
    enet_alpha: f64,
    standardize: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyPredictionIntervalResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;
    let new_x_vec = crate::python::extract_f64_matrix(new_x)?;

    let n = y_vec.len();
    let p = x_vars_vec.len();
    let x = build_pi_matrix(n, p, &x_vars_vec)?;

    let options = crate::regularized::elastic_net::ElasticNetOptions {
        lambda: lambda_val,
        alpha: enet_alpha,
        intercept: true,
        standardize,
        max_iter,
        tol,
        ..Default::default()
    };

    let fit = crate::regularized::elastic_net::elastic_net_fit(&x, &y_vec, &options)
        .map_err(|e| pyo3::PyErr::from(PythonError::from(e)))?;

    let new_x_refs: Vec<&[f64]> = new_x_vec.iter().map(|v| v.as_slice()).collect();

    let result = crate::prediction_intervals::elastic_net_prediction_intervals(
        &fit, &x_vars_vec, &new_x_refs, alpha,
    ).map_err(|e| pyo3::PyErr::from(PythonError::from(e)))?;

    Ok(PyPredictionIntervalResult {
        predicted: result.predicted,
        lower_bound: result.lower_bound,
        upper_bound: result.upper_bound,
        se_pred: result.se_pred,
        leverage: result.leverage,
        alpha: result.alpha,
        df_residuals: result.df_residuals,
    })
}

// ============================================================================
// Helper
// ============================================================================

#[cfg(feature = "python")]
fn build_pi_matrix(n: usize, p: usize, x_vars: &[Vec<f64>]) -> Result<crate::linalg::Matrix, pyo3::PyErr> {
    if n <= p + 1 {
        return Err(pyo3::PyErr::from(PythonError {
            message: format!("Insufficient data: need at least {} observations for {} predictors", p + 2, p),
        }));
    }
    let mut x_data = vec![1.0; n * (p + 1)];
    for (j, x_var) in x_vars.iter().enumerate() {
        for (i, &val) in x_var.iter().enumerate() {
            x_data[i * (p + 1) + j + 1] = val;
        }
    }
    Ok(crate::linalg::Matrix::new(n, p + 1, x_data))
}
