// ============================================================================
// Regularized Regression (Native Types API)
// ============================================================================

use crate::python::error::PythonError;

// ============================================================================
// Helper: Build Matrix from x_vars
// ============================================================================

#[cfg(feature = "python")]
fn build_matrix(n: usize, p: usize, x_vars: &[Vec<f64>]) -> Result<crate::linalg::Matrix, PythonError> {
    if n <= p + 1 {
        return Err(PythonError {
            message: format!("Insufficient data: need at least {} observations for {} predictors", p + 2, p),
        });
    }
    let mut x_data = vec![1.0; n * (p + 1)];
    for (j, x_var) in x_vars.iter().enumerate() {
        if x_var.len() != n {
            return Err(PythonError {
                message: format!("x_vars[{}] has {} elements, expected {}", j, x_var.len(), n),
            });
        }
        for (i, &val) in x_var.iter().enumerate() {
            x_data[i * (p + 1) + j + 1] = val;
        }
    }
    Ok(crate::linalg::Matrix::new(n, p + 1, x_data))
}

// ============================================================================
// Ridge Regression
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (y, x_vars, lambda_val=1.0, standardize=true))]
fn ridge_regression(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    lambda_val: f64,
    standardize: bool,
) -> PyResult<PyRidgeResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;

    let n = y_vec.len();
    let p = x_vars_vec.len();

    let x = build_matrix(n, p, &x_vars_vec)?;

    let options = crate::regularized::ridge::RidgeFitOptions {
        lambda: lambda_val,
        intercept: true,
        standardize,
        max_iter: 100000,
        tol: 1e-7,
        warm_start: None,
        weights: None,
    };

    let result = crate::regularized::ridge::ridge_fit(&x, &y_vec, &options)
        .map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;

    Ok(PyRidgeResult {
        intercept: result.intercept,
        coefficients: result.coefficients,
        lambda: result.lambda,
        fitted_values: result.fitted_values,
        residuals: result.residuals,
        r_squared: result.r_squared,
        mse: result.mse,
        effective_df: result.df,
        log_likelihood: result.log_likelihood,
        aic: result.aic,
        bic: result.bic,
    })
}

// ============================================================================
// Lasso Regression
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (y, x_vars, lambda_val=0.1, standardize=true, max_iter=1000, tol=1e-7))]
fn lasso_regression(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    lambda_val: f64,
    standardize: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyLassoResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;

    let n = y_vec.len();
    let p = x_vars_vec.len();

    let x = build_matrix(n, p, &x_vars_vec)?;

    let options = crate::regularized::lasso::LassoFitOptions {
        lambda: lambda_val,
        intercept: true,
        standardize,
        max_iter,
        tol,
        ..Default::default()
    };

    let result = crate::regularized::lasso::lasso_fit(&x, &y_vec, &options)
        .map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;

    Ok(PyLassoResult {
        intercept: result.intercept,
        coefficients: result.coefficients,
        lambda: result.lambda,
        fitted_values: result.fitted_values,
        residuals: result.residuals,
        n_nonzero: result.n_nonzero,
        converged: result.converged,
        n_iterations: result.iterations,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        mse: result.mse,
        rmse: result.rmse,
        mae: result.mae,
        log_likelihood: result.log_likelihood,
        aic: result.aic,
        bic: result.bic,
    })
}

// ============================================================================
// Elastic Net Regression
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (y, x_vars, lambda_val=0.1, alpha=0.5, standardize=true, max_iter=1000, tol=1e-7))]
fn elastic_net_regression(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    lambda_val: f64,
    alpha: f64,
    standardize: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyElasticNetResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;

    let n = y_vec.len();
    let p = x_vars_vec.len();

    let x = build_matrix(n, p, &x_vars_vec)?;

    let options = crate::regularized::elastic_net::ElasticNetOptions {
        lambda: lambda_val,
        alpha,
        intercept: true,
        standardize,
        max_iter,
        tol,
        ..Default::default()
    };

    let result = crate::regularized::elastic_net::elastic_net_fit(&x, &y_vec, &options)
        .map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;

    Ok(PyElasticNetResult {
        intercept: result.intercept,
        coefficients: result.coefficients,
        lambda: result.lambda,
        alpha: result.alpha,
        fitted_values: result.fitted_values,
        residuals: result.residuals,
        n_nonzero: result.n_nonzero,
        converged: result.converged,
        n_iterations: result.iterations,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        mse: result.mse,
        rmse: result.rmse,
        mae: result.mae,
        log_likelihood: result.log_likelihood,
        aic: result.aic,
        bic: result.bic,
    })
}

// ============================================================================
// Lambda Path Generation
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction]
fn make_lambda_path(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    n_lambda: usize,
    lambda_min_ratio: f64,
) -> PyResult<PyLambdaPathResult> {
    // Extract data
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;

    let n = y_vec.len();
    let p = x_vars_vec.len();

    let x = build_matrix(n, p, &x_vars_vec)?;

    // Standardize X for lambda path computation
    let x_mean: Vec<f64> = (0..x.cols)
        .map(|j| {
            if j == 0 {
                1.0
            } else {
                (0..n).map(|i| x.get(i, j)).sum::<f64>() / n as f64
            }
        })
        .collect();

    let x_standardized: Vec<f64> = (0..x.cols)
        .map(|j| {
            if j == 0 {
                0.0
            } else {
                let mean = x_mean[j];
                let variance =
                    (0..n).map(|i| (x.get(i, j) - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                variance.sqrt()
            }
        })
        .collect();

    let mut x_standardized_data = vec![1.0; n * (p + 1)];
    for j in 0..x.cols {
        for i in 0..n {
            if j == 0 {
                x_standardized_data[i * (p + 1)] = 1.0;
            } else {
                let std = x_standardized[j];
                if std > 1e-10 {
                    x_standardized_data[i * (p + 1) + j] = (x.get(i, j) - x_mean[j]) / std;
                } else {
                    x_standardized_data[i * (p + 1) + j] = 0.0;
                }
            }
        }
    }
    let x_standardized = crate::linalg::Matrix::new(n, p + 1, x_standardized_data);

    // Center y
    let y_mean: f64 = y_vec.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y_vec.iter().map(|&yi| yi - y_mean).collect();

    // Configure lambda path options
    let options = crate::regularized::path::LambdaPathOptions {
        nlambda: n_lambda.max(1),
        lambda_min_ratio: if lambda_min_ratio > 0.0 {
            Some(lambda_min_ratio)
        } else {
            None
        },
        alpha: 1.0,
        ..Default::default()
    };

    let lambda_path =
        crate::regularized::path::make_lambda_path(&x_standardized, &y_centered, &options, None, Some(0));

    let lambda_max = lambda_path.first().copied().unwrap_or(0.0);
    let lambda_min = lambda_path.last().copied().unwrap_or(0.0);
    let n_lambda = lambda_path.len();

    Ok(PyLambdaPathResult {
        lambda_path,
        lambda_max,
        lambda_min,
        n_lambda,
    })
}
