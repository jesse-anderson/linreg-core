// ============================================================================
// Polynomial Regression Python Bindings
// ============================================================================

// ============================================================================
// Polynomial OLS Regression
// ============================================================================

/// Fit polynomial regression using OLS.
///
/// Polynomial regression models the relationship between y and x as:
///     y = β₀ + β₁x + β₂x² + … + β_d·x^d + ε
///
/// Args:
///     y: Response variable values (list, tuple, or numpy array)
///     x: Single predictor variable values (list, tuple, or numpy array)
///     degree: Polynomial degree (>= 1)
///     center: Whether to center x before creating polynomial features.
///         Centering reduces multicollinearity between x, x², x³, etc.
///         Recommended for degree >= 3. (default: False)
///     standardize: Whether to standardize polynomial features (z-score).
///         Useful for regularization but not required for plain OLS. (default: False)
///     intercept: Whether to include an intercept term. (default: True)
///
/// Returns:
///     PolynomialResult with coefficients, statistics, and prediction methods
///
/// Example:
///     >>> fit = polynomial_regression(y, x, degree=2, center=True)
///     >>> print(fit.summary())
///     >>> predictions = fit.predict([6.0, 7.0, 8.0])
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "polynomial_regression", signature = (
    y, x, degree=2, center=false, standardize=false, intercept=true
))]
fn py_polynomial_regression(
    y: &Bound<PyAny>,
    x: &Bound<PyAny>,
    degree: usize,
    center: bool,
    standardize: bool,
    intercept: bool,
) -> PyResult<PyPolynomialResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vec = crate::python::extract_f64_sequence(x)?;

    let options = crate::polynomial::PolynomialOptions {
        degree,
        center,
        standardize,
        intercept,
    };

    let fit = crate::polynomial::polynomial_regression(&y_vec, &x_vec, &options)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Extract VIF values
    let vif: Vec<f64> = fit.ols_output.vif.iter().map(|v| v.vif).collect();

    Ok(PyPolynomialResult {
        degree: fit.degree,
        centered: fit.centered,
        x_mean: fit.x_mean,
        x_std: fit.x_std,
        standardized: fit.standardized,
        n_features: fit.n_features,
        feature_names: fit.feature_names,
        feature_means: fit.feature_means,
        feature_stds: fit.feature_stds,
        coefficients: fit.ols_output.coefficients,
        standard_errors: fit.ols_output.std_errors,
        t_statistics: fit.ols_output.t_stats,
        p_values: fit.ols_output.p_values,
        r_squared: fit.ols_output.r_squared,
        r_squared_adjusted: fit.ols_output.adj_r_squared,
        f_statistic: fit.ols_output.f_statistic,
        f_p_value: fit.ols_output.f_p_value,
        residuals: fit.ols_output.residuals,
        standardized_residuals: fit.ols_output.standardized_residuals,
        leverage: fit.ols_output.leverage,
        vif,
        n_observations: fit.ols_output.n,
        n_predictors: fit.ols_output.k,
        degrees_of_freedom: fit.ols_output.df,
        mse: fit.ols_output.mse,
        rmse: fit.ols_output.rmse,
    })
}

// ============================================================================
// Polynomial Prediction
// ============================================================================

/// Predict using a fitted polynomial regression model.
///
/// This is a standalone function that requires the original PolynomialResult.
/// For convenience, you can also use the .predict() method on PolynomialResult.
///
/// Args:
///     fit: Fitted PolynomialResult object
///     x_new: New x values to predict at (list, tuple, or numpy array)
///
/// Returns:
///     List of predicted y values
///
/// Example:
///     >>> fit = polynomial_regression(y, x, degree=2)
///     >>> predictions = polynomial_predict(fit, [6.0, 7.0, 8.0])
#[cfg(feature = "python")]
#[pyfunction]
fn polynomial_predict(
    py: Python,
    fit: &PyPolynomialResult,
    x_new: &Bound<PyAny>,
) -> PyResult<PyObject> {
    fit.predict(py, x_new)
}

// ============================================================================
// Polynomial Ridge Regression
// ============================================================================

/// Fit polynomial Ridge regression (L2 penalty).
///
/// Ridge regularization helps with multicollinearity in polynomial features.
///
/// Args:
///     y: Response variable values
///     x: Single predictor variable values
///     degree: Polynomial degree (>= 1)
///     lambda_val: Regularization strength (>= 0). Higher values shrink coefficients more. (default: 1.0)
///     center: Whether to center x before expansion. (default: True)
///     standardize: Whether to standardize features. Recommended. (default: True)
///
/// Returns:
///     RidgeResult with coefficients on the original scale
///
/// Example:
///     >>> fit = polynomial_ridge(y, x, degree=3, lambda_val=0.5)
///     >>> print(fit.intercept, fit.coefficients)
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "polynomial_ridge", signature = (
    y, x, degree=2, lambda_val=1.0, center=true, standardize=true
))]
fn py_polynomial_ridge(
    y: &Bound<PyAny>,
    x: &Bound<PyAny>,
    degree: usize,
    lambda_val: f64,
    center: bool,
    standardize: bool,
) -> PyResult<PyRidgeResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vec = crate::python::extract_f64_sequence(x)?;

    let fit = crate::polynomial::polynomial_ridge(&y_vec, &x_vec, degree, lambda_val, center, standardize)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(PyRidgeResult {
        intercept: fit.intercept,
        coefficients: fit.coefficients,
        lambda: fit.lambda,
        fitted_values: fit.fitted_values,
        residuals: fit.residuals,
        r_squared: fit.r_squared,
        mse: fit.mse,
        effective_df: fit.df,
        log_likelihood: fit.log_likelihood,
        aic: fit.aic,
        bic: fit.bic,
    })
}

// ============================================================================
// Polynomial Lasso Regression
// ============================================================================

/// Fit polynomial Lasso regression (L1 penalty).
///
/// Lasso can perform variable selection among polynomial terms,
/// potentially eliminating higher-order terms.
///
/// Args:
///     y: Response variable values
///     x: Single predictor variable values
///     degree: Polynomial degree (>= 1)
///     lambda_val: Regularization strength (>= 0). Higher values zero more coefficients. (default: 0.1)
///     center: Whether to center x before expansion. (default: True)
///     standardize: Whether to standardize features. Recommended. (default: True)
///
/// Returns:
///     LassoResult with coefficients (some may be exactly zero)
///
/// Example:
///     >>> fit = polynomial_lasso(y, x, degree=5, lambda_val=0.05)
///     >>> print(f"Non-zero coefficients: {fit.n_nonzero}")
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "polynomial_lasso", signature = (
    y, x, degree=2, lambda_val=0.1, center=true, standardize=true
))]
fn py_polynomial_lasso(
    y: &Bound<PyAny>,
    x: &Bound<PyAny>,
    degree: usize,
    lambda_val: f64,
    center: bool,
    standardize: bool,
) -> PyResult<PyLassoResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vec = crate::python::extract_f64_sequence(x)?;

    let fit = crate::polynomial::polynomial_lasso(&y_vec, &x_vec, degree, lambda_val, center, standardize)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(PyLassoResult {
        intercept: fit.intercept,
        coefficients: fit.coefficients,
        lambda: fit.lambda,
        fitted_values: fit.fitted_values,
        residuals: fit.residuals,
        n_nonzero: fit.n_nonzero,
        converged: fit.converged,
        n_iterations: fit.iterations,
        r_squared: fit.r_squared,
        adj_r_squared: fit.adj_r_squared,
        mse: fit.mse,
        rmse: fit.rmse,
        mae: fit.mae,
        log_likelihood: fit.log_likelihood,
        aic: fit.aic,
        bic: fit.bic,
    })
}

// ============================================================================
// Polynomial Elastic Net Regression
// ============================================================================

/// Fit polynomial Elastic Net regression (L1 + L2 penalty).
///
/// Elastic Net combines Ridge (L2) and Lasso (L1) penalties.
///
/// Args:
///     y: Response variable values
///     x: Single predictor variable values
///     degree: Polynomial degree (>= 1)
///     lambda: Regularization strength (>= 0). (default: 0.1)
///     alpha: L1/L2 mixing parameter. 0 = Ridge, 1 = Lasso. (default: 0.5)
///     center: Whether to center x before expansion. (default: True)
///     standardize: Whether to standardize features. Recommended. (default: True)
///
/// Returns:
///     ElasticNetResult with coefficients
///
/// Example:
///     >>> fit = polynomial_elastic_net(y, x, degree=4, lambda=0.1, alpha=0.5)
///     >>> print(f"Non-zero coefficients: {fit.n_nonzero}")
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "polynomial_elastic_net", signature = (
    y, x, degree=2, lambda_val=0.1, alpha=0.5, center=true, standardize=true
))]
fn py_polynomial_elastic_net(
    y: &Bound<PyAny>,
    x: &Bound<PyAny>,
    degree: usize,
    lambda_val: f64,
    alpha: f64,
    center: bool,
    standardize: bool,
) -> PyResult<PyElasticNetResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vec = crate::python::extract_f64_sequence(x)?;

    let fit = crate::polynomial::polynomial_elastic_net(
        &y_vec, &x_vec, degree, lambda_val, alpha, center, standardize,
    )
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(PyElasticNetResult {
        intercept: fit.intercept,
        coefficients: fit.coefficients,
        lambda: fit.lambda,
        alpha: fit.alpha,
        fitted_values: fit.fitted_values,
        residuals: fit.residuals,
        n_nonzero: fit.n_nonzero,
        converged: fit.converged,
        n_iterations: fit.iterations,
        r_squared: fit.r_squared,
        adj_r_squared: fit.adj_r_squared,
        mse: fit.mse,
        rmse: fit.rmse,
        mae: fit.mae,
        log_likelihood: fit.log_likelihood,
        aic: fit.aic,
        bic: fit.bic,
    })
}
