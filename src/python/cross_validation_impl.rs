// ============================================================================
// Cross Validation Python Bindings
// ============================================================================

// ============================================================================
// Helper: Convert CVResult to PyCVResult
// ============================================================================

#[cfg(feature = "python")]
fn cv_result_to_py(py: Python, result: crate::cross_validation::CVResult) -> PyResult<PyCVResult> {
    use pyo3::types::PyList;

    let fold_list = PyList::empty_bound(py);
    for fold in &result.fold_results {
        let py_fold = PyFoldResult {
            fold_index: fold.fold_index,
            train_size: fold.train_size,
            test_size: fold.test_size,
            mse: fold.mse,
            rmse: fold.rmse,
            mae: fold.mae,
            r_squared: fold.r_squared,
            train_r_squared: fold.train_r_squared,
        };
        fold_list.append(pyo3::IntoPy::<pyo3::PyObject>::into_py(py_fold, py))?;
    }

    Ok(PyCVResult {
        n_folds: result.n_folds,
        n_samples: result.n_samples,
        mean_mse: result.mean_mse,
        std_mse: result.std_mse,
        mean_rmse: result.mean_rmse,
        std_rmse: result.std_rmse,
        mean_mae: result.mean_mae,
        std_mae: result.std_mae,
        mean_r_squared: result.mean_r_squared,
        std_r_squared: result.std_r_squared,
        mean_train_r_squared: result.mean_train_r_squared,
        fold_results: fold_list.into(),
        fold_coefficients: result.fold_coefficients,
    })
}

// ============================================================================
// OLS Cross Validation
// ============================================================================

/// Perform K-Fold Cross Validation for OLS regression.
///
/// Args:
///     y: Response variable values
///     x_vars: List of predictor variable lists
///     names: Variable names (including "Intercept" as first element)
///     n_folds: Number of folds (default: 5)
///     shuffle: Whether to shuffle data before splitting (default: False)
///     seed: Random seed for reproducible shuffling (default: None)
///
/// Returns:
///     CVResult with mean/std metrics and per-fold FoldResult objects
///
/// Example:
///     >>> cv = kfold_cv_ols(y, [x1, x2], ["Intercept", "X1", "X2"], n_folds=5, seed=42)
///     >>> print(f"RMSE: {cv.mean_rmse:.4f} +/- {cv.std_rmse:.4f}")
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "kfold_cv_ols", signature = (y, x_vars, names, n_folds=5, shuffle=false, seed=None))]
fn py_kfold_cv_ols(
    py: Python,
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    names: Vec<String>,
    n_folds: usize,
    shuffle: bool,
    seed: Option<u64>,
) -> PyResult<PyCVResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;

    let options = crate::cross_validation::KFoldOptions { n_folds, shuffle, seed };

    let result = crate::cross_validation::kfold_cv_ols(&y_vec, &x_vars_vec, &names, &options)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    cv_result_to_py(py, result)
}

// ============================================================================
// Ridge Cross Validation
// ============================================================================

/// Perform K-Fold Cross Validation for Ridge regression.
///
/// Args:
///     y: Response variable values
///     x_vars: List of predictor variable lists
///     lambda_val: Regularization strength (default: 1.0)
///     standardize: Whether to standardize predictors (default: True)
///     n_folds: Number of folds (default: 5)
///     shuffle: Whether to shuffle data before splitting (default: False)
///     seed: Random seed for reproducible shuffling (default: None)
///
/// Returns:
///     CVResult with mean/std metrics and per-fold FoldResult objects
///
/// Example:
///     >>> cv = kfold_cv_ridge(y, [x1, x2], lambda_val=1.0, n_folds=5, seed=42)
///     >>> print(f"RMSE: {cv.mean_rmse:.4f} +/- {cv.std_rmse:.4f}")
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "kfold_cv_ridge", signature = (y, x_vars, lambda_val=1.0, standardize=true, n_folds=5, shuffle=false, seed=None))]
fn py_kfold_cv_ridge(
    py: Python,
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    lambda_val: f64,
    standardize: bool,
    n_folds: usize,
    shuffle: bool,
    seed: Option<u64>,
) -> PyResult<PyCVResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;

    let options = crate::cross_validation::KFoldOptions { n_folds, shuffle, seed };

    let result = crate::cross_validation::kfold_cv_ridge(&x_vars_vec, &y_vec, lambda_val, standardize, &options)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    cv_result_to_py(py, result)
}

// ============================================================================
// Lasso Cross Validation
// ============================================================================

/// Perform K-Fold Cross Validation for Lasso regression.
///
/// Args:
///     y: Response variable values
///     x_vars: List of predictor variable lists
///     lambda_val: Regularization strength (default: 0.1)
///     standardize: Whether to standardize predictors (default: True)
///     n_folds: Number of folds (default: 5)
///     shuffle: Whether to shuffle data before splitting (default: False)
///     seed: Random seed for reproducible shuffling (default: None)
///
/// Returns:
///     CVResult with mean/std metrics and per-fold FoldResult objects
///
/// Example:
///     >>> cv = kfold_cv_lasso(y, [x1, x2], lambda_val=0.1, n_folds=5, seed=42)
///     >>> print(f"RMSE: {cv.mean_rmse:.4f} +/- {cv.std_rmse:.4f}")
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "kfold_cv_lasso", signature = (y, x_vars, lambda_val=0.1, standardize=true, n_folds=5, shuffle=false, seed=None))]
fn py_kfold_cv_lasso(
    py: Python,
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    lambda_val: f64,
    standardize: bool,
    n_folds: usize,
    shuffle: bool,
    seed: Option<u64>,
) -> PyResult<PyCVResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;

    let options = crate::cross_validation::KFoldOptions { n_folds, shuffle, seed };

    let result = crate::cross_validation::kfold_cv_lasso(&x_vars_vec, &y_vec, lambda_val, standardize, &options)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    cv_result_to_py(py, result)
}

// ============================================================================
// Elastic Net Cross Validation
// ============================================================================

/// Perform K-Fold Cross Validation for Elastic Net regression.
///
/// Args:
///     y: Response variable values
///     x_vars: List of predictor variable lists
///     lambda_val: Regularization strength (default: 0.1)
///     alpha: L1/L2 mixing parameter — 0 = Ridge, 1 = Lasso (default: 0.5)
///     standardize: Whether to standardize predictors (default: True)
///     n_folds: Number of folds (default: 5)
///     shuffle: Whether to shuffle data before splitting (default: False)
///     seed: Random seed for reproducible shuffling (default: None)
///
/// Returns:
///     CVResult with mean/std metrics and per-fold FoldResult objects
///
/// Example:
///     >>> cv = kfold_cv_elastic_net(y, [x1, x2], lambda_val=0.1, alpha=0.5, n_folds=5, seed=42)
///     >>> print(f"RMSE: {cv.mean_rmse:.4f} +/- {cv.std_rmse:.4f}")
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "kfold_cv_elastic_net", signature = (y, x_vars, lambda_val=0.1, alpha=0.5, standardize=true, n_folds=5, shuffle=false, seed=None))]
fn py_kfold_cv_elastic_net(
    py: Python,
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    lambda_val: f64,
    alpha: f64,
    standardize: bool,
    n_folds: usize,
    shuffle: bool,
    seed: Option<u64>,
) -> PyResult<PyCVResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_vars_vec = crate::python::extract_f64_matrix(x_vars)?;

    let options = crate::cross_validation::KFoldOptions { n_folds, shuffle, seed };

    let result = crate::cross_validation::kfold_cv_elastic_net(&x_vars_vec, &y_vec, lambda_val, alpha, standardize, &options)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    cv_result_to_py(py, result)
}
