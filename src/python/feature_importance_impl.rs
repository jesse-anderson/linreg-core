// ============================================================================
// Feature Importance Python Bindings
// ============================================================================

#[cfg(feature = "python")]
use pyo3::prelude::*;

// ============================================================================
// Standardized Coefficients
// ============================================================================

/// Compute standardized coefficients for feature importance.
///
/// Standardized coefficients (beta*) represent the change in Y (in standard deviations)
/// for a one standard deviation change in X, making coefficients comparable
/// across predictors with different units/scales.
///
/// Args:
///     coefficients: Model coefficients including intercept as first element
///     x_vars: Predictor variables (list of lists or 2D array-like)
///     variable_names: Optional names for each predictor variable (default: None, uses X1, X2, ...)
///     y_std: Standard deviation of response variable Y (default: 1.0)
///
/// Returns:
///     StandardizedCoefficientsResult with variable names, standardized coefficients, and y_std
///
/// Example:
///     >>> fit = ols_regression(y, x_vars, names)
///     >>> std_coefs = standardized_coefficients(fit.coefficients, x_vars)
///     >>> print(std_coefs.summary())
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    coefficients,
    x_vars,
    variable_names=None,
    y_std=1.0
))]
fn py_standardized_coefficients(
    coefficients: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    variable_names: Option<Vec<String>>,
    y_std: f64,
) -> PyResult<PyStandardizedCoefficientsResult> {
    let coef_vec = crate::python::extract_f64_sequence(coefficients)?;
    let x_matrix = crate::python::extract_f64_matrix(x_vars)?;

    // If no names provided, generate default ones
    let names = variable_names.unwrap_or_else(|| {
        (1..=x_matrix.len())
            .map(|i| format!("X{}", i))
            .collect()
    });

    let result = crate::feature_importance::standardized_coefficients_named(
        &coef_vec, &x_matrix, &names, y_std
    )
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(PyStandardizedCoefficientsResult {
        variable_names: result.variable_names,
        standardized_coefficients: result.standardized_coefficients,
        y_std: result.y_std,
    })
}

// ============================================================================
// SHAP Values
// ============================================================================

/// Compute SHAP (SHapley Additive exPlanations) values for linear models.
///
/// SHAP values decompose predictions into the contribution of each feature.
/// For linear models: SHAP = coef * (x - mean(x))
///
/// Args:
///     coefficients: Model coefficients including intercept as first element
///     x_vars: Predictor variables (list of lists or 2D array-like)
///     variable_names: Optional names for each predictor variable (default: None)
///
/// Returns:
///     ShapResult with SHAP values matrix, base value, and mean absolute SHAP
///
/// Example:
///     >>> fit = ols_regression(y, x_vars, names)
///     >>> shap = shap_values_linear(fit.coefficients, x_vars)
///     >>> print(shap.summary())
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    coefficients,
    x_vars,
    variable_names=None
))]
fn py_shap_values_linear(
    coefficients: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    variable_names: Option<Vec<String>>,
) -> PyResult<PyShapResult> {
    let coef_vec = crate::python::extract_f64_sequence(coefficients)?;
    let x_matrix = crate::python::extract_f64_matrix(x_vars)?;

    let result = if let Some(names) = variable_names {
        crate::feature_importance::shap_values_linear_named(&x_matrix, &coef_vec, &names)
    } else {
        crate::feature_importance::shap_values_linear(&x_matrix, &coef_vec)
    }
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(PyShapResult {
        variable_names: result.variable_names,
        shap_values: result.shap_values,
        base_value: result.base_value,
        mean_abs_shap: result.mean_abs_shap,
    })
}

// ============================================================================
// VIF Ranking
// ============================================================================

/// Compute VIF (Variance Inflation Factor) based ranking of predictors.
///
/// VIF measures how much the variance of a coefficient is inflated due to
/// multicollinearity. Lower VIF = less redundant = more uniquely important.
///
/// Args:
///     ols_result: Fitted OLS result object containing VIF values
///
/// Returns:
///     VifRankingResult with variable names and VIF values
///
/// Example:
///     >>> fit = ols_regression(y, x_vars, names)
///     >>> vif_rank = vif_ranking(fit)
///     >>> print(vif_rank.summary())
#[cfg(feature = "python")]
#[pyfunction]
fn py_vif_ranking(
    ols_result: &PyOLSResult,
) -> PyResult<PyVifRankingResult> {
    // Reconstruct VifResult from the OLS result
    let vif_results: Vec<crate::core::VifResult> = ols_result
        .vif
        .iter()
        .enumerate()
        .map(|(i, &vif_val)| {
            let variable_name = if i < ols_result.vif.len() {
                // Use a generic name since we don't have the original names
                format!("X{}", i + 1)
            } else {
                format!("X{}", i + 1)
            };

            let (interpretation, rsquared) = if vif_val < 1.0 {
                ("Negative correlation (unusual)".to_string(), 0.0)
            } else if vif_val <= 5.0 {
                ("Low multicollinearity".to_string(), 1.0 - 1.0 / vif_val)
            } else if vif_val <= 10.0 {
                ("Moderate multicollinearity".to_string(), 1.0 - 1.0 / vif_val)
            } else {
                ("High multicollinearity".to_string(), 1.0 - 1.0 / vif_val)
            };

            crate::core::VifResult {
                variable: variable_name,
                vif: vif_val,
                rsquared,
                interpretation,
            }
        })
        .collect();

    let result = crate::feature_importance::vif_ranking(&vif_results);

    Ok(PyVifRankingResult {
        variable_names: result.variable_names,
        vif_values: result.vif_values,
    })
}

/// Compute VIF ranking from a list of VIF values.
///
/// This is a convenience function when you have VIF values from another source.
///
/// Args:
///     vif_values: List of VIF values
///     variable_names: Optional names for each variable (default: ["X1", "X2", ...])
///
/// Returns:
///     VifRankingResult with variable names and VIF values
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (vif_values, variable_names=None))]
fn py_vif_ranking_from_values(
    vif_values: Vec<f64>,
    variable_names: Option<Vec<String>>,
) -> PyResult<PyVifRankingResult> {
    let names = variable_names.unwrap_or_else(|| {
        (1..=vif_values.len())
            .map(|i| format!("X{}", i))
            .collect()
    });

    if names.len() != vif_values.len() {
        return Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!(
                "variable_names length ({}) must equal vif_values length ({})",
                names.len(),
                vif_values.len()
            ),
        ));
    }

    // Create VifResult objects
    let vif_results: Vec<crate::core::VifResult> = vif_values
        .iter()
        .zip(names.iter())
        .map(|(&vif, name)| {
            let (interpretation, rsquared) = if vif < 1.0 {
                ("Negative correlation".to_string(), 0.0)
            } else {
                (
                    if vif <= 5.0 {
                        "Low multicollinearity"
                    } else if vif <= 10.0 {
                        "Moderate multicollinearity"
                    } else {
                        "High multicollinearity"
                    }
                    .to_string(),
                    1.0 - 1.0 / vif,
                )
            };

            crate::core::VifResult {
                variable: name.clone(),
                vif,
                rsquared,
                interpretation,
            }
        })
        .collect();

    let result = crate::feature_importance::vif_ranking(&vif_results);

    Ok(PyVifRankingResult {
        variable_names: result.variable_names,
        vif_values: result.vif_values,
    })
}

// ============================================================================
// Permutation Importance
// ============================================================================

/// Compute permutation importance for OLS regression.
///
/// Permutation importance measures the decrease in R² when each predictor
/// is randomly shuffled. Higher values indicate more important features.
///
/// Args:
///     y: Response variable values
///     x_vars: Predictor variables (list of lists or 2D array-like)
///     ols_result: Fitted OLS result object
///     n_permutations: Number of permutation iterations per feature (default: 50)
///     seed: Random seed for reproducibility (default: None)
///     compute_intervals: Whether to compute confidence intervals (default: False)
///     interval_confidence: Confidence level for intervals (default: 0.95)
///     variable_names: Optional names for each predictor variable (default: None)
///
/// Returns:
///     PermutationImportanceResult with importance scores and metadata
///
/// Example:
///     >>> fit = ols_regression(y, x_vars, names)
///     >>> importance = permutation_importance(y, x_vars, fit, n_permutations=100, seed=42)
///     >>> print(importance.summary())
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    y,
    x_vars,
    ols_result,
    n_permutations=50,
    seed=None,
    compute_intervals=false,
    interval_confidence=0.95,
    variable_names=None
))]
fn py_permutation_importance_ols(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    ols_result: &PyOLSResult,
    n_permutations: usize,
    seed: Option<u64>,
    compute_intervals: bool,
    interval_confidence: f64,
    variable_names: Option<Vec<String>>,
) -> PyResult<PyPermutationImportanceResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_matrix = crate::python::extract_f64_matrix(x_vars)?;

    let options = crate::feature_importance::PermutationImportanceOptions {
        n_permutations,
        seed,
        compute_intervals,
        interval_confidence,
    };

    // Reconstruct RegressionOutput from PyOLSResult
    let n_coef = ols_result.coefficients.len();
    let regression_output = crate::core::RegressionOutput {
        coefficients: ols_result.coefficients.clone(),
        std_errors: ols_result.standard_errors.clone(),
        t_stats: ols_result.t_statistics.clone(),
        p_values: ols_result.p_values.clone(),
        conf_int_lower: vec![0.0; n_coef],
        conf_int_upper: vec![0.0; n_coef],
        r_squared: ols_result.r_squared,
        adj_r_squared: ols_result.r_squared_adjusted,
        f_statistic: ols_result.f_statistic,
        f_p_value: ols_result.f_p_value,
        mse: ols_result.mse,
        rmse: ols_result.rmse,
        mae: 0.0,
        std_error: ols_result.rmse,
        residuals: ols_result.residuals.clone(),
        standardized_residuals: ols_result.standardized_residuals.clone(),
        predictions: vec![0.0; ols_result.n_observations],
        leverage: ols_result.leverage.clone(),
        vif: ols_result
            .vif
            .iter()
            .enumerate()
            .map(|(i, &v)| crate::core::VifResult {
                variable: format!("X{}", i + 1),
                vif: v,
                rsquared: if v > 1.0 { 1.0 - 1.0 / v } else { 0.0 },
                interpretation: if v <= 5.0 {
                    "Low multicollinearity".to_string()
                } else if v <= 10.0 {
                    "Moderate multicollinearity".to_string()
                } else {
                    "High multicollinearity".to_string()
                },
            })
            .collect(),
        n: ols_result.n_observations,
        k: ols_result.n_predictors,
        df: ols_result.degrees_of_freedom,
        variable_names: vec![],
        log_likelihood: 0.0,
        aic: 0.0,
        bic: 0.0,
    };

    let result = if let Some(names) = variable_names {
        crate::feature_importance::permutation_importance_ols_named(
            &y_vec,
            &x_matrix,
            &regression_output,
            &options,
            &names,
        )
    } else {
        crate::feature_importance::permutation_importance_ols(
            &y_vec,
            &x_matrix,
            &regression_output,
            &options,
        )
    }
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(PyPermutationImportanceResult {
        variable_names: result.variable_names,
        importance: result.importance,
        baseline_score: result.baseline_score,
        n_permutations: result.n_permutations,
        seed: result.seed,
        importance_std_err: result.importance_std_err,
        interval_lower: result.interval_lower,
        interval_upper: result.interval_upper,
        interval_confidence: result.interval_confidence,
    })
}

// ============================================================================
// Combined Feature Importance Function
// ============================================================================

/// Compute all feature importance metrics at once.
///
/// This is a convenience function that computes standardized coefficients,
/// SHAP values, VIF ranking, and permutation importance in a single call.
///
/// Args:
///     y: Response variable values
///     x_vars: Predictor variables (list of lists or 2D array-like)
///     variable_names: Names for each predictor variable
///     y_std: Standard deviation of Y for standardized coefficients (default: 1.0)
///     n_permutations: Number of permutations for permutation importance (default: 50)
///     seed: Random seed for permutation importance (default: None)
///
/// Returns:
///     Dictionary with keys: 'standardized_coefficients', 'shap_values',
///     'vif_ranking', 'permutation_importance'
///
/// Example:
///     >>> fit = ols_regression(y, x_vars, names)
///     >>> importance = feature_importance_ols(y, x_vars, names, n_permutations=50)
///     >>> print(importance['shap_values'].summary())
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    y,
    x_vars,
    variable_names,
    y_std=1.0,
    n_permutations=50,
    seed=None
))]
fn py_feature_importance_ols(
    py: Python,
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    variable_names: Vec<String>,
    y_std: f64,
    n_permutations: usize,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    use pyo3::types::PyDict;

    // First, fit the OLS model
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_matrix = crate::python::extract_f64_matrix(x_vars)?;

    // Create full variable names including intercept
    let mut full_names = vec!["Intercept".to_string()];
    full_names.extend(variable_names.clone());

    // Fit OLS
    let ols_fit = crate::core::ols_regression(&y_vec, &x_matrix, &full_names)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Compute standardized coefficients
    let std_coefs = crate::feature_importance::standardized_coefficients_named(
        &ols_fit.coefficients,
        &x_matrix,
        &variable_names,
        y_std,
    )
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let py_std_coefs = PyStandardizedCoefficientsResult {
        variable_names: std_coefs.variable_names,
        standardized_coefficients: std_coefs.standardized_coefficients,
        y_std: std_coefs.y_std,
    };

    // Compute SHAP values
    let shap = crate::feature_importance::shap_values_linear_named(
        &x_matrix,
        &ols_fit.coefficients,
        &variable_names,
    )
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let py_shap = PyShapResult {
        variable_names: shap.variable_names,
        shap_values: shap.shap_values,
        base_value: shap.base_value,
        mean_abs_shap: shap.mean_abs_shap,
    };

    // Compute VIF ranking
    let vif_rank = crate::feature_importance::vif_ranking(&ols_fit.vif);

    let py_vif = PyVifRankingResult {
        variable_names: vif_rank.variable_names,
        vif_values: vif_rank.vif_values,
    };

    // Compute permutation importance
    let perm_options = crate::feature_importance::PermutationImportanceOptions {
        n_permutations,
        seed,
        compute_intervals: false,
        interval_confidence: 0.95,
    };

    let perm = crate::feature_importance::permutation_importance_ols_named(
        &y_vec,
        &x_matrix,
        &ols_fit,
        &perm_options,
        &variable_names,
    )
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let py_perm = PyPermutationImportanceResult {
        variable_names: perm.variable_names,
        importance: perm.importance,
        baseline_score: perm.baseline_score,
        n_permutations: perm.n_permutations,
        seed: perm.seed,
        importance_std_err: perm.importance_std_err,
        interval_lower: perm.interval_lower,
        interval_upper: perm.interval_upper,
        interval_confidence: perm.interval_confidence,
    };

    // Create result dictionary - convert result structs to Python objects
    let dict = PyDict::new_bound(py);
    dict.set_item("standardized_coefficients", pyo3::Py::new(py, py_std_coefs)?)?;
    dict.set_item("shap_values", pyo3::Py::new(py, py_shap)?)?;
    dict.set_item("vif_ranking", pyo3::Py::new(py, py_vif)?)?;
    dict.set_item("permutation_importance", pyo3::Py::new(py, py_perm)?)?;

    Ok(dict.into())
}

// ============================================================================
// SHAP Values for Regularized Models
// ============================================================================

/// Compute SHAP values for Ridge regression.
///
/// Args:
///     x_vars: Predictor variables (list of lists or 2D array-like)
///     ridge_result: Fitted Ridge result object
///     variable_names: Optional names for each predictor variable (default: None)
///
/// Returns:
///     ShapResult with SHAP values matrix, base value, and mean absolute SHAP
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    x_vars,
    ridge_result,
    variable_names=None
))]
fn py_shap_values_ridge(
    x_vars: &Bound<PyAny>,
    ridge_result: &PyRidgeResult,
    variable_names: Option<Vec<String>>,
) -> PyResult<PyShapResult> {
    let x_matrix = crate::python::extract_f64_matrix(x_vars)?;

    // Reconstruct RidgeFit
    let ridge_fit = crate::regularized::RidgeFit {
        intercept: ridge_result.intercept,
        coefficients: ridge_result.coefficients.clone(),
        lambda: ridge_result.lambda,
        fitted_values: ridge_result.fitted_values.clone(),
        residuals: ridge_result.residuals.clone(),
        df: ridge_result.effective_df, // Map effective_df to df in RidgeFit
        r_squared: ridge_result.r_squared,
        adj_r_squared: 0.0, // Not exposed in PyRidgeResult
        mse: ridge_result.mse,
        rmse: ridge_result.mse.sqrt(), // Compute rmse from mse
        mae: 0.0, // Not exposed in PyRidgeResult
        log_likelihood: ridge_result.log_likelihood,
        aic: ridge_result.aic,
        bic: ridge_result.bic,
    };

    let result = crate::feature_importance::shap_values_ridge(&x_matrix, &ridge_fit)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Use provided names or default
    let names = variable_names.unwrap_or(result.variable_names);

    Ok(PyShapResult {
        variable_names: names,
        shap_values: result.shap_values,
        base_value: result.base_value,
        mean_abs_shap: result.mean_abs_shap,
    })
}

/// Compute SHAP values for Lasso regression.
///
/// Args:
///     x_vars: Predictor variables (list of lists or 2D array-like)
///     lasso_result: Fitted Lasso result object
///     variable_names: Optional names for each predictor variable (default: None)
///
/// Returns:
///     ShapResult with SHAP values matrix, base value, and mean absolute SHAP
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    x_vars,
    lasso_result,
    variable_names=None
))]
fn py_shap_values_lasso(
    x_vars: &Bound<PyAny>,
    lasso_result: &PyLassoResult,
    variable_names: Option<Vec<String>>,
) -> PyResult<PyShapResult> {
    let x_matrix = crate::python::extract_f64_matrix(x_vars)?;

    // Reconstruct LassoFit
    let lasso_fit = crate::regularized::LassoFit {
        intercept: lasso_result.intercept,
        coefficients: lasso_result.coefficients.clone(),
        lambda: lasso_result.lambda,
        fitted_values: lasso_result.fitted_values.clone(),
        residuals: lasso_result.residuals.clone(),
        n_nonzero: lasso_result.n_nonzero,
        converged: lasso_result.converged,
        iterations: lasso_result.n_iterations,
        r_squared: lasso_result.r_squared,
        adj_r_squared: lasso_result.adj_r_squared,
        mse: lasso_result.mse,
        rmse: lasso_result.rmse,
        mae: lasso_result.mae,
        log_likelihood: lasso_result.log_likelihood,
        aic: lasso_result.aic,
        bic: lasso_result.bic,
    };

    let result = crate::feature_importance::shap_values_lasso(&x_matrix, &lasso_fit)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Use provided names or default
    let names = variable_names.unwrap_or(result.variable_names);

    Ok(PyShapResult {
        variable_names: names,
        shap_values: result.shap_values,
        base_value: result.base_value,
        mean_abs_shap: result.mean_abs_shap,
    })
}

/// Compute SHAP values for Elastic Net regression.
///
/// Args:
///     x_vars: Predictor variables (list of lists or 2D array-like)
///     elastic_net_result: Fitted Elastic Net result object
///     variable_names: Optional names for each predictor variable (default: None)
///
/// Returns:
///     ShapResult with SHAP values matrix, base value, and mean absolute SHAP
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    x_vars,
    elastic_net_result,
    variable_names=None
))]
fn py_shap_values_elastic_net(
    x_vars: &Bound<PyAny>,
    elastic_net_result: &PyElasticNetResult,
    variable_names: Option<Vec<String>>,
) -> PyResult<PyShapResult> {
    let x_matrix = crate::python::extract_f64_matrix(x_vars)?;

    // Reconstruct ElasticNetFit
    let enet_fit = crate::regularized::ElasticNetFit {
        intercept: elastic_net_result.intercept,
        coefficients: elastic_net_result.coefficients.clone(),
        lambda: elastic_net_result.lambda,
        alpha: elastic_net_result.alpha,
        fitted_values: elastic_net_result.fitted_values.clone(),
        residuals: elastic_net_result.residuals.clone(),
        n_nonzero: elastic_net_result.n_nonzero,
        converged: elastic_net_result.converged,
        iterations: elastic_net_result.n_iterations,
        r_squared: elastic_net_result.r_squared,
        adj_r_squared: elastic_net_result.adj_r_squared,
        mse: elastic_net_result.mse,
        rmse: elastic_net_result.rmse,
        mae: elastic_net_result.mae,
        log_likelihood: elastic_net_result.log_likelihood,
        aic: elastic_net_result.aic,
        bic: elastic_net_result.bic,
    };

    let result = crate::feature_importance::shap_values_elastic_net(&x_matrix, &enet_fit)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Use provided names or default
    let names = variable_names.unwrap_or(result.variable_names);

    Ok(PyShapResult {
        variable_names: names,
        shap_values: result.shap_values,
        base_value: result.base_value,
        mean_abs_shap: result.mean_abs_shap,
    })
}

/// Compute SHAP values for polynomial regression.
///
/// Args:
///     x: Single predictor variable values (list or array-like)
///     polynomial_result: Fitted Polynomial result object
///     variable_names: Optional names for each polynomial term (default: None)
///
/// Returns:
///     ShapResult with SHAP values matrix, base value, and mean absolute SHAP
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (x, polynomial_result, variable_names=None))]
fn py_shap_values_polynomial(
    x: &Bound<PyAny>,
    polynomial_result: &PyPolynomialResult,
    variable_names: Option<Vec<String>>,
) -> PyResult<PyShapResult> {
    let x_vec = crate::python::extract_f64_sequence(x)?;

    // Reconstruct PolynomialFit
    let n_coef = polynomial_result.coefficients.len();
    let polynomial_fit = crate::polynomial::PolynomialFit {
        ols_output: crate::core::RegressionOutput {
            coefficients: polynomial_result.coefficients.clone(),
            std_errors: polynomial_result.standard_errors.clone(),
            t_stats: polynomial_result.t_statistics.clone(),
            p_values: polynomial_result.p_values.clone(),
            conf_int_lower: vec![0.0; n_coef],
            conf_int_upper: vec![0.0; n_coef],
            r_squared: polynomial_result.r_squared,
            adj_r_squared: polynomial_result.r_squared_adjusted,
            f_statistic: polynomial_result.f_statistic,
            f_p_value: polynomial_result.f_p_value,
            mse: polynomial_result.mse,
            rmse: polynomial_result.rmse,
            mae: 0.0,
            std_error: polynomial_result.rmse,
            residuals: polynomial_result.residuals.clone(),
            standardized_residuals: polynomial_result.standardized_residuals.clone(),
            predictions: vec![0.0; polynomial_result.n_observations],
            leverage: polynomial_result.leverage.clone(),
            vif: polynomial_result
                .vif
                .iter()
                .enumerate()
                .map(|(i, &v)| crate::core::VifResult {
                    variable: if i < polynomial_result.feature_names.len() && i > 0 {
                        polynomial_result.feature_names[i].clone()
                    } else {
                        format!("X{}", i)
                    },
                    vif: v,
                    rsquared: if v > 1.0 { 1.0 - 1.0 / v } else { 0.0 },
                    interpretation: if v <= 5.0 {
                        "Low multicollinearity".to_string()
                    } else if v <= 10.0 {
                        "Moderate multicollinearity".to_string()
                    } else {
                        "High multicollinearity".to_string()
                    },
                })
                .collect(),
            n: polynomial_result.n_observations,
            k: polynomial_result.n_predictors,
            df: polynomial_result.degrees_of_freedom,
            variable_names: polynomial_result.feature_names.clone(),
            log_likelihood: 0.0,
            aic: 0.0,
            bic: 0.0,
        },
        degree: polynomial_result.degree,
        centered: polynomial_result.centered,
        x_mean: polynomial_result.x_mean,
        x_std: polynomial_result.x_std,
        standardized: polynomial_result.standardized,
        n_features: polynomial_result.n_features,
        feature_names: polynomial_result.feature_names.clone(),
        feature_means: polynomial_result.feature_means.clone(),
        feature_stds: polynomial_result.feature_stds.clone(),
    };

    let result = crate::feature_importance::shap_values_polynomial(&x_vec, &polynomial_fit)
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Use provided names or default
    let names = variable_names.unwrap_or(result.variable_names);

    Ok(PyShapResult {
        variable_names: names,
        shap_values: result.shap_values,
        base_value: result.base_value,
        mean_abs_shap: result.mean_abs_shap,
    })
}

// ============================================================================
// Permutation Importance for Regularized Models
// ============================================================================

/// Compute permutation importance for Ridge regression.
///
/// Args:
///     y: Response variable values
///     x_vars: Predictor variables (list of lists or 2D array-like)
///     ridge_result: Fitted Ridge result object
///     n_permutations: Number of permutation iterations per feature (default: 50)
///     seed: Random seed for reproducibility (default: None)
///     compute_intervals: Whether to compute confidence intervals (default: False)
///     interval_confidence: Confidence level for intervals (default: 0.95)
///     variable_names: Optional names for each predictor variable (default: None)
///
/// Returns:
///     PermutationImportanceResult with importance scores and metadata
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    y,
    x_vars,
    ridge_result,
    n_permutations=50,
    seed=None,
    compute_intervals=false,
    interval_confidence=0.95,
    variable_names=None
))]
fn py_permutation_importance_ridge(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    ridge_result: &PyRidgeResult,
    n_permutations: usize,
    seed: Option<u64>,
    compute_intervals: bool,
    interval_confidence: f64,
    variable_names: Option<Vec<String>>,
) -> PyResult<PyPermutationImportanceResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_matrix = crate::python::extract_f64_matrix(x_vars)?;

    let options = crate::feature_importance::PermutationImportanceOptions {
        n_permutations,
        seed,
        compute_intervals,
        interval_confidence,
    };

    // Reconstruct RidgeFit
    let ridge_fit = crate::regularized::RidgeFit {
        intercept: ridge_result.intercept,
        coefficients: ridge_result.coefficients.clone(),
        lambda: ridge_result.lambda,
        fitted_values: ridge_result.fitted_values.clone(),
        residuals: ridge_result.residuals.clone(),
        df: ridge_result.effective_df, // Map effective_df to df in RidgeFit
        r_squared: ridge_result.r_squared,
        adj_r_squared: 0.0, // Not exposed in PyRidgeResult
        mse: ridge_result.mse,
        rmse: ridge_result.mse.sqrt(), // Compute rmse from mse
        mae: 0.0, // Not exposed in PyRidgeResult
        log_likelihood: ridge_result.log_likelihood,
        aic: ridge_result.aic,
        bic: ridge_result.bic,
    };

    let result = crate::feature_importance::permutation_importance_ridge(
        &y_vec,
        &x_matrix,
        &ridge_fit,
        &options,
    )
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Use provided names or default
    let names = variable_names.unwrap_or(result.variable_names);

    Ok(PyPermutationImportanceResult {
        variable_names: names,
        importance: result.importance,
        baseline_score: result.baseline_score,
        n_permutations: result.n_permutations,
        seed: result.seed,
        importance_std_err: result.importance_std_err,
        interval_lower: result.interval_lower,
        interval_upper: result.interval_upper,
        interval_confidence: result.interval_confidence,
    })
}

/// Compute permutation importance for Lasso regression.
///
/// Args:
///     y: Response variable values
///     x_vars: Predictor variables (list of lists or 2D array-like)
///     lasso_result: Fitted Lasso result object
///     n_permutations: Number of permutation iterations per feature (default: 50)
///     seed: Random seed for reproducibility (default: None)
///     compute_intervals: Whether to compute confidence intervals (default: False)
///     interval_confidence: Confidence level for intervals (default: 0.95)
///     variable_names: Optional names for each predictor variable (default: None)
///
/// Returns:
///     PermutationImportanceResult with importance scores and metadata
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    y,
    x_vars,
    lasso_result,
    n_permutations=50,
    seed=None,
    compute_intervals=false,
    interval_confidence=0.95,
    variable_names=None
))]
fn py_permutation_importance_lasso(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    lasso_result: &PyLassoResult,
    n_permutations: usize,
    seed: Option<u64>,
    compute_intervals: bool,
    interval_confidence: f64,
    variable_names: Option<Vec<String>>,
) -> PyResult<PyPermutationImportanceResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_matrix = crate::python::extract_f64_matrix(x_vars)?;

    let options = crate::feature_importance::PermutationImportanceOptions {
        n_permutations,
        seed,
        compute_intervals,
        interval_confidence,
    };

    // Reconstruct LassoFit
    let lasso_fit = crate::regularized::LassoFit {
        intercept: lasso_result.intercept,
        coefficients: lasso_result.coefficients.clone(),
        lambda: lasso_result.lambda,
        fitted_values: lasso_result.fitted_values.clone(),
        residuals: lasso_result.residuals.clone(),
        n_nonzero: lasso_result.n_nonzero,
        converged: lasso_result.converged,
        iterations: lasso_result.n_iterations,
        r_squared: lasso_result.r_squared,
        adj_r_squared: lasso_result.adj_r_squared,
        mse: lasso_result.mse,
        rmse: lasso_result.rmse,
        mae: lasso_result.mae,
        log_likelihood: lasso_result.log_likelihood,
        aic: lasso_result.aic,
        bic: lasso_result.bic,
    };

    let result = crate::feature_importance::permutation_importance_lasso(
        &y_vec,
        &x_matrix,
        &lasso_fit,
        &options,
    )
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Use provided names or default
    let names = variable_names.unwrap_or(result.variable_names);

    Ok(PyPermutationImportanceResult {
        variable_names: names,
        importance: result.importance,
        baseline_score: result.baseline_score,
        n_permutations: result.n_permutations,
        seed: result.seed,
        importance_std_err: result.importance_std_err,
        interval_lower: result.interval_lower,
        interval_upper: result.interval_upper,
        interval_confidence: result.interval_confidence,
    })
}

/// Compute permutation importance for Elastic Net regression.
///
/// Args:
///     y: Response variable values
///     x_vars: Predictor variables (list of lists or 2D array-like)
///     elastic_net_result: Fitted Elastic Net result object
///     n_permutations: Number of permutation iterations per feature (default: 50)
///     seed: Random seed for reproducibility (default: None)
///     compute_intervals: Whether to compute confidence intervals (default: False)
///     interval_confidence: Confidence level for intervals (default: 0.95)
///     variable_names: Optional names for each predictor variable (default: None)
///
/// Returns:
///     PermutationImportanceResult with importance scores and metadata
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    y,
    x_vars,
    elastic_net_result,
    n_permutations=50,
    seed=None,
    compute_intervals=false,
    interval_confidence=0.95,
    variable_names=None
))]
fn py_permutation_importance_elastic_net(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    elastic_net_result: &PyElasticNetResult,
    n_permutations: usize,
    seed: Option<u64>,
    compute_intervals: bool,
    interval_confidence: f64,
    variable_names: Option<Vec<String>>,
) -> PyResult<PyPermutationImportanceResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_matrix = crate::python::extract_f64_matrix(x_vars)?;

    let options = crate::feature_importance::PermutationImportanceOptions {
        n_permutations,
        seed,
        compute_intervals,
        interval_confidence,
    };

    // Reconstruct ElasticNetFit
    let enet_fit = crate::regularized::ElasticNetFit {
        intercept: elastic_net_result.intercept,
        coefficients: elastic_net_result.coefficients.clone(),
        lambda: elastic_net_result.lambda,
        alpha: elastic_net_result.alpha,
        fitted_values: elastic_net_result.fitted_values.clone(),
        residuals: elastic_net_result.residuals.clone(),
        n_nonzero: elastic_net_result.n_nonzero,
        converged: elastic_net_result.converged,
        iterations: elastic_net_result.n_iterations,
        r_squared: elastic_net_result.r_squared,
        adj_r_squared: elastic_net_result.adj_r_squared,
        mse: elastic_net_result.mse,
        rmse: elastic_net_result.rmse,
        mae: elastic_net_result.mae,
        log_likelihood: elastic_net_result.log_likelihood,
        aic: elastic_net_result.aic,
        bic: elastic_net_result.bic,
    };

    let result = crate::feature_importance::permutation_importance_elastic_net(
        &y_vec,
        &x_matrix,
        &enet_fit,
        &options,
    )
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Use provided names or default
    let names = variable_names.unwrap_or(result.variable_names);

    Ok(PyPermutationImportanceResult {
        variable_names: names,
        importance: result.importance,
        baseline_score: result.baseline_score,
        n_permutations: result.n_permutations,
        seed: result.seed,
        importance_std_err: result.importance_std_err,
        interval_lower: result.interval_lower,
        interval_upper: result.interval_upper,
        interval_confidence: result.interval_confidence,
    })
}

/// Compute permutation importance for LOESS regression.
///
/// Args:
///     y: Response variable values
///     x_vars: Predictor variables (list of lists or 2D array-like)
///     span: Span parameter used in original fit
///     degree: Degree of polynomial used in original fit
///     n_permutations: Number of permutation iterations per feature (default: 50)
///     seed: Random seed for reproducibility (default: None)
///     variable_names: Optional names for each predictor variable (default: None)
///
/// Returns:
///     PermutationImportanceResult with importance scores and metadata
///
/// Note:
///     This is computationally expensive as it re-fits the LOESS model
///     for each permutation of each feature.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    y,
    x_vars,
    span=0.75,
    degree=1,
    n_permutations=50,
    seed=None,
    variable_names=None
))]
fn py_permutation_importance_loess(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    span: f64,
    degree: usize,
    n_permutations: usize,
    seed: Option<u64>,
    variable_names: Option<Vec<String>>,
) -> PyResult<PyPermutationImportanceResult> {
    let y_vec = crate::python::extract_f64_sequence(y)?;
    let x_matrix = crate::python::extract_f64_matrix(x_vars)?;

    let options = crate::feature_importance::PermutationImportanceOptions {
        n_permutations,
        seed,
        compute_intervals: false,
        interval_confidence: 0.95,
    };

    let result = crate::feature_importance::permutation_importance_loess(
        &y_vec,
        &x_matrix,
        span,
        degree,
        &options,
    )
        .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Use provided names or default
    let names = variable_names.unwrap_or(result.variable_names);

    Ok(PyPermutationImportanceResult {
        variable_names: names,
        importance: result.importance,
        baseline_score: result.baseline_score,
        n_permutations: result.n_permutations,
        seed: result.seed,
        importance_std_err: result.importance_std_err,
        interval_lower: result.interval_lower,
        interval_upper: result.interval_upper,
        interval_confidence: result.interval_confidence,
    })
}
