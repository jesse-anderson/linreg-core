// ============================================================================
// Regression Result Classes for Python Bindings
// ============================================================================
// OLS, Ridge, Lasso, ElasticNet, WLS, LOESS, Polynomial

#[cfg(feature = "python")]
use pyo3::prelude::*;

// ============================================================================
// OLSResult - Ordinary Least Squares regression results
// ============================================================================

/// Result class for OLS regression.
///
/// Attributes match statsmodels OLS results for familiarity.
#[cfg(feature = "python")]
#[pyclass(name = "OLSResult")]
pub struct PyOLSResult {
    /// Estimated regression coefficients (including intercept)
    #[pyo3(get, set)]
    pub coefficients: Vec<f64>,

    /// Standard errors of coefficients
    #[pyo3(get, set)]
    pub standard_errors: Vec<f64>,

    /// t-statistics for coefficient significance tests
    #[pyo3(get, set)]
    pub t_statistics: Vec<f64>,

    /// Two-tailed p-values for coefficients
    #[pyo3(get, set)]
    pub p_values: Vec<f64>,

    /// Coefficient of determination (R²)
    #[pyo3(get, set)]
    pub r_squared: f64,

    /// Adjusted R² (accounts for number of predictors)
    #[pyo3(get, set)]
    pub r_squared_adjusted: f64,

    /// F-statistic for overall model significance
    #[pyo3(get, set)]
    pub f_statistic: f64,

    /// p-value for F-statistic
    #[pyo3(get, set)]
    pub f_p_value: f64,

    /// Raw residuals (y - ŷ)
    #[pyo3(get, set)]
    pub residuals: Vec<f64>,

    /// Standardized residuals
    #[pyo3(get, set)]
    pub standardized_residuals: Vec<f64>,

    /// Leverage values (hat matrix diagonal)
    #[pyo3(get, set)]
    pub leverage: Vec<f64>,

    /// Variance Inflation Factors (excludes intercept)
    #[pyo3(get, set)]
    pub vif: Vec<f64>,

    /// Number of observations
    #[pyo3(get, set)]
    pub n_observations: usize,

    /// Number of predictor variables (excluding intercept)
    #[pyo3(get, set)]
    pub n_predictors: usize,

    /// Residual degrees of freedom
    #[pyo3(get, set)]
    pub degrees_of_freedom: usize,

    /// Mean squared error
    #[pyo3(get, set)]
    pub mse: f64,

    /// Root mean squared error
    #[pyo3(get, set)]
    pub rmse: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyOLSResult {
    #[new]
    fn new(
        coefficients: Vec<f64>,
        standard_errors: Vec<f64>,
        t_statistics: Vec<f64>,
        p_values: Vec<f64>,
        r_squared: f64,
        r_squared_adjusted: f64,
        f_statistic: f64,
        f_p_value: f64,
        residuals: Vec<f64>,
        standardized_residuals: Vec<f64>,
        leverage: Vec<f64>,
        vif: Vec<f64>,
        n_observations: usize,
        n_predictors: usize,
        degrees_of_freedom: usize,
        mse: f64,
        rmse: f64,
    ) -> Self {
        Self {
            coefficients,
            standard_errors,
            t_statistics,
            p_values,
            r_squared,
            r_squared_adjusted,
            f_statistic,
            f_p_value,
            residuals,
            standardized_residuals,
            leverage,
            vif,
            n_observations,
            n_predictors,
            degrees_of_freedom,
            mse,
            rmse,
        }
    }

    /// Return a formatted summary of regression results.
    ///
    /// Returns a string similar to statsmodels summary().
    fn summary(&self) -> String {
        format!(
            "OLS Regression Results\n\
             ======================\n\
             R-squared: {:.4}\n\
             Adj. R-squared: {:.4}\n\
             F-statistic: {:.4}\n\
             F p-value: {:.4e}\n\
             Observations: {}\n\
             Predictors: {}\n\
             Degrees of freedom: {}\n\
             MSE: {:.6}\n\
             RMSE: {:.6}",
            self.r_squared,
            self.r_squared_adjusted,
            self.f_statistic,
            self.f_p_value,
            self.n_observations,
            self.n_predictors,
            self.degrees_of_freedom,
            self.mse,
            self.rmse
        )
    }

    /// Convert result to a dictionary for JSON serialization.
    ///
    /// Returns a dict with all result attributes.
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("coefficients", &self.coefficients)?;
        dict.set_item("standard_errors", &self.standard_errors)?;
        dict.set_item("t_statistics", &self.t_statistics)?;
        dict.set_item("p_values", &self.p_values)?;
        dict.set_item("r_squared", self.r_squared)?;
        dict.set_item("r_squared_adjusted", self.r_squared_adjusted)?;
        dict.set_item("f_statistic", self.f_statistic)?;
        dict.set_item("f_p_value", self.f_p_value)?;
        dict.set_item("residuals", &self.residuals)?;
        dict.set_item("standardized_residuals", &self.standardized_residuals)?;
        dict.set_item("leverage", &self.leverage)?;
        dict.set_item("vif", &self.vif)?;
        dict.set_item("n_observations", self.n_observations)?;
        dict.set_item("n_predictors", self.n_predictors)?;
        dict.set_item("degrees_of_freedom", self.degrees_of_freedom)?;
        dict.set_item("mse", self.mse)?;
        dict.set_item("rmse", self.rmse)?;
        Ok(dict.into())
    }

    /// String representation shows key statistics.
    fn __str__(&self) -> String {
        self.summary()
    }

    /// Repr shows class name and key info.
    fn __repr__(&self) -> String {
        format!(
            "OLSResult(n={}, p={}, R²={:.4})",
            self.n_observations, self.n_predictors, self.r_squared
        )
    }
}

// ============================================================================
// RidgeResult - Ridge regression results
// ============================================================================

/// Result class for Ridge regression.
#[cfg(feature = "python")]
#[pyclass(name = "RidgeResult")]
pub struct PyRidgeResult {
    /// Intercept coefficient
    #[pyo3(get, set)]
    pub intercept: f64,

    /// Slope coefficients (excluding intercept)
    #[pyo3(get, set)]
    pub coefficients: Vec<f64>,

    /// Lambda (regularization strength) used
    #[pyo3(get, set)]
    pub lambda: f64,

    /// Fitted values (predictions on training data)
    #[pyo3(get, set)]
    pub fitted_values: Vec<f64>,

    /// Residuals (y - fitted_values)
    #[pyo3(get, set)]
    pub residuals: Vec<f64>,

    /// R-squared
    #[pyo3(get, set)]
    pub r_squared: f64,

    /// Mean squared error
    #[pyo3(get, set)]
    pub mse: f64,

    /// Effective degrees of freedom
    #[pyo3(get, set)]
    pub effective_df: f64,

    /// Log-likelihood of the model
    #[pyo3(get, set)]
    pub log_likelihood: f64,

    /// Akaike Information Criterion
    #[pyo3(get, set)]
    pub aic: f64,

    /// Bayesian Information Criterion
    #[pyo3(get, set)]
    pub bic: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRidgeResult {
    #[new]
    fn new(
        intercept: f64,
        coefficients: Vec<f64>,
        lambda: f64,
        fitted_values: Vec<f64>,
        residuals: Vec<f64>,
        r_squared: f64,
        mse: f64,
        effective_df: f64,
        log_likelihood: f64,
        aic: f64,
        bic: f64,
    ) -> Self {
        Self {
            intercept,
            coefficients,
            lambda,
            fitted_values,
            residuals,
            r_squared,
            mse,
            effective_df,
            log_likelihood,
            aic,
            bic,
        }
    }

    fn summary(&self) -> String {
        format!(
            "Ridge Regression Results\n\
             ========================\n\
             Lambda: {:.4}\n\
             Intercept: {:.6}\n\
             Coefficients: {:?}\n\
             R-squared: {:.4}\n\
             MSE: {:.6}\n\
             Effective df: {:.2}\n\
             Log-likelihood: {:.4}\n\
             AIC: {:.4}\n\
             BIC: {:.4}",
            self.lambda, self.intercept, self.coefficients, self.r_squared, self.mse, self.effective_df,
            self.log_likelihood, self.aic, self.bic
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("intercept", self.intercept)?;
        dict.set_item("coefficients", &self.coefficients)?;
        dict.set_item("lambda", self.lambda)?;
        dict.set_item("fitted_values", &self.fitted_values)?;
        dict.set_item("residuals", &self.residuals)?;
        dict.set_item("r_squared", self.r_squared)?;
        dict.set_item("mse", self.mse)?;
        dict.set_item("effective_df", self.effective_df)?;
        dict.set_item("log_likelihood", self.log_likelihood)?;
        dict.set_item("aic", self.aic)?;
        dict.set_item("bic", self.bic)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "RidgeResult(lambda={:.4}, R²={:.4})",
            self.lambda, self.r_squared
        )
    }
}

// ============================================================================
// LassoResult - Lasso regression results
// ============================================================================

/// Result class for Lasso regression.
#[cfg(feature = "python")]
#[pyclass(name = "LassoResult")]
pub struct PyLassoResult {
    /// Intercept coefficient
    #[pyo3(get, set)]
    pub intercept: f64,

    /// Slope coefficients (some may be exactly zero)
    #[pyo3(get, set)]
    pub coefficients: Vec<f64>,

    /// Lambda (regularization strength) used
    #[pyo3(get, set)]
    pub lambda: f64,

    /// Fitted values (predictions on training data)
    #[pyo3(get, set)]
    pub fitted_values: Vec<f64>,

    /// Residuals (y - fitted_values)
    #[pyo3(get, set)]
    pub residuals: Vec<f64>,

    /// Number of non-zero coefficients (excluding intercept)
    #[pyo3(get, set)]
    pub n_nonzero: usize,

    /// Whether the algorithm converged
    #[pyo3(get, set)]
    pub converged: bool,

    /// Number of iterations performed
    #[pyo3(get, set)]
    pub n_iterations: usize,

    /// R-squared
    #[pyo3(get, set)]
    pub r_squared: f64,

    /// Adjusted R-squared
    #[pyo3(get, set)]
    pub adj_r_squared: f64,

    /// Mean squared error
    #[pyo3(get, set)]
    pub mse: f64,

    /// Root mean squared error
    #[pyo3(get, set)]
    pub rmse: f64,

    /// Mean absolute error
    #[pyo3(get, set)]
    pub mae: f64,

    /// Log-likelihood of the model
    #[pyo3(get, set)]
    pub log_likelihood: f64,

    /// Akaike Information Criterion
    #[pyo3(get, set)]
    pub aic: f64,

    /// Bayesian Information Criterion
    #[pyo3(get, set)]
    pub bic: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyLassoResult {
    #[new]
    fn new(
        intercept: f64,
        coefficients: Vec<f64>,
        lambda: f64,
        fitted_values: Vec<f64>,
        residuals: Vec<f64>,
        n_nonzero: usize,
        converged: bool,
        n_iterations: usize,
        r_squared: f64,
        adj_r_squared: f64,
        mse: f64,
        rmse: f64,
        mae: f64,
        log_likelihood: f64,
        aic: f64,
        bic: f64,
    ) -> Self {
        Self {
            intercept,
            coefficients,
            lambda,
            fitted_values,
            residuals,
            n_nonzero,
            converged,
            n_iterations,
            r_squared,
            adj_r_squared,
            mse,
            rmse,
            mae,
            log_likelihood,
            aic,
            bic,
        }
    }

    fn summary(&self) -> String {
        format!(
            "Lasso Regression Results\n\
             =======================\n\
             Lambda: {:.4}\n\
             Intercept: {:.6}\n\
             Coefficients: {:?}\n\
             Non-zero coefficients: {}\n\
             Converged: {}\n\
             Iterations: {}\n\
             R-squared: {:.4}\n\
             Adj. R-squared: {:.4}\n\
             MSE: {:.6}\n\
             RMSE: {:.6}\n\
             MAE: {:.6}\n\
             Log-likelihood: {:.4}\n\
             AIC: {:.4}\n\
             BIC: {:.4}",
            self.lambda,
            self.intercept,
            self.coefficients,
            self.n_nonzero,
            self.converged,
            self.n_iterations,
            self.r_squared,
            self.adj_r_squared,
            self.mse,
            self.rmse,
            self.mae,
            self.log_likelihood,
            self.aic,
            self.bic
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("intercept", self.intercept)?;
        dict.set_item("coefficients", &self.coefficients)?;
        dict.set_item("lambda", self.lambda)?;
        dict.set_item("fitted_values", &self.fitted_values)?;
        dict.set_item("residuals", &self.residuals)?;
        dict.set_item("n_nonzero", self.n_nonzero)?;
        dict.set_item("converged", self.converged)?;
        dict.set_item("n_iterations", self.n_iterations)?;
        dict.set_item("r_squared", self.r_squared)?;
        dict.set_item("adj_r_squared", self.adj_r_squared)?;
        dict.set_item("mse", self.mse)?;
        dict.set_item("rmse", self.rmse)?;
        dict.set_item("mae", self.mae)?;
        dict.set_item("log_likelihood", self.log_likelihood)?;
        dict.set_item("aic", self.aic)?;
        dict.set_item("bic", self.bic)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "LassoResult(lambda={:.4}, nonzero={}, converged={})",
            self.lambda, self.n_nonzero, self.converged
        )
    }
}

// ============================================================================
// ElasticNetResult - Elastic Net regression results
// ============================================================================

/// Result class for Elastic Net regression.
#[cfg(feature = "python")]
#[pyclass(name = "ElasticNetResult")]
pub struct PyElasticNetResult {
    /// Intercept coefficient
    #[pyo3(get, set)]
    pub intercept: f64,

    /// Slope coefficients
    #[pyo3(get, set)]
    pub coefficients: Vec<f64>,

    /// Lambda (regularization strength) used
    #[pyo3(get, set)]
    pub lambda: f64,

    /// Alpha mixing parameter (0 = Ridge, 1 = Lasso)
    #[pyo3(get, set)]
    pub alpha: f64,

    /// Fitted values (predictions on training data)
    #[pyo3(get, set)]
    pub fitted_values: Vec<f64>,

    /// Residuals (y - fitted_values)
    #[pyo3(get, set)]
    pub residuals: Vec<f64>,

    /// Number of non-zero coefficients (excluding intercept)
    #[pyo3(get, set)]
    pub n_nonzero: usize,

    /// Whether the algorithm converged
    #[pyo3(get, set)]
    pub converged: bool,

    /// Number of iterations performed
    #[pyo3(get, set)]
    pub n_iterations: usize,

    /// R-squared
    #[pyo3(get, set)]
    pub r_squared: f64,

    /// Adjusted R-squared
    #[pyo3(get, set)]
    pub adj_r_squared: f64,

    /// Mean squared error
    #[pyo3(get, set)]
    pub mse: f64,

    /// Root mean squared error
    #[pyo3(get, set)]
    pub rmse: f64,

    /// Mean absolute error
    #[pyo3(get, set)]
    pub mae: f64,

    /// Log-likelihood of the model
    #[pyo3(get, set)]
    pub log_likelihood: f64,

    /// Akaike Information Criterion
    #[pyo3(get, set)]
    pub aic: f64,

    /// Bayesian Information Criterion
    #[pyo3(get, set)]
    pub bic: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyElasticNetResult {
    #[new]
    fn new(
        intercept: f64,
        coefficients: Vec<f64>,
        lambda: f64,
        alpha: f64,
        fitted_values: Vec<f64>,
        residuals: Vec<f64>,
        n_nonzero: usize,
        converged: bool,
        n_iterations: usize,
        r_squared: f64,
        adj_r_squared: f64,
        mse: f64,
        rmse: f64,
        mae: f64,
        log_likelihood: f64,
        aic: f64,
        bic: f64,
    ) -> Self {
        Self {
            intercept,
            coefficients,
            lambda,
            alpha,
            fitted_values,
            residuals,
            n_nonzero,
            converged,
            n_iterations,
            r_squared,
            adj_r_squared,
            mse,
            rmse,
            mae,
            log_likelihood,
            aic,
            bic,
        }
    }

    fn summary(&self) -> String {
        format!(
            "Elastic Net Regression Results\n\
             ==============================\n\
             Lambda: {:.4}\n\
             Alpha: {:.2}\n\
             Intercept: {:.6}\n\
             Coefficients: {:?}\n\
             Non-zero coefficients: {}\n\
             Converged: {}\n\
             Iterations: {}\n\
             R-squared: {:.4}\n\
             Adj. R-squared: {:.4}\n\
             MSE: {:.6}\n\
             RMSE: {:.6}\n\
             MAE: {:.6}\n\
             Log-likelihood: {:.4}\n\
             AIC: {:.4}\n\
             BIC: {:.4}",
            self.lambda,
            self.alpha,
            self.intercept,
            self.coefficients,
            self.n_nonzero,
            self.converged,
            self.n_iterations,
            self.r_squared,
            self.adj_r_squared,
            self.mse,
            self.rmse,
            self.mae,
            self.log_likelihood,
            self.aic,
            self.bic
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("intercept", self.intercept)?;
        dict.set_item("coefficients", &self.coefficients)?;
        dict.set_item("lambda", self.lambda)?;
        dict.set_item("alpha", self.alpha)?;
        dict.set_item("fitted_values", &self.fitted_values)?;
        dict.set_item("residuals", &self.residuals)?;
        dict.set_item("n_nonzero", self.n_nonzero)?;
        dict.set_item("converged", self.converged)?;
        dict.set_item("n_iterations", self.n_iterations)?;
        dict.set_item("r_squared", self.r_squared)?;
        dict.set_item("adj_r_squared", self.adj_r_squared)?;
        dict.set_item("mse", self.mse)?;
        dict.set_item("rmse", self.rmse)?;
        dict.set_item("mae", self.mae)?;
        dict.set_item("log_likelihood", self.log_likelihood)?;
        dict.set_item("aic", self.aic)?;
        dict.set_item("bic", self.bic)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "ElasticNetResult(lambda={:.4}, alpha={:.2}, nonzero={})",
            self.lambda, self.alpha, self.n_nonzero
        )
    }
}

// ============================================================================
// WLSResult - Weighted Least Squares regression results
// ============================================================================

/// Result class for WLS regression with weights.
///
/// Provides comprehensive statistics matching R's `summary(lm(y ~ x, weights=w))` output.
#[cfg(feature = "python")]
#[pyclass(name = "WLSResult")]
pub struct PyWlsResult {
    // ============================================================
    // Coefficient Statistics (matching R's coefficients table)
    // ============================================================
    /// Coefficient values (including intercept)
    #[pyo3(get, set)]
    pub coefficients: Vec<f64>,

    /// Standard errors of the coefficients
    #[pyo3(get, set)]
    pub standard_errors: Vec<f64>,

    /// t-statistics for coefficient significance tests
    #[pyo3(get, set)]
    pub t_statistics: Vec<f64>,

    /// Two-tailed p-values for coefficients
    #[pyo3(get, set)]
    pub p_values: Vec<f64>,

    /// Lower bounds of 95% confidence intervals for coefficients
    #[pyo3(get, set)]
    pub conf_int_lower: Vec<f64>,

    /// Upper bounds of 95% confidence intervals for coefficients
    #[pyo3(get, set)]
    pub conf_int_upper: Vec<f64>,

    // ============================================================
    // Model Fit Statistics
    // ============================================================
    /// R-squared (coefficient of determination)
    #[pyo3(get, set)]
    pub r_squared: f64,

    /// Adjusted R-squared
    #[pyo3(get, set)]
    pub r_squared_adjusted: f64,

    /// F-statistic for overall model significance
    #[pyo3(get, set)]
    pub f_statistic: f64,

    /// p-value for F-statistic
    #[pyo3(get, set)]
    pub f_p_value: f64,

    /// Residual standard error
    #[pyo3(get, set)]
    pub residual_std_error: f64,

    /// Degrees of freedom for residuals
    #[pyo3(get, set)]
    pub df_residuals: isize,

    /// Degrees of freedom for the model
    #[pyo3(get, set)]
    pub df_model: isize,

    // ============================================================
    // Predictions and Diagnostics
    // ============================================================
    /// Fitted values
    #[pyo3(get, set)]
    pub fitted_values: Vec<f64>,

    /// Residuals
    #[pyo3(get, set)]
    pub residuals: Vec<f64>,

    /// Mean squared error
    #[pyo3(get, set)]
    pub mse: f64,

    /// Root mean squared error
    #[pyo3(get, set)]
    pub rmse: f64,

    /// Mean absolute error
    #[pyo3(get, set)]
    pub mae: f64,

    // ============================================================
    // Sample Information
    // ============================================================
    /// Number of observations
    #[pyo3(get, set)]
    pub n_observations: usize,

    /// Number of predictors (excluding intercept)
    #[pyo3(get, set)]
    pub n_predictors: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyWlsResult {
    #[new]
    #[pyo3(signature = (coefficients, standard_errors, t_statistics, p_values, conf_int_lower, conf_int_upper, r_squared, r_squared_adjusted, f_statistic, f_p_value, residual_std_error, df_residuals, df_model, fitted_values, residuals, mse, rmse, mae, n_observations, n_predictors))]
    fn new(
        coefficients: Vec<f64>,
        standard_errors: Vec<f64>,
        t_statistics: Vec<f64>,
        p_values: Vec<f64>,
        conf_int_lower: Vec<f64>,
        conf_int_upper: Vec<f64>,
        r_squared: f64,
        r_squared_adjusted: f64,
        f_statistic: f64,
        f_p_value: f64,
        residual_std_error: f64,
        df_residuals: isize,
        df_model: isize,
        fitted_values: Vec<f64>,
        residuals: Vec<f64>,
        mse: f64,
        rmse: f64,
        mae: f64,
        n_observations: usize,
        n_predictors: usize,
    ) -> Self {
        Self {
            coefficients,
            standard_errors,
            t_statistics,
            p_values,
            conf_int_lower,
            conf_int_upper,
            r_squared,
            r_squared_adjusted,
            f_statistic,
            f_p_value,
            residual_std_error,
            df_residuals,
            df_model,
            fitted_values,
            residuals,
            mse,
            rmse,
            mae,
            n_observations,
            n_predictors,
        }
    }

    /// Get the predicted values as a Python list.
    fn get_fitted_values(&self) -> PyResult<Vec<f64>> {
        Ok(self.fitted_values.clone())
    }

    /// Get the residuals as a Python list.
    fn get_residuals(&self) -> PyResult<Vec<f64>> {
        Ok(self.residuals.clone())
    }

    /// Summary of the WLS regression results.
    fn summary(&self) -> String {
        let coeff_table: String = self.coefficients
            .iter()
            .zip(&self.standard_errors)
            .zip(&self.t_statistics)
            .zip(&self.p_values)
            .zip(&self.conf_int_lower)
            .zip(&self.conf_int_upper)
            .enumerate()
            .map(|(i, (((((&coef, &se), &t), &p), &lower), &upper))| {
                format!("  [{:2}] coef={:8.4}, SE={:8.4}, t={:6.3}, p={:8.4e}, 95% CI=[{:8.4}, {:8.4}]",
                    i, coef, se, t, p, lower, upper)
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "WLS Regression Results\n\
             =====================\n\
             Coefficients:\n{}\n\
             \n\
             R-squared: {:.4}\n\
             Adjusted R-squared: {:.4}\n\
             F-statistic: {:.4} (p-value: {:.4e})\n\
             \n\
             Residual Standard Error: {:.4}\n\
             Degrees of Freedom: {} (model), {} (residuals)",
            coeff_table,
            self.r_squared,
            self.r_squared_adjusted,
            self.f_statistic,
            self.f_p_value,
            self.residual_std_error,
            self.df_model,
            self.df_residuals
        )
    }

    /// Convert results to a Python dictionary.
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);

        dict.set_item("coefficients", &self.coefficients)?;
        dict.set_item("standard_errors", &self.standard_errors)?;
        dict.set_item("t_statistics", &self.t_statistics)?;
        dict.set_item("p_values", &self.p_values)?;
        dict.set_item("conf_int_lower", &self.conf_int_lower)?;
        dict.set_item("conf_int_upper", &self.conf_int_upper)?;
        dict.set_item("r_squared", self.r_squared)?;
        dict.set_item("r_squared_adjusted", self.r_squared_adjusted)?;
        dict.set_item("f_statistic", self.f_statistic)?;
        dict.set_item("f_p_value", self.f_p_value)?;
        dict.set_item("residual_std_error", self.residual_std_error)?;
        dict.set_item("df_residuals", self.df_residuals)?;
        dict.set_item("df_model", self.df_model)?;
        dict.set_item("fitted_values", &self.fitted_values)?;
        dict.set_item("residuals", &self.residuals)?;
        dict.set_item("mse", self.mse)?;
        dict.set_item("rmse", self.rmse)?;
        dict.set_item("mae", self.mae)?;
        dict.set_item("n_observations", self.n_observations)?;
        dict.set_item("n_predictors", self.n_predictors)?;

        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "WLSResult(n={}, k={}, R²={:.4})",
            self.n_observations, self.n_predictors, self.r_squared
        )
    }
}

// ============================================================================
// LoessResult - LOESS regression result
// ============================================================================

/// Result class for LOESS (Locally Estimated Scatterplot Smoothing) regression.
///
/// Attributes:
///     fitted: Fitted values (smoothed y values) at each observation point.
///     residuals: Residuals (y - fitted).
///     span: Span parameter used for fitting.
///     degree: Degree of polynomial used (0=constant, 1=linear, 2=quadratic).
///     robust_iterations: Number of robustness iterations performed.
///     surface: Surface computation method used ("direct" or "interpolate").
///     mse: Mean squared error of residuals.
///     rmse: Root mean squared error.
///     n_observations: Number of observations.
///
/// Methods:
///     summary(): Return a formatted summary of the LOESS fit.
///     to_dict(): Convert result to a dictionary.
///     __str__(): String representation (same as summary).
///     __repr__(): Concise representation.
#[cfg(feature = "python")]
#[pyclass(name = "LoessResult")]
pub struct PyLoessResult {
    /// Fitted values (smoothed y values) at each observation point
    #[pyo3(get, set)]
    pub fitted: Vec<f64>,

    /// Residuals (y - fitted)
    #[pyo3(get, set)]
    pub residuals: Vec<f64>,

    /// Span parameter used for fitting
    #[pyo3(get, set)]
    pub span: f64,

    /// Degree of polynomial used for fitting (0=constant, 1=linear, 2=quadratic)
    #[pyo3(get, set)]
    pub degree: usize,

    /// Number of robustness iterations performed
    #[pyo3(get, set)]
    pub robust_iterations: usize,

    /// Surface computation method used ("direct" or "interpolate")
    #[pyo3(get, set)]
    pub surface: String,

    /// Mean squared error of residuals
    #[pyo3(get, set)]
    pub mse: f64,

    /// Root mean squared error
    #[pyo3(get, set)]
    pub rmse: f64,

    /// Number of observations
    #[pyo3(get, set)]
    pub n_observations: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyLoessResult {
    #[new]
    #[pyo3(text_signature = "(fitted, residuals, span, degree, robust_iterations, surface, mse, rmse, n_observations)")]
    fn new(
        fitted: Vec<f64>,
        residuals: Vec<f64>,
        span: f64,
        degree: usize,
        robust_iterations: usize,
        surface: String,
        mse: f64,
        rmse: f64,
        n_observations: usize,
    ) -> Self {
        Self {
            fitted,
            residuals,
            span,
            degree,
            robust_iterations,
            surface,
            mse,
            rmse,
            n_observations,
        }
    }

    /// Predict at new x values using the fitted LOESS model.
    ///
    /// Args:
    ///     new_x: New x values to predict at (can be scalar or list)
    ///
    /// Returns:
    ///     Predicted y values at the new x points
    fn predict(&self, _py: Python, new_x: &Bound<PyAny>) -> PyResult<PyObject> {
        // Extract the new_x values (for validation - ensures input is well-formed)
        let _new_x_vec = crate::python::extract_f64_sequence(new_x)?;

        // We need access to the original x data for prediction
        // This is a limitation - in a full implementation, we'd store original x
        // For now, return an error indicating the limitation
        use pyo3::exceptions::PyNotImplementedError;
        Err(PyNotImplementedError::new_err(
            "predict() requires the original x data. Use loess_predict() function directly with original data."
        ))
    }

    fn summary(&self) -> String {
        format!(
            "LOESS Regression Results\n\
             ========================\n\
             Span: {:.2}\n\
             Degree: {}\n\
             Robust iterations: {}\n\
             Surface: {}\n\
             Observations: {}\n\
             MSE: {:.6}\n\
             RMSE: {:.6}",
            self.span,
            self.degree,
            self.robust_iterations,
            self.surface,
            self.n_observations,
            self.mse,
            self.rmse
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("fitted", &self.fitted)?;
        dict.set_item("residuals", &self.residuals)?;
        dict.set_item("span", self.span)?;
        dict.set_item("degree", self.degree)?;
        dict.set_item("robust_iterations", self.robust_iterations)?;
        dict.set_item("surface", &self.surface)?;
        dict.set_item("mse", self.mse)?;
        dict.set_item("rmse", self.rmse)?;
        dict.set_item("n_observations", self.n_observations)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "LoessResult(span={:.2}, degree={}, n={})",
            self.span, self.degree, self.n_observations
        )
    }
}

// ============================================================================
// PolynomialResult - Polynomial regression results
// ============================================================================

/// Result class for polynomial regression.
///
/// Wraps the OLS output with polynomial-specific metadata.
#[cfg(feature = "python")]
#[pyclass(name = "PolynomialResult")]
pub struct PyPolynomialResult {
    /// Polynomial degree used
    #[pyo3(get, set)]
    pub degree: usize,

    /// Whether x was centered before fitting
    #[pyo3(get, set)]
    pub centered: bool,

    /// Mean of x used for centering (0.0 if no centering)
    #[pyo3(get, set)]
    pub x_mean: f64,

    /// Standard deviation of x (1.0 if not standardized)
    #[pyo3(get, set)]
    pub x_std: f64,

    /// Whether features were standardized
    #[pyo3(get, set)]
    pub standardized: bool,

    /// Number of polynomial features (excluding intercept)
    #[pyo3(get, set)]
    pub n_features: usize,

    /// Feature names (Intercept, x, x², ...)
    #[pyo3(get, set)]
    pub feature_names: Vec<String>,

    /// Per-feature means used for standardization
    #[pyo3(get, set)]
    pub feature_means: Vec<f64>,

    /// Per-feature standard deviations used for standardization
    #[pyo3(get, set)]
    pub feature_stds: Vec<f64>,

    // OLS output fields
    /// Coefficient values (including intercept)
    #[pyo3(get, set)]
    pub coefficients: Vec<f64>,

    /// Standard errors of the coefficients
    #[pyo3(get, set)]
    pub standard_errors: Vec<f64>,

    /// t-statistics for coefficient significance tests
    #[pyo3(get, set)]
    pub t_statistics: Vec<f64>,

    /// Two-tailed p-values for coefficients
    #[pyo3(get, set)]
    pub p_values: Vec<f64>,

    /// R-squared
    #[pyo3(get, set)]
    pub r_squared: f64,

    /// Adjusted R-squared
    #[pyo3(get, set)]
    pub r_squared_adjusted: f64,

    /// F-statistic
    #[pyo3(get, set)]
    pub f_statistic: f64,

    /// p-value for F-statistic
    #[pyo3(get, set)]
    pub f_p_value: f64,

    /// Raw residuals
    #[pyo3(get, set)]
    pub residuals: Vec<f64>,

    /// Standardized residuals
    #[pyo3(get, set)]
    pub standardized_residuals: Vec<f64>,

    /// Leverage values
    #[pyo3(get, set)]
    pub leverage: Vec<f64>,

    /// Variance Inflation Factors
    #[pyo3(get, set)]
    pub vif: Vec<f64>,

    /// Number of observations
    #[pyo3(get, set)]
    pub n_observations: usize,

    /// Number of predictors (equals degree)
    #[pyo3(get, set)]
    pub n_predictors: usize,

    /// Residual degrees of freedom
    #[pyo3(get, set)]
    pub degrees_of_freedom: usize,

    /// Mean squared error
    #[pyo3(get, set)]
    pub mse: f64,

    /// Root mean squared error
    #[pyo3(get, set)]
    pub rmse: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPolynomialResult {
    #[new]
    #[pyo3(signature = (
        degree, centered, x_mean, x_std, standardized, n_features,
        feature_names, feature_means, feature_stds,
        coefficients, standard_errors, t_statistics, p_values,
        r_squared, r_squared_adjusted, f_statistic, f_p_value,
        residuals, standardized_residuals, leverage, vif,
        n_observations, n_predictors, degrees_of_freedom, mse, rmse
    ))]
    fn new(
        degree: usize,
        centered: bool,
        x_mean: f64,
        x_std: f64,
        standardized: bool,
        n_features: usize,
        feature_names: Vec<String>,
        feature_means: Vec<f64>,
        feature_stds: Vec<f64>,
        coefficients: Vec<f64>,
        standard_errors: Vec<f64>,
        t_statistics: Vec<f64>,
        p_values: Vec<f64>,
        r_squared: f64,
        r_squared_adjusted: f64,
        f_statistic: f64,
        f_p_value: f64,
        residuals: Vec<f64>,
        standardized_residuals: Vec<f64>,
        leverage: Vec<f64>,
        vif: Vec<f64>,
        n_observations: usize,
        n_predictors: usize,
        degrees_of_freedom: usize,
        mse: f64,
        rmse: f64,
    ) -> Self {
        Self {
            degree,
            centered,
            x_mean,
            x_std,
            standardized,
            n_features,
            feature_names,
            feature_means,
            feature_stds,
            coefficients,
            standard_errors,
            t_statistics,
            p_values,
            r_squared,
            r_squared_adjusted,
            f_statistic,
            f_p_value,
            residuals,
            standardized_residuals,
            leverage,
            vif,
            n_observations,
            n_predictors,
            degrees_of_freedom,
            mse,
            rmse,
        }
    }

    /// Predict at new x values using this fitted model.
    ///
    /// Args:
    ///     x_new: New x values to predict at (list, tuple, or numpy array)
    ///
    /// Returns:
    ///     List of predicted y values
    pub fn predict(&self, py: Python, x_new: &Bound<PyAny>) -> PyResult<PyObject> {
        let x_vec = crate::python::extract_f64_sequence(x_new)?;

        // Helper to get VIF interpretation
        fn vif_interpretation(vif_val: f64) -> &'static str {
            if vif_val < 5.0 { "Low multicollinearity" }
            else if vif_val < 10.0 { "Moderate multicollinearity" }
            else { "High multicollinearity" }
        }

        // Reconstruct the PolynomialFit for prediction
        let n_coef = self.coefficients.len();
        let ols_output = crate::core::RegressionOutput {
            coefficients: self.coefficients.clone(),
            std_errors: self.standard_errors.clone(),
            t_stats: self.t_statistics.clone(),
            p_values: self.p_values.clone(),
            conf_int_lower: vec![0.0; n_coef],
            conf_int_upper: vec![0.0; n_coef],
            r_squared: self.r_squared,
            adj_r_squared: self.r_squared_adjusted,
            f_statistic: self.f_statistic,
            f_p_value: self.f_p_value,
            mse: self.mse,
            rmse: self.rmse,
            mae: 0.0,
            std_error: self.rmse,
            residuals: self.residuals.clone(),
            standardized_residuals: self.standardized_residuals.clone(),
            predictions: vec![0.0; self.n_observations],
            leverage: self.leverage.clone(),
            vif: self.vif.iter().enumerate().map(|(i, &v)| crate::core::VifResult {
                variable: if i < self.feature_names.len() && i > 0 {
                    self.feature_names[i].clone()
                } else {
                    format!("X{}", i)
                },
                vif: v,
                rsquared: 0.0,
                interpretation: vif_interpretation(v).to_string(),
            }).collect(),
            n: self.n_observations,
            k: self.n_predictors,
            df: self.degrees_of_freedom,
            variable_names: self.feature_names.clone(),
            log_likelihood: 0.0,
            aic: 0.0,
            bic: 0.0,
        };

        let fit = crate::polynomial::PolynomialFit {
            ols_output,
            degree: self.degree,
            centered: self.centered,
            x_mean: self.x_mean,
            x_std: self.x_std,
            standardized: self.standardized,
            n_features: self.n_features,
            feature_names: self.feature_names.clone(),
            feature_means: self.feature_means.clone(),
            feature_stds: self.feature_stds.clone(),
        };

        let predictions = crate::polynomial::predict(&fit, &x_vec)
            .map_err(|e| pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        use pyo3::types::PyList;
        Ok(PyList::new_bound(py, &predictions).into())
    }

    fn summary(&self) -> String {
        let coeff_table: String = self.coefficients
            .iter()
            .zip(&self.standard_errors)
            .zip(&self.t_statistics)
            .zip(&self.p_values)
            .zip(&self.feature_names)
            .enumerate()
            .map(|(i, zip_val)| {
                let ((((coef, se), t), p), name) = zip_val;
                format!("  [{:2}] {:12} coef={:8.4}, SE={:8.4}, t={:6.3}, p={:8.4e}",
                    i, name, coef, se, t, p)
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "Polynomial Regression Results (degree={})\n\
             =========================================\n\
             Centered: {}\n\
             Standardized: {}\n\
             X mean: {:.6}\n\
             \n\
             Coefficients:\n{}\n\
             \n\
             R-squared: {:.4}\n\
             Adjusted R-squared: {:.4}\n\
             F-statistic: {:.4} (p-value: {:.4e})\n\
             \n\
             Observations: {}\n\
             Degrees of freedom: {}\n\
             MSE: {:.6}\n\
             RMSE: {:.6}",
            self.degree,
            self.centered,
            self.standardized,
            self.x_mean,
            coeff_table,
            self.r_squared,
            self.r_squared_adjusted,
            self.f_statistic,
            self.f_p_value,
            self.n_observations,
            self.degrees_of_freedom,
            self.mse,
            self.rmse
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);

        // Polynomial-specific fields
        dict.set_item("degree", self.degree)?;
        dict.set_item("centered", self.centered)?;
        dict.set_item("x_mean", self.x_mean)?;
        dict.set_item("x_std", self.x_std)?;
        dict.set_item("standardized", self.standardized)?;
        dict.set_item("n_features", self.n_features)?;
        dict.set_item("feature_names", &self.feature_names)?;
        dict.set_item("feature_means", &self.feature_means)?;
        dict.set_item("feature_stds", &self.feature_stds)?;

        // OLS output fields
        dict.set_item("coefficients", &self.coefficients)?;
        dict.set_item("standard_errors", &self.standard_errors)?;
        dict.set_item("t_statistics", &self.t_statistics)?;
        dict.set_item("p_values", &self.p_values)?;
        dict.set_item("r_squared", self.r_squared)?;
        dict.set_item("r_squared_adjusted", self.r_squared_adjusted)?;
        dict.set_item("f_statistic", self.f_statistic)?;
        dict.set_item("f_p_value", self.f_p_value)?;
        dict.set_item("residuals", &self.residuals)?;
        dict.set_item("standardized_residuals", &self.standardized_residuals)?;
        dict.set_item("leverage", &self.leverage)?;
        dict.set_item("vif", &self.vif)?;
        dict.set_item("n_observations", self.n_observations)?;
        dict.set_item("n_predictors", self.n_predictors)?;
        dict.set_item("degrees_of_freedom", self.degrees_of_freedom)?;
        dict.set_item("mse", self.mse)?;
        dict.set_item("rmse", self.rmse)?;

        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "PolynomialResult(degree={}, R²={:.4})",
            self.degree, self.r_squared
        )
    }
}
