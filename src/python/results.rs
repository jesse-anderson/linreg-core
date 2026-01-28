// ============================================================================
// Result Classes for Python Bindings
// ============================================================================
// This module defines Python-accessible result classes that wrap
// the core Rust result types, providing a Pythonic API with
// attribute access and convenience methods.

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
        n_nonzero: usize,
        converged: bool,
        n_iterations: usize,
        r_squared: f64,
        log_likelihood: f64,
        aic: f64,
        bic: f64,
    ) -> Self {
        Self {
            intercept,
            coefficients,
            lambda,
            n_nonzero,
            converged,
            n_iterations,
            r_squared,
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
        dict.set_item("n_nonzero", self.n_nonzero)?;
        dict.set_item("converged", self.converged)?;
        dict.set_item("n_iterations", self.n_iterations)?;
        dict.set_item("r_squared", self.r_squared)?;
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
        n_nonzero: usize,
        converged: bool,
        n_iterations: usize,
        r_squared: f64,
        log_likelihood: f64,
        aic: f64,
        bic: f64,
    ) -> Self {
        Self {
            intercept,
            coefficients,
            lambda,
            alpha,
            n_nonzero,
            converged,
            n_iterations,
            r_squared,
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
        dict.set_item("n_nonzero", self.n_nonzero)?;
        dict.set_item("converged", self.converged)?;
        dict.set_item("n_iterations", self.n_iterations)?;
        dict.set_item("r_squared", self.r_squared)?;
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
// LambdaPathResult - Lambda path generation results
// ============================================================================

/// Result class for lambda path generation.
#[cfg(feature = "python")]
#[pyclass(name = "LambdaPathResult")]
pub struct PyLambdaPathResult {
    /// Generated lambda sequence (decreasing order)
    #[pyo3(get, set)]
    pub lambda_path: Vec<f64>,

    /// Maximum lambda value
    #[pyo3(get, set)]
    pub lambda_max: f64,

    /// Minimum lambda value
    #[pyo3(get, set)]
    pub lambda_min: f64,

    /// Number of lambda values
    #[pyo3(get, set)]
    pub n_lambda: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyLambdaPathResult {
    #[new]
    fn new(lambda_path: Vec<f64>, lambda_max: f64, lambda_min: f64, n_lambda: usize) -> Self {
        Self {
            lambda_path,
            lambda_max,
            lambda_min,
            n_lambda,
        }
    }

    fn summary(&self) -> String {
        format!(
            "Lambda Path Results\n\
             ===================\n\
             Lambda max: {:.6}\n\
             Lambda min: {:.6}\n\
             Number of values: {}",
            self.lambda_max, self.lambda_min, self.n_lambda
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("lambda_path", &self.lambda_path)?;
        dict.set_item("lambda_max", self.lambda_max)?;
        dict.set_item("lambda_min", self.lambda_min)?;
        dict.set_item("n_lambda", self.n_lambda)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "LambdaPathResult(n={}, max={:.2e}, min={:.2e})",
            self.n_lambda, self.lambda_max, self.lambda_min
        )
    }
}

// ============================================================================
// DiagnosticResult - Base diagnostic test result
// ============================================================================

/// Base result class for diagnostic tests.
#[cfg(feature = "python")]
#[pyclass(name = "DiagnosticResult")]
pub struct PyDiagnosticResult {
    /// Test statistic
    #[pyo3(get, set)]
    pub statistic: f64,

    /// p-value for the test
    #[pyo3(get, set)]
    pub p_value: f64,

    /// Name of the test
    #[pyo3(get, set)]
    pub test_name: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDiagnosticResult {
    #[new]
    fn new(statistic: f64, p_value: f64, test_name: String) -> Self {
        Self {
            statistic,
            p_value,
            test_name,
        }
    }

    fn summary(&self) -> String {
        format!(
            "{} Test\n\
             =========\n\
             Statistic: {:.4}\n\
             p-value: {:.4e}",
            self.test_name, self.statistic, self.p_value
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("statistic", self.statistic)?;
        dict.set_item("p_value", self.p_value)?;
        dict.set_item("test_name", &self.test_name)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "DiagnosticResult({}, stat={:.4}, p={:.2e})",
            self.test_name, self.statistic, self.p_value
        )
    }
}

// ============================================================================
// DurbinWatsonResult - Durbin-Watson test result
// ============================================================================

/// Result class for Durbin-Watson test.
#[cfg(feature = "python")]
#[pyclass(name = "DurbinWatsonResult")]
pub struct PyDurbinWatsonResult {
    /// DW statistic (0 to 4, ~2 indicates no autocorrelation)
    #[pyo3(get, set)]
    pub statistic: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDurbinWatsonResult {
    #[new]
    fn new(statistic: f64) -> Self {
        Self { statistic }
    }

    fn summary(&self) -> String {
        let interpretation = if self.statistic < 1.5 {
            "Positive autocorrelation likely"
        } else if self.statistic > 2.5 {
            "Negative autocorrelation likely"
        } else {
            "No significant autocorrelation"
        };
        format!(
            "Durbin-Watson Test\n\
             ===================\n\
             Statistic: {:.4}\n\
             Interpretation: {}",
            self.statistic, interpretation
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("statistic", self.statistic)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!("DurbinWatsonResult(stat={:.4})", self.statistic)
    }
}

// ============================================================================
// CooksDistanceResult - Cook's distance result
// ============================================================================

/// Result class for Cook's distance analysis.
#[cfg(feature = "python")]
#[pyclass(name = "CooksDistanceResult")]
pub struct PyCooksDistanceResult {
    /// Cook's distances (one per observation)
    #[pyo3(get, set)]
    pub distances: Vec<f64>,

    /// Number of parameters (including intercept)
    #[pyo3(get, set)]
    pub p: usize,

    /// Mean squared error of the model
    #[pyo3(get, set)]
    pub mse: f64,

    /// Common threshold: 4/n (observations above this are potentially influential)
    #[pyo3(get, set)]
    pub threshold_4_over_n: f64,

    /// Conservative threshold: 4/(n-p-1)
    #[pyo3(get, set)]
    pub threshold_4_over_df: f64,

    /// Absolute threshold: D_i > 1 indicates high influence
    #[pyo3(get, set)]
    pub threshold_1: f64,

    /// Indices of observations exceeding 4/n threshold
    #[pyo3(get, set)]
    pub influential_4_over_n: Vec<usize>,

    /// Indices of observations exceeding conservative threshold
    #[pyo3(get, set)]
    pub influential_4_over_df: Vec<usize>,

    /// Indices of observations exceeding D_i > 1 threshold
    #[pyo3(get, set)]
    pub influential_1: Vec<usize>,

    /// Interpretation of results
    #[pyo3(get, set)]
    pub interpretation: String,

    /// Guidance for handling influential observations
    #[pyo3(get, set)]
    pub guidance: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCooksDistanceResult {
    #[new]
    #[pyo3(text_signature = "(distances, p, mse, threshold_4_over_n, threshold_4_over_df, threshold_1, influential_4_over_n, influential_4_over_df, influential_1, interpretation, guidance)")]
    fn new(
        distances: Vec<f64>,
        p: usize,
        mse: f64,
        threshold_4_over_n: f64,
        threshold_4_over_df: f64,
        threshold_1: f64,
        influential_4_over_n: Vec<usize>,
        influential_4_over_df: Vec<usize>,
        influential_1: Vec<usize>,
        interpretation: String,
        guidance: String,
    ) -> Self {
        Self {
            distances,
            p,
            mse,
            threshold_4_over_n,
            threshold_4_over_df,
            threshold_1,
            influential_4_over_n,
            influential_4_over_df,
            influential_1,
            interpretation,
            guidance,
        }
    }

    fn summary(&self) -> String {
        format!(
            "Cook's Distance\n\
             ===============\n\
             Threshold (4/n): {:.4}\n\
             Threshold (4/df): {:.4}\n\
             Threshold (>1): {:.4}\n\
             Influential (4/n): {} observations: {:?}\n\
             Influential (4/df): {} observations: {:?}\n\
             Influential (>1): {} observations: {:?}\n\
             {}\n\
             {}",
            self.threshold_4_over_n,
            self.threshold_4_over_df,
            self.threshold_1,
            self.influential_4_over_n.len(),
            self.influential_4_over_n,
            self.influential_4_over_df.len(),
            self.influential_4_over_df,
            self.influential_1.len(),
            self.influential_1,
            self.interpretation,
            self.guidance
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("distances", &self.distances)?;
        dict.set_item("p", self.p)?;
        dict.set_item("mse", self.mse)?;
        dict.set_item("threshold_4_over_n", self.threshold_4_over_n)?;
        dict.set_item("threshold_4_over_df", self.threshold_4_over_df)?;
        dict.set_item("threshold_1", self.threshold_1)?;
        dict.set_item("influential_4_over_n", &self.influential_4_over_n)?;
        dict.set_item("influential_4_over_df", &self.influential_4_over_df)?;
        dict.set_item("influential_1", &self.influential_1)?;
        dict.set_item("interpretation", &self.interpretation)?;
        dict.set_item("guidance", &self.guidance)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "CooksDistanceResult(n={}, p={}, influential_4_over_n={})",
            self.distances.len(),
            self.p,
            self.influential_4_over_n.len()
        )
    }
}

// ============================================================================
// BreuschGodfreyResult - Breusch-Godfrey test result
// ============================================================================

/// Result class for Breusch-Godfrey test for higher-order serial correlation.
#[cfg(feature = "python")]
#[pyclass(name = "BreuschGodfreyResult")]
pub struct PyBreuschGodfreyResult {
    /// Name of the test
    #[pyo3(get, set)]
    pub test_name: String,

    /// Maximum order of serial correlation tested
    #[pyo3(get, set)]
    pub order: usize,

    /// Type of test statistic computed
    #[pyo3(get, set)]
    pub test_type: String,

    /// Test statistic value (LM or F)
    #[pyo3(get, set)]
    pub statistic: f64,

    /// P-value for the test
    #[pyo3(get, set)]
    pub p_value: f64,

    /// Degrees of freedom
    #[pyo3(get, set)]
    pub df: Vec<f64>,

    /// Whether the null hypothesis was not rejected (no serial correlation)
    #[pyo3(get, set)]
    pub passed: bool,

    /// Interpretation of the test result
    #[pyo3(get, set)]
    pub interpretation: String,

    /// Guidance for the user
    #[pyo3(get, set)]
    pub guidance: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBreuschGodfreyResult {
    #[new]
    #[pyo3(text_signature = "(test_name, order, test_type, statistic, p_value, df, passed, interpretation, guidance)")]
    pub fn new(
        test_name: String,
        order: usize,
        test_type: String,
        statistic: f64,
        p_value: f64,
        df: Vec<f64>,
        passed: bool,
        interpretation: String,
        guidance: String,
    ) -> Self {
        Self {
            test_name,
            order,
            test_type,
            statistic,
            p_value,
            df,
            passed,
            interpretation,
            guidance,
        }
    }

    fn summary(&self) -> String {
        format!(
            "Breusch-Godfrey Test\n\
             =====================\n\
             Order: {}\n\
             Type: {}\n\
             Statistic: {:.4}\n\
             p-value: {:.4e}\n\
             Degrees of freedom: {:?}\n\
             Passed: {}\n\
             {}\n\
             {}",
            self.order, self.test_type, self.statistic, self.p_value,
            self.df, self.passed, self.interpretation, self.guidance
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("test_name", &self.test_name)?;
        dict.set_item("order", self.order)?;
        dict.set_item("test_type", &self.test_type)?;
        dict.set_item("statistic", self.statistic)?;
        dict.set_item("p_value", self.p_value)?;
        dict.set_item("df", &self.df)?;
        dict.set_item("passed", self.passed)?;
        dict.set_item("interpretation", &self.interpretation)?;
        dict.set_item("guidance", &self.guidance)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "BreuschGodfreyResult(order={}, stat={:.4}, p={:.2e}, passed={})",
            self.order, self.statistic, self.p_value, self.passed
        )
    }
}

// ============================================================================
// RainbowTestResult - Rainbow test result with both methods
// ============================================================================

/// Result class for Rainbow test supporting both R and Python variants.
/// Has flattened fields to avoid PyO3 nesting issues.
#[cfg(feature = "python")]
#[pyclass(name = "RainbowTestResult")]
pub struct PyRainbowTestResult {
    /// Name of the test
    #[pyo3(get, set)]
    pub test_name: String,

    /// Whether R result is available
    #[pyo3(get, set)]
    pub has_r_result: bool,

    /// R method statistic (available if has_r_result is true)
    #[pyo3(get, set)]
    pub r_statistic: f64,

    /// R method p-value (available if has_r_result is true)
    #[pyo3(get, set)]
    pub r_p_value: f64,

    /// Whether Python result is available
    #[pyo3(get, set)]
    pub has_python_result: bool,

    /// Python method statistic (available if has_python_result is true)
    #[pyo3(get, set)]
    pub python_statistic: f64,

    /// Python method p-value (available if has_python_result is true)
    #[pyo3(get, set)]
    pub python_p_value: f64,

    /// Interpretation of the test result
    #[pyo3(get, set)]
    pub interpretation: String,

    /// Guidance based on the test result
    #[pyo3(get, set)]
    pub guidance: String,
}

#[cfg(feature = "python")]
impl PyRainbowTestResult {
    /// Internal constructor from core result (not exposed to Python)
    pub fn from_core_result(result: crate::diagnostics::RainbowTestOutput) -> Self {
        let (has_r, r_stat, r_pval) = if let Some(ref r) = result.r_result {
            (true, r.statistic, r.p_value)
        } else {
            (false, 0.0, 0.0)
        };

        let (has_py, py_stat, py_pval) = if let Some(ref p) = result.python_result {
            (true, p.statistic, p.p_value)
        } else {
            (false, 0.0, 0.0)
        };

        // Direct struct construction to avoid PyO3's private `new` method
        PyRainbowTestResult {
            test_name: result.test_name,
            has_r_result: has_r,
            r_statistic: r_stat,
            r_p_value: r_pval,
            has_python_result: has_py,
            python_statistic: py_stat,
            python_p_value: py_pval,
            interpretation: result.interpretation,
            guidance: result.guidance,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRainbowTestResult {
    #[new]
    fn new(
        test_name: String,
        has_r_result: bool,
        r_statistic: f64,
        r_p_value: f64,
        has_python_result: bool,
        python_statistic: f64,
        python_p_value: f64,
        interpretation: String,
        guidance: String,
    ) -> Self {
        Self {
            test_name,
            has_r_result,
            r_statistic,
            r_p_value,
            has_python_result,
            python_statistic,
            python_p_value,
            interpretation,
            guidance,
        }
    }

    fn summary(&self) -> String {
        let methods: Vec<&str> = [
            if self.has_r_result { Some("R") } else { None },
            if self.has_python_result { Some("Python") } else { None },
        ]
        .iter()
        .filter_map(|&x| x)
        .collect();

        format!(
            "Rainbow Test\n\
             ============\n\
             Methods: {}\n\
             {}\n\
             {}",
            methods.join(", "),
            self.interpretation,
            self.guidance
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("test_name", &self.test_name)?;

        if self.has_r_result {
            let r_dict = PyDict::new_bound(py);
            r_dict.set_item("statistic", self.r_statistic)?;
            r_dict.set_item("p_value", self.r_p_value)?;
            dict.set_item("r_result", r_dict)?;
        } else {
            dict.set_item("r_result", py.None())?;
        }

        if self.has_python_result {
            let py_dict = PyDict::new_bound(py);
            py_dict.set_item("statistic", self.python_statistic)?;
            py_dict.set_item("p_value", self.python_p_value)?;
            dict.set_item("python_result", py_dict)?;
        } else {
            dict.set_item("python_result", py.None())?;
        }

        dict.set_item("interpretation", &self.interpretation)?;
        dict.set_item("guidance", &self.guidance)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "RainbowTestResult(has_r={}, has_python={})",
            self.has_r_result, self.has_python_result
        )
    }
}

// ============================================================================
// WhiteTestResult - White test result with both methods
// ============================================================================

/// Result class for White test supporting both R and Python variants.
/// Has flattened fields to avoid PyO3 nesting issues.
#[cfg(feature = "python")]
#[pyclass(name = "WhiteTestResult")]
pub struct PyWhiteTestResult {
    /// Name of the test
    #[pyo3(get, set)]
    pub test_name: String,

    /// Whether R result is available
    #[pyo3(get, set)]
    pub has_r_result: bool,

    /// R method statistic (available if has_r_result is true)
    #[pyo3(get, set)]
    pub r_statistic: f64,

    /// R method p-value (available if has_r_result is true)
    #[pyo3(get, set)]
    pub r_p_value: f64,

    /// Whether Python result is available
    #[pyo3(get, set)]
    pub has_python_result: bool,

    /// Python method statistic (available if has_python_result is true)
    #[pyo3(get, set)]
    pub python_statistic: f64,

    /// Python method p-value (available if has_python_result is true)
    #[pyo3(get, set)]
    pub python_p_value: f64,

    /// Interpretation of the test result
    #[pyo3(get, set)]
    pub interpretation: String,

    /// Guidance based on the test result
    #[pyo3(get, set)]
    pub guidance: String,
}

#[cfg(feature = "python")]
impl PyWhiteTestResult {
    /// Internal constructor from core result (not exposed to Python)
    pub fn from_core_result(result: crate::diagnostics::WhiteTestOutput) -> Self {
        let (has_r, r_stat, r_pval) = if let Some(ref r) = result.r_result {
            (true, r.statistic, r.p_value)
        } else {
            (false, 0.0, 0.0)
        };

        let (has_py, py_stat, py_pval) = if let Some(ref p) = result.python_result {
            (true, p.statistic, p.p_value)
        } else {
            (false, 0.0, 0.0)
        };

        // Direct struct construction to avoid PyO3's private `new` method
        PyWhiteTestResult {
            test_name: result.test_name,
            has_r_result: has_r,
            r_statistic: r_stat,
            r_p_value: r_pval,
            has_python_result: has_py,
            python_statistic: py_stat,
            python_p_value: py_pval,
            interpretation: result.interpretation,
            guidance: result.guidance,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyWhiteTestResult {
    #[new]
    fn new(
        test_name: String,
        has_r_result: bool,
        r_statistic: f64,
        r_p_value: f64,
        has_python_result: bool,
        python_statistic: f64,
        python_p_value: f64,
        interpretation: String,
        guidance: String,
    ) -> Self {
        Self {
            test_name,
            has_r_result,
            r_statistic,
            r_p_value,
            has_python_result,
            python_statistic,
            python_p_value,
            interpretation,
            guidance,
        }
    }

    fn summary(&self) -> String {
        let methods: Vec<&str> = [
            if self.has_r_result { Some("R") } else { None },
            if self.has_python_result { Some("Python") } else { None },
        ]
        .iter()
        .filter_map(|&x| x)
        .collect();

        format!(
            "White Test\n\
             ==========\n\
             Methods: {}\n\
             {}\n\
             {}",
            methods.join(", "),
            self.interpretation,
            self.guidance
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("test_name", &self.test_name)?;

        if self.has_r_result {
            let r_dict = PyDict::new_bound(py);
            r_dict.set_item("statistic", self.r_statistic)?;
            r_dict.set_item("p_value", self.r_p_value)?;
            dict.set_item("r_result", r_dict)?;
        } else {
            dict.set_item("r_result", py.None())?;
        }

        if self.has_python_result {
            let py_dict = PyDict::new_bound(py);
            py_dict.set_item("statistic", self.python_statistic)?;
            py_dict.set_item("p_value", self.python_p_value)?;
            dict.set_item("python_result", py_dict)?;
        } else {
            dict.set_item("python_result", py.None())?;
        }

        dict.set_item("interpretation", &self.interpretation)?;
        dict.set_item("guidance", &self.guidance)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "WhiteTestResult(has_r={}, has_python={})",
            self.has_r_result, self.has_python_result
        )
    }
}

// ============================================================================
// CSVResult - CSV parsing result
// ============================================================================

/// Result class for CSV parsing.
#[cfg(feature = "python")]
#[pyclass(name = "CSVResult")]
pub struct PyCSVResult {
    /// Column headers from the CSV
    #[pyo3(get, set)]
    pub headers: Vec<String>,

    /// Parsed data as Python object (list of dicts)
    #[pyo3(get)]
    pub data: PyObject,

    /// Names of columns that contain numeric data
    #[pyo3(get, set)]
    pub numeric_columns: Vec<String>,

    /// Number of rows parsed
    #[pyo3(get, set)]
    pub n_rows: usize,

    /// Number of columns parsed
    #[pyo3(get, set)]
    pub n_cols: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCSVResult {
    #[new]
    fn new(
        headers: Vec<String>,
        data: PyObject,
        numeric_columns: Vec<String>,
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        Self {
            headers,
            data,
            numeric_columns,
            n_rows,
            n_cols,
        }
    }

    /// Get the parsed data as a Python list of dicts.
    fn get_data(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.data.clone_ref(py))
    }

    fn summary(&self) -> String {
        format!(
            "CSV Parsing Results\n\
             ====================\n\
             Rows: {}\n\
             Columns: {}\n\
             Headers: {:?}\n\
             Numeric columns: {:?}",
            self.n_rows, self.n_cols, self.headers, self.numeric_columns
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        use pyo3::types::PyList;
        let dict = PyDict::new_bound(py);

        // Set headers as Python list
        let headers_list = PyList::new_bound(py, &self.headers);
        dict.set_item("headers", headers_list)?;

        // Set numeric_columns as Python list
        let numeric_list = PyList::new_bound(py, &self.numeric_columns);
        dict.set_item("numeric_columns", numeric_list)?;

        // Set n_rows and n_cols
        dict.set_item("n_rows", self.n_rows)?;
        dict.set_item("n_cols", self.n_cols)?;

        // Set data (already a Python object)
        dict.set_item("data", self.data.clone_ref(py))?;

        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "CSVResult(n_rows={}, n_cols={}, numeric_cols={})",
            self.n_rows, self.n_cols, self.numeric_columns.len()
        )
    }
}
