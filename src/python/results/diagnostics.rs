// ============================================================================
// Diagnostic Test Result Classes for Python Bindings
// ============================================================================
// DiagnosticResult, DurbinWatson, CooksDistance, Dfbetas, Dffits,
// VifDetail, VifTestResult, BreuschGodfrey, RainbowTest, WhiteTest

#[cfg(feature = "python")]
use pyo3::prelude::*;

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
// DfbetasResult - DFBETAS analysis result
// ============================================================================

/// Result class for DFBETAS analysis (influence of observations on coefficients).
#[cfg(feature = "python")]
#[pyclass(name = "DfbetasResult")]
pub struct PyDfbetasResult {
    /// DFBETAS matrix: n rows (observations) x p columns (parameters)
    #[pyo3(get, set)]
    pub dfbetas: Vec<Vec<f64>>,

    /// Number of observations
    #[pyo3(get, set)]
    pub n: usize,

    /// Number of parameters (including intercept)
    #[pyo3(get, set)]
    pub p: usize,

    /// Common threshold: 2/sqrt(n)
    #[pyo3(get, set)]
    pub threshold: f64,

    /// Map of coefficient indices (1-based) to list of influential observation indices (1-based)
    #[pyo3(get, set)]
    pub influential_observations: std::collections::HashMap<usize, Vec<usize>>,

    /// Interpretation of results
    #[pyo3(get, set)]
    pub interpretation: String,

    /// Guidance for handling influential observations
    #[pyo3(get, set)]
    pub guidance: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDfbetasResult {
    #[new]
    #[pyo3(text_signature = "(dfbetas, n, p, threshold, influential_observations, interpretation, guidance)")]
    fn new(
        dfbetas: Vec<Vec<f64>>,
        n: usize,
        p: usize,
        threshold: f64,
        influential_observations: std::collections::HashMap<usize, Vec<usize>>,
        interpretation: String,
        guidance: String,
    ) -> Self {
        Self {
            dfbetas,
            n,
            p,
            threshold,
            influential_observations,
            interpretation,
            guidance,
        }
    }

    fn summary(&self) -> String {
        let total_influential: usize = self.influential_observations.values().map(|v| v.len()).sum();
        let mut details = Vec::new();
        for (coef_idx, obs_list) in &self.influential_observations {
            let coef_name = if *coef_idx == 1 { "intercept" } else { &format!("X{}", coef_idx - 1) };
            details.push(format!("{}: {:?}", coef_name, obs_list));
        }
        let details_str = if details.is_empty() {
            "None".to_string()
        } else {
            details.join(", ")
        };
        format!(
            "DFBETAS\n\
             =======\n\
             Observations (n): {}\n\
             Parameters (p): {}\n\
             Threshold (2/√n): {:.4}\n\
             Influential observations: {}\n\
             Details: {}\n\
             {}\n\
             {}",
            self.n,
            self.p,
            self.threshold,
            total_influential,
            details_str,
            self.interpretation,
            self.guidance
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("dfbetas", &self.dfbetas)?;
        dict.set_item("n", self.n)?;
        dict.set_item("p", self.p)?;
        dict.set_item("threshold", self.threshold)?;
        dict.set_item("influential_observations", &self.influential_observations)?;
        dict.set_item("interpretation", &self.interpretation)?;
        dict.set_item("guidance", &self.guidance)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        let total_influential: usize = self.influential_observations.values().map(|v| v.len()).sum();
        format!(
            "DfbetasResult(n={}, p={}, influential={})",
            self.n,
            self.p,
            total_influential
        )
    }
}

// ============================================================================
// DffitsResult - DFFITS analysis result
// ============================================================================

/// Result class for DFFITS analysis (influence of observations on fitted values).
#[cfg(feature = "python")]
#[pyclass(name = "DffitsResult")]
pub struct PyDffitsResult {
    /// DFFITS value for each observation
    #[pyo3(get, set)]
    pub dffits: Vec<f64>,

    /// Number of observations
    #[pyo3(get, set)]
    pub n: usize,

    /// Number of parameters (including intercept)
    #[pyo3(get, set)]
    pub p: usize,

    /// Common threshold: 2*sqrt(p/n)
    #[pyo3(get, set)]
    pub threshold: f64,

    /// Indices of observations exceeding |DFFITS| > threshold (1-based indexing)
    #[pyo3(get, set)]
    pub influential_observations: Vec<usize>,

    /// Interpretation of results
    #[pyo3(get, set)]
    pub interpretation: String,

    /// Guidance for handling influential observations
    #[pyo3(get, set)]
    pub guidance: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDffitsResult {
    #[new]
    #[pyo3(text_signature = "(dffits, n, p, threshold, influential_observations, interpretation, guidance)")]
    fn new(
        dffits: Vec<f64>,
        n: usize,
        p: usize,
        threshold: f64,
        influential_observations: Vec<usize>,
        interpretation: String,
        guidance: String,
    ) -> Self {
        Self {
            dffits,
            n,
            p,
            threshold,
            influential_observations,
            interpretation,
            guidance,
        }
    }

    fn summary(&self) -> String {
        format!(
            "DFFITS\n\
             =======\n\
             Observations (n): {}\n\
             Parameters (p): {}\n\
             Threshold (2*√(p/n)): {:.4}\n\
             Influential observations: {} - {:?}\n\
             {}\n\
             {}",
            self.n,
            self.p,
            self.threshold,
            self.influential_observations.len(),
            self.influential_observations,
            self.interpretation,
            self.guidance
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("dffits", &self.dffits)?;
        dict.set_item("n", self.n)?;
        dict.set_item("p", self.p)?;
        dict.set_item("threshold", self.threshold)?;
        dict.set_item("influential_observations", &self.influential_observations)?;
        dict.set_item("interpretation", &self.interpretation)?;
        dict.set_item("guidance", &self.guidance)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "DffitsResult(n={}, p={}, influential={})",
            self.n,
            self.p,
            self.influential_observations.len()
        )
    }
}

// ============================================================================
// VifDetail - Individual VIF result for a single variable
// ============================================================================

/// Detail class for a single variable's VIF result.
#[cfg(feature = "python")]
#[pyclass(name = "VifDetail")]
pub struct PyVifDetail {
    /// Variable name
    #[pyo3(get, set)]
    pub variable: String,

    /// VIF value for this variable
    #[pyo3(get, set)]
    pub vif: f64,

    /// R-squared from regressing this variable on others
    #[pyo3(get, set)]
    pub rsquared: f64,

    /// Interpretation of this VIF value
    #[pyo3(get, set)]
    pub interpretation: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyVifDetail {
    #[new]
    #[pyo3(text_signature = "(variable, vif, rsquared, interpretation)")]
    fn new(variable: String, vif: f64, rsquared: f64, interpretation: String) -> Self {
        Self {
            variable,
            vif,
            rsquared,
            interpretation,
        }
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("variable", &self.variable)?;
        dict.set_item("vif", self.vif)?;
        dict.set_item("rsquared", self.rsquared)?;
        dict.set_item("interpretation", &self.interpretation)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        format!(
            "VifDetail(variable={}, VIF={:.4}, R²={:.4}, {})",
            self.variable, self.vif, self.rsquared, self.interpretation
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "VifDetail(variable='{}', vif={:.4}, rsquared={:.4})",
            self.variable, self.vif, self.rsquared
        )
    }
}

// ============================================================================
// VifTestResult - VIF test result
// ============================================================================

/// Result class for Variance Inflation Factor (VIF) analysis.
#[cfg(feature = "python")]
#[pyclass(name = "VifTestResult")]
pub struct PyVifTestResult {
    /// Maximum VIF value across all variables
    #[pyo3(get, set)]
    pub max_vif: f64,

    /// Individual VIF results for each variable (as Python list)
    #[pyo3(get)]
    pub vif_results: PyObject,

    /// Overall interpretation
    #[pyo3(get, set)]
    pub interpretation: String,

    /// Guidance for addressing multicollinearity
    #[pyo3(get, set)]
    pub guidance: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyVifTestResult {
    // No #[new] method - objects are constructed from Rust code
    // The vif_test function directly creates PyVifTestResult instances

    fn summary(&self) -> String {
        format!(
            "VIF Test\n\
            ========\n\
             Max VIF: {:.4}\n\
             {}\n\
             {}",
            self.max_vif,
            self.interpretation,
            self.guidance
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("max_vif", self.max_vif)?;
        dict.set_item("vif_results", &self.vif_results)?;
        dict.set_item("interpretation", &self.interpretation)?;
        dict.set_item("guidance", &self.guidance)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "VifTestResult(max_vif={:.4})",
            self.max_vif
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
