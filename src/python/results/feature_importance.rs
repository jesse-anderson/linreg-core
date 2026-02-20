// ============================================================================
// Feature Importance Result Classes for Python Bindings
// ============================================================================

#[cfg(feature = "python")]
use pyo3::prelude::*;

// ============================================================================
// StandardizedCoefficientsResult
// ============================================================================

/// Result class for standardized coefficients.
///
/// Standardized coefficients (beta*) represent the change in Y (in standard deviations)
/// for a one standard deviation change in X, making coefficients comparable
/// across predictors with different units/scales.
#[cfg(feature = "python")]
#[pyclass(name = "StandardizedCoefficientsResult")]
pub struct PyStandardizedCoefficientsResult {
    /// Names of predictor variables (excluding intercept)
    #[pyo3(get, set)]
    pub variable_names: Vec<String>,

    /// Standardized coefficients (one per predictor)
    #[pyo3(get, set)]
    pub standardized_coefficients: Vec<f64>,

    /// Standard deviation of the response variable Y
    #[pyo3(get, set)]
    pub y_std: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyStandardizedCoefficientsResult {
    #[new]
    fn new(
        variable_names: Vec<String>,
        standardized_coefficients: Vec<f64>,
        y_std: f64,
    ) -> Self {
        Self {
            variable_names,
            standardized_coefficients,
            y_std,
        }
    }

    /// Returns the ranking of variables by absolute standardized coefficient value.
    ///
    /// Returns a list of (variable_name, absolute_value) tuples, sorted by importance
    /// (highest absolute coefficient first).
    fn ranking(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<_> = self
            .variable_names
            .iter()
            .zip(self.standardized_coefficients.iter())
            .map(|(name, &coef)| (name.clone(), coef.abs()))
            .collect();
        // Sort by absolute value descending, treat NaN as 0
        ranked.sort_by(|a, b| {
            let a_val = if a.1.is_nan() { 0.0 } else { a.1 };
            let b_val = if b.1.is_nan() { 0.0 } else { b.1 };
            b_val.partial_cmp(&a_val).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Summary of standardized coefficients.
    fn summary(&self) -> String {
        let ranking = self.ranking();
        let ranking_str: String = ranking
            .iter()
            .enumerate()
            .map(|(i, (name, val))| format!("  {}. {}: {:.4}", i + 1, name, val))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "Standardized Coefficients\n\
             =========================\n\
             Y std dev: {:.4}\n\
             \n\
             Ranking by absolute value:\n{}\n\
             \n\
             Raw coefficients:\n{}",
            self.y_std,
            ranking_str,
            self.variable_names
                .iter()
                .zip(self.standardized_coefficients.iter())
                .map(|(name, coef)| format!("  {}: {:.4}", name, coef))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Convert result to a dictionary.
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("variable_names", &self.variable_names)?;
        dict.set_item("standardized_coefficients", &self.standardized_coefficients)?;
        dict.set_item("y_std", self.y_std)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "StandardizedCoefficientsResult(n={})",
            self.variable_names.len()
        )
    }
}

// ============================================================================
// ShapResult
// ============================================================================

/// Result class for SHAP (SHapley Additive exPlanations) values.
///
/// SHAP values decompose predictions into the contribution of each feature.
#[cfg(feature = "python")]
#[pyclass(name = "ShapResult")]
pub struct PyShapResult {
    /// Names of predictor variables (excluding intercept)
    #[pyo3(get, set)]
    pub variable_names: Vec<String>,

    /// SHAP values matrix (n_observations × n_features)
    /// `shap_values\[i]\[j\]` = contribution of feature `j` to observation `i`
    #[pyo3(get, set)]
    pub shap_values: Vec<Vec<f64>>,

    /// Base value (mean prediction / intercept contribution)
    #[pyo3(get, set)]
    pub base_value: f64,

    /// Mean absolute SHAP values per feature (global importance)
    #[pyo3(get, set)]
    pub mean_abs_shap: Vec<f64>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyShapResult {
    #[new]
    fn new(
        variable_names: Vec<String>,
        shap_values: Vec<Vec<f64>>,
        base_value: f64,
        mean_abs_shap: Vec<f64>,
    ) -> Self {
        Self {
            variable_names,
            shap_values,
            base_value,
            mean_abs_shap,
        }
    }

    /// Returns the ranking of variables by mean absolute SHAP value.
    ///
    /// Returns a list of (variable_name, mean_abs_shap) tuples, sorted by importance
    /// (highest mean absolute SHAP first).
    fn ranking(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<_> = self
            .variable_names
            .iter()
            .zip(self.mean_abs_shap.iter())
            .map(|(name, &shap)| (name.clone(), shap))
            .collect();
        // Sort by mean absolute SHAP descending, treat NaN as 0
        ranked.sort_by(|a, b| {
            let a_val = if a.1.is_nan() { 0.0 } else { a.1 };
            let b_val = if b.1.is_nan() { 0.0 } else { b.1 };
            b_val.partial_cmp(&a_val).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Gets the SHAP decomposition for a single observation.
    ///
    /// Args:
    ///     observation_index: Index of the observation (0-based)
    ///
    /// Returns a list of (variable_name, shap_value) tuples for that observation.
    fn observation_contribution(&self, observation_index: usize) -> Vec<(String, f64)> {
        if observation_index >= self.shap_values.len() {
            return vec![];
        }

        self.variable_names
            .iter()
            .zip(&self.shap_values[observation_index])
            .map(|(name, &shap)| (name.clone(), shap))
            .collect()
    }

    /// Summary of SHAP values.
    fn summary(&self) -> String {
        let ranking = self.ranking();
        let ranking_str: String = ranking
            .iter()
            .enumerate()
            .map(|(i, (name, val))| format!("  {}. {}: {:.4}", i + 1, name, val))
            .collect::<Vec<_>>()
            .join("\n");

        let n_obs = self.shap_values.len();

        format!(
            "SHAP Values Result\n\
             ==================\n\
             Observations: {}\n\
             Features: {}\n\
             Base value: {:.4}\n\
             \n\
             Global importance (mean |SHAP|):\n{}\n\
             \n\
             Mean absolute SHAP:\n{}",
            n_obs,
            self.variable_names.len(),
            self.base_value,
            ranking_str,
            self.variable_names
                .iter()
                .zip(self.mean_abs_shap.iter())
                .map(|(name, shap)| format!("  {}: {:.4}", name, shap))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Convert result to a dictionary.
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("variable_names", &self.variable_names)?;
        dict.set_item("shap_values", &self.shap_values)?;
        dict.set_item("base_value", self.base_value)?;
        dict.set_item("mean_abs_shap", &self.mean_abs_shap)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapResult(n_obs={}, n_features={}, base_value={:.4})",
            self.shap_values.len(),
            self.variable_names.len(),
            self.base_value
        )
    }
}

// ============================================================================
// VifRankingResult
// ============================================================================

/// Result class for VIF (Variance Inflation Factor) ranking.
///
/// VIF measures how much the variance of a coefficient is inflated due to
/// multicollinearity. Lower VIF = less redundant = more uniquely important.
#[cfg(feature = "python")]
#[pyclass(name = "VifRankingResult")]
pub struct PyVifRankingResult {
    /// Names of predictor variables
    #[pyo3(get, set)]
    pub variable_names: Vec<String>,

    /// VIF values for each predictor (lower = less redundant)
    #[pyo3(get, set)]
    pub vif_values: Vec<f64>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyVifRankingResult {
    #[new]
    fn new(variable_names: Vec<String>, vif_values: Vec<f64>) -> Self {
        Self {
            variable_names,
            vif_values,
        }
    }

    /// Returns the ranking of variables by VIF value (ascending).
    ///
    /// Lower VIF values indicate less multicollinearity and are ranked higher.
    /// Returns a list of (variable_name, vif_value) tuples.
    fn ranking(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<_> = self
            .variable_names
            .iter()
            .zip(self.vif_values.iter())
            .map(|(name, &vif)| (name.clone(), vif))
            .collect();
        // Sort by VIF ascending (lower is better), treat NaN as infinity (worst)
        ranked.sort_by(|a, b| {
            let a_val = if a.1.is_nan() { f64::INFINITY } else { a.1 };
            let b_val = if b.1.is_nan() { f64::INFINITY } else { b.1 };
            a_val.partial_cmp(&b_val).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Get interpretation for a VIF value.
    ///
    /// Args:
    ///     vif: VIF value to interpret
    ///
    /// Returns interpretation string.
    #[staticmethod]
    fn interpret_vif(vif: f64) -> String {
        if vif < 1.0 {
            "Negative correlation (unusual)".to_string()
        } else if vif <= 5.0 {
            "Low multicollinearity (acceptable)".to_string()
        } else if vif <= 10.0 {
            "Moderate multicollinearity (review)".to_string()
        } else {
            "High multicollinearity (problematic)".to_string()
        }
    }

    /// Get interpretations for all VIF values.
    fn interpretations(&self) -> Vec<String> {
        self.vif_values
            .iter()
            .map(|&vif| Self::interpret_vif(vif))
            .collect()
    }

    /// Summary of VIF ranking.
    fn summary(&self) -> String {
        let ranking = self.ranking();
        let ranking_str: String = ranking
            .iter()
            .enumerate()
            .map(|(i, (name, vif))| {
                format!(
                    "  {}. {}: {:.2} - {}",
                    i + 1,
                    name,
                    vif,
                    Self::interpret_vif(*vif)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "VIF Ranking\n\
             ===========\n\
             Features: {}\n\
             \n\
             Ranking (lowest VIF first = most independent):\n{}\n\
             \n\
             VIF values:\n{}",
            self.variable_names.len(),
            ranking_str,
            self.variable_names
                .iter()
                .zip(self.vif_values.iter())
                .map(|(name, vif)| {
                    format!(
                        "  {}: {:.2} - {}",
                        name,
                        vif,
                        Self::interpret_vif(*vif)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Convert result to a dictionary.
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("variable_names", &self.variable_names)?;
        dict.set_item("vif_values", &self.vif_values)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "VifRankingResult(n={})",
            self.variable_names.len()
        )
    }
}

// ============================================================================
// PermutationImportanceResult
// ============================================================================

/// Result class for permutation importance.
///
/// Permutation importance measures the decrease in model performance (typically R²)
/// when a single predictor's values are randomly shuffled.
#[cfg(feature = "python")]
#[pyclass(name = "PermutationImportanceResult")]
pub struct PyPermutationImportanceResult {
    /// Names of predictor variables
    #[pyo3(get, set)]
    pub variable_names: Vec<String>,

    /// Importance scores (baseline_score - score_when_shuffled)
    #[pyo3(get, set)]
    pub importance: Vec<f64>,

    /// Baseline model performance (R² on original data)
    #[pyo3(get, set)]
    pub baseline_score: f64,

    /// Number of permutations performed
    #[pyo3(get, set)]
    pub n_permutations: usize,

    /// Random seed used (None if not provided)
    #[pyo3(get, set)]
    pub seed: Option<u64>,

    /// Standard error of importance scores (None if intervals were not computed)
    #[pyo3(get, set)]
    pub importance_std_err: Option<Vec<f64>>,

    /// Lower bounds of confidence intervals (None if not computed)
    #[pyo3(get, set)]
    pub interval_lower: Option<Vec<f64>>,

    /// Upper bounds of confidence intervals (None if not computed)
    #[pyo3(get, set)]
    pub interval_upper: Option<Vec<f64>>,

    /// Confidence level used for intervals (None if not computed)
    #[pyo3(get, set)]
    pub interval_confidence: Option<f64>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPermutationImportanceResult {
    #[new]
    #[pyo3(signature = (
        variable_names,
        importance,
        baseline_score,
        n_permutations,
        seed=None,
        importance_std_err=None,
        interval_lower=None,
        interval_upper=None,
        interval_confidence=None
    ))]
    fn new(
        variable_names: Vec<String>,
        importance: Vec<f64>,
        baseline_score: f64,
        n_permutations: usize,
        seed: Option<u64>,
        importance_std_err: Option<Vec<f64>>,
        interval_lower: Option<Vec<f64>>,
        interval_upper: Option<Vec<f64>>,
        interval_confidence: Option<f64>,
    ) -> Self {
        Self {
            variable_names,
            importance,
            baseline_score,
            n_permutations,
            seed,
            importance_std_err,
            interval_lower,
            interval_upper,
            interval_confidence,
        }
    }

    /// Returns the ranking of variables by permutation importance.
    ///
    /// Higher importance = more important (larger performance drop when shuffled).
    /// Returns a list of (variable_name, importance) tuples.
    fn ranking(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<_> = self
            .variable_names
            .iter()
            .zip(self.importance.iter())
            .map(|(name, &imp)| (name.clone(), imp))
            .collect();
        // Sort by importance descending (higher is better), treat NaN as 0
        ranked.sort_by(|a, b| {
            let a_val = if a.1.is_nan() { 0.0 } else { a.1 };
            let b_val = if b.1.is_nan() { 0.0 } else { b.1 };
            b_val.partial_cmp(&a_val).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Summary of permutation importance.
    fn summary(&self) -> String {
        let ranking = self.ranking();
        let ranking_str: String = ranking
            .iter()
            .enumerate()
            .map(|(i, (name, imp))| format!("  {}. {}: {:.4}", i + 1, name, imp))
            .collect::<Vec<_>>()
            .join("\n");

        let seed_str = self.seed.map_or("None".to_string(), |s| format!("Some({})", s));
        let interval_str = if let (Some(lower), Some(upper), Some(conf)) = (
            &self.interval_lower,
            &self.interval_upper,
            self.interval_confidence,
        ) {
            let intervals: String = self
                .variable_names
                .iter()
                .zip(lower.iter())
                .zip(upper.iter())
                .map(|((name, &l), &u)| format!("  {}: [{:.4}, {:.4}]", name, l, u))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "\nConfidence Intervals ({}%):\n{}",
                conf * 100.0,
                intervals
            )
        } else {
            String::new()
        };

        format!(
            "Permutation Importance\n\
             ======================\n\
             Baseline R²: {:.4}\n\
             Permutations: {}\n\
             Seed: {}\n\
             Features: {}\n\
             \n\
             Ranking (highest importance first):\n{}\n\
             \n\
             Importance scores:\n{}\n\
             {}",
            self.baseline_score,
            self.n_permutations,
            seed_str,
            self.variable_names.len(),
            ranking_str,
            self.variable_names
                .iter()
                .zip(self.importance.iter())
                .map(|(name, imp)| format!("  {}: {:.4}", name, imp))
                .collect::<Vec<_>>()
                .join("\n"),
            interval_str
        )
    }

    /// Convert result to a dictionary.
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("variable_names", &self.variable_names)?;
        dict.set_item("importance", &self.importance)?;
        dict.set_item("baseline_score", self.baseline_score)?;
        dict.set_item("n_permutations", self.n_permutations)?;
        dict.set_item("seed", self.seed)?;

        if let Some(ref std_err) = self.importance_std_err {
            dict.set_item("importance_std_err", std_err)?;
        }
        if let Some(ref lower) = self.interval_lower {
            dict.set_item("interval_lower", lower)?;
        }
        if let Some(ref upper) = self.interval_upper {
            dict.set_item("interval_upper", upper)?;
        }
        if let Some(conf) = self.interval_confidence {
            dict.set_item("interval_confidence", conf)?;
        }

        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "PermutationImportanceResult(n={}, baseline_R²={:.4})",
            self.variable_names.len(),
            self.baseline_score
        )
    }
}
