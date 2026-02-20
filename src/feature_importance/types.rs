//! Shared types for feature importance metrics.

use crate::core::VifResult;
use serde::{Deserialize, Serialize};

/// Output from standardized coefficients calculation.
///
/// Standardized coefficients (beta*) represent the change in Y (in standard deviations)
/// for a one standard deviation change in X. This makes coefficients comparable
/// across predictors with different units/scales.
///
/// # Interpretation
///
/// - `beta_star[i]` = "A 1 SD increase in Xᵢ leads to beta_star[i] SD change in Y"
/// - Absolute values indicate relative importance
/// - Sign indicates direction of relationship
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::StandardizedCoefficientsOutput;
/// let output = StandardizedCoefficientsOutput {
///     variable_names: vec!["X1".to_string(), "X2".to_string()],
///     standardized_coefficients: vec![0.5, -0.3],
///     y_std: 1.2,
/// };
///
/// // X1 has a stronger positive effect than X2's negative effect
/// assert!(output.standardized_coefficients[0].abs() > output.standardized_coefficients[1].abs());
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StandardizedCoefficientsOutput {
    /// Names of predictor variables (excluding intercept)
    pub variable_names: Vec<String>,
    /// Standardized coefficients (one per predictor)
    pub standardized_coefficients: Vec<f64>,
    /// Standard deviation of the response variable Y
    pub y_std: f64,
}

impl StandardizedCoefficientsOutput {
    /// Returns the ranking of variables by absolute standardized coefficient value.
    ///
    /// # Returns
    ///
    /// A vector of (variable_name, absolute_value) tuples, sorted by importance
    /// (highest absolute coefficient first).
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::feature_importance::StandardizedCoefficientsOutput;
    /// let output = StandardizedCoefficientsOutput {
    ///     variable_names: vec!["X1".to_string(), "X2".to_string(), "X3".to_string()],
    ///     standardized_coefficients: vec![0.3, -0.8, 0.1],
    ///     y_std: 1.0,
    /// };
    ///
    /// let ranking = output.ranking();
    /// assert_eq!(ranking[0].0, "X2"); // Highest absolute value
    /// assert_eq!(ranking[2].0, "X3"); // Lowest absolute value
    /// ```
    pub fn ranking(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<_> = self
            .variable_names
            .iter()
            .zip(self.standardized_coefficients.iter())
            .map(|(name, &coef)| (name.clone(), coef.abs()))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

/// Output from VIF (Variance Inflation Factor) ranking.
///
/// VIF measures how much the variance of a coefficient is inflated due to
/// multicollinearity. Lower VIF = less redundant = more uniquely important.
///
/// # Interpretation
///
/// - VIF = 1: No correlation with other predictors
/// - VIF 1-5: Low multicollinearity (acceptable)
/// - VIF 5-10: Moderate multicollinearity (review)
/// - VIF > 10: High multicollinearity (problematic)
///
/// # Note
///
/// Unlike other importance metrics, **lower VIF is better**. This ranking
/// sorts by VIF ascending (least redundant first).
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::VifRankingOutput;
/// let output = VifRankingOutput {
///     variable_names: vec!["X1".to_string(), "X2".to_string()],
///     vif_values: vec![1.2, 8.5],
/// };
///
/// // X1 is less redundant (better) than X2
/// assert_eq!(output.ranking()[0].0, "X1");
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VifRankingOutput {
    /// Names of predictor variables
    pub variable_names: Vec<String>,
    /// VIF values for each predictor (lower = less redundant)
    pub vif_values: Vec<f64>,
}

impl VifRankingOutput {
    /// Returns the ranking of variables by VIF value (ascending).
    ///
    /// Lower VIF values indicate less multicollinearity and are ranked higher.
    ///
    /// # Returns
    ///
    /// A vector of (variable_name, vif_value) tuples, sorted by VIF ascending
    /// (lowest/most independent first).
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::feature_importance::VifRankingOutput;
    /// let output = VifRankingOutput {
    ///     variable_names: vec!["X1".to_string(), "X2".to_string(), "X3".to_string()],
    ///     vif_values: vec![1.2, 8.5, 2.3],
    /// };
    ///
    /// let ranking = output.ranking();
    /// assert_eq!(ranking[0].0, "X1"); // Lowest VIF (best)
    /// assert_eq!(ranking[2].0, "X2"); // Highest VIF (most redundant)
    /// ```
    pub fn ranking(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<_> = self
            .variable_names
            .iter()
            .zip(self.vif_values.iter())
            .map(|(name, &vif)| (name.clone(), vif))
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Creates VifRankingOutput from existing VifResult vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::core::VifResult;
    /// # use linreg_core::feature_importance::VifRankingOutput;
    /// let vif_results = vec![
    ///     VifResult {
    ///         variable: "X1".to_string(),
    ///         vif: 1.5,
    ///         rsquared: 0.33,
    ///         interpretation: "Low multicollinearity".to_string(),
    ///     },
    ///     VifResult {
    ///         variable: "X2".to_string(),
    ///         vif: 3.2,
    ///         rsquared: 0.69,
    ///         interpretation: "Low multicollinearity".to_string(),
    ///     },
    /// ];
    ///
    /// let output = VifRankingOutput::from_vif_results(&vif_results);
    /// assert_eq!(output.variable_names.len(), 2);
    /// ```
    pub fn from_vif_results(vif_results: &[VifResult]) -> Self {
        let variable_names = vif_results.iter().map(|v| v.variable.clone()).collect();
        let vif_values = vif_results.iter().map(|v| v.vif).collect();

        VifRankingOutput {
            variable_names,
            vif_values,
        }
    }
}

/// Output from SHAP (SHapley Additive exPlanations) calculation.
///
/// SHAP values decompose predictions into the contribution of each feature.
/// For linear models, this has a closed-form solution: SHAPᵢ = coefᵢ × (xᵢ - mean(xᵢ))
///
/// # Properties
///
/// - **Local accuracy**: Σ SHAPᵢ + base_value = prediction
/// - **Missingness**: Features not in model have SHAP = 0
/// - **Consistency**: Guaranteed by the exact linear solution
///
/// # Structure
///
/// - `shap_values[i][j]` = contribution of feature j to observation i's prediction
/// - `base_value` = mean prediction (intercept contribution)
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::ShapOutput;
/// let output = ShapOutput {
///     variable_names: vec!["X1".to_string(), "X2".to_string()],
///     shap_values: vec![
///         vec![0.5, -0.2],  // Observation 0
///         vec![0.3, 0.1],   // Observation 1
///     ],
///     base_value: 5.0,
///     mean_abs_shap: vec![0.4, 0.15],
/// };
///
/// // For observation 0: prediction = base_value + 0.5 + (-0.2) = 5.3
/// assert!((output.base_value + output.shap_values[0][0] + output.shap_values[0][1] - 5.3).abs() < 0.01);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapOutput {
    /// Names of predictor variables (excluding intercept)
    pub variable_names: Vec<String>,
    /// SHAP values matrix (n_observations × n_features)
    /// shap_values[i][j] = contribution of feature j to observation i
    pub shap_values: Vec<Vec<f64>>,
    /// Base value (mean prediction / intercept contribution)
    pub base_value: f64,
    /// Mean absolute SHAP values per feature (global importance)
    pub mean_abs_shap: Vec<f64>,
}

impl ShapOutput {
    /// Returns the ranking of variables by mean absolute SHAP value.
    ///
    /// Higher mean |SHAP| = more important on average.
    ///
    /// # Returns
    ///
    /// A vector of (variable_name, mean_abs_shap) tuples, sorted by importance
    /// (highest mean absolute SHAP first).
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::feature_importance::ShapOutput;
    /// let output = ShapOutput {
    ///     variable_names: vec!["X1".to_string(), "X2".to_string(), "X3".to_string()],
    ///     shap_values: vec![
    ///         vec![0.8, 0.1, -0.05],
    ///         vec![0.6, -0.2, 0.0],
    ///     ],
    ///     base_value: 5.0,
    ///     mean_abs_shap: vec![0.7, 0.15, 0.025],
    /// };
    ///
    /// let ranking = output.ranking();
    /// assert_eq!(ranking[0].0, "X1"); // Most important
    /// assert_eq!(ranking[2].0, "X3"); // Least important
    /// ```
    pub fn ranking(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<_> = self
            .variable_names
            .iter()
            .zip(self.mean_abs_shap.iter())
            .map(|(name, &shap)| (name.clone(), shap))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Gets the SHAP decomposition for a single observation.
    ///
    /// # Arguments
    ///
    /// * `observation_index` - Index of the observation (0-based)
    ///
    /// # Returns
    ///
    /// A vector of (variable_name, shap_value) tuples for that observation.
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::feature_importance::ShapOutput;
    /// let output = ShapOutput {
    ///     variable_names: vec!["X1".to_string(), "X2".to_string()],
    ///     shap_values: vec![vec![0.5, -0.2]],
    ///     base_value: 5.0,
    ///     mean_abs_shap: vec![0.5, 0.2],
    /// };
    ///
    /// let decomposition = output.observation_contribution(0);
    /// assert_eq!(decomposition[0].0, "X1");
    /// assert_eq!(decomposition[0].1, 0.5);
    /// ```
    pub fn observation_contribution(&self, observation_index: usize) -> Vec<(String, f64)> {
        if observation_index >= self.shap_values.len() {
            return vec![];
        }

        self.variable_names
            .iter()
            .zip(&self.shap_values[observation_index])
            .map(|(name, &shap)| (name.clone(), shap))
            .collect()
    }
}

/// Output from permutation importance calculation.
///
/// Permutation importance measures the decrease in model performance (typically R²)
/// when a single predictor's values are randomly shuffled.
///
/// # Interpretation
///
/// - Higher values = more important (shuffling causes larger performance drop)
/// - Values close to 0 = feature has no effect on predictions
/// - Negative values possible = feature happened to help by chance in this sample
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::PermutationImportanceOutput;
/// let output = PermutationImportanceOutput {
///     variable_names: vec!["X1".to_string(), "X2".to_string()],
///     importance: vec![0.15, 0.03],
///     baseline_score: 0.85,
///     n_permutations: 50,
///     seed: Some(42),
///     importance_std_err: None,
///     interval_lower: None,
///     interval_upper: None,
///     interval_confidence: None,
/// };
///
/// // X1 is more important (shuffling causes 15% R² drop)
/// assert!(output.importance[0] > output.importance[1]);
/// // Baseline R² was 0.85
/// assert_eq!(output.baseline_score, 0.85);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PermutationImportanceOutput {
    /// Names of predictor variables
    pub variable_names: Vec<String>,
    /// Importance scores (baseline_score - score_when_shuffled)
    pub importance: Vec<f64>,
    /// Baseline model performance (R² on original data)
    pub baseline_score: f64,
    /// Number of permutations performed
    pub n_permutations: usize,
    /// Random seed used (if provided)
    pub seed: Option<u64>,
    /// Standard error of importance scores (if intervals were computed)
    pub importance_std_err: Option<Vec<f64>>,
    /// Lower bounds of confidence intervals (if computed)
    pub interval_lower: Option<Vec<f64>>,
    /// Upper bounds of confidence intervals (if computed)
    pub interval_upper: Option<Vec<f64>>,
    /// Confidence level used for intervals (if computed)
    pub interval_confidence: Option<f64>,
}

impl PermutationImportanceOutput {
    /// Returns the ranking of variables by permutation importance.
    ///
    /// Higher importance = more important (larger performance drop when shuffled).
    ///
    /// # Returns
    ///
    /// A vector of (variable_name, importance) tuples, sorted by importance
    /// (most important first).
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::feature_importance::PermutationImportanceOutput;
    /// let output = PermutationImportanceOutput {
    ///     variable_names: vec!["X1".to_string(), "X2".to_string(), "X3".to_string()],
    ///     importance: vec![0.15, 0.03, 0.22],
    ///     baseline_score: 0.85,
    ///     n_permutations: 50,
    ///     seed: Some(42),
    ///     importance_std_err: None,
    ///     interval_lower: None,
    ///     interval_upper: None,
    ///     interval_confidence: None,
    /// };
    ///
    /// let ranking = output.ranking();
    /// assert_eq!(ranking[0].0, "X3"); // Most important
    /// assert_eq!(ranking[2].0, "X2"); // Least important
    /// ```
    pub fn ranking(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<_> = self
            .variable_names
            .iter()
            .zip(self.importance.iter())
            .map(|(name, &imp)| (name.clone(), imp))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

/// Options for permutation importance calculation.
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::PermutationImportanceOptions;
/// let options = PermutationImportanceOptions {
///     n_permutations: 100,
///     seed: Some(42),
///     compute_intervals: true,
///     interval_confidence: 0.95,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug)]
pub struct PermutationImportanceOptions {
    /// Number of permutation iterations per feature
    pub n_permutations: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to compute confidence intervals
    pub compute_intervals: bool,
    /// Confidence level for intervals (e.g., 0.95 for 95% CI)
    pub interval_confidence: f64,
}

impl Default for PermutationImportanceOptions {
    fn default() -> Self {
        PermutationImportanceOptions {
            n_permutations: 50,
            seed: None,
            compute_intervals: false,
            interval_confidence: 0.95,
        }
    }
}
