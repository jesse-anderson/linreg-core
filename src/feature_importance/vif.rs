//! VIF (Variance Inflation Factor) ranking for feature importance.
//!
//! VIF measures how much the variance of a coefficient is inflated due to
//! multicollinearity. Lower VIF indicates the feature is less redundant
//! with other predictors.
//!
//! # Interpretation
//!
//! - VIF = 1: No correlation with other predictors (ideal)
//! - VIF 1-5: Low multicollinearity (acceptable)
//! - VIF 5-10: Moderate multicollinearity (review)
//! - VIF > 10: High multicollinearity (problematic)
//!
//! # Note
//!
//! Unlike other importance metrics, **lower VIF is better** when assessing
//! uniqueness of information. Features with low VIF provide independent
//! information not captured by other predictors.

use crate::core::VifResult;
use crate::feature_importance::types::VifRankingOutput;

/// Computes VIF-based ranking of predictor variables.
///
/// VIF (Variance Inflation Factor) measures how much the variance of a
/// coefficient is inflated due to multicollinearity. Lower VIF values
/// indicate less redundant information.
///
/// # Arguments
///
/// * `vif_results` - VIF results from [`crate::core::calculate_vif`] or
///   [`crate::core::RegressionOutput::vif`]
///
/// # Returns
///
/// A [`VifRankingOutput`] containing variable names, VIF values, and
/// a ranking method that sorts by VIF ascending (least redundant first).
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::vif_ranking;
/// # use linreg_core::core::VifResult;
/// let vif_results = vec![
///     VifResult {
///         variable: "X1".to_string(),
///         vif: 1.2,
///         rsquared: 0.17,
///         interpretation: "Low multicollinearity".to_string(),
///     },
///     VifResult {
///         variable: "X2".to_string(),
///         vif: 8.5,
///         rsquared: 0.88,
///         interpretation: "High multicollinearity".to_string(),
///     },
/// ];
///
/// let ranking = vif_ranking(&vif_results);
///
/// // X1 is ranked higher (lower VIF = less redundant)
/// assert_eq!(ranking.ranking()[0].0, "X1");
/// ```
pub fn vif_ranking(vif_results: &[VifResult]) -> VifRankingOutput {
    VifRankingOutput::from_vif_results(vif_results)
}

/// Computes VIF-based ranking with importance scores.
///
/// This function converts VIF values to "importance" scores where
/// higher values indicate more unique information (inverse of VIF).
///
/// # Arguments
///
/// * `vif_results` - VIF results from regression output
///
/// # Returns
///
/// A tuple of (variable_names, importance_scores) where importance is
/// computed as `1 / VIF`. Higher values = more unique information.
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::vif::vif_importance_scores;
/// # use linreg_core::core::VifResult;
/// let vif_results = vec![
///     VifResult {
///         variable: "X1".to_string(),
///         vif: 1.0,
///         rsquared: 0.0,
///         interpretation: "No correlation".to_string(),
///     },
///     VifResult {
///         variable: "X2".to_string(),
///         vif: 5.0,
///         rsquared: 0.8,
///         interpretation: "Moderate multicollinearity".to_string(),
///     },
/// ];
///
/// let (names, scores) = vif_importance_scores(&vif_results);
///
/// // X1 has higher importance score (lower VIF)
/// assert!(scores[0] > scores[1]);
/// assert_eq!(names[0], "X1");
/// ```
pub fn vif_importance_scores(vif_results: &[VifResult]) -> (Vec<String>, Vec<f64>) {
    let mut names = Vec::with_capacity(vif_results.len());
    let mut scores = Vec::with_capacity(vif_results.len());

    for vif in vif_results {
        names.push(vif.variable.clone());
        // Importance = 1/VIF (higher = less redundant)
        // For infinite VIF, use 0 as importance
        let importance = if vif.vif.is_finite() && vif.vif > 0.0 {
            1.0 / vif.vif
        } else {
            0.0
        };
        scores.push(importance);
    }

    (names, scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_vif_result(name: &str, vif: f64) -> VifResult {
        VifResult {
            variable: name.to_string(),
            vif,
            rsquared: if vif.is_finite() { 1.0 - 1.0 / vif } else { 1.0 },
            interpretation: "Test".to_string(),
        }
    }

    #[test]
    fn test_vif_ranking_basic() {
        let vif_results = vec![
            create_vif_result("X1", 2.5),
            create_vif_result("X2", 8.0),
            create_vif_result("X3", 1.2),
        ];

        let ranking = vif_ranking(&vif_results);

        assert_eq!(ranking.variable_names, vec!["X1", "X2", "X3"]);
        assert_eq!(ranking.vif_values, vec![2.5, 8.0, 1.2]);

        let ranked = ranking.ranking();
        // Should be sorted by VIF ascending
        assert_eq!(ranked[0].0, "X3"); // VIF=1.2 (lowest)
        assert_eq!(ranked[1].0, "X1"); // VIF=2.5
        assert_eq!(ranked[2].0, "X2"); // VIF=8.0 (highest)
    }

    #[test]
    fn test_vif_importance_scores() {
        let vif_results = vec![
            create_vif_result("X1", 1.0),
            create_vif_result("X2", 5.0),
            create_vif_result("X3", 10.0),
        ];

        let (names, scores) = vif_importance_scores(&vif_results);

        assert_eq!(names, vec!["X1", "X2", "X3"]);
        assert_eq!(scores[0], 1.0);  // 1/1.0
        assert_eq!(scores[1], 0.2);  // 1/5.0
        assert_eq!(scores[2], 0.1);  // 1/10.0
    }

    #[test]
    fn test_vif_ranking_with_high_vif() {
        let vif_results = vec![
            create_vif_result("X1", 1.5),
            create_vif_result("X2", 15.0), // High VIF
            create_vif_result("X3", 3.0),
        ];

        let ranking = vif_ranking(&vif_results);
        let ranked = ranking.ranking();

        // X1 should be ranked highest (lowest VIF)
        assert_eq!(ranked[0], ("X1".to_string(), 1.5));
        // X2 should be ranked lowest (highest VIF)
        assert_eq!(ranked[2], ("X2".to_string(), 15.0));
    }

    #[test]
    fn test_vif_empty() {
        let vif_results = vec![];
        let ranking = vif_ranking(&vif_results);

        assert!(ranking.variable_names.is_empty());
        assert!(ranking.vif_values.is_empty());
        assert!(ranking.ranking().is_empty());
    }
}
