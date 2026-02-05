// ============================================================================
// Variance Inflation Factor (VIF) Diagnostic
// ============================================================================
//
// VIF measures how much the variance of a regression coefficient is inflated
// due to multicollinearity among the predictor variables.
//
// VIF = 1 / (1 - R²_j)
//
// where R²_j is the R-squared from regressing predictor j on all other predictors.
//
// Interpretation:
// - VIF = 1: No correlation with other predictors
// - VIF > 5: Moderate multicollinearity (concerning)
// - VIF > 10: High multicollinearity (severe)

use super::types::{VifDetail, VifDiagnosticResult};
use crate::core::calculate_vif;
use crate::error::{Error, Result};

/// Performs VIF analysis for multicollinearity detection.
///
/// This diagnostic computes the Variance Inflation Factor for each predictor variable.
/// High VIF values indicate that a predictor is highly correlated with other predictors,
/// which can make regression coefficients unstable and difficult to interpret.
///
/// # Arguments
///
/// * `y` - Dependent variable values (not directly used in VIF, but needed for validation)
/// * `x_vars` - Independent variables (each vec is a column)
///
/// # Returns
///
/// A [`VifDiagnosticResult`] containing:
/// - Maximum VIF across all predictors
/// - Detailed VIF results for each predictor
/// - Interpretation and guidance
///
/// # Errors
///
/// * [`Error::InsufficientData`] - if fewer than 2 predictor variables
///
/// # Interpretation Guidelines
///
/// | VIF Value | Interpretation |
/// |-----------|----------------|
/// | 1.0 | No correlation |
/// | 1-5 | Low correlation (acceptable) |
/// | 5-10 | Moderate correlation (concerning) |
/// | > 10 | High correlation (severe) |
///
/// # Example
///
/// ```
/// # use linreg_core::diagnostics::vif_test;
/// let y = vec![2.5, 3.7, 4.2, 5.1, 6.3, 7.0, 8.2, 9.1];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0, 2.0, 4.0, 5.0];
/// let x3 = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]; // Correlated with x1
///
/// let result = vif_test(&y, &[x1, x2, x3]).unwrap();
///
/// println!("Max VIF: {}", result.max_vif);
/// for detail in &result.vif_results {
///     println!("{}: VIF = {}", detail.variable, detail.vif);
/// }
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn vif_test(y: &[f64], x_vars: &[Vec<f64>]) -> Result<VifDiagnosticResult> {
    let n = y.len();
    let k = x_vars.len();

    // Need at least 2 predictors to compute VIF
    if k < 2 {
        return Err(Error::InsufficientData {
            required: 2,
            available: k,
        });
    }

    // Validate dimensions
    for (i, var) in x_vars.iter().enumerate() {
        if var.len() != n {
            return Err(Error::InvalidInput(format!(
                "x_vars[{}] has {} elements, expected {}",
                i,
                var.len(),
                n
            )));
        }
    }

    // Check for finite values
    for (i, var) in x_vars.iter().enumerate() {
        for (j, &val) in var.iter().enumerate() {
            if !val.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "x_vars[{}] contains non-finite value at index {}",
                    i, j
                )));
            }
        }
    }

    // Create variable names
    let names: Vec<String> = (0..=k)
        .map(|i| {
            if i == 0 {
                "Intercept".to_string()
            } else {
                format!("X{}", i)
            }
        })
        .collect();

    // Call existing VIF calculation from core.rs
    let vif_results = calculate_vif(x_vars, &names, n);

    // Convert to our VifDetail format
    let details: Vec<VifDetail> = vif_results
        .iter()
        .map(|v| VifDetail {
            variable: v.variable.clone(),
            vif: v.vif,
            rsquared: v.rsquared,
            interpretation: v.interpretation.clone(),
        })
        .collect();

    // Find maximum VIF (including infinite values for perfect multicollinearity)
    let max_vif = details
        .iter()
        .map(|d| d.vif)
        .fold(0.0_f64, |acc, v| {
            if v.is_infinite() && v > 0.0 {
                f64::INFINITY
            } else {
                acc.max(v)
            }
        });

    // Determine if there are concerning VIF values
    let high_vif_count = details.iter().filter(|d| d.vif > 10.0 || d.vif.is_infinite()).count();
    let moderate_vif_count = details.iter().filter(|d| d.vif > 5.0 && d.vif <= 10.0).count();

    let (interpretation, guidance) = if high_vif_count > 0 {
        (
            format!(
                "Found {} variable(s) with VIF > 10 (severe multicollinearity). Maximum VIF = {:.2}.",
                high_vif_count, max_vif
            ),
            "Consider removing or combining highly correlated predictors. High multicollinearity makes coefficient estimates unstable and difficult to interpret.",
        )
    } else if moderate_vif_count > 0 {
        (
            format!(
                "Found {} variable(s) with VIF > 5 (moderate multicollinearity). Maximum VIF = {:.2}.",
                moderate_vif_count, max_vif
            ),
            "Monitor these variables. Moderate multicollinearity may indicate redundant predictors. Consider dimensionality reduction if interpretation becomes problematic.",
        )
    } else {
        (
            format!(
                "All VIF values are within acceptable range (VIF ≤ 5). Maximum VIF = {:.2}.",
                max_vif
            ),
            "No concerning multicollinearity detected. Coefficient estimates should be stable.",
        )
    };

    Ok(VifDiagnosticResult {
        test_name: "Variance Inflation Factor (VIF)".to_string(),
        max_vif,
        vif_results: details,
        interpretation,
        guidance: guidance.to_string(),
    })
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vif_low_correlation() {
        let y = vec![2.5, 3.7, 4.2, 5.1, 6.3, 7.0, 8.2, 9.1, 10.5, 11.2];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0]; // Not well correlated with x1

        let result = vif_test(&y, &[x1, x2]).unwrap();

        assert_eq!(result.test_name, "Variance Inflation Factor (VIF)");
        assert!(result.max_vif < 5.0, "Max VIF should be low for uncorrelated predictors");
        assert_eq!(result.vif_results.len(), 2);
        assert!(result.passed_interpretation());
    }

    #[test]
    fn test_vif_high_correlation() {
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]; // Perfectly correlated with x1

        let result = vif_test(&y, &[x1, x2]).unwrap();

        assert!(result.max_vif > 10.0, "VIF should be high for perfectly correlated predictors");
        assert!(!result.passed_interpretation());
    }

    #[test]
    fn test_vif_insufficient_predictors() {
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];

        let result = vif_test(&y, &[x1]);

        assert!(result.is_err());
    }

    #[test]
    fn test_vif_mismatched_dimensions() {
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let x2 = vec![1.0, 2.0, 3.0]; // Wrong length

        let result = vif_test(&y, &[x1, x2]);

        assert!(result.is_err());
    }

    #[test]
    fn test_vif_detail_structure() {
        let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let result = vif_test(&y, &[x1, x2]).unwrap();

        assert_eq!(result.vif_results.len(), 2);
        for detail in &result.vif_results {
            assert!(!detail.variable.is_empty());
            assert!(detail.vif >= 1.0);
            assert!(detail.rsquared >= 0.0 && detail.rsquared <= 1.0);
            assert!(!detail.interpretation.is_empty());
        }
    }
}

// Helper method for tests
impl VifDiagnosticResult {
    #[allow(dead_code)]
    fn passed_interpretation(&self) -> bool {
        self.max_vif < 10.0 && self.guidance.contains("No concerning multicollinearity")
    }
}
