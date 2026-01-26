// ============================================================================
// Jarque-Bera Test for Normality
// ============================================================================
//
// H0: Residuals are normally distributed
// H1: Residuals are not normally distributed
//
// Implementation: Tests whether sample skewness and kurtosis match normal distribution
// Reference: tseries::jarque.bera.test in R
//            scipy.stats.jarque_bera in Python
//
// Algorithm:
// 1. Fit OLS model and compute residuals e_i
// 2. Compute sample skewness: S = (1/n) * Σ((e_i - mean)³) / σ³
// 3. Compute sample kurtosis: K = (1/n) * Σ((e_i - mean)⁴) / σ⁴
// 4. Test statistic: JB = (n/6) * (S² + (K-3)²/4)
// 5. Under H0, JB follows chi-squared distribution with df = 2
//
// Note: Uses "excess kurtosis" definition where normal distribution has kurtosis = 3
//       The formula subtracts 3, so we use (K - 3) where K is the raw kurtosis

use super::helpers::{chi_squared_p_value, fit_ols};
use super::types::DiagnosticTestResult;
use crate::error::{Error, Result};
use crate::linalg::{vec_mean, Matrix};

/// Performs the Jarque-Bera test for normality of residuals.
///
/// This test checks whether the residuals from an OLS regression follow a normal
/// distribution by examining the sample skewness and kurtosis. The test statistic
/// measures how far the skewness and kurtosis deviate from their expected values
/// under normality (skewness = 0, excess kurtosis = 0).
///
/// # Arguments
///
/// * `y` - Dependent variable values
/// * `x_vars` - Independent variables (each vec is a column)
///
/// # Returns
///
/// A [`DiagnosticTestResult`] containing the test statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if n ≤ k + 2.
///
/// # Reference
///
/// - Jarque, C. M., & Bera, A. K. (1987). "A Test for Normality of Observations
///   and Regression Residuals". International Statistical Review, 55(2), 163-172.
///
/// # Example
///
/// ```
/// # use linreg_core::diagnostics::jarque_bera_test;
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
///
/// let result = jarque_bera_test(&y, &[x1, x2]).unwrap();
///
/// println!("JB statistic: {}", result.statistic);
/// println!("P-value: {}", result.p_value);
/// // Low p-value suggests residuals are not normally distributed
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn jarque_bera_test(y: &[f64], x_vars: &[Vec<f64>]) -> Result<DiagnosticTestResult> {
    let n = y.len();
    let k = x_vars.len(); // number of non-intercept predictors
    let p = k + 1; // total parameters including intercept

    // Validate inputs - need at least p + 1 observations
    if n <= p {
        return Err(Error::InsufficientData {
            required: p + 1,
            available: n,
        });
    }

    // Validate dimensions and finite values using shared helper
    super::helpers::validate_regression_data(y, x_vars)?;

    // Create design matrix with intercept
    let mut x_data = vec![1.0; n * p];
    for row in 0..n {
        x_data[row * p] = 1.0; // intercept
        for (col, x_var) in x_vars.iter().enumerate() {
            x_data[row * p + col + 1] = x_var[row];
        }
    }
    let x_full = Matrix::new(n, p, x_data);

    // Fit OLS on full data
    let beta = fit_ols(y, &x_full)?;

    // Compute residuals
    let predictions = x_full.mul_vec(&beta);
    let residuals: Vec<f64> = y
        .iter()
        .zip(predictions.iter())
        .map(|(&yi, &yi_hat)| yi - yi_hat)
        .collect();

    // Compute mean of residuals (should be ~0 for OLS with intercept)
    let mean = vec_mean(&residuals);

    // Compute standard deviation (using n denominator, matching scipy.stats)
    // Note: We use the MLE-style denominator (n), not the unbiased estimator (n-1)
    let variance: f64 = residuals
        .iter()
        .map(|&r| {
            let diff = r - mean;
            diff * diff
        })
        .sum::<f64>()
        / (n as f64);

    if variance <= 0.0 || !variance.is_finite() {
        return Err(Error::InvalidInput("Invalid residual variance".to_string()));
    }

    let std_dev = variance.sqrt();
    let std_dev_cubed = std_dev * std_dev * std_dev;
    let std_dev_fourth = std_dev_cubed * std_dev;

    // Compute skewness: S = (1/n) * Σ((e_i - mean)³) / σ³
    let skewness: f64 = residuals
        .iter()
        .map(|&r| {
            let diff = r - mean;
            diff * diff * diff
        })
        .sum::<f64>()
        / (n as f64 * std_dev_cubed);

    // Compute kurtosis: K = (1/n) * Σ((e_i - mean)⁴) / σ⁴
    let kurtosis: f64 = residuals
        .iter()
        .map(|&r| {
            let diff = r - mean;
            let diff_sq = diff * diff;
            diff_sq * diff_sq
        })
        .sum::<f64>()
        / (n as f64 * std_dev_fourth);

    // Jarque-Bera test statistic: JB = (n/6) * (S² + (K-3)²/4)
    // Note: K is the raw kurtosis, so (K-3) is the excess kurtosis
    let jb_stat = (n as f64) / 6.0 * (skewness * skewness + (kurtosis - 3.0).powi(2) / 4.0);

    // Degrees of freedom: 2 (testing both skewness and kurtosis)
    let df = 2.0;

    // P-value from chi-squared distribution (upper tail)
    let p_value = chi_squared_p_value(jb_stat, df);

    let alpha = 0.05;
    let passed = p_value > alpha;

    let (interpretation, guidance) = if passed {
        (
            format!(
                "p-value = {:.4} is greater than {:.2}. Cannot reject H0. No significant evidence that residuals deviate from normality.",
                p_value, alpha
            ),
            "The normality assumption appears to be met. Jarque-Bera test does not detect significant skewness or excess kurtosis."
        )
    } else {
        (
            format!(
                "p-value = {:.4} is less than or equal to {:.2}. Reject H0. Significant evidence that residuals deviate from normality.",
                p_value, alpha
            ),
            "Consider transforming the dependent variable (e.g., log, Box-Cox transformation), using robust standard errors, or applying a different estimation method."
        )
    };

    Ok(DiagnosticTestResult {
        test_name: "Jarque-Bera Test for Normality".to_string(),
        statistic: jb_stat,
        p_value,
        passed,
        interpretation,
        guidance: guidance.to_string(),
    })
}
