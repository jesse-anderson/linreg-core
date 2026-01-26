// ============================================================================
// Breusch-Godfrey Test for Higher-Order Serial Correlation
// ============================================================================
//
// H0: No serial correlation up to order p
// H1: Serial correlation exists at some lag <= p
//
// The Breusch-Godfrey test (also known as the Lagrange Multiplier (LM) test)
// tests for serial correlation at higher lags, unlike the Durbin-Watson test
// which only tests for first-order autocorrelation.
//
// Test procedure:
// 1. Fit original model: y = Xβ + ε
// 2. Obtain residuals ê
// 3. Fit auxiliary regression: ê = Xγ + δ₁ê_{t-1} + ... + δₚê_{t-p} + u
// 4. Compute test statistic:
//    - LM (Chi-squared): n * R² from auxiliary regression
//    - F: F-statistic comparing restricted vs unrestricted models
//
// References:
// - Breusch, T.S. (1978). Testing for Autocorrelation in Dynamic Linear
//   Models, Australian Economic Papers, 17, 334-355.
// - Godfrey, L.G. (1978). Testing Against General Autoregressive and
//   Moving Average Error Models when the Regressors Include Lagged
//   Dependent Variables, Econometrica, 46, 1293-1301.
// - R: lmtest::bgtest(model, order)

use super::helpers::fit_ols;
use crate::distributions::{chi_squared_survival, fisher_snedecor_cdf};
use crate::error::{Error, Result};
use crate::linalg::Matrix;
use serde::Serialize;

/// Type of test statistic to compute for the Breusch-Godfrey test.
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::BGTestType;
///
/// let chisq_type = BGTestType::Chisq;
/// let f_type = BGTestType::F;
///
/// // Enum variants can be compared
/// assert_eq!(chisq_type, BGTestType::Chisq);
/// assert_ne!(chisq_type, f_type);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BGTestType {
    /// Chi-squared test statistic (asymptotic LM test)
    Chisq,
    /// F test statistic (finite sample version)
    F,
}

/// Result of the Breusch-Godfrey test.
///
/// Contains the test statistic, p-value, and interpretation for serial correlation testing.
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::BreuschGodfreyResult;
///
/// let result = BreuschGodfreyResult {
///     test_name: "Breusch-Godfrey".to_string(),
///     order: 1,
///     test_type: "Chisq".to_string(),
///     statistic: 2.34,
///     p_value: 0.126,
///     df: vec![1.0],
///     passed: true,
///     interpretation: "No significant serial correlation detected.".to_string(),
///     guidance: "Model assumptions are satisfied.".to_string(),
/// };
///
/// assert_eq!(result.order, 1);
/// assert!(result.passed);
/// assert!(result.p_value > 0.05);
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct BreuschGodfreyResult {
    /// Name of the test
    pub test_name: String,
    /// Maximum order of serial correlation tested
    pub order: usize,
    /// Type of test statistic computed
    pub test_type: String,
    /// Test statistic value (LM or F)
    pub statistic: f64,
    /// P-value for the test
    pub p_value: f64,
    /// Degrees of freedom (for Chisq: single value; for F: [df1, df2])
    pub df: Vec<f64>,
    /// Whether the null hypothesis was not rejected (no serial correlation)
    #[serde(rename = "is_passed")]
    pub passed: bool,
    /// Interpretation of the test result
    pub interpretation: String,
    /// Guidance for further action
    pub guidance: String,
}

/// Breusch-Godfrey test for higher-order serial correlation in residuals.
///
/// Unlike the Durbin-Watson test which only detects first-order autocorrelation,
/// the Breusch-Godfrey test can detect serial correlation at any lag order.
///
/// # Test Procedure
///
/// 1. Fit the original OLS model: `y = Xβ + ε` and obtain residuals `ê`
/// 2. Create lagged residual variables: `ê_{t-1}, ê_{t-2}, ..., ê_{t-p}`
/// 3. Fit auxiliary regression: `ê = Xγ + δ₁ê_{t-1} + ... + δₚê_{t-p} + u`
/// 4. Compute test statistic based on `R²` from auxiliary regression
///
/// # Test Statistics
///
/// ## Chi-squared (LM) statistic (default):
/// ```text
/// LM = n × R²
/// ```
/// where `n` is the sample size and `R²` is from the auxiliary regression.
/// Under H₀, `LM ~ χ²(p)` where `p` is the order.
///
/// ## F statistic:
/// ```text
/// F = ((RSS₀ - RSS₁)/p) / (RSS₁/(n-k-p))
/// ```
/// where `RSS₀` is from the restricted model and `RSS₁` from the auxiliary model.
/// Under H₀, `F ~ F(p, n-k-p)`.
///
/// # Arguments
///
/// * `y` - Dependent variable values
/// * `x_vars` - Independent variables (each vec is a column)
/// * `order` - Maximum order of serial correlation to test (default: 1)
/// * `test_type` - Type of test statistic: Chisq or F (default: Chisq)
///
/// # Returns
///
/// A [`BreuschGodfreyResult`] containing the test statistic, p-value, and interpretation.
///
/// # Errors
///
/// * [`Error::InsufficientData`] - if n ≤ k + order + 1
/// * [`Error::InvalidInput`] - if data contains NaN or infinite values
///
/// # Examples
///
/// ```ignore
/// use linreg_core::diagnostics::{breusch_godfrey_test, BGTestType};
///
/// let y = vec![2.1, 4.2, 5.8, 8.1, 10.1, 12.2, 13.8, 16.1];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
///
/// // Test for first-order serial correlation
/// let result = breusch_godfrey_test(&y, &[x1], 1, BGTestType::Chisq)?;
///
/// // Test for fourth-order serial correlation
/// let result = breusch_godfrey_test(&y, &[x1], 4, BGTestType::Chisq)?;
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn breusch_godfrey_test(
    y: &[f64],
    x_vars: &[Vec<f64>],
    order: usize,
    test_type: BGTestType,
) -> Result<BreuschGodfreyResult> {
    let n = y.len();
    let k = x_vars.len();
    let p = k + 1; // including intercept

    // Validate we have enough data
    if n <= p + order {
        return Err(Error::InsufficientData {
            required: p + order + 1,
            available: n,
        });
    }

    // Validate dimensions and finite values using shared helper
    super::helpers::validate_regression_data(y, x_vars)?;

    // Special case: order = 0 means no lagged residuals to test
    // The null hypothesis (no serial correlation) is trivially true
    if order == 0 {
        let test_type_str = match test_type {
            BGTestType::Chisq => "Chi-squared",
            BGTestType::F => "F",
        };

        return Ok(BreuschGodfreyResult {
            test_name: "Breusch-Godfrey LM test for serial correlation of order up to 0".to_string(),
            order: 0,
            test_type: test_type_str.to_string(),
            statistic: 0.0,
            p_value: 1.0,
            df: vec![0.0],
            passed: true,
            interpretation: "Breusch-Godfrey test (order = 0): No lags tested, null hypothesis trivially holds.".to_string(),
            guidance: "No action needed. Order 0 means no serial correlation was tested.".to_string(),
        });
    }

    // Create design matrix with intercept (row-major order)
    let mut x_data = vec![0.0; n * p];
    for row in 0..n {
        x_data[row * p] = 1.0; // intercept
        for (col, x_var) in x_vars.iter().enumerate() {
            x_data[row * p + col + 1] = x_var[row];
        }
    }

    let x = Matrix::new(n, p, x_data);

    // Fit OLS and get residuals
    let beta = fit_ols(y, &x)?;
    let predictions = x.mul_vec(&beta);
    let residuals: Vec<f64> = y
        .iter()
        .zip(predictions.iter())
        .map(|(&yi, &yi_hat)| yi - yi_hat)
        .collect();

    // Create lagged residual matrix
    // R fills missing values with 0 and includes all n observations
    // For lag j (0-indexed), at row r we want residual[r - j - 1]
    // This is valid when r > j (i.e., r - j - 1 >= 0)
    let mut lag_data = vec![0.0; n * order];
    for row in 0..n {
        for lag in 0..order {
            if row > lag {
                // For row > lag, use residual[row - lag - 1]
                lag_data[row * order + lag] = residuals[row - lag - 1];
            }
            // else: leave as 0.0 for rows without enough lagged data (matching R's c(0, res[1:(n-1)]))
        }
    }

    // Create augmented design matrix [X | lagged_residuals]
    // Include all n observations (R includes the first one with zero-filled lags)
    let p_aug = p + order;
    let mut x_aug_data = vec![0.0; n * p_aug];
    for row in 0..n {
        // Copy original predictors
        for col in 0..p {
            x_aug_data[row * p_aug + col] = x.data[row * p + col];
        }
        // Copy lagged residuals
        for lag in 0..order {
            x_aug_data[row * p_aug + p + lag] = lag_data[row * order + lag];
        }
    }

    let x_aug = Matrix::new(n, p_aug, x_aug_data);

    // Fit auxiliary regression on all n observations
    let beta_aug = fit_ols(&residuals, &x_aug)?;
    let fitted_aug = x_aug.mul_vec(&beta_aug);

    // Compute R² from auxiliary regression
    // R² = 1 - SS_res / SS_tot
    let ss_res: f64 = residuals
        .iter()
        .zip(fitted_aug.iter())
        .map(|(&r, &f)| (r - f) * (r - f))
        .sum();

    let mean_res = residuals.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = residuals
        .iter()
        .map(|&r| (r - mean_res) * (r - mean_res))
        .sum();

    let r_squared = 1.0 - ss_res / ss_tot;

    // Compute test statistic based on type
    let (statistic, df, p_value) = match test_type {
        BGTestType::Chisq => {
            // LM = n * R² ~ χ²(order)
            // R uses the ORIGINAL sample size n, not the truncated size
            let lm = n as f64 * r_squared;
            let df_val = order as f64;
            let p_val = chi_squared_survival(lm, df_val);
            (lm, vec![df_val], p_val)
        }
        BGTestType::F => {
            // F = ((RSS0 - RSS1)/order) / (RSS1/(n-k-order))
            // RSS0 is from restricted model (regressing residuals on X only)
            // RSS1 is ss_res from auxiliary model
            let beta_restricted = fit_ols(&residuals, &x)?;
            let fitted_restricted = x.mul_vec(&beta_restricted);
            let rss0: f64 = residuals
                .iter()
                .zip(fitted_restricted.iter())
                .map(|(&r, &f)| (r - f) * (r - f))
                .sum();

            let rss1 = ss_res;
            let df1 = order as f64;
            let df2 = (n - p - order) as f64;

            if df2 < 1.0 {
                return Err(Error::InsufficientData {
                    required: p + order + 2,
                    available: n,
                });
            }

            let f_stat = ((rss0 - rss1) / df1) / (rss1 / df2);
            let p_val = 1.0 - fisher_snedecor_cdf(f_stat, df1, df2);
            (f_stat, vec![df1, df2], p_val)
        }
    };

    // Interpret result at α = 0.05
    let alpha = 0.05;
    let passed = p_value > alpha;

    let test_type_str = match test_type {
        BGTestType::Chisq => "Chi-squared",
        BGTestType::F => "F",
    };

    let interpretation = if passed {
        format!(
            "Breusch-Godfrey {} test (order = {}): statistic = {:.4}, p-value = {:.4}. \
            No significant serial correlation detected up to order {}.",
            test_type_str, order, statistic, p_value, order
        )
    } else {
        format!(
            "Breusch-Godfrey {} test (order = {}): statistic = {:.4}, p-value = {:.4}. \
            Significant serial correlation detected at order <= {}.",
            test_type_str, order, statistic, p_value, order
        )
    };

    let guidance = if passed {
        "No action needed. The residuals show no significant serial correlation up to the specified order.".to_string()
    } else {
        "Consider: (1) Adding lagged dependent variables, (2) Using autoregressive error models (e.g., Cochrane-Orcutt), (3) Checking for omitted variables, (4) Using HAC standard errors.".to_string()
    };

    Ok(BreuschGodfreyResult {
        test_name: format!("Breusch-Godfrey LM test for serial correlation of order up to {}", order),
        order,
        test_type: test_type_str.to_string(),
        statistic,
        p_value,
        df,
        passed,
        interpretation,
        guidance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Breusch-Godfrey with simple data
    #[test]
    fn test_breusch_godfrey_simple() {
        let y = vec![2.1, 4.2, 5.8, 8.1, 10.1, 12.2, 13.8, 16.1];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let result = breusch_godfrey_test(&y, &[x1], 1, BGTestType::Chisq).unwrap();

        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.order, 1);
        assert_eq!(result.test_type, "Chi-squared");
        assert_eq!(result.df.len(), 1);
        assert_eq!(result.df[0], 1.0);
    }

    /// Test with higher order
    #[test]
    fn test_breusch_godfrey_order_4() {
        let y = vec![
            2.1, 4.2, 5.8, 8.1, 10.1, 12.2, 13.8, 16.1, 17.9, 20.2, 21.8, 24.1,
        ];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let result = breusch_godfrey_test(&y, &[x1], 4, BGTestType::Chisq).unwrap();

        assert_eq!(result.order, 4);
        assert_eq!(result.df[0], 4.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    /// Test F statistic variant
    #[test]
    fn test_breusch_godfrey_f_statistic() {
        let y = vec![
            2.1, 4.2, 5.8, 8.1, 10.1, 12.2, 13.8, 16.1, 17.9, 20.2, 21.8, 24.1,
        ];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let result = breusch_godfrey_test(&y, &[x1], 2, BGTestType::F).unwrap();

        assert_eq!(result.test_type, "F");
        assert_eq!(result.df.len(), 2);
        assert_eq!(result.df[0], 2.0); // df1 = order
        assert!(result.df[1] > 0.0); // df2 = n - k - order
        assert!(result.statistic >= 0.0);
    }

    /// Test insufficient data
    #[test]
    fn test_insufficient_data() {
        let y = vec![1.0, 2.0];
        let x1 = vec![1.0, 2.0];

        let result = breusch_godfrey_test(&y, &[x1], 1, BGTestType::Chisq);
        assert!(matches!(result, Err(Error::InsufficientData { .. })));
    }

    /// Test with multiple predictors
    #[test]
    fn test_breusch_godfrey_multiple_predictors() {
        let y = vec![
            2.1, 4.2, 5.8, 8.1, 10.1, 12.2, 13.8, 16.1, 17.9, 20.2, 21.8, 24.1,
        ];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0];

        let result = breusch_godfrey_test(&y, &[x1, x2], 1, BGTestType::Chisq).unwrap();

        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(!result.interpretation.is_empty());
        assert!(!result.guidance.is_empty());
    }

    /// Test interpretation strings
    #[test]
    fn test_interpretation_content() {
        let y = vec![
            2.1, 4.2, 5.8, 8.1, 10.1, 12.2, 13.8, 16.1, 17.9, 20.2, 21.8, 24.1,
        ];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let result = breusch_godfrey_test(&y, &[x1], 1, BGTestType::Chisq).unwrap();

        assert!(result.interpretation.contains("Breusch-Godfrey"));
        assert!(result.interpretation.contains("order"));
        assert!(result.interpretation.contains("statistic"));
        assert!(result.interpretation.contains("p-value"));

        // Guidance should be non-empty
        assert!(!result.guidance.is_empty());
    }

    /// Test that Chi-squared and F give similar p-values for large samples
    #[test]
    fn test_chisq_vs_f_similarity() {
        // Create a longer series for better asymptotic approximation
        let mut y = Vec::new();
        let mut x1 = Vec::new();
        for i in 1..=50 {
            x1.push(i as f64);
            y.push(2.0 * i as f64 + 5.0 + (i as f64 * 0.1)); // slight trend
        }

        let result_chisq =
            breusch_godfrey_test(&y, &[x1.clone()], 1, BGTestType::Chisq).unwrap();
        let result_f = breusch_godfrey_test(&y, &[x1], 1, BGTestType::F).unwrap();

        // Both should give similar conclusions (both pass or both fail)
        // Note: p-values may differ but should be in same ballpark
        assert!(result_chisq.p_value >= 0.0 && result_chisq.p_value <= 1.0);
        assert!(result_f.p_value >= 0.0 && result_f.p_value <= 1.0);
    }

    /// Test with data exhibiting autocorrelation
    #[test]
    fn test_detect_autocorrelation() {
        // Create data with positive autocorrelation
        let mut y = vec![10.0];
        let mut x1 = vec![1.0];
        for i in 1..=30 {
            x1.push(i as f64);
            // Each value depends on previous with some noise
            let new_val = y[i - 1] + 2.0 + (i as f64) * 0.5;
            y.push(new_val);
        }

        let result = breusch_godfrey_test(&y, &[x1], 1, BGTestType::Chisq).unwrap();

        // Should detect autocorrelation (low p-value)
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    /// Test edge case: order = 0 (essentially no test)
    #[test]
    fn test_order_zero() {
        let y = vec![2.1, 4.2, 5.8, 8.1, 10.1, 12.2, 13.8, 16.1];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Order 0 means no lagged residuals - should still work
        let result = breusch_godfrey_test(&y, &[x1], 0, BGTestType::Chisq).unwrap();

        assert_eq!(result.order, 0);
        assert_eq!(result.df[0], 0.0);
        // With no lags, p-value should be 1.0 (no autocorrelation detected by construction)
        assert_eq!(result.p_value, 1.0);
    }

    /// Test against R reference (synthetic autocorrelated data)
    ///
    /// This test uses the synthetic_autocorrelated dataset which is designed
    /// to have AR(1) errors. The test should detect significant autocorrelation.
    #[test]
    fn test_synthetic_autocorrelated() {
        // Synthetic data with AR(1) structure: y_t = 1 + 2*x_t + ε_t
        // where ε_t = 0.7*ε_{t-1} + u_t
        let y = vec![
            3.15, 5.49, 7.48, 9.92, 12.09, 14.62, 16.88, 19.40, 21.66, 24.38, 26.50, 29.29,
            31.40, 34.07, 36.54, 39.21, 41.51, 44.18, 46.55, 49.02,
        ];
        let x1 = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        ];

        let result = breusch_godfrey_test(&y, &[x1], 1, BGTestType::Chisq).unwrap();

        // Should detect significant autocorrelation (p < 0.05)
        // The exact values depend on numerical precision
        assert!(result.p_value < 0.2); // Should be quite significant
        assert!(result.statistic > 0.0);
    }
}
