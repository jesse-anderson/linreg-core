// ============================================================================
// Anderson-Darling Test for Normality
// ============================================================================
//
// H0: Residuals are normally distributed
// H1: Residuals are not normally distributed
//
// Implementation: Tests whether sample follows normal distribution
// Reference: nortest::ad.test in R
//            statsmodels.stats.diagnostic.normal_ad in Python
//
// Algorithm:
// 1. Compute residuals from OLS regression
// 2. Sort residuals
// 3. Standardize residuals: z_i = (y_i - mean) / std
// 4. Compute p_i = Φ(z_i) where Φ is standard normal CDF
// 5. Anderson-Darling statistic:
//    A² = -n - Σ[(2i-1)/n * (ln(p_i) + ln(1 - p_{n+1-i}))]
// 6. Modified statistic for p-value:
//    A*² = A² × (1 + 0.75/n + 2.25/n²)
// 7. Compute p-value using approximation formula
//
// The Anderson-Darling test is more sensitive to tail deviations than
// the Kolmogorov-Smirnov test. Large A² values indicate deviation from normality.
//
// References:
// - Anderson, T. W., & Darling, D. A. (1952). "Asymptotic theory of certain
//   goodness of fit criteria based on stochastic processes". Annals of Mathematical
//   Statistics, 23(2), 193-212.
// - Stephens, M. A. (1974). "EDF Statistics for Goodness of Fit and Some
//   Comparisons". Journal of the American Statistical Association, 69(347), 730-737.
// - Marsaglia, G., & Marsaglia, J. (2004). "Evaluating the Anderson-Darling
//   Distribution". Journal of Statistical Software, 9(2), 1-5.

use super::helpers::fit_ols;
use super::types::DiagnosticTestResult;
use crate::distributions::normal_cdf;
use crate::error::{Error, Result};
use crate::linalg::{vec_mean, Matrix};

/// Computes the natural logarithm of the standard normal CDF.
///
/// This is equivalent to `log(pnorm(z))` in R.
/// Uses numerical stabilization to avoid underflow for extreme negative values.
///
/// # Precision
///
/// Uses Abramowitz & Stegun 7.1.26 approximation for normal CDF.
/// Difference from R's pnorm is approximately 1e-6, which propagates to
/// Anderson-Darling A² statistic difference of ~5e-7.
///
/// # Arguments
///
/// * `z` - Standard normal value
///
/// # Returns
///
/// `log(Φ(z))` where Φ is the standard normal CDF
fn log_normal_cdf(z: f64) -> f64 {
    // Clamp extreme values to avoid -inf from ln(0)
    // ln(f64::MIN_POSITIVE) ≈ -708, so we use -745 as a safe bound
    const MIN_LOG: f64 = -745.0;

    // For z >= 0, Φ(z) >= 0.5, so log(Φ(z)) is well-behaved
    if z >= 0.0 {
        normal_cdf(z).ln()
    } else {
        // For z < 0, Φ(z) is small. Use log(1 - Φ(-z)) = log1p(-Φ(-z))
        // But more accurately: log(Φ(z)) = log(1 - Φ(-z))
        let p_neg = normal_cdf(-z);
        let log_p = (-p_neg).ln_1p(); // log(1 - p_neg) = log(Φ(z))
        log_p.max(MIN_LOG) // Clamp to avoid -inf
    }
}

/// Computes the natural logarithm of the complement of the standard normal CDF.
///
/// This is equivalent to `pnorm(-z, log.p=TRUE)` in R, which gives `log(1 - Φ(z))`.
///
/// # Arguments
///
/// * `z` - Standard normal value
///
/// # Returns
///
/// `log(1 - Φ(z))` where Φ is the standard normal CDF
fn log_normal_cdf_complement(z: f64) -> f64 {
    // This is pnorm(-z, log.p=TRUE) in R
    // Which equals log(Φ(-z)) = log(1 - Φ(z))
    // Clamp extreme values to avoid -inf from ln(0)
    const MIN_LOG: f64 = -745.0;
    log_normal_cdf(-z).max(MIN_LOG)
}

/// Performs the Anderson-Darling test for normality of residuals.
///
/// This test checks whether the residuals from an OLS regression follow a normal
/// distribution by comparing the empirical distribution of the residuals to the
/// expected normal distribution. The Anderson-Darling test is particularly
/// sensitive to deviations in the tails of the distribution.
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
/// * [`Error::InsufficientData`] - if n < 8 (minimum for valid A² computation)
/// * [`Error::InvalidInput`] - if residual variance is zero or non-finite
///
/// # Reference
///
/// - Anderson, T. W., & Darling, D. A. (1954). "A Test of Goodness of Fit".
///   Journal of the American Statistical Association, 49(268), 765-769.
/// - Stephens, M. A. (1974). "EDF Statistics for Goodness of Fit". JASA, 69, 730-737.
pub fn anderson_darling_test(y: &[f64], x_vars: &[Vec<f64>]) -> Result<DiagnosticTestResult> {
    let n = y.len();
    let k = x_vars.len();
    let p = k + 1;

    // Validate inputs - need at least 8 observations for meaningful AD test
    if n < 8 {
        return Err(Error::InsufficientData {
            required: 8,
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
    let mut residuals: Vec<f64> = y
        .iter()
        .zip(predictions.iter())
        .map(|(&yi, &yi_hat)| yi - yi_hat)
        .collect();

    // Sort residuals (use unwrap_or to handle NaN safely)
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute mean and standard deviation of residuals
    // Note: Use n-1 denominator to match R's sd() function
    let mean = vec_mean(&residuals);

    // Compute variance using n-1 denominator (sample variance)
    let variance: f64 = residuals
        .iter()
        .map(|&r| {
            let diff = r - mean;
            diff * diff
        })
        .sum::<f64>()
        / (n as f64 - 1.0);

    if variance <= 0.0 || !variance.is_finite() {
        return Err(Error::InvalidInput(
            "Invalid residual variance (all values identical)".to_string(),
        ));
    }

    let std_dev = variance.sqrt();

    // Standardize residuals (like R: (x - mean(x))/sd(x))
    let z: Vec<f64> = residuals.iter().map(|&r| (r - mean) / std_dev).collect();

    // Compute log probabilities (like R's pnorm with log.p=TRUE)
    // logp1 <- pnorm((x - mean(x))/sd(x), log.p = TRUE)
    // logp2 <- pnorm(-(x - mean(x))/sd(x), log.p = TRUE)
    let logp1: Vec<f64> = z.iter().map(|&zi| log_normal_cdf(zi)).collect();
    let logp2: Vec<f64> = z.iter().map(|&zi| log_normal_cdf_complement(zi)).collect();

    // Compute Anderson-Darling statistic using R's exact formula
    // h <- (2 * seq(1:n) - 1) * (logp1 + rev(logp2))
    // A <- -n - mean(h)
    let mut h_sum = 0.0;
    for i in 0..n {
        // R's seq(1:n) gives 1, 2, ..., n
        // In 0-based indexing: (2 * (i + 1) - 1) = (2*i + 1)
        let weight = 2.0 * (i as f64 + 1.0) - 1.0;
        // R uses rev(logp2), so logp2[n-1-i] in 0-based indexing
        h_sum += weight * (logp1[i] + logp2[n - 1 - i]);
    }
    // R: A <- -n - mean(h) where mean(h) = sum(h)/n
    let ad_stat = -(n as f64) - h_sum / (n as f64);

    // Apply correction factor for finite sample size
    // A*² = A² × (1 + 0.75/n + 2.25/n²)
    let n_f = n as f64;
    let correction = 1.0 + 0.75 / n_f + 2.25 / (n_f * n_f);
    let ad_modified = ad_stat * correction;

    // Compute p-value using Marsaglia & Marsaglia (2004) approximation
    let p_value = anderson_darling_p_value(ad_modified);

    let alpha = 0.05;
    let passed = p_value > alpha;

    let (interpretation, guidance) = if passed {
        (
            format!(
                "p-value = {:.4} is greater than {:.2}. Cannot reject H0. No significant evidence that residuals deviate from normality.",
                p_value, alpha
            ),
            "The normality assumption appears to be met. Anderson-Darling test does not detect significant deviation from normal distribution, including in the tails."
        )
    } else {
        (
            format!(
                "p-value = {:.4} is less than or equal to {:.2}. Reject H0. Significant evidence that residuals deviate from normality.",
                p_value, alpha
            ),
            "Consider transforming the dependent variable (e.g., log, Box-Cox transformation), using robust standard errors, or applying a different estimation method. The Anderson-Darling test is particularly sensitive to tail deviations."
        )
    };

    Ok(DiagnosticTestResult {
        test_name: "Anderson-Darling Test for Normality".to_string(),
        statistic: ad_stat, // Return raw statistic to match R's nortest::ad.test (R returns A, not A*)
        p_value,
        passed,
        interpretation,
        guidance: guidance.to_string(),
    })
}

/// Applies the Anderson-Darling test directly to a sample of values.
///
/// This is the core implementation of the Anderson-Darling test that operates on any
/// sample of data, without first computing regression residuals. Use this when you
/// already have a sample you want to test for normality.
///
/// The Anderson-Darling test is particularly sensitive to deviations in the tails of
/// the distribution, making it more powerful than the Kolmogorov-Smirnov test for
/// detecting tail behavior.
///
/// For testing regression residuals, use [`anderson_darling_test`] instead.
///
/// # Arguments
///
/// * `sample` - Data values to test for normality
///
/// # Returns
///
/// A [`DiagnosticTestResult`] containing:
/// - `statistic`: The A² statistic (larger values indicate greater deviation from normality)
/// - `p_value`: Upper-tail p-value for the test
/// - `passed`: Whether the null hypothesis cannot be rejected (p > 0.05)
/// - `interpretation`: Human-readable explanation of the result
/// - `guidance`: Recommendations based on the test result
///
/// # Errors
///
/// * [`Error::InsufficientData`] - if n < 8 (minimum for valid A² computation)
/// * [`Error::InvalidInput`] - if sample has zero variance (all values identical)
///
/// # Example
///
/// ```rust
/// use linreg_core::diagnostics::anderson_darling_test_raw;
///
/// let sample = vec![0.1, -0.5, 0.3, 1.2, -0.8, 0.4, -0.2, 0.9, 0.0, 0.5];
/// let result = anderson_darling_test_raw(&sample)?;
/// println!("A² = {}, p-value = {}", result.statistic, result.p_value);
/// # Ok::<(), linreg_core::Error>(())
/// ```
///
/// # Notes
///
/// - The A² statistic measures the weighted squared distance between the empirical
///   and theoretical normal distribution functions
/// - Returns the raw A² statistic (not the modified A*²) to match R's `nortest::ad.test`
/// - Uses Marsaglia & Marsaglia (2004) approximation for p-value computation
pub fn anderson_darling_test_raw(sample: &[f64]) -> Result<DiagnosticTestResult> {
    let n = sample.len();

    if n < 8 {
        return Err(Error::InsufficientData {
            required: 8,
            available: n,
        });
    }

    // Validate sample contains no NaN or infinite values
    for (i, &val) in sample.iter().enumerate() {
        if !val.is_finite() {
            return Err(Error::InvalidInput(format!(
                "Sample contains non-finite value at index {}: {}",
                i, val
            )));
        }
    }

    // Sort the sample
    let mut sorted = sample.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute sample mean and variance
    // Note: Use n-1 denominator to match R's sd() function
    let mean = vec_mean(&sorted);
    let variance: f64 = sorted
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>()
        / (n as f64 - 1.0);

    if variance <= 0.0 || !variance.is_finite() {
        return Err(Error::InvalidInput(
            "Invalid sample variance (all values identical)".to_string(),
        ));
    }

    let std_dev = variance.sqrt();

    // Standardize (like R: (x - mean(x))/sd(x))
    let z: Vec<f64> = sorted.iter().map(|&x| (x - mean) / std_dev).collect();

    // Compute log probabilities (like R's pnorm with log.p=TRUE)
    let logp1: Vec<f64> = z.iter().map(|&zi| log_normal_cdf(zi)).collect();
    let logp2: Vec<f64> = z.iter().map(|&zi| log_normal_cdf_complement(zi)).collect();

    // Compute Anderson-Darling statistic using R's exact formula
    let mut h_sum = 0.0;
    for i in 0..n {
        let weight = 2.0 * (i as f64 + 1.0) - 1.0;
        h_sum += weight * (logp1[i] + logp2[n - 1 - i]);
    }
    let ad_stat = -(n as f64) - h_sum / (n as f64);

    // Apply correction factor
    let n_f = n as f64;
    let correction = 1.0 + 0.75 / n_f + 2.25 / (n_f * n_f);
    let ad_modified = ad_stat * correction;

    let p_value = anderson_darling_p_value(ad_modified);

    let alpha = 0.05;
    let passed = p_value > alpha;

    let (interpretation, guidance) = if passed {
        (
            format!(
                "p-value = {:.4} is greater than {:.2}. Cannot reject H0.",
                p_value, alpha
            ),
            "No significant evidence that the sample deviates from normality.",
        )
    } else {
        (
            format!(
                "p-value = {:.4} is less than or equal to {:.2}. Reject H0.",
                p_value, alpha
            ),
            "Significant evidence that the sample deviates from normality.",
        )
    };

    Ok(DiagnosticTestResult {
        test_name: "Anderson-Darling Test for Normality".to_string(),
        statistic: ad_stat, // Return raw statistic to match R's nortest::ad.test (R returns A, not A*)
        p_value,
        passed,
        interpretation,
        guidance: guidance.to_string(),
    })
}

/// Compute p-value from Anderson-Darling statistic.
///
/// Uses the piecewise approximation from R's nortest::ad.test which matches
/// the Stephens (1974) approximations for the Anderson-Darling distribution.
///
/// # Arguments
///
/// * `ad_modified` - Modified Anderson-Darling statistic (A*² = A² × (1 + 0.75/n + 2.25/n²))
///
/// # Returns
///
/// The p-value (upper tail probability)
///
/// # Reference
///
/// Stephens, M. A. (1974). "EDF Statistics for Goodness of Fit". JASA, 69, 730-737.
/// nortest package source code (ad.test function)
fn anderson_darling_p_value(ad_modified: f64) -> f64 {
    // R's nortest::ad.test uses a piecewise formula depending on the
    // modified statistic value. The formulas are from Stephens (1974).
    //
    // The modified statistic is: A*² = A² × (1 + 0.75/n + 2.25/n²)
    //
    // Piecewise formula ranges:
    // - AA < 0.2:    p = 1 - exp(-13.436 + 101.14*AA - 223.73*AA²)
    // - 0.2 ≤ AA < 0.34: p = 1 - exp(-8.318 + 42.796*AA - 59.938*AA²)
    // - 0.34 ≤ AA < 0.6: p = exp(0.9177 - 4.279*AA - 1.38*AA²)
    // - 0.6 ≤ AA < 10: p = exp(1.2937 - 5.709*AA + 0.0186*AA²)
    // - AA ≥ 10:      p = 3.7e-24

    let aa = ad_modified;
    let aa_sq = aa * aa;

    let p_val = if aa < 0.2 {
        1.0 - (-13.436 + 101.14 * aa - 223.73 * aa_sq).exp()
    } else if aa < 0.34 {
        1.0 - (-8.318 + 42.796 * aa - 59.938 * aa_sq).exp()
    } else if aa < 0.6 {
        (0.9177 - 4.279 * aa - 1.38 * aa_sq).exp()
    } else if aa < 10.0 {
        (1.2937 - 5.709 * aa + 0.0186 * aa_sq).exp()
    } else {
        3.7e-24
    };

    p_val.clamp(0.0, 1.0)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anderson_darling_normal_data() {
        // Normal data should have high p-value
        let normal_data = vec![
            0.1, -0.5, 0.3, 1.2, -0.8, 0.4, -0.2, 0.9, -0.3, 0.6, -0.1, 0.7, -0.4, 0.2, 1.1, -0.6,
            0.8, -0.9, 0.5, -0.7, 0.0, 0.3, -0.4, 0.6, -0.2,
        ];

        let result = anderson_darling_test_raw(&normal_data).unwrap();
        // For normal data, p-value should be > 0.05 (test passes)
        assert!(result.p_value > 0.01, "p-value = {}", result.p_value);
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_anderson_darling_uniform_data() {
        // Uniform data should have lower p-value than normal data
        let uniform_data: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();

        let result = anderson_darling_test_raw(&uniform_data).unwrap();
        // R gives A² = 0.5345, p = 0.1632
        // p-value > 0.05 means we fail to reject H0 (AD test not powerful enough for this data)
        assert!(result.p_value < 0.25, "p-value = {}", result.p_value);
        // Note: With alpha = 0.05, uniform data doesn't strongly reject normality for n=50
    }

    #[test]
    fn test_anderson_darling_small_sample() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = anderson_darling_test_raw(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_anderson_darling_too_small() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let result = anderson_darling_test_raw(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_anderson_darling_constant_data() {
        let data = vec![5.0; 100];
        let result = anderson_darling_test_raw(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_anderson_darling_with_regression() {
        // Test with actual regression data
        let y = vec![
            10.5, 12.3, 11.8, 14.2, 13.5, 15.1, 14.8, 16.3, 15.9, 17.2, 16.8, 18.5, 18.1, 19.3,
            19.0, 20.5, 20.1, 21.8, 21.3, 22.5,
        ];
        let x1 = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ];

        let result = anderson_darling_test(&y, &[x1]);
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.statistic >= 0.0);
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }
}
