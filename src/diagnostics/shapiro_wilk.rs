// ============================================================================
// Shapiro-Wilk Test for Normality
// ============================================================================
//
// H0: Residuals are normally distributed
// H1: Residuals are not normally distributed
//
// This implementation uses Royston's algorithm (AS R94, 1995) to compute the
// Shapiro-Wilk test statistic and p-value for any sample size 3 ≤ n ≤ 5000.
//
// The W statistic is computed as the squared correlation between the ordered
// data and the expected normal order statistics coefficients.
//
// References:
// - Shapiro, S. S., & Wilk, M. B. (1965). "An analysis of variance test for
//   normality (complete samples)". Biometrika, 52(3-4), 591-611.
// - Royston, P. (1982). "An extension of Shapiro and Wilk's W test for
//   normality to large samples". Applied Statistics, 31(2), 115-124.
// - Royston, P. (1995). "Remark AS R94: A Remark on Algorithm AS 181:
//   The W-test for Normality". Applied Statistics, 44, 547-551.

use crate::error::{Error, Result};
use super::types::DiagnosticTestResult;

/// Performs the Shapiro-Wilk test for normality of residuals.
pub fn shapiro_wilk_test(
    y: &[f64],
    x_vars: &[Vec<f64>],
) -> Result<DiagnosticTestResult> {
    let n = y.len();

    if n < 3 {
        return Err(Error::InsufficientData { required: 3, available: n });
    }

    if n > 5000 {
        return Err(Error::InvalidInput(
            "Shapiro-Wilk test only supports n ≤ 5000. Use Jarque-Bera test for larger samples.".to_string()
        ));
    }

    // Validate dimensions and finite values using shared helper
    super::helpers::validate_regression_data(y, x_vars)?;

    let residuals = compute_residuals(y, x_vars)?;
    shapiro_wilk_test_raw(&residuals)
}

/// Applies the Shapiro-Wilk test directly to a sample of values.
///
/// This is the core implementation of the Shapiro-Wilk test that operates on any
/// sample of data, without first computing regression residuals. Use this when you
/// already have a sample you want to test for normality.
///
/// For testing regression residuals, use [`shapiro_wilk_test`] instead.
///
/// # Arguments
///
/// * `sample` - Data values to test for normality
///
/// # Returns
///
/// A [`DiagnosticTestResult`] containing:
/// - `statistic`: The W statistic (0 ≤ W ≤ 1, closer to 1 indicates normality)
/// - `p_value`: Two-tailed p-value for the test
/// - `passed`: Whether the null hypothesis cannot be rejected (p > 0.05)
/// - `interpretation`: Human-readable explanation of the result
/// - `guidance`: Recommendations based on the test result
///
/// # Errors
///
/// * [`Error::InsufficientData`] - if n < 3 (minimum for Shapiro-Wilk) or n > 5000
/// * [`Error::InvalidInput`] - if sample has zero variance (all values identical)
///
/// # Example
///
/// ```rust
/// use linreg_core::diagnostics::shapiro_wilk_test_raw;
///
/// let sample = vec![0.1, -0.5, 0.3, 1.2, -0.8, 0.4, -0.2, 0.9];
/// let result = shapiro_wilk_test_raw(&sample)?;
/// println!("W = {}, p-value = {}", result.statistic, result.p_value);
/// # Ok::<(), linreg_core::Error>(())
/// ```
///
/// # Notes
///
/// - W = 1 indicates perfect normality; W < 1 indicates deviation from normality
/// - Uses Royston's algorithm (AS R94, 1995) for p-value computation
/// - Limited to n ≤ 5000 (same as R's `shapiro.test()`)
pub fn shapiro_wilk_test_raw(sample: &[f64]) -> Result<DiagnosticTestResult> {
    let n = sample.len();

    if n < 3 {
        return Err(Error::InsufficientData { required: 3, available: n });
    }

    if n > 5000 {
        return Err(Error::InvalidInput(
            "Shapiro-Wilk test only supports n ≤ 5000.".to_string()
        ));
    }

    // Validate sample contains no NaN or infinite values
    for (i, &val) in sample.iter().enumerate() {
        if !val.is_finite() {
            return Err(Error::InvalidInput(format!(
                "Sample contains non-finite value at index {}: {}", i, val
            )));
        }
    }

    // Sort the sample
    let mut sorted = sample.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Check for constant values
    let range = sorted.last().unwrap() - sorted.first().unwrap();
    if range == 0.0 || !range.is_finite() {
        return Err(Error::InvalidInput("Invalid sample variance (all values identical)".to_string()));
    }

    // Compute W statistic and p-value using Royston's algorithm
    let (w, p_value) = royston_swilk(&sorted)?;

    let alpha = 0.05;
    let passed = p_value > alpha;

    let (interpretation, guidance) = if passed {
        (
            format!(
                "p-value = {:.4} is greater than {:.2}. Cannot reject H0. No significant evidence that residuals deviate from normality.",
                p_value, alpha
            ),
            "The normality assumption appears to be met."
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
        test_name: "Shapiro-Wilk Test for Normality".to_string(),
        statistic: w,
        p_value,
        passed,
        interpretation,
        guidance: guidance.to_string(),
    })
}

/// Compute residuals from OLS regression.
fn compute_residuals(y: &[f64], x_vars: &[Vec<f64>]) -> Result<Vec<f64>> {
    let n = y.len();
    let k = x_vars.len();
    let p = k + 1;

    let mut xt_x = vec![0.0; p * p];
    let mut xt_y = vec![0.0; p];

    for i in 0..n {
        for j in 0..p {
            let x_ij = if j == 0 { 1.0 } else { x_vars[j - 1][i] };
            xt_y[j] += x_ij * y[i];

            for l in 0..p {
                let x_il = if l == 0 { 1.0 } else { x_vars[l - 1][i] };
                xt_x[j * p + l] += x_ij * x_il;
            }
        }
    }

    let beta = solve_spd(&xt_x, &xt_y, p)?;

    let residuals: Vec<f64> = (0..n).map(|i| {
        let mut y_pred = beta[0];
        for j in 1..p {
            y_pred += beta[j] * x_vars[j - 1][i];
        }
        y[i] - y_pred
    }).collect();

    Ok(residuals)
}

/// Solve Ax = b for symmetric positive definite A using Cholesky decomposition.
fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(Error::SingularMatrix);
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }

    Ok(x)
}

/// Compute Shapiro-Wilk W statistic and p-value using Royston (1995) algorithm.
///
/// This is the algorithm used by R's shapiro.test() function.
/// Based on Applied Statistics algorithms AS181, R94.
#[allow(clippy::needless_range_loop)]
fn royston_swilk(x: &[f64]) -> Result<(f64, f64)> {
    let n = x.len();
    let nn2 = n / 2;
    let mut a = vec![0.0; nn2 + 1]; // 1-based indexing like R

    // For n = 3, the coefficient is simply sqrt(1/2)
    if n == 3 {
        a[1] = std::f64::consts::FRAC_1_SQRT_2;
    } else {
        compute_swilk_coefficients_into(&mut a, n)?;
    }

    // Check for zero range
    let range = x[n - 1] - x[0];
    if range < 1e-19 {
        return Ok((1.0, 0.0));
    }

    // Scale x by range (normalize to [0, 1])
    let small = 1e-19;

    // Calculate mean of scaled x and mean of coefficients (following R code)
    let mut xx = x[0] / range;
    let mut sx = xx;
    let mut sa = -a[1];

    for i in 1..n {
        let xi = x[i] / range;
        if xx - xi > small {
            // This would set ifault = 7 in R, but we just continue
        }
        sx += xi;
        let j = n - 1 - i;
        if i != j {
            let idx = i.min(j) + 1; // +1 for 1-based indexing
            let sign = if i > j { 1.0 } else { -1.0 };
            sa += sign * a[idx];
        }
        xx = xi;
    }

    sa /= n as f64;
    sx /= n as f64;

    // Calculate W as squared correlation between data and coefficients
    let mut ssa = 0.0;
    let mut ssx = 0.0;
    let mut sax = 0.0;

    for i in 0..n {
        let j = n - 1 - i;
        let asa = if i != j {
            let idx = i.min(j) + 1; // +1 for 1-based indexing
            let sign = if i > j { 1.0 } else { -1.0 };
            sign * a[idx] - sa
        } else {
            -sa
        };
        let xsx = x[i] / range - sx;
        ssa += asa * asa;
        ssx += xsx * xsx;
        sax += asa * xsx;
    }

    // W = 1 - w1 (to avoid rounding error for W near 1)
    let ssassx = (ssa * ssx).sqrt();
    let w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx);
    let w = (1.0 - w1).clamp(0.0, 1.0);

    // Calculate p-value
    let pw = royston_p_value(w, n);

    Ok((w, pw))
}

/// Compute Shapiro-Wilk coefficients using Royston (1995) algorithm.
///
/// Writes coefficients into a\[1..=nn2\] (1-based indexing to match R).
/// The array a must have size nn2 + 1 (with a\[0\] unused).
#[allow(clippy::needless_range_loop)]
fn compute_swilk_coefficients_into(a: &mut [f64], n: usize) -> Result<()> {
    let nn2 = n / 2;

    let an = n as f64;
    let an25 = an + 0.25;
    let mut summ2 = 0.0;

    // Compute initial a[i] values using normal quantiles
    // R code: for (i = 1; i <= nn2; i++) a[i] = qnorm((i - 0.375) / an25, ...)
    for i in 1..=nn2 {
        let p = (i as f64 - 0.375) / an25;
        a[i] = normal_quantile(p)?;
        summ2 += a[i] * a[i];
    }

    summ2 *= 2.0;
    let ssumm2 = summ2.sqrt();
    let rsn = 1.0 / an.sqrt();

    // Polynomial coefficients from Royston (1995) - using nord (size of array)
    let c1: [f64; 6] = [0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
    let c2: [f64; 6] = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];

    // R code: a1 = poly(c1, 6, rsn) - a[1] / ssumm2;
    let a1 = poly(&c1, rsn) - a[1] / ssumm2;

    // Normalize a[]
    // R code: if (n > 5) { i1 = 3; a2 = -a[2] / ssumm2 + poly(c2, 6, rsn); ...
    let (i1, fac) = if n > 5 {
        let a2 = -a[2] / ssumm2 + poly(&c2, rsn);
        let fac = ((summ2 - 2.0 * a[1] * a[1] - 2.0 * a[2] * a[2]) /
                   (1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2)).sqrt();
        a[2] = a2;
        (3, fac)
    } else {
        // n <= 5
        let fac = ((summ2 - 2.0 * a[1] * a[1]) /
                   (1.0 - 2.0 * a1 * a1)).sqrt();
        (2, fac)
    };

    a[1] = a1;
    // R code: for (i = i1; i <= nn2; i++) a[i] /= -fac;
    for i in i1..=nn2 {
        a[i] /= -fac;
    }

    Ok(())
}

/// Evaluates a polynomial at x.
///
/// Polynomial coefficients are in cc\[0..nord\], with cc\[0\] being the constant term.
fn poly(cc: &[f64], x: f64) -> f64 {
    let nord = cc.len();
    let mut ret_val = cc[0];
    if nord > 1 {
        let mut p = x * cc[nord - 1];
        for j in (1..nord-1).rev() {
            p = (p + cc[j]) * x;
        }
        ret_val += p;
    }
    ret_val
}

/// Standard normal quantile function (inverse CDF) using Acklam's algorithm.
#[allow(clippy::excessive_precision)]
fn normal_quantile(p: f64) -> Result<f64> {
    if p <= 0.0 { return Ok(f64::NEG_INFINITY); }
    if p >= 1.0 { return Ok(f64::INFINITY); }
    if (p - 0.5).abs() < 1e-15 { return Ok(0.0); }

    // Acklam's algorithm coefficients
    const A: [f64; 6] = [
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p > P_LOW && p < P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        return Ok(num / den);
    }

    // For lower and upper regions, use rational approximation directly
    // The Acklam algorithm formula is x = num/den (lower) or x = -num/den (upper)
    // There is NO subtraction from q!
    let (num, den, sign) = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        let n = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let d = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        (n, d, 1.0)  // Lower tail: x = num/den (will be negative)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let n = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let d = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        (n, d, -1.0)  // Upper tail: x = -num/den (will be positive)
    };

    Ok(sign * num / den)
}

/// Compute p-value using Royston's transformation of the W statistic.
///
/// From Royston (1995) Algorithm AS R94.
fn royston_p_value(w: f64, n: usize) -> f64 {
    if w >= 1.0 {
        return 1.0;
    }
    if w <= 0.0 || !w.is_finite() {
        return 0.0;
    }

    // For n = 3, use exact formula
    if n == 3 {
        let pi6 = 1.90985931710274;  // = 6/pi
        let stqr = std::f64::consts::FRAC_PI_3; // = asin(sqrt(3/4)) = π/3
        let mut pw = pi6 * (w.sqrt().asin() - stqr);
        if pw < 0.0 {
            pw = 0.0;
        }
        return pw;
    }

    let an = n as f64;
    let w1 = 1.0 - w;
    let y = w1.ln();
    let xx = an.ln();

    // Polynomial coefficients from Royston (1995)
    let g: [f64; 2] = [-2.273, 0.459];
    let c3: [f64; 4] = [0.544, -0.39978, 0.025054, -0.0006714];
    let c4: [f64; 4] = [1.3822, -0.77857, 0.062767, -0.0020322];
    let c5: [f64; 4] = [-1.5861, -0.31082, -0.083751, 0.0038915];
    let c6: [f64; 3] = [-0.4803, -0.082676, 0.0030302];

    let (y_for_p, m, s) = if n <= 11 {
        let gamma = poly(&g, an);
        if y >= gamma {
            return 1e-99;
        }
        let y_trans = -((gamma - y).ln());
        let m = poly(&c3, an);
        let s = poly(&c4, an).exp();
        (y_trans, m, s)
    } else {
        // n >= 12
        let m = poly(&c5, xx);
        let s = poly(&c6, xx).exp();
        (y, m, s)
    };

    // P-value from standard normal upper tail
    normal_cdf_upper(y_for_p, m, s)
}

/// Standard normal upper tail CDF: P(Z > z) where Z ~ N(mean, sd)
fn normal_cdf_upper(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    // P(Z > z) = 1 - Phi(z) where Phi is standard normal CDF
    1.0 - normal_cdf(z)
}

/// Standard normal CDF using the error function approximation.
/// Uses Abramowitz & Stegun 7.1.26 polynomial approximation for erf(x).
/// Phi(x) = 0.5 * (1 + erf(x/sqrt(2)))
fn normal_cdf(x: f64) -> f64 {
    use std::f64::consts::FRAC_1_SQRT_2;
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    // For normal CDF, we need to use erf(x/sqrt(2))
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = (x * FRAC_1_SQRT_2).abs();  // x/sqrt(2)

    let t = 1.0 / (1.0 + P * x_abs);
    // Using Horner's method: ((((A5*t + A4)*t + A3)*t + A2)*t + A1)
    let poly = ((((A5 * t + A4) * t) + A3) * t + A2) * t + A1;
    // erf approximation
    let erf_approx = sign * (1.0 - poly * t * (-x_abs * x_abs).exp());
    // Phi(x) = 0.5 * (1 + erf(x/sqrt(2)))
    0.5 * (1.0 + erf_approx)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shapiro_wilk_normal_data() {
        let normal_data = vec![
            0.1, -0.5, 0.3, 1.2, -0.8, 0.4, -0.2, 0.9, -0.3, 0.6,
            -0.1, 0.7, -0.4, 0.2, 1.1, -0.6, 0.8, -0.9, 0.5, -0.7,
        ];
        let result = shapiro_wilk_test_raw(&normal_data).unwrap();
        assert!(result.p_value > 0.01);
        assert!(result.statistic > 0.0 && result.statistic <= 1.0);
    }

    #[test]
    fn test_shapiro_wilk_small_sample() {
        let data = vec![1.0, 2.0, 3.0];
        let result = shapiro_wilk_test_raw(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_shapiro_wilk_too_small() {
        let data = vec![1.0, 2.0];
        let result = shapiro_wilk_test_raw(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_shapiro_wilk_too_large() {
        let data: Vec<f64> = (0..6000).map(|i| i as f64).collect();
        let result = shapiro_wilk_test_raw(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_shapiro_wilk_constant_data() {
        let data = vec![5.0; 100];
        let result = shapiro_wilk_test_raw(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_shapiro_wilk_uniform_data() {
        let uniform_data: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();
        let result = shapiro_wilk_test_raw(&uniform_data).unwrap();
        // Uniform data should have lower W (rejects normality)
        // scipy gives W=0.956 for this data
        assert!(result.statistic < 0.97, "W = {}", result.statistic);
        // At alpha=0.05, this may or may not reject depending on exact value
        // scipy gives p=0.058, so it doesn't reject
    }

    #[test]
    fn test_normal_quantile() {
        assert_eq!(normal_quantile(0.5).unwrap(), 0.0);
        assert!((normal_quantile(0.975).unwrap() - 1.959963984540054).abs() < 1e-6);
        assert!((normal_quantile(0.025).unwrap() - (-1.959963984540054)).abs() < 1e-6);
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }

    #[test]
    fn test_poly() {
        let coeffs = [1.0, 2.0, 3.0];  // 1 + 2x + 3x^2
        assert!((poly(&coeffs, 0.0) - 1.0).abs() < 1e-10);
        assert!((poly(&coeffs, 1.0) - 6.0).abs() < 1e-10);  // 1 + 2 + 3
        assert!((poly(&coeffs, 2.0) - 17.0).abs() < 1e-10); // 1 + 4 + 12
    }

    #[test]
    fn test_w_matches_known_values() {
        // Test data: 1, 2, ..., n
        for n in [3, 5, 10, 20] {
            let data: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let result = shapiro_wilk_test_raw(&data).unwrap();

            // These are approximate expected values from scipy/R
            let expected_w = match n {
                3 => 1.0,
                5 => 0.9868,
                10 => 0.9702,
                20 => 0.9604,
                _ => unreachable!(),
            };

            assert!((result.statistic - expected_w).abs() < 0.001,
                    "n={}: W={} != {}", n, result.statistic, expected_w);
        }
    }
}
