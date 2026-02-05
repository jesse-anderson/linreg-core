// ============================================================================
// Harvey-Collier Test for Linearity (Functional Form)
// ============================================================================
//
// H0: The relationship is linear (no functional misspecification)
// H1: The relationship is non-linear (functional form misspecification)
//
// This test computes recursive residuals from an expanding-window OLS fit
// and performs a t-test that the mean of the (scaled) recursive residuals
// is zero.

use super::helpers::two_tailed_p_value;
use super::types::{DiagnosticTestResult, HarveyCollierMethod};
use crate::error::{Error, Result};
use crate::linalg::{vec_mean, Matrix};

/// Performs the Harvey-Collier test for linearity (functional form).
///
/// This test checks for functional form misspecification by computing **recursive residuals**
/// from an expanding-window OLS fit and then performing a **t-test that the mean of the
/// (scaled) recursive residuals is zero**. Systematic deviation from zero provides evidence
/// against linear specification.
///
/// # Arguments
///
/// * `y` - Dependent variable values
/// * `x_vars` - Independent variables (each vec is a column)
/// * `method` - Which variant to use: R (lmtest::harvtest) or Python (statsmodels)
///
/// # Returns
///
/// A [`DiagnosticTestResult`] containing the test statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if n ≤ k + 2.
///
/// # Example
///
/// ```
/// # use linreg_core::diagnostics::{harvey_collier_test, HarveyCollierMethod};
/// // Data with some non-linear pattern
/// let y = vec![1.0, 2.1, 3.5, 6.2, 9.8, 15.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let x2 = vec![1.0, 3.0, 2.0, 4.0, 2.5, 5.0];
///
/// let result = harvey_collier_test(&y, &[x1, x2], HarveyCollierMethod::R).unwrap();
///
/// println!("t-statistic: {}", result.statistic);
/// println!("P-value: {}", result.p_value);
/// // Low p-value suggests non-linearity (functional form misspecification)
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn harvey_collier_test(
    y: &[f64],
    x_vars: &[Vec<f64>],
    method: HarveyCollierMethod,
) -> Result<DiagnosticTestResult> {
    match method {
        HarveyCollierMethod::R => harvey_collier_test_r(y, x_vars),
        HarveyCollierMethod::Python => harvey_collier_test_python(y, x_vars),
    }
}

/// Harvey-Collier test using R's lmtest::harvtest-esque steps.
///
/// This implementation matches R's lmtest package output, using all recursive residuals
/// without skipping any elements.
#[allow(clippy::needless_range_loop)]
fn harvey_collier_test_r(y: &[f64], x_vars: &[Vec<f64>]) -> Result<DiagnosticTestResult> {
    let n = y.len();
    let k = x_vars.len(); // number of *non-intercept* regressors
    let p = k + 1; // number of columns in design matrix, incl intercept (R's "k")

    // Validate inputs
    // Need at least p + 2 observations so that:
    // - recursive residuals length = n - p >= 2 (variance defined)
    // - df = n - p - 1 >= 1
    if n <= p + 1 {
        return Err(Error::InsufficientData {
            required: p + 2,
            available: n,
        });
    }

    // Validate dimensions and finite values using shared helper
    super::helpers::validate_regression_data(y, x_vars)?;

    // Create design matrix (with intercept)
    let mut x_data = vec![0.0; n * p];
    for row in 0..n {
        x_data[row * p] = 1.0;
        for (col, x_var) in x_vars.iter().enumerate() {
            x_data[row * p + col + 1] = x_var[row];
        }
    }

    // NOTE: R's harvtest does NOT sort by fitted values when order.by = NULL (the default).
    // The data is used in its original order. Only when order.by is explicitly provided
    // does R reorder the observations. We follow R's default behavior here.
    let y_sorted = y.to_vec();
    let x_sorted = x_data;

    // ------------------------------------------------------------------------
    // Recursive residuals: match lmtest::harvtest rec.res() exactly
    //
    // R source (lmtest):
    //   Xr1 <- X[1:q,]
    //   xr  <- X[q+1,]
    //   X1  <- chol2inv(qr.R(qr(Xr1)))
    //   fr  <- 1 + t(xr) %*% X1 %*% xr
    //   betar <- X1 %*% t(Xr1) %*% y[1:q]
    //   w[1] <- (y[q+1] - t(xr) %*% betar) / sqrt(fr)
    //   for r in (q+2):n:
    //     X1 <- X1 - (X1 %*% outer(xr,xr) %*% X1) / fr
    //     betar <- betar + X1 %*% xr * w[r-q-1] * sqrt(fr)
    //     xr <- X[r,]
    //     fr <- 1 + t(xr) %*% X1 %*% xr
    //     w[r-q] <- (y[r] - t(xr) %*% betar) / sqrt(fr)
    // ------------------------------------------------------------------------

    // Build Xr1 = first p rows (p×p)
    let x_r1 = Matrix::new(p, p, x_sorted[0..(p * p)].to_vec());

    // X1 = chol2inv(qr.R(qr(Xr1)))  -> your linalg provides chol2inv_from_qr()
    let mut x1 = match x_r1.chol2inv_from_qr() {
        Some(inv) => inv,
        None => return Err(Error::SingularMatrix),
    };

    // betar = X1 * t(Xr1) * y[0..p]
    let xt = x_r1.transpose();
    let xty = xt.mul_vec(&y_sorted[0..p]);
    let mut betar = x1.mul_vec(&xty);

    // xr = row p (0-based). This corresponds to R's X[q+1,]
    let mut xr: Vec<f64> = (0..p).map(|j| x_sorted[p * p + j]).collect();

    // fr = 1 + t(xr) * X1 * xr
    let mut fr = 1.0;
    for i in 0..p {
        for j in 0..p {
            fr += xr[i] * x1.get(i, j) * xr[j];
        }
    }

    // w has length (n - p)
    let mut w = vec![0.0_f64; n - p];

    // w[0] corresponds to R's w[1] = residual at y[q+1]
    let y_pred = (0..p).map(|j| betar[j] * xr[j]).sum::<f64>();
    w[0] = (y_sorted[p] - y_pred) / fr.sqrt();

    // Loop r = (p+1)..(n-1) (0-based). This corresponds to R's r = (q+2):n (1-based).
    for r in (p + 1)..n {
        // Update X1: X1 <- X1 - (X1 %*% outer(xr,xr) %*% X1) / fr
        // Compute u = X1 * xr (using current X1 and current xr)
        let mut u = vec![0.0; p];
        for i in 0..p {
            for j in 0..p {
                u[i] += x1.get(i, j) * xr[j];
            }
        }
        for i in 0..p {
            for j in 0..p {
                let update = (u[i] * u[j]) / fr;
                x1.set(i, j, x1.get(i, j) - update);
            }
        }

        // Update betar: betar <- betar + X1 %*% xr * w[r-q-1] * sqrt(fr)
        // Here (r - p - 1) is the 0-based index for w[r-q-1] in R.
        let w_prev = w[r - p - 1];
        let scale = w_prev * fr.sqrt();

        // Compute v = X1 * xr using updated X1 (matches R ordering)
        let mut v = vec![0.0; p];
        for i in 0..p {
            for j in 0..p {
                v[i] += x1.get(i, j) * xr[j];
            }
        }
        for i in 0..p {
            betar[i] += v[i] * scale;
        }

        // xr <- X[r,]
        xr = (0..p).map(|j| x_sorted[r * p + j]).collect();

        // fr <- 1 + t(xr) * X1 * xr
        fr = 1.0;
        for i in 0..p {
            for j in 0..p {
                fr += xr[i] * x1.get(i, j) * xr[j];
            }
        }

        // w[r-q] in R corresponds to w[r-p] in 0-based indexing
        let y_pred_r = (0..p).map(|j| betar[j] * xr[j]).sum::<f64>();
        w[r - p] = (y_sorted[r] - y_pred_r) / fr.sqrt();
    }

    // ------------------------------------------------------------------------
    // Harvey-Collier statistic and p-value: match lmtest::harvtest
    //
    // resr <- w
    // sigma <- sqrt(var(resr)*(length(resr)-1)/(n-k-1))   where k = ncol(X) = p
    // resr <- resr/sigma
    // harv <- abs(sum(resr)/sqrt(n-k)) / sqrt(var(resr))
    // df <- n-k-1
    // p.value <- 2 * pt(harv, df, lower.tail=FALSE)
    // ------------------------------------------------------------------------

    let n_rr = w.len();
    if n_rr < 2 {
        return Err(Error::InsufficientData {
            required: 2,
            available: n_rr,
        });
    }

    // Sample variance of w (R's var())
    let mean_w = vec_mean(&w);
    let var_w = w.iter().map(|&v| (v - mean_w).powi(2)).sum::<f64>() / ((n_rr - 1) as f64);
    if !var_w.is_finite() || var_w <= 0.0 {
        return Err(Error::InvalidInput(
            "Invalid variance in Harvey-Collier test".to_string(),
        ));
    }

    // In R, sigma simplifies to sqrt(var_w) because (length-1)/(n-p-1) cancels,
    // but we keep the exact structure for readability.
    let df = (n - p - 1) as f64; // IMPORTANT: p includes intercept (matches R)
    let sigma = (var_w * ((n_rr - 1) as f64) / df).sqrt();

    // Scale recursive residuals
    let resr: Vec<f64> = w.iter().map(|&v| v / sigma).collect();

    // Compute harv statistic
    let sum_resr = resr.iter().sum::<f64>();
    let mean_resr = sum_resr / (n_rr as f64);
    let var_resr = resr.iter().map(|&v| (v - mean_resr).powi(2)).sum::<f64>() / ((n_rr - 1) as f64);
    if !var_resr.is_finite() || var_resr <= 0.0 {
        return Err(Error::InvalidInput(
            "Invalid scaled variance in Harvey-Collier test".to_string(),
        ));
    }

    // n - k in R uses k = ncol(X) = p, so n - k = n - p = length(resr)
    let harv = (sum_resr / ((n - p) as f64).sqrt()).abs() / var_resr.sqrt();

    // Two-tailed p-value (R uses 2 * pt(harv, df, lower.tail=FALSE))
    let p_value = two_tailed_p_value(harv, df);

    let alpha = 0.05;
    let passed = p_value > alpha;

    let (interpretation, guidance) = if passed {
        (
            format!(
                "p-value = {:.4} is greater than {:.2}. Cannot reject H0. No significant evidence of functional misspecification.",
                p_value, alpha
            ),
            "The linear model specification appears appropriate."
        )
    } else {
        (
            format!(
                "p-value = {:.4} is less than or equal to {:.2}. Reject H0. Significant evidence of functional misspecification.",
                p_value, alpha
            ),
            "Consider adding polynomial terms, transforming variables, or using non-linear modeling."
        )
    };

    Ok(DiagnosticTestResult {
        test_name: "Harvey-Collier Test for Linearity".to_string(),
        statistic: harv,
        p_value,
        passed,
        interpretation,
        guidance: guidance.to_string(),
    })
}

/// Harvey-Collier test w/ Python's statsmodels algorithm-esque setup.
///
/// This implementation matches Python's statsmodels.stats.diagnostic.linear_harvey_collier.
/// The key difference from R's method is that Python skips the first 3 elements of the
/// standardized recursive residuals when computing the t-statistic.
#[allow(clippy::needless_range_loop)]
fn harvey_collier_test_python(y: &[f64], x_vars: &[Vec<f64>]) -> Result<DiagnosticTestResult> {
    let n = y.len();
    let k = x_vars.len(); // number of *non-intercept* regressors
    let p = k + 1; // number of columns in design matrix, incl intercept (R's "k")

    // Validate inputs
    // Need at least p + 2 observations so that:
    // - recursive residuals length = n - p >= 2 (variance defined)
    // - df = n - p - 1 >= 1
    if n <= p + 1 {
        return Err(Error::InsufficientData {
            required: p + 2,
            available: n,
        });
    }

    // Validate dimensions and finite values using shared helper
    super::helpers::validate_regression_data(y, x_vars)?;

    // Create design matrix (with intercept)
    let mut x_data = vec![0.0; n * p];
    for row in 0..n {
        x_data[row * p] = 1.0;
        for (col, x_var) in x_vars.iter().enumerate() {
            x_data[row * p + col + 1] = x_var[row];
        }
    }

    // NOTE: Python's statsmodels does NOT sort by fitted values when order.by = NULL.
    // The data is used in its original order, same as R.
    let y_sorted = y.to_vec();
    let x_sorted = x_data;

    // Python's variant uses the same recursive residuals computation as R,
    // but differs in the final test statistic: it skips the first 3 elements
    // of the standardized recursive residuals when computing the t-statistic.

    // Build Xr1 = first p rows (p×p)
    let x_r1 = Matrix::new(p, p, x_sorted[0..(p * p)].to_vec());

    let mut x1 = match x_r1.chol2inv_from_qr() {
        Some(inv) => inv,
        None => return Err(Error::SingularMatrix),
    };

    let xt = x_r1.transpose();
    let xty = xt.mul_vec(&y_sorted[0..p]);
    let mut betar = x1.mul_vec(&xty);

    // xr = row p (0-based). This corresponds to R's X[q+1,]
    let mut xr: Vec<f64> = (0..p).map(|j| x_sorted[p * p + j]).collect();

    let mut fr = 1.0;
    for i in 0..p {
        for j in 0..p {
            fr += xr[i] * x1.get(i, j) * xr[j];
        }
    }

    // w has length (n - p)
    let mut w = vec![0.0_f64; n - p];

    // w[0] corresponds to R's w[1] = residual at y[q+1]
    let y_pred = (0..p).map(|j| betar[j] * xr[j]).sum::<f64>();
    w[0] = (y_sorted[p] - y_pred) / fr.sqrt();

    // Loop r = (p+1)..(n-1) (0-based). This corresponds to R's r = (q+2):n (1-based).
    for r in (p + 1)..n {
        // Update X1: X1 <- X1 - (X1 %*% outer(xr,xr) %*% X1) / fr
        // Compute u = X1 * xr (using current X1 and current xr)
        let mut u = vec![0.0; p];
        for i in 0..p {
            for j in 0..p {
                u[i] += x1.get(i, j) * xr[j];
            }
        }
        for i in 0..p {
            for j in 0..p {
                let update = (u[i] * u[j]) / fr;
                x1.set(i, j, x1.get(i, j) - update);
            }
        }

        // Update betar: betar <- betar + X1 %*% xr * w[r-q-1] * sqrt(fr)
        // Here (r - p - 1) is the 0-based index for w[r-q-1] in R.
        let w_prev = w[r - p - 1];
        let scale = w_prev * fr.sqrt();

        // Compute v = X1 * xr using updated X1 
        let mut v = vec![0.0; p];
        for i in 0..p {
            for j in 0..p {
                v[i] += x1.get(i, j) * xr[j];
            }
        }
        for i in 0..p {
            betar[i] += v[i] * scale;
        }

        // xr <- X[r,]
        xr = (0..p).map(|j| x_sorted[r * p + j]).collect();

        // fr <- 1 + t(xr) * X1 * xr
        fr = 1.0;
        for i in 0..p {
            for j in 0..p {
                fr += xr[i] * x1.get(i, j) * xr[j];
            }
        }

        // w[r-q] in R corresponds to w[r-p] in 0-based indexing
        let y_pred_r = (0..p).map(|j| betar[j] * xr[j]).sum::<f64>();
        w[r - p] = (y_sorted[r] - y_pred_r) / fr.sqrt();
    }

    // ------------------------------------------------------------------------
    // Harvey-Collier statistic and p-value: Python's statsmodels variant
    //
    // Python's statsmodels.stats.diagnostic.linear_harvey_collier uses:
    //   rr = recursive_olsresiduals(res, skip=skip, alpha=0.95, order_by=order_by)
    //   rr[3] is rresid_standardized (N(0,sigma2) distributed)
    //   ttest_1samp(rr[3][3:], 0) - NOTE: always skips first 3 elements!
    //
    // This is different from R, which uses ALL recursive residuals.
    // ------------------------------------------------------------------------

    let n_rr = w.len();
    if n_rr < 4 {
        // Python needs at least 4 elements (skip 3, then at least 1 for the test)
        return Err(Error::InsufficientData {
            required: 4,
            available: n,
        });
    }

    // Sample variance of w (R's var())
    let mean_w = vec_mean(&w);
    let var_w = w.iter().map(|&v| (v - mean_w).powi(2)).sum::<f64>() / ((n_rr - 1) as f64);
    if !var_w.is_finite() || var_w <= 0.0 {
        return Err(Error::InvalidInput(
            "Invalid variance in Harvey-Collier test".to_string(),
        ));
    }

    // In Python: sigma = sqrt(var(w) * (n_rr - 1) / (n - p - 1))
    let df = (n - p - 1) as f64;
    let sigma = (var_w * ((n_rr - 1) as f64) / df).sqrt();

    // Scale recursive residuals (same as R)
    let resr: Vec<f64> = w.iter().map(|&v| v / sigma).collect();

    // PYTHON-SPECIFIC: Skip first element to match Python's rr[3][3:] behavior
    // Python's nobs-length array has valid values at indices skip..nobs-1.
    // Python's rr[3][3:] takes indices 3..nobs-1.
    // Our resr has n-p values corresponding to indices skip..nobs-1.
    // To match indices 3..nobs-1, we skip (3-skip) elements from our resr array.
    let skip_offset = 3usize.saturating_sub(p);
    let resr_test: Vec<f64> = resr.iter().skip(skip_offset).cloned().collect();

    let n_test = resr_test.len();
    if n_test < 2 {
        return Err(Error::InsufficientData {
            required: 5, // Need n >= p + 4 to have at least 2 elements after skipping 3
            available: n,
        });
    }

    // Compute mean and variance of the truncated resr
    let sum_resr = resr_test.iter().sum::<f64>();
    let mean_resr = sum_resr / (n_test as f64);
    let var_resr = resr_test
        .iter()
        .map(|&v| (v - mean_resr).powi(2))
        .sum::<f64>()
        / ((n_test - 1) as f64);

    if !var_resr.is_finite() || var_resr <= 0.0 {
        return Err(Error::InvalidInput(
            "Invalid scaled variance in Harvey-Collier test (Python)".to_string(),
        ));
    }

    // Python's t-statistic (from scipy.stats.ttest_1samp): t = mean / (std / sqrt(n))
    // where std = sqrt(var), so t = mean * sqrt(n) / sqrt(var) = sum / sqrt(n*var)
    // Note: The t-statistic preserves the sign of the mean!
    let harv = sum_resr / (n_test as f64).sqrt() / var_resr.sqrt();

    // Two-tailed p-value: 2 * pt(|harv|, df, lower.tail=FALSE)
    let p_value = two_tailed_p_value(harv.abs(), df);

    let alpha = 0.05;
    let passed = p_value > alpha;

    let (interpretation, guidance) = if passed {
        (
            format!(
                "p-value = {:.4} is greater than {:.2}. Cannot reject H0. No significant evidence of functional misspecification.",
                p_value, alpha
            ),
            "The linear model specification appears appropriate."
        )
    } else {
        (
            format!(
                "p-value = {:.4} is less than or equal to {:.2}. Reject H0. Significant evidence of functional misspecification.",
                p_value, alpha
            ),
            "Consider adding polynomial terms, transforming variables, or using non-linear modeling."
        )
    };

    Ok(DiagnosticTestResult {
        test_name: "Harvey-Collier Test for Linearity (Python variant)".to_string(),
        statistic: harv,
        p_value,
        passed,
        interpretation,
        guidance: guidance.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harvey_collier_insufficient_data() {
        // n=2, k=1 (p=2), need n > p+1 = 3, so n=2 is insufficient
        let y = vec![1.0, 2.0];
        let x = vec![1.0, 2.0];

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R);
        assert!(result.is_err());

        if let Err(Error::InsufficientData { required, available }) = result {
            assert_eq!(required, 4); // p + 2 = 2 + 2
            assert_eq!(available, 2);
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    #[test]
    fn test_harvey_collier_exact_minimum_data() {
        // n=4, k=1 (p=2), need n > p+1 = 3, so n=4 is exactly minimum
        // Add tiny noise for Harvey-Collier numerical stability
        let y = vec![1.001, 1.999, 3.002, 3.998];
        let x = vec![1.0, 2.0, 3.0, 4.0];

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.p_value > 0.0);
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_harvey_collier_linear_relationship() {
        // Linear relationship with small noise - Harvey-Collier needs residual variance
        let y: Vec<f64> = (1..=30).map(|i| {
            1.0 + 2.0 * i as f64 + 0.01 * ((i % 7) as f64 - 3.0) // Small deterministic noise
        }).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();
        // For perfectly linear relationship, p-value should be large
        assert!(result.p_value > 0.05);
        assert!(result.passed);
    }

    #[test]
    fn test_harvey_collier_quadratic_relationship() {
        // Quadratic relationship - should detect non-linearity (low p-value)
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + i as f64 + 0.1 * i as f64 * i as f64).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();
        // For quadratic relationship with linear model, p-value should be small
        assert!(result.p_value < 0.1);
    }

    #[test]
    fn test_harvey_collier_multiple_predictors() {
        // Multiple predictors with linear relationship + small noise
        // Use i and i^2 as predictors to avoid collinearity
        let y: Vec<f64> = (1..=30).map(|i| {
            1.0 + 2.0 * i as f64 + 0.3 * (i as f64) * (i as f64) + 0.01 * ((i % 5) as f64 - 2.0)
        }).collect();
        let x1: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let x2: Vec<f64> = (1..=30).map(|i| (i as f64) * (i as f64)).collect();

        let result = harvey_collier_test(&y, &[x1, x2], HarveyCollierMethod::R).unwrap();
        // For correctly specified linear model, p-value should be large
        assert!(result.p_value > 0.01);
    }

    #[test]
    fn test_harvey_collier_multiple_predictors_nonlinear() {
        // Multiple predictors with non-linear relationship
        // Use i and i^2 as predictors (not collinear)
        let y: Vec<f64> = (1..=30).map(|i| {
            1.0 + i as f64 + (i as f64 / 2.0) + 0.05 * i as f64 * i as f64
        }).collect();
        let x1: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let x2: Vec<f64> = (1..=30).map(|i| (i as f64) * (i as f64)).collect();

        let result = harvey_collier_test(&y, &[x1, x2], HarveyCollierMethod::R).unwrap();
        // Should detect some non-linearity
        assert!(result.p_value < 0.15);
    }

    #[test]
    fn test_harvey_collier_three_predictors() {
        // Three predictors - use uncorrelated predictors to avoid singular matrix
        let y: Vec<f64> = (1..=20).map(|i| {
            1.0 + 2.0 * i as f64 + 0.3 * (i as f64).powi(2) + 0.5 * ((i % 4) as f64)
        }).collect();
        let x1: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let x2: Vec<f64> = (1..=20).map(|i| (i as f64) * (i as f64)).collect();
        let x3: Vec<f64> = (1..=20).map(|i| (i % 4) as f64).collect();

        let result = harvey_collier_test(&y, &[x1, x2, x3], HarveyCollierMethod::R);
        assert!(result.is_ok());
    }

    #[test]
    fn test_harvey_collier_small_dataset() {
        // Minimum viable dataset (n=5, k=1, p=2, n-p-1=2)
        let y = vec![1.0, 2.5, 4.0, 5.5, 7.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.p_value > 0.0);
        assert!(result.statistic.is_finite());
    }

    #[test]
    fn test_harvey_collier_constant_y() {
        // Constant y value (edge case)
        let y = vec![5.0; 30];
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R);
        // This might error due to singular matrix
        assert!(result.is_err() || result.unwrap().p_value >= 0.0);
    }

    #[test]
    fn test_harvey_collier_interpretation_and_guidance() {
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + 2.0 * i as f64).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();

        // Check interpretation contains expected phrases
        assert!(result.interpretation.contains("p-value"));
        assert!(result.guidance.contains("appropriate") || result.guidance.contains("polynomial"));

        // Test name
        assert_eq!(result.test_name, "Harvey-Collier Test for Linearity");
    }

    #[test]
    fn test_harvey_collier_perfect_collinearity() {
        // Perfect collinearity (x2 = 2 * x1)
        let y: Vec<f64> = (1..=10).map(|i| 1.0 + 2.0 * i as f64).collect();
        let x1: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let x2: Vec<f64> = (1..=10).map(|i| 2.0 * i as f64).collect();

        let result = harvey_collier_test(&y, &[x1, x2], HarveyCollierMethod::R);
        // Should handle collinearity (might error or produce a result)
        match result {
            Ok(r) => {
                assert!(r.p_value >= 0.0);
            }
            Err(Error::SingularMatrix) => {
                // Expected for perfectly collinear data
            }
            Err(_) => {
                panic!("Unexpected error type");
            }
        }
    }

    #[test]
    fn test_harvey_collier_exponential_relationship() {
        // Exponential relationship - strong non-linearity
        let y: Vec<f64> = (0..30).map(|i| (1.0 + 0.1 * i as f64).exp()).collect();
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();
        // Should definitely detect non-linearity
        assert!(result.p_value < 0.05);
        assert!(!result.passed);
    }

    #[test]
    fn test_harvey_collier_logarithmic_relationship() {
        // Logarithmic relationship - non-linear
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + 2.0 * (i as f64).ln()).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();
        // Should detect non-linearity
        assert!(result.p_value < 0.15);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_harvey_collier_noisy_linear() {
        // Noisy linear relationship
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let y: Vec<f64> = (1..=50).map(|i| {
            1.0 + 2.0 * i as f64 + (rng.gen::<f64>() - 0.5) * 2.0
        }).collect();
        let x: Vec<f64> = (1..=50).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();
        // For noisy linear, p-value should be reasonable
        assert!(result.p_value > 0.0);
    }

    #[test]
    fn test_harvey_collier_sin_relationship() {
        // Sine wave - non-linear relationship
        // NOTE: Without sorting by fitted values (matching R's behavior), the test may not
        // always detect the non-linearity depending on the data order. We just verify the test runs.
        let y: Vec<f64> = (1..=40).map(|i| 5.0 + 2.0 * (0.2 * i as f64).sin()).collect();
        let x: Vec<f64> = (1..=40).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();
        // Just verify the test produces valid results
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.statistic.is_finite());
    }

    #[test]
    fn test_harvey_collier_single_predictor_minimum_obs() {
        // n=4, k=1 (p=2), n-p-1 = 1 (barely enough for variance calculation)
        let y = vec![1.0, 2.0, 4.0, 5.0];
        let x = vec![1.0, 2.0, 3.0, 4.0];

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R);
        assert!(result.is_ok());
    }

    #[test]
    fn test_harvey_collier_result_structure() {
        let y: Vec<f64> = (1..=20).map(|i| 1.0 + 2.0 * i as f64).collect();
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();

        // Verify result structure
        assert!(!result.test_name.is_empty());
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0);
        assert!(result.p_value <= 1.0);
        assert!(result.interpretation.contains("p-value"));
        assert!(!result.guidance.is_empty());
    }

    #[test]
    fn test_harvey_collier_large_dataset() {
        // Larger dataset to test scalability
        let y: Vec<f64> = (1..=200).map(|i| 1.0 + 2.0 * i as f64 + 0.001 * i as f64 * i as f64).collect();
        let x: Vec<f64> = (1..=200).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();
        // With large n, should detect the slight non-linearity
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_harvey_collier_negative_values() {
        // Data with negative values + small noise
        let y: Vec<f64> = (-10..=10).map(|i| 1.0 + 2.0 * i as f64 + 0.01 * ((i as f64) % 3.0 - 1.0)).collect();
        let x: Vec<f64> = (-10..=10).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();
        assert!(result.p_value > 0.01); // Should pass for linear relationship
    }

    #[test]
    fn test_harvey_collier_step_function() {
        // Step function - definitely non-linear
        // NOTE: Harvey-Collier is not very sensitive to step functions
        // R gets p ≈ 0.45 for this data, so we don't expect a low p-value
        let y: Vec<f64> = (1..=30).map(|i| if i < 15 { 5.0 } else { 10.0 }).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();
        // Harvey-Collier doesn't reliably detect step functions - just verify it runs
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
}
