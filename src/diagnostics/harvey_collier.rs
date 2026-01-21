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

use crate::error::{Error, Result};
use crate::linalg::{Matrix, vec_mean};
use super::types::DiagnosticTestResult;
use super::helpers::{fit_ols, two_tailed_p_value};

/// Performs the Harvey-Collier test for linearity (functional form).
///
/// This test checks for functional form misspecification by computing **recursive residuals**
/// from an expanding-window OLS fit and then performing a **t-test that the mean of the
/// (scaled) recursive residuals is zero**. Systematic deviation from zero provides evidence
/// against linear specification.
///
/// **Ordering matters:** the test is applied after ordering observations (in this tool's
/// implementation, by fitted values unless an explicit ordering is provided upstream).
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
#[allow(clippy::needless_range_loop)]
pub fn harvey_collier_test(
    y: &[f64],
    x_vars: &[Vec<f64>],
) -> Result<DiagnosticTestResult> {
    let n = y.len();
    let k = x_vars.len();   // number of *non-intercept* regressors
    let p = k + 1;          // number of columns in design matrix, incl intercept (R's "k")

    // Validate inputs
    // Need at least p + 2 observations so that:
    // - recursive residuals length = n - p >= 2 (variance defined)
    // - df = n - p - 1 >= 1
    if n <= p + 1 {
        return Err(Error::InsufficientData { required: p + 2, available: n });
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

    // Fit full model to get fitted values and order observations by fitted (your chosen behavior)
    // NOTE: R's harvtest orders by `order.by` if provided; otherwise assumes already-ordered.
    // Here we mimic "order.by = fitted" behavior.
    let x_full = Matrix::new(n, p, x_data.clone());
    let beta_full = fit_ols(y, &x_full)?;
    let fitted = x_full.mul_vec(&beta_full);

    // Sort indices by fitted values
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        fitted[a].partial_cmp(&fitted[b]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Reorder y and X by sorted indices
    let mut y_sorted = vec![0.0; n];
    let mut x_sorted = vec![0.0; n * p];
    for (i, &idx) in sorted_indices.iter().enumerate() {
        y_sorted[i] = y[idx];
        for j in 0..p {
            x_sorted[i * p + j] = x_data[idx * p + j];
        }
    }

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
        return Err(Error::InsufficientData { required: 2, available: n_rr });
    }

    // Sample variance of w (R's var())
    let mean_w = vec_mean(&w);
    let var_w = w.iter().map(|&v| (v - mean_w).powi(2)).sum::<f64>() / ((n_rr - 1) as f64);
    if !var_w.is_finite() || var_w <= 0.0 {
        return Err(Error::InvalidInput("Invalid variance in Harvey-Collier test".to_string()));
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
        return Err(Error::InvalidInput("Invalid scaled variance in Harvey-Collier test".to_string()));
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
