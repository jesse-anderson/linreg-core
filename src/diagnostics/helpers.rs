// ============================================================================
// Diagnostic Test Helper Functions
// ============================================================================
//
// Shared helper functions used across multiple diagnostic tests.

use crate::error::{Error, Result};
use crate::linalg::{Matrix, vec_sub};
use crate::distributions::{student_t_cdf, fisher_snedecor_cdf, chi_squared_survival};

/// Computes a two-tailed p-value from a t-statistic.
///
/// This function calculates the probability of observing a t-statistic as extreme
/// as the one provided, assuming a two-tailed test. It uses the Student's t
/// cumulative distribution function.
///
/// # Arguments
///
/// * `t` - The t-statistic value
/// * `df` - Degrees of freedom (must be positive)
///
/// # Returns
///
/// The two-tailed p-value in the range `[0, 2]`. For extreme values (`|t| > 100`),
/// returns `0.0` to avoid numerical underflow.
///
/// # Examples
///
/// ```
/// use linreg_core::diagnostics::two_tailed_p_value;
///
/// // t = 2.0 with 10 degrees of freedom
/// let p = two_tailed_p_value(2.0, 10.0);
/// assert!(p > 0.05 && p < 0.10);
/// ```
pub fn two_tailed_p_value(t: f64, df: f64) -> f64 {
    if t.abs() > 100.0 {
        return 0.0;
    }

    let cdf = student_t_cdf(t, df);
    if t >= 0.0 { 2.0 * (1.0 - cdf) } else { 2.0 * cdf }
}

/// Computes a p-value from an F-statistic.
///
/// Calculates the upper-tail probability of observing an F-statistic as extreme
/// as the one provided, using the Fisher-Snedecor (F) distribution.
///
/// # Arguments
///
/// * `f_stat` - The F-statistic value (must be non-negative)
/// * `df1` - Numerator degrees of freedom
/// * `df2` - Denominator degrees of freedom
///
/// # Returns
///
/// The p-value (upper tail probability) in the range `[0, 1]`. Returns `1.0` for
/// non-positive F-statistics.
///
/// # Examples
///
/// ```
/// use linreg_core::diagnostics::f_p_value;
///
/// // F = 5.0 with df1 = 2, df2 = 10
/// let p = f_p_value(5.0, 2.0, 10.0);
/// assert!(p > 0.0 && p < 0.05);
/// ```
pub fn f_p_value(f_stat: f64, df1: f64, df2: f64) -> f64 {
    if f_stat <= 0.0 {
        return 1.0;
    }
    1.0 - fisher_snedecor_cdf(f_stat, df1, df2)
}

/// Computes a p-value from a chi-squared statistic (upper tail).
///
/// Calculates the probability of observing a chi-squared statistic as extreme
/// as the one provided, using the chi-squared distribution.
///
/// # Arguments
///
/// * `stat` - The chi-squared statistic value (must be non-negative)
/// * `df` - Degrees of freedom
///
/// # Returns
///
/// The upper-tail p-value in the range `[0, 1]`.
///
/// # Examples
///
/// ```ignore
/// use linreg_core::diagnostics::helpers::chi_squared_p_value;
///
/// // chi-squared = 10.0 with df = 5
/// let p = chi_squared_p_value(10.0, 5.0);
/// assert!(p > 0.0 && p < 1.0);
/// ```
pub fn chi_squared_p_value(stat: f64, df: f64) -> f64 {
    chi_squared_survival(stat, df)
}

/// Computes the residual sum of squares (RSS) from a fitted model.
///
/// The RSS is the sum of squared differences between observed and predicted
/// values: `RSS = Σ(yᵢ - ŷᵢ)²`, where `ŷᵢ = Xᵢβ`.
///
/// This is a measure of model fit - lower values indicate better fit. The RSS
/// is used in many diagnostic tests including the Rainbow test and likelihood
/// ratio tests.
///
/// # Arguments
///
/// * `y` - Observed response values (n observations)
/// * `x` - Design matrix (n × p)
/// * `beta` - Coefficient vector (p elements)
///
/// # Returns
///
/// The residual sum of squares as a non-negative value.
///
/// # Examples
///
/// ```ignore
/// use linreg_core::diagnostics::helpers::compute_rss;
/// use linreg_core::linalg::Matrix;
///
/// let y = vec![2.0, 4.0, 6.0];
/// let x = Matrix::new(3, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
/// let beta = vec![0.0, 2.0];  // y = 2*x
/// let rss = compute_rss(&y, &x, &beta).unwrap();
/// assert_eq!(rss, 0.0);  // Perfect fit
/// ```
pub fn compute_rss(y: &[f64], x: &Matrix, beta: &[f64]) -> Result<f64> {
    // predictions = x * beta
    let predictions = x.mul_vec(beta);
    let residuals = vec_sub(y, &predictions);
    Ok(residuals.iter().map(|&r| r * r).sum())
}

/// Fits an OLS regression model and returns the coefficient estimates.
///
/// This function provides a robust OLS fitting procedure that first attempts
/// standard QR decomposition, then falls back to ridge regression if numerical
/// instability is detected (e.g., due to multicollinearity).
///
/// The ridge fallback uses a very small regularization parameter (λ = 0.0001)
/// to maintain numerical stability while minimizing distortion of the estimates.
///
/// # Arguments
///
/// * `y` - Response variable (n observations)
/// * `x` - Design matrix (n × p, should include intercept column if needed)
///
/// # Returns
///
/// A vector of coefficient estimates (p elements).
///
/// # Errors
///
/// * [`Error::SingularMatrix`] - if the design matrix is singular and ridge
///   regression also fails
///
/// # Examples
///
/// ```ignore
/// use linreg_core::diagnostics::helpers::fit_ols;
/// use linreg_core::linalg::Matrix;
///
/// let y = vec![2.1, 4.0, 5.9];
/// let x = Matrix::new(3, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
/// let beta = fit_ols(&y, &x).unwrap();
/// assert_eq!(beta.len(), 2);  // Intercept and slope
/// ```
pub fn fit_ols(y: &[f64], x: &Matrix) -> Result<Vec<f64>> {
    // First try standard QR decomposition OLS
    let result = try_fit_ols_qr(y, x);
    if result.is_ok() {
        return result;
    }

    // If QR fails due to multicollinearity, use ridge regression
    // Use a very small lambda to minimize distortion while maintaining stability
    fit_ols_ridge(y, x, 0.0001)
}

/// Standard QR decomposition OLS solver.
///
/// Solves the normal equations using QR decomposition: `Xβ = y`. This is the
/// preferred method for OLS estimation due to its numerical stability.
///
/// The algorithm computes `X = QR` where Q is orthogonal and R is upper
/// triangular, then solves `Rβ = Qᵀy` via back-substitution.
///
/// # Arguments
///
/// * `y` - Response variable (n observations)
/// * `x` - Design matrix (n × p)
///
/// # Returns
///
/// A vector of coefficient estimates (p elements).
///
/// # Errors
///
/// * [`Error::SingularMatrix`] - if the design matrix is singular (p > n or
///   R is not invertible)
fn try_fit_ols_qr(y: &[f64], x: &Matrix) -> Result<Vec<f64>> {
    let p = x.cols;
    let n = x.rows;

    // When p > n, we have an underdetermined system (more predictors than observations)
    // Fall back to ridge regression for numerical stability
    if p > n {
        return Err(Error::SingularMatrix);
    }

    let (q, r) = x.qr();

    // Q^T * y
    let qty = q.transpose().mul_vec(y);

    // Take first p elements
    let rhs_vec = qty[0..p].to_vec();
    let rhs_mat = Matrix::new(p, 1, rhs_vec);

    // Extract upper triangle of R
    let mut r_upper = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            r_upper.set(i, j, r.get(i, j));
        }
    }

    match r_upper.invert_upper_triangular() {
        Some(r_inv) => Ok(r_inv.matmul(&rhs_mat).data),
        None => Err(Error::SingularMatrix),
    }
}

/// Ridge regression fallback for multicollinear data.
///
/// Solves the ridge regression problem: `(X'X + λI)β = X'y`. This adds a small
/// positive constant to the diagonal of `X'X`, ensuring invertibility even when
/// the design matrix is rank-deficient.
///
/// Ridge regression is used as a fallback when standard QR decomposition fails
/// due to multicollinearity or numerical singularity.
///
/// # Arguments
///
/// * `y` - Response variable (n observations)
/// * `x` - Design matrix (n × p)
/// * `lambda` - Regularization parameter (small positive value, e.g., 0.0001)
///
/// # Returns
///
/// A vector of ridge-regularized coefficient estimates (p elements).
///
/// # Errors
///
/// * [`Error::SingularMatrix`] - if the ridge-adjusted matrix is still singular
fn fit_ols_ridge(y: &[f64], x: &Matrix, lambda: f64) -> Result<Vec<f64>> {
    let p = x.cols;

    // Solve: (X'X + lambda*I) * beta = X'y
    let xt = x.transpose();
    let xtx = xt.matmul(x);

    // Add ridge to diagonal
    let mut xtx_ridge_data = xtx.data.clone();
    for i in 0..p {
        xtx_ridge_data[i * p + i] += lambda;
    }
    let xtx_ridge = Matrix::new(p, p, xtx_ridge_data);

    // X'y
    let xty = xt.mul_vec(y);

    // Invert and solve
    let xtx_inv = xtx_ridge.invert().ok_or(Error::SingularMatrix)?;
    let beta_mat = xtx_inv.mul_vec(&xty);
    Ok(beta_mat)
}
