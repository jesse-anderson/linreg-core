//! Weighted Least Squares (WLS) regression
//!
//! This module provides WLS regression using the weighted least squares solver
//! from the LOESS module. WLS is useful when:
//! - Observations have different precision/variances (heteroscedasticity)
//! - You want to incorporate robustness weights from a previous fit
//! - Certain observations should be given more influence
//!
//! The output format matches R's `lm()` function with weights, providing:
//! - Coefficient estimates with standard errors, t-values, and p-values
//! - F-statistic and p-value for overall model significance
//! - Residual standard error, R², adjusted R²

use crate::{
    core::{f_p_value, t_critical_quantile},
    distributions::student_t_cdf,
    error::{Error, Result},
    linalg::Matrix,
    serialization::types::ModelType,
    impl_serialization,
};
use serde::{Deserialize, Serialize};

/// WLS regression result
///
/// Contains the fitted coefficients and comprehensive model fit statistics
/// matching R's `summary(lm(y ~ x, weights=w))` output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WlsFit {
    // ============================================================
    // Coefficient Statistics (matching R's coefficients table)
    // ============================================================
    /// Coefficient values (including intercept as first element)
    pub coefficients: Vec<f64>,

    /// Standard errors of the coefficients
    pub standard_errors: Vec<f64>,

    /// t-statistics for coefficient significance tests
    pub t_statistics: Vec<f64>,

    /// Two-tailed p-values for coefficients
    pub p_values: Vec<f64>,

    /// Lower bounds of 95% confidence intervals for coefficients
    pub conf_int_lower: Vec<f64>,

    /// Upper bounds of 95% confidence intervals for coefficients
    pub conf_int_upper: Vec<f64>,

    // ============================================================
    // Model Fit Statistics
    // ============================================================
    /// R-squared (coefficient of determination)
    pub r_squared: f64,

    /// Adjusted R-squared
    pub adj_r_squared: f64,

    /// F-statistic for overall model significance
    pub f_statistic: f64,

    /// p-value for F-statistic
    pub f_p_value: f64,

    /// Residual standard error (sigma-hat estimate)
    pub residual_std_error: f64,

    /// Degrees of freedom for residuals
    pub df_residuals: isize,

    /// Degrees of freedom for the model
    pub df_model: isize,

    // ============================================================
    // Predictions and Diagnostics
    // ============================================================
    /// Fitted values (predicted values)
    pub fitted_values: Vec<f64>,

    /// Residuals (y - ŷ)
    pub residuals: Vec<f64>,

    /// Mean squared error
    pub mse: f64,

    /// Root mean squared error
    pub rmse: f64,

    /// Mean absolute error
    pub mae: f64,

    // ============================================================
    // Sample Information
    // ============================================================
    /// Number of observations
    pub n: usize,

    /// Number of predictors (excluding intercept)
    pub k: usize,
}

/// Perform weighted least squares regression
///
/// Fits a linear model using weighted least squares, where each observation
/// can have a different weight. The output format matches R's `lm()` function
/// with the `weights` parameter, providing comprehensive statistics including
/// coefficient standard errors, t-statistics, p-values, and F-test.
///
/// # Arguments
///
/// * `y` - Response variable (n observations)
/// * `x_vars` - Predictor variables (p vectors, each of length n)
/// * `weights` - Observation weights (n weights, must be non-negative)
///
/// # Returns
///
/// `WlsFit` containing coefficients, fitted values, and comprehensive fit statistics
///
/// # Errors
///
/// - `Error::InsufficientData` if n <= k + 1
/// - `Error::InvalidInput` if weights contain negative values or dimensions don't match
/// - `Error::SingularMatrix` if the design matrix is singular even with weighting
///
/// # Example
///
/// ```
/// use linreg_core::weighted_regression::{wls_regression, WlsFit};
///
/// let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // Equal weights = OLS
///
/// let fit: WlsFit = wls_regression(&y, &[x1], &weights)?;
///
/// // Access coefficients and statistics
/// println!("Intercept: {} (SE: {}, t: {}, p: {})",
///     fit.coefficients[0],
///     fit.standard_errors[0],
///     fit.t_statistics[0],
///     fit.p_values[0]
/// );
/// println!("F-statistic: {} (p: {})", fit.f_statistic, fit.f_p_value);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn wls_regression(
    y: &[f64],
    x_vars: &[Vec<f64>],
    weights: &[f64],
) -> Result<WlsFit> {
    let n = y.len();
    let k = x_vars.len();

    // Validate minimum sample size
    if n <= k + 1 {
        return Err(Error::InsufficientData {
            required: k + 2,
            available: n,
        });
    }

    // Validate dimensions
    for (i, x_var) in x_vars.iter().enumerate() {
        if x_var.len() != n {
            return Err(Error::InvalidInput(format!(
                "x[{}] has {} elements, expected {}",
                i,
                x_var.len(),
                n
            )));
        }
    }

    if weights.len() != n {
        return Err(Error::InvalidInput(format!(
            "weights has {} elements, expected {}",
            weights.len(),
            n
        )));
    }

    // Check for negative weights
    for (i, &w) in weights.iter().enumerate() {
        if w < 0.0 {
            return Err(Error::InvalidInput(format!(
                "weights[{}] is negative ({}), weights must be non-negative",
                i, w
            )));
        }
    }

    // Check for zero total weight
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum <= 0.0 {
        return Err(Error::InvalidInput(
            "Sum of weights is zero or negative".to_string()
        ));
    }

    // Build design matrix: include intercept column
    let mut x_data = Vec::with_capacity(n * (k + 1));
    for i in 0..n {
        x_data.push(1.0); // Intercept
        for j in 0..k {
            x_data.push(x_vars[j][i]);
        }
    }
    let x = Matrix::new(n, k + 1, x_data);

    // Call the WLS solver with decomposition info (single decomposition, no duplicate QR)
    let decomp = crate::loess::wls::weighted_least_squares_with_decomposition(&x, y, weights)?;
    let coefficients = decomp.coefficients;

    // Compute fitted values
    let fitted_values: Vec<f64> = (0..n)
        .map(|i| {
            let mut y_hat = coefficients[0]; // Intercept
            for j in 0..k {
                y_hat += coefficients[j + 1] * x_vars[j][i];
            }
            y_hat
        })
        .collect();

    // Compute residuals
    let residuals: Vec<f64> = y.iter().zip(fitted_values.iter())
        .map(|(yi, y_hat)| yi - y_hat)
        .collect();

    // ============================================================
    // Compute Degrees of Freedom
    // ============================================================
    let p = k + 1; // Number of coefficients (including intercept)
    let df_residuals = n as isize - p as isize;
    let df_model = k as isize;

    if df_residuals <= 0 {
        return Err(Error::InsufficientData {
            required: p + 1,
            available: n,
        });
    }

    // ============================================================
    // Compute MSE and Residual Standard Error
    // ============================================================
    // RSS = sum of squared residuals
    let rss: f64 = residuals.iter().map(|r| r * r).sum();

    // MSE (using n - p for unbiased estimate, like R)
    let mse = rss / df_residuals as f64;

    // Residual standard error (R's sigma-hat)
    let residual_std_error = mse.sqrt();

    // ============================================================
    // Compute R-squared and Adjusted R-squared
    // ============================================================
    let ss_tot: f64 = {
        let y_mean = y.iter().sum::<f64>() / n as f64;
        y.iter().map(|yi| (yi - y_mean).powi(2)).sum()
    };
    let r_squared = if ss_tot > 0.0 {
        1.0 - (rss / ss_tot)
    } else {
        0.0
    };

    let adj_r_squared = if df_residuals > 1 {
        1.0 - ((1.0 - r_squared) * (n - 1) as f64 / df_residuals as f64)
    } else {
        r_squared
    };

    // ============================================================
    // Compute Covariance Matrix of Coefficients
    // ============================================================
    // Uses decomposition info from the solver (no duplicate QR!)
    let cov = if let Some(ref r_inv) = decomp.r_inv {
        // QR path: Cov = MSE * S^-1 * R^-1 * (R^-1)' * S^-1
        compute_covariance_from_qr(r_inv, &decomp.column_scales, mse, p)
    } else if let Some((ref v, ref singular_values)) = decomp.svd_components {
        // SVD path: Cov = MSE * V * diag(1/σᵢ²) * V'
        compute_covariance_from_svd(v, singular_values, &decomp.column_scales, mse, p)
    } else {
        return Err(Error::SingularMatrix);
    };

    // ============================================================
    // Extract Standard Errors (diagonal of covariance matrix)
    // ============================================================
    let mut standard_errors = Vec::with_capacity(p);
    for i in 0..p {
        let se = cov.get(i, i).sqrt();
        standard_errors.push(se);
    }

    // ============================================================
    // Compute t-statistics and p-values for coefficients
    // ============================================================
    let mut t_statistics = Vec::with_capacity(p);
    let mut p_values = Vec::with_capacity(p);

    for i in 0..p {
        let t = coefficients[i] / standard_errors[i];
        t_statistics.push(t);

        // Two-tailed p-value using Student's t-distribution
        let p = 2.0 * (1.0 - student_t_cdf(t.abs(), df_residuals as f64));
        p_values.push(p);
    }

    // ============================================================
    // Compute 95% Confidence Intervals
    // ============================================================
    let alpha = 0.05;
    let t_critical = t_critical_quantile(df_residuals as f64, alpha);

    let conf_int_lower: Vec<f64> = coefficients
        .iter()
        .zip(&standard_errors)
        .map(|(&b, &se)| b - t_critical * se)
        .collect();
    let conf_int_upper: Vec<f64> = coefficients
        .iter()
        .zip(&standard_errors)
        .map(|(&b, &se)| b + t_critical * se)
        .collect();

    // ============================================================
    // Compute F-statistic and p-value for overall model
    // ============================================================
    // F = ((TSS - RSS) / k) / (RSS / (n - k - 1))
    // where TSS is total sum of squares, RSS is residual sum of squares,
    // and k is the number of predictors (excluding intercept)
    let f_statistic = if ss_tot > rss && k > 0 {
        ((ss_tot - rss) / k as f64) / (rss / df_residuals as f64)
    } else {
        0.0
    };

    let f_p_value = if f_statistic > 0.0 {
        f_p_value(f_statistic, k as f64, df_residuals as f64)
    } else {
        1.0
    };

    // ============================================================
    // Additional Error Metrics
    // ============================================================
    let rmse = mse.sqrt();
    let mae = residuals.iter().map(|r| r.abs()).sum::<f64>() / n as f64;

    Ok(WlsFit {
        coefficients,
        standard_errors,
        t_statistics,
        p_values,
        conf_int_lower,
        conf_int_upper,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_p_value,
        residual_std_error,
        df_residuals,
        df_model,
        fitted_values,
        residuals,
        mse,
        rmse,
        mae,
        n,
        k,
    })
}

/// Compute covariance matrix from QR decomposition
///
/// Formula: Cov(β_orig)_ij = MSE * Σ_l(R^-1_il * R^-1_jl) / (scales\[i\] * scales\[j\])
fn compute_covariance_from_qr(
    r_inv: &Matrix,
    column_scales: &[f64],
    mse: f64,
    p: usize,
) -> Matrix {
    let mut cov = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            let mut sum = 0.0;
            for l in 0..p {
                sum += r_inv.get(i, l) * r_inv.get(j, l);
            }
            cov.set(i, j, mse * sum / (column_scales[i] * column_scales[j]));
        }
    }
    cov
}

/// Compute covariance matrix from SVD decomposition
///
/// Formula: Cov(β) = MSE * V * diag(1/σᵢ²) * V'
/// Then compensate for equilibration: divide by scales\[i\] * scales\[j\]
fn compute_covariance_from_svd(
    v: &Matrix,
    singular_values: &[f64],
    column_scales: &[f64],
    mse: f64,
    p: usize,
) -> Matrix {
    // Use same tolerance as svd_solve in linalg.rs: sigma[0] * 100 * epsilon
    let max_sigma = singular_values.first().copied().unwrap_or(0.0);
    let tol = if max_sigma > 0.0 {
        max_sigma * 100.0 * f64::EPSILON
    } else {
        f64::EPSILON
    };

    let mut cov = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            let mut sum = 0.0;
            for l in 0..singular_values.len().min(p) {
                if singular_values[l] > tol {
                    let inv_sigma_sq = 1.0 / (singular_values[l] * singular_values[l]);
                    sum += v.get(i, l) * v.get(j, l) * inv_sigma_sq;
                }
            }
            cov.set(i, j, mse * sum / (column_scales[i] * column_scales[j]));
        }
    }
    cov
}

// ============================================================================
// Model Serialization Traits
// ============================================================================

// Generate ModelSave and ModelLoad implementations using macro
impl_serialization!(WlsFit, ModelType::WLS, "WLS");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wls_equal_weights_matches_ols() {
        // WLS with equal weights should match OLS
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0; 5]; // Equal weights

        let fit = wls_regression(&y, &[x], &weights).unwrap();

        // For perfect linear y = x, intercept should be ~0, slope ~1
        assert!((fit.coefficients[0] - 0.0).abs() < 1e-10);
        assert!((fit.coefficients[1] - 1.0).abs() < 1e-10);
        assert_eq!(fit.k, 1);
        assert_eq!(fit.n, 5);

        // Check that statistics are computed
        assert!(fit.standard_errors.len() == 2);
        assert!(fit.t_statistics.len() == 2);
        assert!(fit.p_values.len() == 2);
        assert!(fit.f_statistic > 0.0);
        assert!(fit.f_p_value < 0.05); // Should be significant for perfect fit
    }

    #[test]
    fn test_wls_with_weighted_data() {
        // Create data where one point is an outlier
        let y = vec![2.0, 4.0, 6.0, 8.0, 100.0]; // Last point is outlier
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // With low weight on the outlier, the fit should ignore it
        let weights_low = vec![1.0, 1.0, 1.0, 1.0, 0.01];
        let fit_low = wls_regression(&y, &[x.clone()], &weights_low).unwrap();

        // With high weight on the outlier, the fit should be pulled toward it
        let weights_high = vec![1.0, 1.0, 1.0, 1.0, 10.0];
        let fit_high = wls_regression(&y, &[x], &weights_high).unwrap();

        // The low-weight fit should have slope close to 2 (from first 4 points)
        // The high-weight fit should have a much larger slope
        assert!(fit_low.coefficients[1] < fit_high.coefficients[1]);
    }

    #[test]
    fn test_wls_negative_weight_error() {
        let y = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, -1.0, 1.0]; // Negative weight

        let result = wls_regression(&y, &[x], &weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_wls_multiple_predictors() {
        // Use non-collinear predictors (x2 is not a linear function of x1)
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![1.0, 4.0, 2.0, 5.0, 3.0];  // Independent of x1
        let weights = vec![1.0; 5];

        let fit = wls_regression(&y, &[x1, x2], &weights).unwrap();

        assert_eq!(fit.k, 2); // Two predictors
        assert_eq!(fit.coefficients.len(), 3); // Intercept + 2 slopes
        assert_eq!(fit.fitted_values.len(), 5);
        assert_eq!(fit.standard_errors.len(), 3);
        assert_eq!(fit.t_statistics.len(), 3);
        assert_eq!(fit.p_values.len(), 3);
    }

    #[test]
    fn test_wls_insufficient_data() {
        let y = vec![1.0, 2.0];
        let x1 = vec![1.0, 2.0];
        let x2 = vec![0.5, 1.0]; // Second predictor
        let weights = vec![1.0, 1.0];

        // n=2, k=2, need k+2=4 observations
        let result = wls_regression(&y, &[x1, x2], &weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_wls_statistics_completeness() {
        // Verify all statistics are computed
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0; 5];

        let fit = wls_regression(&y, &[x], &weights).unwrap();

        // Check all fields are populated
        assert_eq!(fit.coefficients.len(), 2);
        assert_eq!(fit.standard_errors.len(), 2);
        assert_eq!(fit.t_statistics.len(), 2);
        assert_eq!(fit.p_values.len(), 2);
        assert!(fit.r_squared >= 0.0 && fit.r_squared <= 1.0);
        assert!(fit.adj_r_squared >= 0.0 && fit.adj_r_squared <= 1.0);
        assert!(fit.f_statistic >= 0.0);
        assert!(fit.f_p_value >= 0.0 && fit.f_p_value <= 1.0);
        assert!(fit.residual_std_error >= 0.0);
        assert_eq!(fit.df_residuals, 3); // n=5, p=2, df=5-2=3
        assert_eq!(fit.df_model, 1);
        assert_eq!(fit.fitted_values.len(), 5);
        assert_eq!(fit.residuals.len(), 5);
        assert!(fit.mse >= 0.0);
        assert!(fit.rmse >= 0.0);
        assert!(fit.mae >= 0.0);
        assert_eq!(fit.n, 5);
        assert_eq!(fit.k, 1);
    }

    #[test]
    fn test_wls_zero_sum_weights_error() {
        let y = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 2.0, 3.0];
        let weights = vec![0.0, 0.0, 0.0]; // All zero

        let result = wls_regression(&y, &[x], &weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_wls_svd_fallback_computes_standard_errors() {
        // Near-collinear predictors that trigger SVD fallback
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // x2 = 2*x1 (perfectly collinear)
        let weights = vec![1.0; 5];

        let result = wls_regression(&y, &[x1, x2], &weights);
        // Should either succeed with finite SEs or fail gracefully
        // Previously this would succeed for coefficients but fail for SEs
        match result {
            Ok(fit) => {
                // If it succeeds, SEs should be finite (from SVD covariance path)
                for se in &fit.standard_errors {
                    assert!(se.is_finite(), "Standard error should be finite, got {}", se);
                }
            }
            Err(_) => {
                // Graceful failure is also acceptable for perfectly collinear data
            }
        }
    }
}
