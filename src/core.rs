//! Core OLS regression implementation.
//!
//! This module provides the main Ordinary Least Squares regression functionality
//! that can be used directly in Rust code. Functions accept native Rust slices
//! and return Result types for proper error handling.
//!
//! # Example
//!
//! ```
//! # use linreg_core::core::ols_regression;
//! # use linreg_core::Error;
//! let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let x2 = vec![2.0, 3.0, 3.5, 4.0, 4.5, 5.0];
//! let names = vec![
//!     "Intercept".to_string(),
//!     "X1".to_string(),
//!     "X2".to_string(),
//! ];
//!
//! let result = ols_regression(&y, &[x1, x2], &names)?;
//! # Ok::<(), Error>(())
//! ```

use crate::distributions::{fisher_snedecor_cdf, student_t_cdf, student_t_inverse_cdf};
use crate::error::{Error, Result};
use crate::linalg::{vec_dot, vec_mean, vec_sub, Matrix};
use serde::Serialize;

// ============================================================================
// Numerical Constants
// ============================================================================

/// Minimum threshold for standardized residual denominator to avoid division by zero.
/// When (1 - leverage) is very small, the observation has extremely high leverage
/// and standardized residuals may be unreliable.
const MIN_LEVERAGE_DENOM: f64 = 1e-10;

// ============================================================================
// Result Types
// ============================================================================
//
// Structs containing the output of regression computations.

/// Result of VIF (Variance Inflation Factor) calculation.
///
/// VIF measures how much the variance of an estimated regression coefficient
/// increases due to multicollinearity among the predictors.
///
/// # Fields
///
/// * `variable` - Name of the predictor variable
/// * `vif` - Variance Inflation Factor (VIF > 10 indicates high multicollinearity)
/// * `rsquared` - R-squared from regressing this predictor on all others
/// * `interpretation` - Human-readable interpretation of the VIF value
///
/// # Example
///
/// ```
/// # use linreg_core::core::VifResult;
/// let vif = VifResult {
///     variable: "X1".to_string(),
///     vif: 2.5,
///     rsquared: 0.6,
///     interpretation: "Low multicollinearity".to_string(),
/// };
/// assert_eq!(vif.variable, "X1");
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct VifResult {
    /// Name of the predictor variable
    pub variable: String,
    /// Variance Inflation Factor (VIF > 10 indicates high multicollinearity)
    pub vif: f64,
    /// R-squared from regressing this predictor on all others
    pub rsquared: f64,
    /// Human-readable interpretation of the VIF value
    pub interpretation: String,
}

/// Complete output from OLS regression.
///
/// Contains all coefficients, statistics, diagnostics, and residuals from
/// an Ordinary Least Squares regression.
///
/// # Example
///
/// ```
/// # use linreg_core::core::ols_regression;
/// let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let names = vec!["Intercept".to_string(), "X1".to_string()];
///
/// let result = ols_regression(&y, &[x1], &names).unwrap();
/// assert!(result.r_squared > 0.0);
/// assert_eq!(result.coefficients.len(), 2); // intercept + 1 predictor
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct RegressionOutput {
    /// Regression coefficients (including intercept)
    pub coefficients: Vec<f64>,
    /// Standard errors of coefficients
    pub std_errors: Vec<f64>,
    /// t-statistics for coefficient significance tests
    pub t_stats: Vec<f64>,
    /// Two-tailed p-values for coefficients
    pub p_values: Vec<f64>,
    /// Lower bounds of 95% confidence intervals
    pub conf_int_lower: Vec<f64>,
    /// Upper bounds of 95% confidence intervals
    pub conf_int_upper: Vec<f64>,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Adjusted R-squared (accounts for number of predictors)
    pub adj_r_squared: f64,
    /// F-statistic for overall model significance
    pub f_statistic: f64,
    /// P-value for F-statistic
    pub f_p_value: f64,
    /// Mean squared error of residuals
    pub mse: f64,
    /// Root mean squared error (prediction error in original units)
    pub rmse: f64,
    /// Mean absolute error of residuals
    pub mae: f64,
    /// Standard error of the regression (residual standard deviation)
    pub std_error: f64,
    /// Raw residuals (observed - predicted)
    pub residuals: Vec<f64>,
    /// Standardized residuals (accounting for leverage)
    pub standardized_residuals: Vec<f64>,
    /// Fitted/predicted values
    pub predictions: Vec<f64>,
    /// Leverage values for each observation (diagonal of hat matrix)
    pub leverage: Vec<f64>,
    /// Variance Inflation Factors for detecting multicollinearity
    pub vif: Vec<VifResult>,
    /// Number of observations
    pub n: usize,
    /// Number of predictor variables (excluding intercept)
    pub k: usize,
    /// Degrees of freedom for residuals (n - k - 1)
    pub df: usize,
    /// Names of variables (including intercept)
    pub variable_names: Vec<String>,
    /// Log-likelihood of the model (useful for AIC/BIC calculation and model comparison)
    pub log_likelihood: f64,
    /// Akaike Information Criterion (lower = better model, penalizes complexity)
    pub aic: f64,
    /// Bayesian Information Criterion (lower = better model, penalizes complexity more heavily than AIC)
    pub bic: f64,
}

// ============================================================================
// Statistical Helper Functions
// ============================================================================
//
// Utility functions for computing p-values, critical values, and leverage.

/// Computes a two-tailed p-value from a t-statistic.
///
/// Uses the Student's t-distribution CDF to calculate the probability
/// of observing a t-statistic as extreme as the one provided.
///
/// # Arguments
///
/// * `t` - The t-statistic value
/// * `df` - Degrees of freedom
///
/// # Example
///
/// ```
/// # use linreg_core::core::two_tailed_p_value;
/// let p = two_tailed_p_value(2.0, 20.0);
/// assert!(p > 0.0 && p < 0.1);
/// ```
pub fn two_tailed_p_value(t: f64, df: f64) -> f64 {
    if t.abs() > 100.0 {
        return 0.0;
    }

    let cdf = student_t_cdf(t, df);
    if t >= 0.0 {
        2.0 * (1.0 - cdf)
    } else {
        2.0 * cdf
    }
}

/// Computes the critical t-value for a given significance level and degrees of freedom.
///
/// Returns the t-value such that the area under the t-distribution curve
/// to the right of it equals alpha/2 (two-tailed test).
///
/// # Arguments
///
/// * `df` - Degrees of freedom
/// * `alpha` - Significance level (typically 0.05 for 95% confidence)
///
/// # Example
///
/// ```
/// # use linreg_core::core::t_critical_quantile;
/// let t_crit = t_critical_quantile(20.0, 0.05);
/// assert!(t_crit > 2.0); // approximately 2.086 for df=20, alpha=0.05
/// ```
pub fn t_critical_quantile(df: f64, alpha: f64) -> f64 {
    let p = 1.0 - alpha / 2.0;
    student_t_inverse_cdf(p, df)
}

/// Computes a p-value from an F-statistic.
///
/// Uses the F-distribution CDF to calculate the probability of observing
/// an F-statistic as extreme as the one provided.
///
/// # Arguments
///
/// * `f_stat` - The F-statistic value
/// * `df1` - Numerator degrees of freedom
/// * `df2` - Denominator degrees of freedom
///
/// # Example
///
/// ```
/// # use linreg_core::core::f_p_value;
/// let p = f_p_value(5.0, 2.0, 20.0);
/// assert!(p > 0.0 && p < 0.05);
/// ```
pub fn f_p_value(f_stat: f64, df1: f64, df2: f64) -> f64 {
    if f_stat <= 0.0 {
        return 1.0;
    }
    1.0 - fisher_snedecor_cdf(f_stat, df1, df2)
}

/// Computes leverage values from the design matrix and its inverse.
///
/// Leverage measures how far an observation's predictor values are from
/// the center of the predictor space. High leverage points can have
/// disproportionate influence on the regression results.
///
/// # Arguments
///
/// * `x` - Design matrix (including intercept column)
/// * `xtx_inv` - Inverse of X'X matrix
///
/// # Example
///
/// ```
/// # use linreg_core::core::compute_leverage;
/// # use linreg_core::linalg::Matrix;
/// // Design matrix with intercept: [[1, 1], [1, 2], [1, 3]]
/// let x = Matrix::new(3, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
/// let xtx = x.transpose().matmul(&x);
/// let xtx_inv = xtx.invert().unwrap();
///
/// let leverage = compute_leverage(&x, &xtx_inv);
/// assert_eq!(leverage.len(), 3);
/// // Leverage values should sum to the number of parameters (2)
/// assert!((leverage.iter().sum::<f64>() - 2.0).abs() < 0.01);
/// ```
#[allow(clippy::needless_range_loop)]
pub fn compute_leverage(x: &Matrix, xtx_inv: &Matrix) -> Vec<f64> {
    let n = x.rows;
    let mut leverage = vec![0.0; n];
    for i in 0..n {
        // x_row is (1, cols)
        // temp = x_row * xtx_inv (1, cols)
        // lev = temp * x_row^T (1, 1)

        // Manual row extraction and multiplication
        let mut row_vec = vec![0.0; x.cols];
        for j in 0..x.cols {
            row_vec[j] = x.get(i, j);
        }

        let mut temp_vec = vec![0.0; x.cols];
        for c in 0..x.cols {
            let mut sum = 0.0;
            for k in 0..x.cols {
                sum += row_vec[k] * xtx_inv.get(k, c);
            }
            temp_vec[c] = sum;
        }

        leverage[i] = vec_dot(&temp_vec, &row_vec);
    }
    leverage
}

// ============================================================================
// Model Selection Criteria
// ============================================================================
//
// Log-likelihood, AIC, and BIC for model comparison.

/// Computes the log-likelihood of the OLS model.
///
/// For a linear regression with normally distributed errors, the log-likelihood is:
///
/// ```text
/// log L = -n/2 * ln(2π) - n/2 * ln(SSR/n) - n/2
///       = -n/2 * ln(2π*SSR/n) - n/2
/// ```
///
/// where SSR is the sum of squared residuals and n is the number of observations.
/// This matches R's `logLik.lm()` implementation.
///
/// # Arguments
///
/// * `n` - Number of observations
/// * `mse` - Mean squared error (estimate of σ², but NOT directly used in formula)
/// * `ssr` - Sum of squared residuals
///
/// # Example
///
/// ```
/// # use linreg_core::core::log_likelihood;
/// let ll = log_likelihood(100, 4.5, 450.0);
/// assert!(ll < 0.0);  // Log-likelihood is negative for typical data
/// ```
pub fn log_likelihood(n: usize, _mse: f64, ssr: f64) -> f64 {
    let n_f64 = n as f64;
    let two_pi = 2.0 * std::f64::consts::PI;

    // R's logLik.lm formula: -n/2 * log(2*pi*SSR/n) - n/2
    // This is equivalent to: -n/2 * (log(2*pi) + log(SSR/n) + 1)
    -0.5 * n_f64 * (two_pi * ssr / n_f64).ln() - n_f64 / 2.0
}

/// Computes the Akaike Information Criterion (AIC).
///
/// AIC = 2k - 2logL
///
/// where k is the number of estimated parameters and logL is the log-likelihood.
/// Lower AIC indicates a better model, with a penalty for additional parameters.
///
/// Note: R's AIC for lm models counts k as (n_coef + 1) where n_coef is the
/// number of coefficients (including intercept) and +1 is for the estimated
/// variance parameter. This implementation follows that convention.
///
/// # Arguments
///
/// * `log_likelihood` - Log-likelihood of the model
/// * `n_coef` - Number of coefficients (including intercept)
///
/// # Example
///
/// ```
/// # use linreg_core::core::aic;
/// let aic_value = aic(-150.5, 3);  // 3 coefficients
/// ```
pub fn aic(log_likelihood: f64, n_coef: usize) -> f64 {
    // R's AIC for lm: 2k - 2*logL
    // where k = n_coef + 1 (coefficients + variance parameter)
    let k = n_coef + 1;
    2.0 * k as f64 - 2.0 * log_likelihood
}

/// Computes the Bayesian Information Criterion (BIC).
///
/// BIC = k*ln(n) - 2logL
///
/// where k is the number of estimated parameters, n is the sample size, and
/// logL is the log-likelihood. BIC penalizes model complexity more heavily
/// than AIC for larger sample sizes.
///
/// Note: R's BIC for lm models counts k as (n_coef + 1) where n_coef is the
/// number of coefficients (including intercept) and +1 is for the estimated
/// variance parameter. This implementation follows that convention.
///
/// # Arguments
///
/// * `log_likelihood` - Log-likelihood of the model
/// * `n_coef` - Number of coefficients (including intercept)
/// * `n_obs` - Number of observations
///
/// # Example
///
/// ```
/// # use linreg_core::core::bic;
/// let bic_value = bic(-150.5, 3, 100);  // 3 coefficients, 100 obs
/// ```
pub fn bic(log_likelihood: f64, n_coef: usize, n_obs: usize) -> f64 {
    // R's BIC for lm: k * ln(n) - 2*logL
    // where k = n_coef + 1 (coefficients + variance parameter)
    let k = n_coef + 1;
    k as f64 * (n_obs as f64).ln() - 2.0 * log_likelihood
}

/// Computes AIC using Python/statsmodels convention.
///
/// AIC = 2k - 2logL
///
/// where k is the number of coefficients (NOT including variance parameter).
/// This matches Python's statsmodels OLS.aic behavior.
///
/// # Arguments
///
/// * `log_likelihood` - Log-likelihood of the model
/// * `n_coef` - Number of coefficients (including intercept)
///
/// # Example
///
/// ```
/// # use linreg_core::core::aic_python;
/// let aic_value = aic_python(-150.5, 3);  // 3 coefficients
/// ```
pub fn aic_python(log_likelihood: f64, n_coef: usize) -> f64 {
    // Python's statsmodels AIC: 2k - 2*logL
    // where k = n_coef (does NOT include variance parameter)
    2.0 * n_coef as f64 - 2.0 * log_likelihood
}

/// Computes BIC using Python/statsmodels convention.
///
/// BIC = k*ln(n) - 2logL
///
/// where k is the number of coefficients (NOT including variance parameter).
/// This matches Python's statsmodels OLS.bic behavior.
///
/// # Arguments
///
/// * `log_likelihood` - Log-likelihood of the model
/// * `n_coef` - Number of coefficients (including intercept)
/// * `n_obs` - Number of observations
///
/// # Example
///
/// ```
/// # use linreg_core::core::bic_python;
/// let bic_value = bic_python(-150.5, 3, 100);  // 3 coefficients, 100 obs
/// ```
pub fn bic_python(log_likelihood: f64, n_coef: usize, n_obs: usize) -> f64 {
    // Python's statsmodels BIC: k * ln(n) - 2*logL
    // where k = n_coef (does NOT include variance parameter)
    n_coef as f64 * (n_obs as f64).ln() - 2.0 * log_likelihood
}

// ============================================================================
// VIF Calculation
// ============================================================================
//
// Variance Inflation Factor analysis for detecting multicollinearity.

/// Calculates Variance Inflation Factors for all predictors.
///
/// VIF quantifies the severity of multicollinearity in a regression analysis.
/// A VIF > 10 indicates high multicollinearity that may need to be addressed.
///
/// # Arguments
///
/// * `x_vars` - Predictor variables (each of length n)
/// * `names` - Variable names (including intercept as first element)
/// * `n` - Number of observations
///
/// # Returns
///
/// Vector of VIF results for each predictor (excluding intercept).
///
/// # Example
///
/// ```
/// # use linreg_core::core::calculate_vif;
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];
///
/// let vif_results = calculate_vif(&[x1, x2], &names, 5);
/// assert_eq!(vif_results.len(), 2);
/// // Perfectly collinear variables will have VIF = infinity
/// ```
pub fn calculate_vif(x_vars: &[Vec<f64>], names: &[String], n: usize) -> Vec<VifResult> {
    let k = x_vars.len();
    if k <= 1 {
        return vec![];
    }

    // Standardize predictors (Z-score)
    let mut z_data = vec![0.0; n * k];

    for (j, var) in x_vars.iter().enumerate() {
        let mean = vec_mean(var);
        let variance = var.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ((n - 1) as f64);
        let std_dev = variance.sqrt();

        // Handle constant variables
        if std_dev.abs() < 1e-10 {
            return names
                .iter()
                .skip(1)
                .map(|name| VifResult {
                    variable: name.clone(),
                    vif: f64::INFINITY,
                    rsquared: 1.0,
                    interpretation: "Constant variable (undefined correlation)".to_string(),
                })
                .collect();
        }

        for i in 0..n {
            z_data[i * k + j] = (var[i] - mean) / std_dev;
        }
    }

    let z = Matrix::new(n, k, z_data);

    // Correlation Matrix R = (1/(n-1)) * Z^T * Z
    let z_t = z.transpose();
    let zt_z = z_t.matmul(&z);

    // Scale by 1/(n-1)
    let mut r_corr = zt_z; // Copy
    let factor = 1.0 / ((n - 1) as f64);
    for val in &mut r_corr.data {
        *val *= factor;
    }

    // Invert R using QR on R_corr (since it's symmetric positive definite, Cholesky is better but QR works)
    // Or just generic inversion. We implemented generic inversion for Upper Triangular.
    // Let's use QR: A = QR => A^-1 = R^-1 Q^T
    let (q_corr, r_corr_tri) = r_corr.qr();

    let r_inv_opt = r_corr_tri.invert_upper_triangular();

    let r_inv = match r_inv_opt {
        Some(inv) => inv.matmul(&q_corr.transpose()),
        None => {
            return names
                .iter()
                .skip(1)
                .map(|name| VifResult {
                    variable: name.clone(),
                    vif: f64::INFINITY,
                    rsquared: 1.0,
                    interpretation: "Perfect multicollinearity (singular matrix)".to_string(),
                })
                .collect();
        },
    };

    // Extract diagonals
    let mut results = vec![];
    for j in 0..k {
        let vif = r_inv.get(j, j);
        let vif = if vif < 1.0 { 1.0 } else { vif };
        let rsquared = 1.0 - 1.0 / vif;

        let interpretation = if vif.is_infinite() {
            "Perfect multicollinearity".to_string()
        } else if vif > 10.0 {
            "High multicollinearity - consider removing".to_string()
        } else if vif > 5.0 {
            "Moderate multicollinearity".to_string()
        } else {
            "Low multicollinearity".to_string()
        };

        results.push(VifResult {
            variable: names[j + 1].clone(),
            vif,
            rsquared,
            interpretation,
        });
    }

    results
}

// ============================================================================
// OLS Regression
// ============================================================================
//
// Ordinary Least Squares regression implementation using QR decomposition.

/// Performs Ordinary Least Squares regression using QR decomposition.
///
/// Uses a numerically stable QR decomposition approach to solve the normal
/// equations. Returns comprehensive statistics including coefficients,
/// standard errors, t-statistics, p-values, and diagnostic measures.
///
/// # Arguments
///
/// * `y` - Response variable (n observations)
/// * `x_vars` - Predictor variables (each of length n)
/// * `variable_names` - Names for variables (including intercept)
///
/// # Returns
///
/// A [`RegressionOutput`] containing all regression statistics and diagnostics.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if n ≤ k + 1.
/// Returns [`Error::SingularMatrix`] if perfect multicollinearity exists.
/// Returns [`Error::InvalidInput`] if coefficients are NaN.
///
/// # Example
///
/// ```
/// # use linreg_core::core::ols_regression;
/// # use linreg_core::Error;
/// let y = vec![2.5, 3.7, 4.2, 5.1, 6.3, 7.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0, 2.0];
/// let names = vec![
///     "Intercept".to_string(),
///     "Temperature".to_string(),
///     "Pressure".to_string(),
/// ];
///
/// let result = ols_regression(&y, &[x1, x2], &names)?;
/// println!("R-squared: {}", result.r_squared);
/// # Ok::<(), Error>(())
/// ```
#[allow(clippy::needless_range_loop)]
pub fn ols_regression(
    y: &[f64],
    x_vars: &[Vec<f64>],
    variable_names: &[String],
) -> Result<RegressionOutput> {
    let n = y.len();
    let k = x_vars.len();
    let p = k + 1;

    // Validate inputs
    if n <= k + 1 {
        return Err(Error::InsufficientData {
            required: k + 2,
            available: n,
        });
    }

    // Validate dimensions and finite values using shared helper
    crate::diagnostics::validate_regression_data(y, x_vars)?;

    // Prepare variable names
    let mut names = variable_names.to_vec();
    while names.len() <= k {
        names.push(format!("X{}", names.len()));
    }

    // Create design matrix
    let mut x_data = vec![0.0; n * p];
    for (row, _yi) in y.iter().enumerate() {
        x_data[row * p] = 1.0; // intercept
        for (col, x_var) in x_vars.iter().enumerate() {
            x_data[row * p + col + 1] = x_var[row];
        }
    }

    let x_matrix = Matrix::new(n, p, x_data);

    // QR Decomposition
    let (q, r) = x_matrix.qr();

    // Solve R * beta = Q^T * y
    // extract upper p x p part of R
    let mut r_upper = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            r_upper.set(i, j, r.get(i, j));
        }
    }

    // Q^T * y
    let q_t = q.transpose();
    let qty = q_t.mul_vec(y);

    // Take first p elements of qty
    let rhs_vec = qty[0..p].to_vec();
    let rhs_mat = Matrix::new(p, 1, rhs_vec); // column vector

    // Invert R_upper
    let r_inv = match r_upper.invert_upper_triangular() {
        Some(inv) => inv,
        None => return Err(Error::SingularMatrix),
    };

    let beta_mat = r_inv.matmul(&rhs_mat);
    let beta = beta_mat.data;

    // Validate coefficients
    if beta.iter().any(|&b| b.is_nan()) {
        return Err(Error::InvalidInput("Coefficients contain NaN".to_string()));
    }

    // Compute predictions and residuals
    let predictions = x_matrix.mul_vec(&beta);
    let residuals = vec_sub(y, &predictions);

    // Compute sums of squares
    let y_mean = vec_mean(y);
    let ss_total: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_residual: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
    let ss_regression = ss_total - ss_residual;

    // R-squared and adjusted R-squared
    let r_squared = if ss_total == 0.0 {
        f64::NAN
    } else {
        1.0 - ss_residual / ss_total
    };

    let adj_r_squared = if ss_total == 0.0 {
        f64::NAN
    } else {
        1.0 - (1.0 - r_squared) * ((n - 1) as f64 / (n - k - 1) as f64)
    };

    // Mean squared error and standard error
    let df = n - k - 1;
    let mse = ss_residual / df as f64;
    let std_error = mse.sqrt();

    // Standard errors using (X'X)^-1 = R^-1 (R')^-1
    // xtx_inv = r_inv * r_inv^T
    let xtx_inv = r_inv.matmul(&r_inv.transpose());

    let mut std_errors = vec![0.0; k + 1];
    for i in 0..=k {
        std_errors[i] = (xtx_inv.get(i, i) * mse).sqrt();
        if std_errors[i].is_nan() {
            return Err(Error::InvalidInput(format!(
                "Standard error for coefficient {} is NaN",
                i
            )));
        }
    }

    // T-statistics and p-values
    let t_stats: Vec<f64> = beta
        .iter()
        .zip(&std_errors)
        .map(|(&b, &se)| b / se)
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| two_tailed_p_value(t, df as f64))
        .collect();

    // Confidence intervals
    let alpha = 0.05;
    let t_critical = t_critical_quantile(df as f64, alpha);

    let conf_int_lower: Vec<f64> = beta
        .iter()
        .zip(&std_errors)
        .map(|(&b, &se)| b - t_critical * se)
        .collect();
    let conf_int_upper: Vec<f64> = beta
        .iter()
        .zip(&std_errors)
        .map(|(&b, &se)| b + t_critical * se)
        .collect();

    // Leverage
    let leverage = compute_leverage(&x_matrix, &xtx_inv);

    // Standardized residuals
    let residuals_vec = residuals.clone();
    let standardized_residuals: Vec<f64> = residuals_vec
        .iter()
        .zip(&leverage)
        .map(|(&r, &h)| {
            let factor = (1.0 - h).max(MIN_LEVERAGE_DENOM).sqrt();
            let denom = std_error * factor;
            if denom > MIN_LEVERAGE_DENOM {
                r / denom
            } else {
                0.0
            }
        })
        .collect();

    // F-statistic
    let f_statistic = (ss_regression / k as f64) / mse;
    let f_p_value = f_p_value(f_statistic, k as f64, df as f64);

    // RMSE and MAE
    let rmse = std_error;
    let mae: f64 = residuals_vec.iter().map(|&r| r.abs()).sum::<f64>() / n as f64;

    // VIF
    let vif = calculate_vif(x_vars, &names, n);

    // Model selection criteria (for model comparison)
    let ll = log_likelihood(n, mse, ss_residual);
    let n_coef = k + 1;  // predictors + intercept
    let aic_val = aic(ll, n_coef);
    let bic_val = bic(ll, n_coef, n);

    Ok(RegressionOutput {
        coefficients: beta,
        std_errors,
        t_stats,
        p_values,
        conf_int_lower,
        conf_int_upper,
        r_squared,
        adj_r_squared,
        f_statistic,
        f_p_value,
        mse,
        rmse,
        mae,
        std_error,
        residuals: residuals_vec,
        standardized_residuals,
        predictions,
        leverage,
        vif,
        n,
        k,
        df,
        variable_names: names,
        log_likelihood: ll,
        aic: aic_val,
        bic: bic_val,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aic_bic_formulas_known_values() {
        // Test formulas with simple known inputs
        let ll = -100.0;
        let n_coef = 3; // 3 coefficients (e.g., intercept + 2 predictors)
        let n_obs = 100;

        let aic_val = aic(ll, n_coef);
        let bic_val = bic(ll, n_coef, n_obs);

        // AIC = 2k - 2logL where k = n_coef + 1 (variance parameter)
        // AIC = 2*4 - 2*(-100) = 8 + 200 = 208
        assert!((aic_val - 208.0).abs() < 1e-10);

        // BIC = k*ln(n) - 2logL where k = n_coef + 1
        // BIC = 4*ln(100) - 2*(-100) = 4*4.605... + 200
        let expected_bic = 4.0 * (100.0_f64).ln() + 200.0;
        assert!((bic_val - expected_bic).abs() < 1e-10);
    }

    #[test]
    fn test_bic_greater_than_aic_for_reasonable_n() {
        // For n >= 8, ln(n) > 2, so BIC > AIC (both have -2logL term)
        // BIC uses k*ln(n) while AIC uses 2k, so when ln(n) > 2, BIC > AIC
        let ll = -50.0;
        let n_coef = 2;

        let aic_val = aic(ll, n_coef);
        let bic_val = bic(ll, n_coef, 100); // n=100, ln(100) ≈ 4.6 > 2

        assert!(bic_val > aic_val);
    }

    #[test]
    fn test_log_likelihood_returns_finite() {
        // Ensure log_likelihood returns finite values for valid inputs
        let n = 10;
        let mse = 4.0;
        let ssr = mse * (n - 2) as f64;

        let ll = log_likelihood(n, mse, ssr);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_log_likelihood_increases_with_better_fit() {
        // Lower SSR (better fit) should give higher log-likelihood
        let n = 10;

        // Worse fit (higher residuals)
        let ll_worse = log_likelihood(n, 10.0, 80.0);

        // Better fit (lower residuals)
        let ll_better = log_likelihood(n, 2.0, 16.0);

        assert!(ll_better > ll_worse);
    }

    #[test]
    fn test_model_selection_criteria_present_in_output() {
        // Basic sanity check that the fields are populated
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let names = vec!["Intercept".to_string(), "X1".to_string()];

        let result = ols_regression(&y, &[x1], &names).unwrap();

        // All three should be finite
        assert!(result.log_likelihood.is_finite());
        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());

        // AIC and BIC should be positive for typical cases
        // (since log_likelihood is usually negative and bounded)
        assert!(result.aic > 0.0);
        assert!(result.bic > 0.0);
    }

    #[test]
    fn test_regression_output_has_correct_dimensions() {
        // Verify AIC/BIC use k = n_coef + 1 (coefficients + variance parameter)
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = vec![3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0];
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let result = ols_regression(&y, &[x1, x2], &names).unwrap();

        // n_coef = 3 (intercept + 2 predictors)
        // k = n_coef + 1 = 4 (including variance parameter, following R convention)
        let n_coef = 3;
        let k = n_coef + 1; // R's convention includes variance parameter

        // Verify by recalculating AIC from log_likelihood
        let expected_aic = 2.0 * k as f64 - 2.0 * result.log_likelihood;
        assert!((result.aic - expected_aic).abs() < 1e-10);

        // Verify by recalculating BIC from log_likelihood
        let expected_bic = k as f64 * (result.n as f64).ln() - 2.0 * result.log_likelihood;
        assert!((result.bic - expected_bic).abs() < 1e-10);
    }

    #[test]
    fn test_aic_python_convention() {
        // Python's statsmodels uses k = n_coef (no variance parameter)
        let ll = -100.0;
        let n_coef = 3;

        let aic_py = aic_python(ll, n_coef);
        // AIC = 2k - 2logL where k = n_coef (Python convention)
        // AIC = 2*3 - 2*(-100) = 6 + 200 = 206
        assert!((aic_py - 206.0).abs() < 1e-10);
    }

    #[test]
    fn test_bic_python_convention() {
        // Python's statsmodels uses k = n_coef (no variance parameter)
        let ll = -100.0;
        let n_coef = 3;
        let n_obs = 100;

        let bic_py = bic_python(ll, n_coef, n_obs);
        // BIC = k*ln(n) - 2logL where k = n_coef (Python convention)
        // BIC = 3*ln(100) - 2*(-100) = 3*4.605... + 200
        let expected_bic = 3.0 * (100.0_f64).ln() + 200.0;
        assert!((bic_py - expected_bic).abs() < 1e-10);
    }

    #[test]
    fn test_python_aic_smaller_than_r_aic() {
        // Python convention uses k = n_coef, R uses k = n_coef + 1
        // So Python AIC should be 2 smaller than R AIC
        let ll = -50.0;
        let n_coef = 2;

        let aic_r = aic(ll, n_coef);
        let aic_py = aic_python(ll, n_coef);

        assert_eq!(aic_r - aic_py, 2.0);
    }

    #[test]
    fn test_log_likelihood_formula_matches_r() {
        // Test against R's logLik.lm() formula
        // For a model with n=100, SSR=450, logL = -n/2 * log(2*pi*SSR/n) - n/2
        let n = 100;
        let ssr = 450.0;
        let mse = ssr / (n as f64 - 2.0); // 2 parameters

        let ll = log_likelihood(n, mse, ssr);

        // Calculate expected value manually
        let two_pi = 2.0 * std::f64::consts::PI;
        let expected = -0.5 * n as f64 * (two_pi * ssr / n as f64).ln() - n as f64 / 2.0;

        assert!((ll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_aic_bic_with_perfect_fit() {
        // Perfect fit (zero residuals) - edge case
        let n = 10;
        let ssr = 0.001; // Very small but non-zero to avoid log(0)
        let mse = ssr / (n as f64 - 2.0);

        let ll = log_likelihood(n, mse, ssr);
        let aic_val = aic(ll, 2);
        let bic_val = bic(ll, 2, n);

        // Perfect fit gives very high log-likelihood
        assert!(ll > 0.0);
        // AIC/BIC penalize complexity, so may be negative for very good fits
        assert!(aic_val.is_finite());
        assert!(bic_val.is_finite());
    }

    #[test]
    fn test_aic_bic_model_selection() {
        // Simulate model comparison: simpler model vs complex model
        // Both models fit same data with similar R² but different complexity
        let n = 100;

        // Simple model (2 params): better log-likelihood due to less penalty
        let ll_simple = -150.0;
        let aic_simple = aic(ll_simple, 2);
        let bic_simple = bic(ll_simple, 2, n);

        // Complex model (5 params): slightly better fit but more parameters
        let ll_complex = -148.0; // Better fit (less negative)
        let aic_complex = aic(ll_complex, 5);
        let bic_complex = bic(ll_complex, 5, n);

        // AIC might favor complex model (2*2 - 2*(-150) = 304 vs 2*6 - 2*(-148) = 308)
        // Actually: 4 + 300 = 304 vs 12 + 296 = 308, so simple wins
        assert!(aic_simple < aic_complex);

        // BIC more heavily penalizes complexity, so simple should win
        assert!(bic_simple < bic_complex);
    }

    #[test]
    fn test_log_likelihood_scale_invariance() {
        // Log-likelihood scales with sample size for same per-observation fit quality
        let ssr_per_obs = 1.0;

        let n1 = 50;
        let ssr1 = ssr_per_obs * n1 as f64;
        let ll1 = log_likelihood(n1, ssr1 / (n1 as f64 - 2.0), ssr1);

        let n2 = 100;
        let ssr2 = ssr_per_obs * n2 as f64;
        let ll2 = log_likelihood(n2, ssr2 / (n2 as f64 - 2.0), ssr2);

        // The log-likelihood should become more negative with larger n for the same SSR/n ratio
        // because -n/2 * ln(2*pi*SSR/n) - n/2 becomes more negative as n increases
        assert!(ll2 < ll1);

        // But when normalized by n, they should be similar
        let ll_per_obs1 = ll1 / n1 as f64;
        let ll_per_obs2 = ll2 / n2 as f64;
        assert!((ll_per_obs1 - ll_per_obs2).abs() < 0.1);
    }

    #[test]
    fn test_regularized_regression_has_model_selection_criteria() {
        // Test that Ridge regression also calculates AIC/BIC/log_likelihood
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x_data = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0];
        let x = crate::linalg::Matrix::new(5, 2, x_data);

        let options = crate::regularized::ridge::RidgeFitOptions {
            lambda: 0.1,
            intercept: true,
            standardize: false,
            ..Default::default()
        };

        let fit = crate::regularized::ridge::ridge_fit(&x, &y, &options).unwrap();

        assert!(fit.log_likelihood.is_finite());
        assert!(fit.aic.is_finite());
        assert!(fit.bic.is_finite());
    }

    #[test]
    fn test_elastic_net_regression_has_model_selection_criteria() {
        // Test that Elastic Net regression also calculates AIC/BIC/log_likelihood
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x_data = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0];
        let x = crate::linalg::Matrix::new(5, 2, x_data);

        let options = crate::regularized::elastic_net::ElasticNetOptions {
            lambda: 0.1,
            alpha: 0.5,
            intercept: true,
            standardize: false,
            ..Default::default()
        };

        let fit = crate::regularized::elastic_net::elastic_net_fit(&x, &y, &options).unwrap();

        assert!(fit.log_likelihood.is_finite());
        assert!(fit.aic.is_finite());
        assert!(fit.bic.is_finite());
    }
}
