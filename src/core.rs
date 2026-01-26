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
    })
}
