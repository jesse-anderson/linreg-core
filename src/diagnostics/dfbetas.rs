// ============================================================================
// DFBETAS for Influential Observations
// ============================================================================
//
// DFBETAS measures the influence of each observation on each regression coefficient.
// For each observation i and each coefficient j, it measures the number of standard
// errors that coefficient j changes when observation i is omitted.
//
// Algorithm:
// For each observation i:

#![allow(clippy::needless_range_loop)]
//   1. Fit the model without observation i
//   2. Compute the change in each coefficient
//   3. Standardize by the standard error of the coefficient
//
// Efficient computation (using the "leave-one-out" formula):
//   DFBETA_ij = (r_i / (sigma * sqrt(1 - h_ii))) * ((X'X)^{-1} * x_i')_j / sqrt((X'X)^{-1}_jj)
//   where x_i' is the i-th row of X as a column vector
//
// This avoids refitting the model n times by using the closed-form expression.
//
// Interpretation:
// - |DFBETAS| > 2/sqrt(n): Common threshold for identifying influential observations
// - Larger absolute values indicate greater influence of that observation on that coefficient

use super::types::DfbetasResult;
use crate::error::{Error, Result};
use crate::linalg::{vec_sub, Matrix};
use std::collections::HashMap;

/// Computes DFBETAS for identifying influential observations on specific coefficients.
///
/// DFBETAS measures how much each observation influences each regression coefficient.
/// For each observation and each coefficient, it computes the standardized change
/// in the coefficient when that observation is omitted from the model.
///
/// # Arguments
///
/// * `y` - Dependent variable values
/// * `x_vars` - Independent variables (each vec is a column)
///
/// # Returns
///
/// A [`DfbetasResult`] containing:
/// - Matrix of DFBETAS values (n × p)
/// - Number of observations and parameters
/// - Threshold (2/√n)
/// - Map of influential observations by coefficient
/// - Interpretation and guidance
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if n ≤ p + 1.
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::dfbetas_test;
///
/// let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// let result = dfbetas_test(&y, &[x1])?;
/// println!("DFBETAS matrix: {:?}", result.dfbetas);
/// println!("Threshold: {:.4}", result.threshold);
/// # Ok::<(), linreg_core::error::Error>(())
/// ```
pub fn dfbetas_test(y: &[f64], x_vars: &[Vec<f64>]) -> Result<DfbetasResult> {
    let n = y.len();
    let k = x_vars.len();
    let p = k + 1;

    // Validate inputs
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
        for (col, x_var) in x_vars.iter().enumerate() {
            x_data[row * p + col + 1] = x_var[row];
        }
    }
    let x_full = Matrix::new(n, p, x_data);

    // Fit OLS using QR decomposition
    let (q, r) = x_full.qr();

    // Extract upper p x p part of R
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
    let rhs_mat = Matrix::new(p, 1, rhs_vec);

    // Invert R_upper
    let r_inv = match r_upper.invert_upper_triangular() {
        Some(inv) => inv,
        None => return Err(Error::SingularMatrix),
    };

    let beta = r_inv.matmul(&rhs_mat);
    let beta_data = beta.data;

    // Validate coefficients
    if beta_data.iter().any(|&b| b.is_nan()) {
        return Err(Error::InvalidInput("Coefficients contain NaN".to_string()));
    }

    // Compute predictions and residuals
    let predictions = x_full.mul_vec(&beta_data);
    let residuals = vec_sub(y, &predictions);

    // Compute RSS and MSE
    let rss: f64 = residuals.iter().map(|&r| r * r).sum();
    let df_residual = n - p;
    let mse = rss / (df_residual as f64);
    let _sigma = mse.sqrt();

    // Compute (X'X)^-1 for leverage calculation
    let xtx_inv = r_inv.matmul(&r_inv.transpose());

    // Compute leverage values (diagonal of hat matrix H = X(X'X)^-1X')
    let mut leverage = vec![0.0; n];
    for i in 0..n {
        let mut h_ii = 0.0;
        for j in 0..p {
            let x_ij = if j == 0 { 1.0 } else { x_vars[j - 1][i] };
            // H_ii = sum_j sum_k x_ij * (X'X)^{-1}_jk * x_ik
            // = x_row_i * (X'X)^{-1} * x_col_i
            for k in 0..p {
                let x_ik = if k == 0 { 1.0 } else { x_vars[k - 1][i] };
                h_ii += x_ij * xtx_inv.get(j, k) * x_ik;
            }
        }
        leverage[i] = h_ii;
    }

    // Compute DFBETAS using the efficient leave-one-out formula
    // DFBETAS_ij = DFBETA_ij / (sigma_i * sqrt(diag((X'X)^{-1})_jj))
    // where DFBETA_ij = r_i * ((X'X)^{-1} * x_i')_j / (1 - h_ii)
    // Combining: DFBETAS_ij = r_i * ((X'X)^{-1} * x_i')_j / ((1 - h_ii) * sigma_i * sqrt((X'X)^{-1}_jj))
    let mut dfbetas = vec![vec![0.0; p]; n];

    // Pre-compute sqrt of diagonal elements of (X'X)^{-1}
    let mut xtx_inv_diag_sqrt = vec![0.0; p];
    for j in 0..p {
        let diag_val = xtx_inv.get(j, j);
        xtx_inv_diag_sqrt[j] = diag_val.sqrt();
    }

    for i in 0..n {
        let r_i = residuals[i];
        let h_ii = leverage[i];

        // Avoid division by zero for high leverage points
        let one_minus_h = (1.0 - h_ii).max(f64::EPSILON);

        // Compute LOO sigma (sigma_i) using df = n - p - 1
        let loo_rss = (rss - r_i * r_i / one_minus_h).max(0.0);
        let sigma_i = (loo_rss / (df_residual - 1) as f64).sqrt();

        // For each coefficient j
        for j in 0..p {
            // Compute ((X'X)^{-1} * x_i')_j = sum_k (X'X)^{-1}_jk * x_ik
            let xtx_inv_x = {
                let mut sum = 0.0;
                for k in 0..p {
                    let x_ik = if k == 0 { 1.0 } else { x_vars[k - 1][i] };
                    sum += xtx_inv.get(j, k) * x_ik;
                }
                sum
            };

            // DFBETAS_ij = r_i * xtx_inv_x / ((1 - h_ii) * sigma_i * sqrt(xtx_inv_jj))
            let dfbeta = if sigma_i > 0.0 && xtx_inv_diag_sqrt[j] > 0.0 {
                r_i * xtx_inv_x / (one_minus_h * sigma_i * xtx_inv_diag_sqrt[j])
            } else {
                0.0
            };
            dfbetas[i][j] = dfbeta;
        }
    }

    // Compute threshold: 2/sqrt(n)
    let threshold = 2.0 / (n as f64).sqrt();

    // Identify influential observations for each coefficient
    let mut influential_observations: HashMap<usize, Vec<usize>> = HashMap::new();

    for j in 0..p {
        let mut influential: Vec<usize> = Vec::new();
        for i in 0..n {
            if dfbetas[i][j].abs() > threshold {
                influential.push(i + 1); // 1-based indexing
            }
        }
        if !influential.is_empty() {
            influential_observations.insert(j + 1, influential); // 1-based coefficient index
        }
    }

    // Find maximum absolute DFBETAS value and its location
    let mut max_dfbetas = 0.0;
    let mut max_obs = 0;
    let mut max_coef = 0;

    for i in 0..n {
        for j in 0..p {
            let abs_val = dfbetas[i][j].abs();
            if abs_val > max_dfbetas {
                max_dfbetas = abs_val;
                max_obs = i + 1;
                max_coef = j + 1;
            }
        }
    }

    // Count total influential observations
    let total_influential: usize = influential_observations
        .values()
        .map(|v| v.len())
        .sum();

    // Generate interpretation
    let interpretation = if total_influential == 0 {
        format!(
            "Maximum |DFBETAS| is {:.4} (observation {}, coefficient {}). No observations exceed the threshold of {:.4}. No highly influential observations detected.",
            max_dfbetas, max_obs, max_coef, threshold
        )
    } else {
        let coef_names: Vec<String> = influential_observations
            .keys()
            .map(|k| if *k == 1 { "intercept".to_string() } else { format!("X{}", k - 1) })
            .collect();
        format!(
            "Maximum |DFBETAS| is {:.4} (observation {}, coefficient {}). {} observation(s) exceed the threshold of {:.4} for coefficient(s): {:?}. These observations have high influence on specific coefficients.",
            max_dfbetas, max_obs, max_coef, total_influential, threshold, coef_names
        )
    };

    // Generate guidance
    let guidance = if total_influential == 0 {
        "No influential observations detected. The model coefficients appear stable with respect to individual observations.".to_string()
    } else if total_influential > n / 4 {
        format!(
            "Multiple observations ({}) exceed the DFBETAS threshold. This may indicate model misspecification or the presence of multiple outliers. Consider: 1) examining residual plots, 2) checking data accuracy, 3) testing alternative model specifications.",
            total_influential
        )
    } else {
        let mut details = Vec::new();
        for (coef_idx, obs_list) in &influential_observations {
            let coef_name = if *coef_idx == 1 { "intercept" } else { &format!("X{}", coef_idx - 1) };
            details.push(format!("{}: {:?}", coef_name, obs_list));
        }
        format!(
            "Some observations have high influence on specific coefficients ({}). Investigate these observations: {}. Consider verifying data accuracy and assessing model fit without these points.",
            details.join(", "),
            total_influential
        )
    };

    Ok(DfbetasResult {
        test_name: "DFBETAS".to_string(),
        dfbetas,
        n,
        p,
        threshold,
        influential_observations,
        interpretation,
        guidance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfbetas_simple() {
        // Simple linear regression
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = dfbetas_test(&y, &[x1]).unwrap();

        // Should return n x p matrix (5 x 2)
        assert_eq!(result.dfbetas.len(), 5);
        assert_eq!(result.dfbetas[0].len(), 2);

        // All DFBETAS should be finite
        for row in &result.dfbetas {
            for &val in row {
                assert!(val.is_finite());
            }
        }

        // Threshold should be 2/sqrt(5) ≈ 0.894
        assert!((result.threshold - 2.0 / 5.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_dfbetas_with_outlier() {
        // Data with one clear outlier
        // Using y=20 instead of more extreme values to avoid edge cases
        let y = vec![1.0, 2.0, 3.0, 4.0, 20.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = dfbetas_test(&y, &[x1]).unwrap();

        // All DFBETAS should be finite
        for row in &result.dfbetas {
            for &val in row {
                assert!(val.is_finite());
            }
        }

        // The outlier should have more influence than typical observations
        let outlier_influence: f64 = result.dfbetas[4].iter().map(|&v| v.abs()).sum();
        let typical_influence: f64 = result.dfbetas[1].iter().map(|&v| v.abs()).sum();
        assert!(
            outlier_influence > typical_influence,
            "Outlier should have higher DFBETAS influence than typical observation"
        );
    }

    #[test]
    fn test_dfbetas_insufficient_data() {
        let y = vec![1.0, 2.0];
        let x1 = vec![1.0, 2.0];

        // Need at least p + 1 = 3 observations for simple regression
        let result = dfbetas_test(&y, &[x1]);
        assert!(result.is_err());
    }
}
