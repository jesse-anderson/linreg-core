// ============================================================================
// DFFITS for Influential Observations
// ============================================================================
//
// DFFITS measures the influence of each observation on its own fitted value.
// It is the number of standard errors that the fitted value changes when
// observation i is omitted.
//
// Algorithm:
// For each observation i:
//   DFFITS_i = (y_i_hat - y_i_hat(-i)) / (sigma(-i) * sqrt(h_ii / (1 - h_ii)))
//
// Efficient computation (using the "leave-one-out" formula):
//   DFFITS_i = r_i * sqrt(h_ii / (1 - h_ii)) / sigma
//   where r_i is the studentized residual
//
// Interpretation:
// - |DFFITS| > 2*sqrt(p/n): Common threshold for identifying influential observations
// - Larger absolute values indicate greater influence of that observation on its fitted value

use super::types::DffitsResult;
use crate::error::{Error, Result};
use crate::linalg::{vec_sub, Matrix};

/// Computes DFFITS for identifying influential observations on fitted values.
///
/// DFFITS measures how much each observation influences its own fitted value.
/// It is the standardized change in the fitted value when that observation
/// is omitted from the model.
///
/// # Arguments
///
/// * `y` - Dependent variable values
/// * `x_vars` - Independent variables (each vec is a column)
///
/// # Returns
///
/// A [`DffitsResult`] containing:
/// - Vector of DFFITS values (one per observation)
/// - Number of observations and parameters
/// - Threshold (2*√(p/n))
/// - List of influential observations
/// - Interpretation and guidance
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if n ≤ p + 1.
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::dffits_test;
///
/// let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// let result = dffits_test(&y, &[x1])?;
/// println!("DFFITS: {:?}", result.dffits);
/// println!("Threshold: {:.4}", result.threshold);
/// # Ok::<(), linreg_core::error::Error>(())
/// ```
pub fn dffits_test(y: &[f64], x_vars: &[Vec<f64>]) -> Result<DffitsResult> {
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
            for k in 0..p {
                let x_ik = if k == 0 { 1.0 } else { x_vars[k - 1][i] };
                h_ii += x_ij * xtx_inv.get(j, k) * x_ik;
            }
        }
        leverage[i] = h_ii;
    }

    // Compute DFFITS using the efficient leave-one-out formula
    // R formula: DFFITS_i = r_i * sqrt(h_ii) / (sigma_i * (1 - h_ii))
    // where sigma_i is the LOO standard deviation
    let mut dffits = Vec::with_capacity(n);

    for i in 0..n {
        let r_i = residuals[i];
        let h_ii = leverage[i];

        // Avoid division by zero for high leverage points
        let one_minus_h = (1.0 - h_ii).max(f64::EPSILON);

        // Compute LOO sigma (sigma_i) using df = n - p - 1
        let loo_rss = (rss - r_i * r_i / one_minus_h).max(0.0);
        let sigma_i = (loo_rss / (df_residual - 1) as f64).sqrt();

        // R's formula: r_i * sqrt(h_ii) / (sigma_i * (1 - h_ii))
        let dffit = if sigma_i > 0.0 {
            r_i * h_ii.sqrt() / (sigma_i * one_minus_h)
        } else {
            0.0
        };

        dffits.push(dffit);
    }

    // Compute threshold: 2*sqrt(p/n)
    let threshold = 2.0 * ((p as f64) / (n as f64)).sqrt();

    // Identify influential observations
    let influential_observations: Vec<usize> = dffits
        .iter()
        .enumerate()
        .filter(|(_, &d)| d.abs() > threshold)
        .map(|(i, _)| i + 1) // 1-based indexing
        .collect();

    // Find maximum absolute DFFITS value
    let max_dffits = dffits.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    let max_idx = dffits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i + 1)
        .unwrap_or(0);

    // Generate interpretation
    let interpretation = if influential_observations.is_empty() {
        format!(
            "Maximum |DFFITS| is {:.4} (observation {}). No observations exceed the threshold of {:.4}. No highly influential observations detected.",
            max_dffits, max_idx, threshold
        )
    } else {
        format!(
            "Maximum |DFFITS| is {:.4} (observation {}). {} observation(s) exceed the threshold of {:.4}: {:?}. These observations have high influence on their fitted values.",
            max_dffits, max_idx, influential_observations.len(), threshold, influential_observations
        )
    };

    // Generate guidance
    let guidance = if influential_observations.is_empty() {
        "No influential observations detected. The model fitted values appear stable with respect to individual observations.".to_string()
    } else if influential_observations.len() > n / 4 {
        format!(
            "Multiple observations ({}) exceed the DFFITS threshold. This may indicate model misspecification or the presence of multiple outliers. Consider: 1) examining residual plots, 2) checking data accuracy, 3) testing alternative model specifications.",
            influential_observations.len()
        )
    } else {
        format!(
            "Some observations ({:?}) have high influence on their fitted values. Investigate these observations to ensure they are valid and not data entry errors. Consider assessing model fit without these points.",
            influential_observations
        )
    };

    Ok(DffitsResult {
        test_name: "DFFITS".to_string(),
        dffits,
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
    fn test_dffits_simple() {
        // Simple linear regression
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = dffits_test(&y, &[x1]).unwrap();

        // Should return n DFFITS values
        assert_eq!(result.dffits.len(), 5);

        // All DFFITS should be finite
        for &val in &result.dffits {
            assert!(val.is_finite());
        }

        // Threshold should be 2*sqrt(2/5)
        assert!((result.threshold - 2.0 * (2.0_f64 / 5.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_dffits_with_outlier() {
        // Data with one clear outlier
        // Using y=20 instead of more extreme values to avoid edge cases
        let y = vec![1.0, 2.0, 3.0, 4.0, 20.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = dffits_test(&y, &[x1]).unwrap();

        // All DFFITS should be finite
        for &val in &result.dffits {
            assert!(val.is_finite());
        }

        // The outlier should have more influence than typical observations
        let outlier_dffits = result.dffits[4].abs();
        let typical_dffits = result.dffits[1].abs();
        assert!(
            outlier_dffits > typical_dffits,
            "Outlier should have higher DFFITS than typical observation"
        );
    }

    #[test]
    fn test_dffits_insufficient_data() {
        let y = vec![1.0, 2.0];
        let x1 = vec![1.0, 2.0];

        // Need at least p + 1 = 3 observations for simple regression
        let result = dffits_test(&y, &[x1]);
        assert!(result.is_err());
    }
}
