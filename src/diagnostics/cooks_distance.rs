// ============================================================================
// Cook's Distance for Influential Observations
// ============================================================================
//
// Cook's distance measures the influence of each observation on the regression
// model by comparing coefficient estimates with and without that observation.
//
// Algorithm:
// For each observation i:
//   D_i = (r_i² * h_i) / (p * MSE * (1 - h_i)²)
//
// Where:
//   r_i = residual for observation i
//   h_i = leverage (hat matrix diagonal) for observation i
//   p = number of parameters (including intercept)
//   MSE = mean squared error
//
// Reference: stats::cooks.distance(model) in R
//            statsmodels.stats.outliers_influence.OLSInfluence.cooks_distance in Python
//
// Interpretation:
// - D_i > 1: Highly influential observation (warrants investigation)
// - D_i > 4/n: Common threshold (n = sample size)
// - D_i > 4/(n-p-1): More conservative threshold

use crate::error::{Error, Result};
use crate::linalg::{Matrix, vec_sub};
use crate::core::compute_leverage;
use super::types::CooksDistanceResult;

/// Computes Cook's distance for identifying influential observations.
///
/// Cook's distance measures how much each observation influences the regression
/// model. It compares the fitted values with and without each observation to
/// assess its impact on the coefficient estimates.
///
/// # Arguments
///
/// * `y` - Dependent variable values
/// * `x_vars` - Independent variables (each vec is a column)
///
/// # Returns
///
/// A [`CooksDistanceResult`] containing:
/// - Vector of Cook's distances (one per observation)
/// - Number of parameters (p)
/// - Mean squared error (MSE)
/// - Thresholds: 4/n, 4/(n-p-1), and 1
/// - Lists of influential observations by each threshold
/// - Interpretation of results
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if n ≤ p + 1.
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::cooks_distance_test;
///
/// // Simple linear regression
/// let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// let result = cooks_distance_test(&y, &[x1])?;
/// println!("Max Cook's distance: {:.4}", result.distances.iter().cloned().fold(0.0, f64::max));
/// println!("Influential observations: {:?}", result.influential_1);
/// # Ok::<(), linreg_core::error::Error>(())
/// ```
pub fn cooks_distance_test(
    y: &[f64],
    x_vars: &[Vec<f64>],
) -> Result<CooksDistanceResult> {
    let n = y.len();
    let k = x_vars.len();
    let p = k + 1;

    // Validate inputs
    if n <= p {
        return Err(Error::InsufficientData { required: p + 1, available: n });
    }

    // Create design matrix with intercept
    let mut x_data = vec![1.0; n * p];
    for row in 0..n {
        x_data[row * p] = 1.0;  // intercept
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

    // Compute (X'X)^-1 for leverage calculation
    // Use QR decomposition for stable inversion
    let (_q2, r2) = x_full.qr();
    let mut r_upper2 = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            r_upper2.set(i, j, r2.get(i, j));
        }
    }

    let r_inv2 = match r_upper2.invert_upper_triangular() {
        Some(inv) => inv,
        None => return Err(Error::SingularMatrix),
    };

    let xtx_inv = r_inv2.matmul(&r_inv2.transpose());

    // Compute leverage values
    let leverage = compute_leverage(&x_full, &xtx_inv);

    // Compute Cook's distance for each observation
    let mut distances = Vec::with_capacity(n);
    for i in 0..n {
        let r_i = residuals[i];
        let h_i = leverage[i];

        // Avoid division by zero for high leverage points
        let one_minus_h = (1.0 - h_i).max(1e-10);

        let numerator = r_i * r_i * h_i;
        let denominator = (p as f64) * mse * one_minus_h * one_minus_h;

        let d_i = if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };

        distances.push(d_i);
    }

    // Compute thresholds
    let threshold_4_over_n = 4.0 / (n as f64);
    let threshold_4_over_df = if df_residual > 0 {
        4.0 / (df_residual as f64)
    } else {
        0.0
    };
    let threshold_1 = 1.0;

    // Identify influential observations (using 1-based indexing for output)
    let influential_4_over_n: Vec<usize> = distances.iter()
        .enumerate()
        .filter(|(_, &d)| d > threshold_4_over_n)
        .map(|(i, _)| i + 1)
        .collect();

    let influential_4_over_df: Vec<usize> = distances.iter()
        .enumerate()
        .filter(|(_, &d)| d > threshold_4_over_df)
        .map(|(i, _)| i + 1)
        .collect();

    let influential_1: Vec<usize> = distances.iter()
        .enumerate()
        .filter(|(_, &d)| d > threshold_1)
        .map(|(i, _)| i + 1)
        .collect();

    // Find maximum Cook's distance
    let max_d = distances.iter().cloned().fold(0.0, f64::max);
    let max_idx = distances.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i + 1)
        .unwrap_or(0);

    // Generate interpretation
    let interpretation = if influential_1.is_empty() && max_d < 1.0 {
        format!(
            "Maximum Cook's distance is {:.4} (observation {}). No observations exceed D > 1 threshold. No highly influential observations detected.",
            max_d, max_idx
        )
    } else if !influential_1.is_empty() {
        format!(
            "Maximum Cook's distance is {:.4} (observation {}). {} observation(s) exceed D > 1 threshold: {:?}. These observations have high influence on the model.",
            max_d, max_idx, influential_1.len(), influential_1
        )
    } else {
        format!(
            "Maximum Cook's distance is {:.4} (observation {}). {} observation(s) exceed 4/n threshold ({:.4}): {:?}",
            max_d, max_idx, influential_4_over_n.len(), threshold_4_over_n, influential_4_over_n
        )
    };

    // Generate guidance
    let guidance = if !influential_1.is_empty() {
        format!(
            "Highly influential observations detected (D > 1). Investigate observations {:?}. Consider: 1) verifying data accuracy, 2) checking if these observations represent a different population, 3) assessing model fit without these points, 4) using robust regression methods.",
            influential_1
        )
    } else if (influential_4_over_n.len() as f64) > n as f64 / 5.0 {
        format!(
            "Multiple observations ({}) exceed the 4/n threshold. This may indicate model misspecification or outliers. Consider examining residual plots and checking model assumptions.",
            influential_4_over_n.len()
        )
    } else if !influential_4_over_n.is_empty() {
        format!(
            "Some observations ({:?}) are moderately influential. Examine these points to ensure they are valid and not data entry errors.",
            influential_4_over_n
        )
    } else {
        "No influential observations detected. The model appears stable with respect to individual observations.".to_string()
    };

    Ok(CooksDistanceResult {
        test_name: "Cook's Distance".to_string(),
        distances,
        p,
        mse,
        threshold_4_over_n,
        threshold_4_over_df,
        threshold_1,
        influential_4_over_n,
        influential_4_over_df,
        influential_1,
        interpretation,
        guidance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooks_distance_simple() {
        // Simple linear regression with known values
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = cooks_distance_test(&y, &[x1]).unwrap();

        // Should return 5 distances
        assert_eq!(result.distances.len(), 5);
        // All distances should be finite and non-negative
        for &d in &result.distances {
            assert!(d.is_finite());
            assert!(d >= 0.0);
        }
        // For this well-behaved data, no observations should be highly influential
        assert!(result.influential_1.is_empty());
    }

    #[test]
    fn test_cooks_distance_with_outlier() {
        // Data with one clear outlier
        let y = vec![1.0, 2.0, 3.0, 4.0, 100.0];  // Last value is outlier
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = cooks_distance_test(&y, &[x1]).unwrap();

        // Should return 5 distances
        assert_eq!(result.distances.len(), 5);

        // The last observation should have highest Cook's distance
        let max_idx = result.distances.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();

        // Due to the extreme outlier, the last observation should have highest influence
        assert_eq!(max_idx, 4);
    }

    #[test]
    fn test_cooks_distance_insufficient_data() {
        let y = vec![1.0, 2.0];
        let x1 = vec![1.0, 2.0];

        // Need at least p + 1 = 3 observations for simple regression
        let result = cooks_distance_test(&y, &[x1]);
        assert!(result.is_err());
    }
}
