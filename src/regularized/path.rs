//! Lambda path generation for regularized regression.
//!
//! This module provides utilities for generating a sequence of lambda values
//! for regularization paths, matching glmnet's approach.
//!
//! # Lambda Path Construction
//!
//! glmnet generates a lambda path from `lambda_max` down to `lambda_min`:
//!
//! - `lambda_max`: The smallest lambda for which all penalized coefficients are zero
//! - `lambda_min`: `lambda_min_ratio * lambda_max`
//!
//! For pure ridge (`alpha=0`), `lambda_max` is theoretically infinite, so we use
//! a small `alpha` value to compute a finite starting point.

use crate::linalg::Matrix;

/// Options for generating a lambda path.
///
/// # Fields
///
/// * `nlambda` - Number of lambda values (default: 100)
/// * `lambda_min_ratio` - Ratio for smallest lambda (default: 0.0001 if n < p, else 0.01)
/// * `alpha` - Elastic net mixing parameter (0 = ridge, 1 = lasso)
/// * `eps_for_ridge` - Small alpha to use for computing lambda_max when alpha=0 (default: 0.001)
#[derive(Clone, Debug)]
pub struct LambdaPathOptions {
    /// Number of lambda values to generate
    pub nlambda: usize,
    /// Minimum lambda as a fraction of maximum
    pub lambda_min_ratio: Option<f64>,
    /// Elastic net mixing parameter (0 = ridge, 1 = lasso)
    pub alpha: f64,
    /// Small alpha to use for ridge lambda_max computation
    pub eps_for_ridge: f64,
}

impl Default for LambdaPathOptions {
    fn default() -> Self {
        LambdaPathOptions {
            nlambda: 100,
            lambda_min_ratio: None,
            alpha: 1.0,
            eps_for_ridge: 0.001,
        }
    }
}

/// Computes `lambda_max`: the smallest lambda for which all penalized coefficients are zero.
///
/// For lasso (alpha > 0), lambda_max is the smallest value such that the soft-thresholding
/// operation zeros out all coefficients.
///
/// For standardized X and centered y:
/// ```text
/// lambda_max = max_j |x_j^T y| / (n * alpha)
/// ```
///
/// # Arguments
///
/// * `x` - Standardized design matrix (n × p), first column is intercept if present
/// * `y` - Centered response vector (n elements)
/// * `alpha` - Elastic net mixing parameter
/// * `penalty_factor` - Per-feature penalty factors (optional, defaults to all 1.0)
/// * `intercept_col` - Index of intercept column (typically 0, or None if no intercept)
///
/// # Returns
///
/// The maximum lambda value for the path.
///
/// # Note
///
/// If `alpha = 0` (pure ridge), this returns `f64::INFINITY` since ridge never produces
/// exact zero coefficients. Use [`make_lambda_path`] which handles this case by using
/// a small alpha value.
#[allow(clippy::needless_range_loop)]
pub fn compute_lambda_max(
    x: &Matrix,
    y: &[f64],
    alpha: f64,
    penalty_factor: Option<&[f64]>,
    intercept_col: Option<usize>,
) -> f64 {
    if alpha <= 0.0 {
        return f64::INFINITY;
    }

    let n = x.rows as f64;
    let p = x.cols;

    let mut max_corr: f64 = 0.0;

    for j in 0..p {
        // Skip intercept column
        if let Some(ic) = intercept_col {
            if j == ic {
                continue;
            }
        }

        // Compute absolute correlation: |x_j^T y|
        // Matrix is row-major, so we iterate through rows for each column
        let mut corr = 0.0;
        for i in 0..x.rows {
            corr += x.get(i, j) * y[i];
        }
        let corr = corr.abs();

        // Apply penalty factor if provided
        let effective_corr = if let Some(pf) = penalty_factor {
            if j < pf.len() && pf[j] > 0.0 {
                corr / pf[j]
            } else {
                corr
            }
        } else {
            corr
        };

        max_corr = max_corr.max(effective_corr);
    }

    max_corr / (n * alpha)
}

/// Generates a lambda path from lambda_max down to lambda_min.
///
/// This creates a logarithmically-spaced sequence of lambda values, matching
/// glmnet's approach for regularization paths.
///
/// # Arguments
///
/// * `x` - Standardized design matrix (n × p)
/// * `y` - Centered response vector (n elements)
/// * `options` - Lambda path generation options
/// * `penalty_factor` - Optional per-feature penalty factors
/// * `intercept_col` - Index of intercept column (typically 0)
///
/// # Returns
///
/// A vector of lambda values in **decreasing** order (largest to smallest).
///
/// # Lambda Sequence
///
/// The lambda values are logarithmically spaced:
/// ```text
/// lambda[k] = lambda_max * exp(log(lambda_min_ratio) * k / (nlambda - 1))
/// ```
///
/// For ridge (`alpha ≈ 0`), we use a small alpha value to compute a finite
/// `lambda_max`, then use that lambda sequence for the actual ridge fit.
///
/// # Default lambda_min_ratio
///
/// Following glmnet:
/// - If `n >= p`: `lambda_min_ratio = 0.0001`
/// - If `n < p`: `lambda_min_ratio = 0.01`
pub fn make_lambda_path(
    x: &Matrix,
    y: &[f64],
    options: &LambdaPathOptions,
    penalty_factor: Option<&[f64]>,
    intercept_col: Option<usize>,
) -> Vec<f64> {
    let n = x.rows;
    let p = x.cols;

    // Determine default lambda_min_ratio
    let default_min_ratio = if n >= p { 0.0001 } else { 0.01 };
    let lambda_min_ratio = options.lambda_min_ratio.unwrap_or(default_min_ratio);

    // For pure ridge (alpha = 0), use a small alpha to compute lambda_max
    let alpha_for_lambda_max = if options.alpha <= 0.0 {
        options.eps_for_ridge
    } else {
        options.alpha
    };

    // Compute lambda_max
    let lambda_max = compute_lambda_max(x, y, alpha_for_lambda_max, penalty_factor, intercept_col);

    // Handle the case where lambda_max is infinite or very small
    if !lambda_max.is_finite() || lambda_max <= 0.0 {
        // For ridge with no good lambda_max, return a default path
        return (0..options.nlambda)
            .map(|k| {
                let t = k as f64 / (options.nlambda - 1) as f64;
                10.0_f64.powf(2.0 * (1.0 - t)) // 10^2 down to 10^0
            })
            .collect();
    }

    let _lambda_min = lambda_min_ratio * lambda_max;

    // Generate logarithmically spaced lambdas (decreasing)
    (0..options.nlambda)
        .map(|k| {
            let t = k as f64 / (options.nlambda - 1) as f64;
            lambda_max * (lambda_min_ratio.powf(t))
        })
        .collect()
}

/// Extracts a specific set of lambdas from a path.
///
/// This is useful when you want to evaluate at specific lambda values
/// rather than using the full path.
///
/// # Arguments
///
/// * `full_path` - The complete lambda path (must be in decreasing order)
/// * `indices` - Indices of lambdas to extract
///
/// # Returns
///
/// A new vector containing the specified lambda values.
pub fn extract_lambdas(full_path: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&i| full_path[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_lambda_max() {
        // Simple test: X = [1, x], y = [1, 2, 3]
        let x_data = vec![1.0, -1.0, 1.0, 0.0, 1.0, 1.0];
        let x = Matrix::new(3, 2, x_data);
        let y = vec![1.0, 2.0, 3.0];

        // Center y
        let y_mean: f64 = y.iter().sum::<f64>() / 3.0;
        let y_centered: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

        let lambda_max = compute_lambda_max(&x, &y_centered, 1.0, None, Some(0));

        // x^T y for column 1: -1*0 + 0*0 + 1*0 = 0 (since y is centered)
        // Actually y_centered = [-1, 0, 1], so x^T y = 1*(-1) + 0*0 + 1*1 = 0
        // y = [1,2,3], y_mean = 2, y_centered = [-1, 0, 1]
        // back checks...
        // Column 1 of X: [-1, 0, 1]
        // dot = (-1)*(-1) + 0*0 + 1*1 = 1 + 0 + 1 = 2
        // lambda_max = 2 / (3 * 1) = 2/3
        assert!((lambda_max - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_make_lambda_path_decreasing() {
        let x_data = vec![1.0, -1.0, 1.0, 0.0, 1.0, 1.0];
        let x = Matrix::new(3, 2, x_data);
        let y_centered = vec![-1.0, 0.0, 1.0];

        let options = LambdaPathOptions {
            nlambda: 10,
            lambda_min_ratio: Some(0.01),
            alpha: 1.0,
            eps_for_ridge: 0.001,
        };

        let path = make_lambda_path(&x, &y_centered, &options, None, Some(0));

        assert_eq!(path.len(), 10);

        // Check that path is decreasing
        for i in 1..path.len() {
            assert!(path[i] < path[i - 1]);
        }

        // First value should be lambda_max, last should be lambda_max * 0.01
        let lambda_max = 2.0 / 3.0;
        assert!((path[0] - lambda_max).abs() < 1e-10);
        assert!((path[9] - lambda_max * 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_lambda_max_ridge() {
        let x_data = vec![1.0, -1.0, 1.0, 0.0, 1.0, 1.0];
        let x = Matrix::new(3, 2, x_data);
        let y = vec![-1.0, 0.0, 1.0];

        // For alpha = 0 (ridge), lambda_max should be infinite
        let lambda_max = compute_lambda_max(&x, &y, 0.0, None, Some(0));
        assert!(lambda_max.is_infinite());
    }

    #[test]
    fn test_extract_lambdas() {
        let path = vec![10.0, 5.0, 2.5, 1.25, 0.625];
        let indices = vec![0, 2, 4];
        let extracted = extract_lambdas(&path, &indices);

        assert_eq!(extracted, vec![10.0, 2.5, 0.625]);
    }
}
