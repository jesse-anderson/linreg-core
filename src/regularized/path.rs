//! Lambda path generation for regularized regression.
//!
//! This module provides utilities for generating a sequence of lambda values
//! for regularization paths, matching the R glmnet implementation's output/approach.
//!
//! # Lambda Path Construction
//!
//! The lambda path follows this geometric decay pattern (following the glmnet R implementation):
//!
//! ```text
//! lambda[0] = infinity            // effectively infinite (all coefficients zero)
//! lambda[1] = lambda_decay_factor * lambda_max           // First "real" lambda
//! lambda[k] = lambda[k-1] * lambda_decay_factor    // Geometric decay
//! ```
//!
//! Where (mapping implementation variables to the Friedman et al. paper):
//! - `LAMBDA_EFFECTIVE_INFINITY = infinity`: Sentinel value (effectively infinite lambda)
//! - `lambda_max`: Theoretical maximum lambda (paper: $\lambda_{max}$)
//! - `lambda_decay_factor`: Geometric decay factor (paper: $(\epsilon)^{1/(K-1)}$)
//! - `lambda_min_ratio`: Minimum lambda ratio (paper: $\epsilon$)
//! - `eps = 1.0e-6`: Implementation constant for stability
//!
//! # Note on `LAMBDA_EFFECTIVE_INFINITY`
//!
//! The first lambda value is set to infinity, which produces all-zero coefficients.
//! This matches glmnet's behavior and provides a complete view of the regularization
//! path, showing the exact point where each coefficient enters the model as lambda
//! decreases. Users who don't need this can simply ignore the first element.

use crate::linalg::Matrix;

/// Represents effectively infinite lambda.
///
/// At this lambda value, all penalized coefficients are zero.
const LAMBDA_EFFECTIVE_INFINITY: f64 = f64::INFINITY;

/// epsilon constant - minimum lambda_min_ratio value.
const LAMBDA_MIN_RATIO_MINIMUM: f64 = 1.0e-6;

/// Options for generating a lambda path.
///
/// # Fields
///
/// * `nlambda` - Number of lambda values (default: 100)
/// * `lambda_min_ratio` - Lambda minimum ratio (default: None, which auto-selects based on n vs p)
/// * `alpha` - Elastic net mixing parameter (0 = ridge, 1 = lasso)
/// * `eps_for_ridge` - Small alpha to use for computing lambda_max when alpha=0 (default: 0.001)
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::path::LambdaPathOptions;
/// let opts = LambdaPathOptions {
///     nlambda: 50,
///     lambda_min_ratio: Some(0.01),
///     alpha: 0.5,
///     eps_for_ridge: 0.001,
/// };
///
/// assert_eq!(opts.nlambda, 50);
/// assert_eq!(opts.alpha, 0.5);
/// ```
#[derive(Clone, Debug)]
pub struct LambdaPathOptions {
    /// Number of lambda values to generate
    pub nlambda: usize,
    /// Lambda minimum ratio.
    /// If None, auto-selects: 0.0001 if n >= p, else 0.01
    pub lambda_min_ratio: Option<f64>,
    /// Elastic net mixing parameter (0 = ridge, 1 = lasso)
    pub alpha: f64,
    /// Small value to use for ridge regression max lambda calculation
    /// to avoid numerical issues (default: 1e-3)
    pub eps_for_ridge: f64,
}

impl Default for LambdaPathOptions {
    fn default() -> Self {
        LambdaPathOptions {
            nlambda: 100,
            lambda_min_ratio: None,
            alpha: 1.0,
            eps_for_ridge: 1e-3,
        }
    }
}

/// Computes `lambda_max`: the smallest lambda for which all
/// penalized coefficients are zero.
///
/// Formula:
/// ```text
/// lambda_max = max(|X_j^T y|/vp(j)) / max(alpha, 1e-3)
/// ```
/// where `g(j) = |X_j^T y|` is the absolute correlation.
///
/// Key points:
/// - y is assumed to be STANDARDIZED to unit norm (||y|| = 1)
/// - X columns are assumed to be STANDARDIZED to unit norm
/// - The `1e-3` minimum prevents division issues for pure ridge
///
/// # Arguments
///
/// * `x` - Standardized design matrix (n × p), first column is intercept if present
/// * `y` - Standardized response vector (||y|| = 1)
/// * `alpha` - Elastic net mixing parameter
/// * `penalty_factor` - Per-feature penalty factors (optional, defaults to all 1.0)
/// * `intercept_col` - Index of intercept column (typically 0, or None if no intercept)
///
/// # Returns
///
/// The `lambda_max` value used by glmnet for lambda path construction.
#[allow(clippy::needless_range_loop)]
pub fn compute_lambda_max(
    x: &Matrix,
    y: &[f64],
    alpha: f64,
    penalty_factor: Option<&[f64]>,
    intercept_col: Option<usize>,
) -> f64 {
    // Use max(alpha, 1e-3) to avoid division issues
    let alpha_clamped = alpha.max(1e-3);

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
        let effective_corr = if let Some(penalty_factors) = penalty_factor {
            if j < penalty_factors.len() && penalty_factors[j] > 0.0 {
                corr / penalty_factors[j]
            } else {
                corr
            }
        } else {
            corr
        };

        max_corr = max_corr.max(effective_corr);
    }

    // Formula: lambda_max = max(|X^T y|) / max(alpha, 1e-3)
    max_corr / alpha_clamped
}

/// Generates a lambda path.
///
/// ```text
/// lambda[0] = inf                  // Large value (all coefficients zero)
/// lambda[1] = lambda_decay_factor * lambda_max           // First real lambda
/// lambda[k] = lambda[k-1] * lambda_decay_factor    // Geometric decay
/// ```
///
/// Where:
/// - `lambda_max = max(|X^T y|) / max(alpha, 1e-3)`: Theoretical lambda_max
/// - `lambda_decay_factor = max(lambda_min_ratio, eps)^(1/(nlambda-1))`: Geometric decay factor
/// - `eps = 1.0e-6`: Minimum lambda_min_ratio value
///
/// # Arguments
///
/// * `x` - Standardized design matrix (n × p), columns are unit norm
/// * `y` - Standardized response vector (||y|| = 1)
/// * `options` - Lambda path generation options
/// * `penalty_factor` - Optional per-feature penalty factors
/// * `intercept_col` - Index of intercept column (typically 0)
///
/// # Returns
///
/// A vector of lambda values in **decreasing** order.
///
/// # First Lambda (LAMBDA_EFFECTIVE_INFINITY)
///
/// The first lambda value is set to `infinity`, which effectively produces
/// all-zero coefficients. This matches R's cursed behavior.
/// PROOF OF CONCEPT/R EQUIVALENCE: We may want to make this optional in future versions.
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
    let default_lambda_min_ratio = if n >= p { 0.0001 } else { 0.01 };
    let lambda_min_ratio = options.lambda_min_ratio.unwrap_or(default_lambda_min_ratio);

    // Compute geometric decay factor: lambda_decay_factor = max(lambda_min_ratio, eps)^(1/(nlambda-1))
    let lambda_min_ratio_clamped = lambda_min_ratio.max(LAMBDA_MIN_RATIO_MINIMUM);
    let lambda_decay_factor = lambda_min_ratio_clamped.powf(1.0 / (options.nlambda - 1) as f64);

    // Compute lambda_max = max(|X^T y|) / max(alpha, 1e-3)
    let lambda_max = compute_lambda_max(x, y, options.alpha, penalty_factor, intercept_col);

    // Build lambda path following glmnet's algorithm:
    // lambda[0] = INF
    // lambda[1] = lambda_decay_factor * lambda_max
    // lambda[k] = lambda[k-1] * lambda_decay_factor
    let mut lambdas = Vec::with_capacity(options.nlambda);

    for k in 0..options.nlambda {
        if k == 0 {
            // First lambda: effectively infinite
            lambdas.push(LAMBDA_EFFECTIVE_INFINITY);
        } else if k == 1 {
            // Second lambda: lambda_decay_factor * lambda_max
            lambdas.push(lambda_decay_factor * lambda_max);
        } else {
            // Remaining lambdas: geometric decay
            lambdas.push(lambdas[k - 1] * lambda_decay_factor);
        }
    }

    lambdas
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
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::path::extract_lambdas;
/// let path = vec![1.0, 0.5, 0.25, 0.125, 0.0625];
/// let indices = vec![0, 2, 4];
/// let extracted = extract_lambdas(&path, &indices);
/// assert_eq!(extracted, vec![1.0, 0.25, 0.0625]);
/// ```
pub fn extract_lambdas(full_path: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&i| full_path[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_lambda_max() {
        // Simple test: X = [1, x], y standardized to unit norm
        let x_data = vec![1.0, -1.0, 1.0, 0.0, 1.0, 1.0];
        let x = Matrix::new(3, 2, x_data);

        // Standardize y to unit norm (||y|| = 1)
        let y = vec![-1.0, 0.0, 1.0];
        let y_norm: f64 = y.iter().map(|&yi| yi * yi).sum::<f64>().sqrt();
        let y_standardized: Vec<f64> = y.iter().map(|&yi| yi / y_norm).collect();

        let lambda_max = compute_lambda_max(&x, &y_standardized, 1.0, None, Some(0));

        // Column 1 of X: [-1, 0, 1]
        // y_standardized = [-1/sqrt(2), 0, 1/sqrt(2)]
        // dot = (-1)*(-1/sqrt(2)) + 0*0 + 1*(1/sqrt(2)) = 2/sqrt(2) = sqrt(2)
        // lambda_max = sqrt(2) / max(1.0, 1e-3) = sqrt(2)
        assert!((lambda_max - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_make_lambda_path_glmnet_style() {
        let x_data = vec![1.0, -1.0, 1.0, 0.0, 1.0, 1.0];
        let x = Matrix::new(3, 2, x_data);
        let y = vec![-1.0, 0.0, 1.0];

        // Standardize y to unit norm
        let y_norm: f64 = y.iter().map(|&yi| yi * yi).sum::<f64>().sqrt();
        let y_standardized: Vec<f64> = y.iter().map(|&yi| yi / y_norm).collect();

        let options = LambdaPathOptions {
            nlambda: 10,
            lambda_min_ratio: Some(0.01),
            alpha: 1.0,
            eps_for_ridge: 0.001,
        };

        let path = make_lambda_path(&x, &y_standardized, &options, None, Some(0));

        assert_eq!(path.len(), 10);

        // First lambda should be LAMBDA_EFFECTIVE_INFINITY (effectively infinite)
        assert_eq!(path[0], LAMBDA_EFFECTIVE_INFINITY);

        // Second lambda should be: lambda_decay_factor * lambda_max
        // where lambda_max = sqrt(2) and lambda_decay_factor = 0.01^(1/9)
        let lambda_max = 2.0_f64.sqrt();
        let lambda_decay_factor = 0.01_f64.powf(1.0 / 9.0);
        assert!((path[1] - lambda_decay_factor * lambda_max).abs() < 1e-10);

        // Path should be decreasing (each lambda < previous)
        for i in 1..path.len() {
            assert!(path[i] < path[i - 1]);
        }
    }

    #[test]
    fn test_lambda_max_with_small_alpha() {
        let x_data = vec![1.0, -1.0, 1.0, 0.0, 1.0, 1.0];
        let x = Matrix::new(3, 2, x_data);
        let y = vec![-1.0, 0.0, 1.0];

        let y_norm: f64 = y.iter().map(|&yi| yi * yi).sum::<f64>().sqrt();
        let y_standardized: Vec<f64> = y.iter().map(|&yi| yi / y_norm).collect();

        // For very small alpha, should use max(alpha, 1e-3) = 1e-3
        let lambda_max = compute_lambda_max(&x, &y_standardized, 0.0001, None, Some(0));

        // lambda_max = sqrt(2) / max(0.0001, 1e-3) = sqrt(2) / 1e-3
        let expected = 2.0_f64.sqrt() / 1e-3;
        assert!((lambda_max - expected).abs() < 1e-10);
    }

    #[test]
    fn test_extract_lambdas() {
        let path = vec![10.0, 5.0, 2.5, 1.25, 0.625];
        let indices = vec![0, 2, 4];
        let extracted = extract_lambdas(&path, &indices);

        assert_eq!(extracted, vec![10.0, 2.5, 0.625]);
    }
}
