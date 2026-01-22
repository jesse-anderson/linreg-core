//! Lasso regression (L1-regularized linear regression).
//!
//! This module provides lasso regression implementation using cyclical coordinate
//! descent with soft-thresholding, matching glmnet's approach.
//!
//! # Lasso Regression Objective
//!
//! Lasso regression solves:
//!
//! ```text
//! minimize over (β₀, β):
//!
//!     (1/(2n)) * Σᵢ (yᵢ - β₀ - xᵢᵀβ)² + λ * ||β||₁
//! ```
//!
//! The intercept `β₀` is **not penalized**.
//!
//! # Solution Method
//!
//! Uses cyclical coordinate descent with soft-thresholding:
//!
//! 1. For standardized X, each coordinate update has a closed form
//! 2. Soft-thresholding operator: S(z, γ) = sign(z) * max(|z| - γ, 0)
//! 3. Warm starts along lambda path for efficiency

use crate::error::{Error, Result};
use crate::linalg::Matrix;
use crate::regularized::preprocess::{
    predict, standardize_xy, unstandardize_coefficients, StandardizeOptions,
};

#[cfg(feature = "wasm")]
use serde::Serialize;

/// Soft-thresholding operator: S(z, γ) = sign(z) * max(|z| - γ, 0).
///
/// # Arguments
///
/// * `z` - Input value
/// * `gamma` - Threshold value (must be >= 0)
///
/// # Returns
///
/// The soft-thresholded value.
///
/// # Formula
///
/// ```text
/// S(z, γ) = {
///     z - γ    if z > 0 and |z| > γ
///     z + γ    if z < 0 and |z| > γ
///     0        if |z| <= γ
/// }
/// ```
pub fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if gamma < 0.0 {
        panic!("Soft threshold gamma must be non-negative");
    }
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

/// Options for lasso regression fitting.
///
/// # Fields
///
/// * `lambda` - Regularization strength (single value)
/// * `intercept` - Whether to include an intercept term (default: true)
/// * `standardize` - Whether to standardize predictors (default: true)
/// * `max_iter` - Maximum iterations per lambda (default: 1000)
/// * `tol` - Convergence tolerance (default: 1e-7)
/// * `penalty_factor` - Optional per-feature penalty factors
#[derive(Clone, Debug)]
pub struct LassoFitOptions {
    /// Regularization strength (must be >= 0)
    pub lambda: f64,
    /// Whether to include an intercept
    pub intercept: bool,
    /// Whether to standardize predictors
    pub standardize: bool,
    /// Maximum coordinate descent iterations
    pub max_iter: usize,
    /// Convergence tolerance on coefficient changes
    pub tol: f64,
    /// Per-feature penalty factors (optional)
    pub penalty_factor: Option<Vec<f64>>,
}

impl Default for LassoFitOptions {
    fn default() -> Self {
        LassoFitOptions {
            lambda: 1.0,
            intercept: true,
            standardize: true,
            max_iter: 1000,
            tol: 1e-7,
            penalty_factor: None,
        }
    }
}

/// Result of a lasso regression fit.
///
/// # Fields
///
/// * `lambda` - The lambda value used for fitting
/// * `intercept` - Intercept coefficient (on original scale)
/// * `coefficients` - Slope coefficients (on original scale, may contain zeros)
/// * `fitted_values` - In-sample predictions
/// * `residuals` - Residuals (y - fitted_values)
/// * `n_nonzero` - Number of non-zero coefficients (excluding intercept)
/// * `iterations` - Number of coordinate descent iterations
/// * `converged` - Whether the algorithm converged
/// * `r_squared` - R² (coefficient of determination)
/// * `adj_r_squared` - Adjusted R² (using effective df based on n_nonzero)
/// * `mse` - Mean squared error
/// * `rmse` - Root mean squared error
/// * `mae` - Mean absolute error
#[derive(Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(Serialize))]
pub struct LassoFit {
    /// Lambda value used for fitting
    pub lambda: f64,
    /// Intercept on original scale
    pub intercept: f64,
    /// Slope coefficients on original scale
    pub coefficients: Vec<f64>,
    /// Fitted values
    pub fitted_values: Vec<f64>,
    /// Residuals
    pub residuals: Vec<f64>,
    /// Number of non-zero coefficients
    pub n_nonzero: usize,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// R² (coefficient of determination)
    pub r_squared: f64,
    /// Adjusted R² (penalized for effective number of parameters)
    pub adj_r_squared: f64,
    /// Mean squared error
    pub mse: f64,
    /// Root mean squared error
    pub rmse: f64,
    /// Mean absolute error
    pub mae: f64,
}

/// Fits lasso regression for a single lambda value.
///
/// # Arguments
///
/// * `x` - Design matrix (n × p). Should include intercept column if `intercept=true`.
/// * `y` - Response vector (n elements)
/// * `options` - Lasso fitting options
///
/// # Returns
///
/// A [`LassoFit`] containing the fit results.
///
/// # Errors
///
/// Returns an error if:
/// - `lambda < 0`
/// - Dimensions don't match
/// - Maximum iterations reached without convergence
///
/// # Algorithm
///
/// Uses cyclical coordinate descent:
/// 1. Standardize X and center y (if requested)
/// 2. Initialize coefficients (zeros or warm start)
/// 3. For each feature j:
///    - Compute partial residual: r = y - X_{-j} * beta_{-j}
///    - Compute correlation: rho_j = X_j^T * r / n
///    - Apply soft-thresholding: beta_j = S(rho_j, lambda) / (1 + 0)
///    - (For lasso with standardized X, denominator is 1)
/// 4. Check for convergence
/// 5. Unstandardize coefficients
///
/// # Example
///
/// ```rust,no_run
/// use linreg_core::linalg::Matrix;
/// use linreg_core::regularized::lasso::{lasso_fit, LassoFitOptions};
///
/// let x = Matrix::new(3, 2, vec![
///     1.0, 2.0,
///     1.0, 3.0,
///     1.0, 4.0,
/// ]);
/// let y = vec![3.0, 5.0, 7.0];
///
/// let options = LassoFitOptions {
///     lambda: 1.0,
///     intercept: true,
///     standardize: true,
///     ..Default::default()
/// };
///
/// let fit = lasso_fit(&x, &y, &options).unwrap();
/// println!("Non-zero coefficients: {}", fit.n_nonzero);
/// ```
pub fn lasso_fit(x: &Matrix, y: &[f64], options: &LassoFitOptions) -> Result<LassoFit> {
    if options.lambda < 0.0 {
        return Err(Error::InvalidInput(
            "Lambda must be non-negative for lasso regression".to_string(),
        ));
    }

    let n = x.rows;
    let p = x.cols;

    if y.len() != n {
        return Err(Error::DimensionMismatch(format!(
            "Length of y ({}) must match number of rows in X ({})",
            y.len(),
            n
        )));
    }

    // Handle zero lambda: just do OLS
    if options.lambda == 0.0 {
        return lasso_ols_fit(x, y, options);
    }

    // Standardize X and center y
    let std_options = StandardizeOptions {
        intercept: options.intercept,
        standardize_x: options.standardize,
        standardize_y: false,
    };

    let (x_std, y_centered, std_info) = standardize_xy(x, y, &std_options);

    // Initialize coefficients to zero
    let mut beta_std = vec![0.0; p];

    // Determine which columns are penalized
    let start_col = if options.intercept { 1 } else { 0 };

    // Run coordinate descent
    let (iterations, converged) = coordinate_descent(
        &x_std,
        &y_centered,
        &mut beta_std,
        options.lambda,
        start_col,
        options.max_iter,
        options.tol,
        options.penalty_factor.as_deref(),
    )?;

    // Unstandardize coefficients (beta_orig now contains only slope coefficients)
    let (intercept, beta_orig) = unstandardize_coefficients(&beta_std, &std_info);

    // Count non-zero coefficients (beta_orig already excludes intercept col coefficient)
    let n_nonzero = beta_orig.iter().filter(|&&b| b.abs() > 0.0).count();

    // Compute fitted values and residuals
    let fitted = predict(x, intercept, &beta_orig);
    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, yh)| yi - yh)
        .collect();

    // Compute model fit statistics
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
    let r_squared = if ss_tot > 1e-10 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    // For lasso, effective df = (intercept) + n_nonzero
    // Adjusted R² uses effective degrees of freedom
    let eff_df = 1.0 + n_nonzero as f64; // intercept + non-zero coefficients
    let adj_r_squared = if ss_tot > 1e-10 && n > eff_df as usize {
        1.0 - (1.0 - r_squared) * ((n - 1) as f64 / (n as f64 - eff_df))
    } else {
        r_squared
    };

    let mse = ss_res / (n as f64 - eff_df).max(1.0);
    let rmse = mse.sqrt();
    let mae: f64 = residuals.iter().map(|r| r.abs()).sum::<f64>() / n as f64;

    Ok(LassoFit {
        lambda: options.lambda,
        intercept,
        coefficients: beta_orig,
        fitted_values: fitted,
        residuals,
        n_nonzero,
        iterations,
        converged,
        r_squared,
        adj_r_squared,
        mse,
        rmse,
        mae,
    })
}

/// Coordinate descent for lasso.
///
/// # Arguments
///
/// * `x` - Standardized design matrix
/// * `y` - Centered response
/// * `beta` - Coefficient vector (modified in place)
/// * `lambda` - Regularization strength
/// * `start_col` - First penalized column index
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
/// * `penalty_factor` - Optional per-feature penalties
///
/// # Returns
///
/// A tuple `(iterations, converged)` indicating the number of iterations
/// and whether convergence was achieved.
fn coordinate_descent(
    x: &Matrix,
    y: &[f64],
    beta: &mut [f64],
    lambda: f64,
    start_col: usize,
    max_iter: usize,
    tol: f64,
    penalty_factor: Option<&[f64]>,
) -> Result<(usize, bool)> {
    let n = x.rows;
    let p = x.cols;

    let mut residuals: Vec<f64> = y.to_vec();
    let mut converged = false;

    // Initialize with current beta values
    for iter in 0..max_iter {
        let _beta_old = beta.to_vec();
        let mut max_change: f64 = 0.0;

        // Update each coordinate
        for j in start_col..p {
            // Skip if penalty factor is infinite (always excluded)
            if let Some(pf) = penalty_factor {
                if j < pf.len() && pf[j] == f64::INFINITY {
                    beta[j] = 0.0;
                    continue;
                }
            }

            // Compute rho_j = x_j^T * r / n (where r includes x_j * beta_j)
            // Actually: r = y - X*beta, and we want x_j^T * (r + x_j * beta_j) / n
            // This equals x_j^T * (y - X_{-j} * beta_{-j}) / n

            // First, remove the contribution of feature j from residuals
            let old_beta_j = beta[j];
            for i in 0..n {
                residuals[i] += x.get(i, j) * old_beta_j;
            }

            // Compute rho_j = x_j^T * residuals / n
            let mut rho_j = 0.0;
            for i in 0..n {
                rho_j += x.get(i, j) * residuals[i];
            }
            rho_j /= n as f64;

            // Get penalty factor for this feature
            let pf = penalty_factor
                .and_then(|pf| pf.get(j))
                .copied()
                .unwrap_or(1.0);

            // Apply soft-thresholding
            // For standardized X, denominator is 1
            let threshold = lambda * pf;
            let new_beta_j = soft_threshold(rho_j, threshold);

            // Update residuals with new coefficient
            for i in 0..n {
                residuals[i] -= x.get(i, j) * new_beta_j;
            }

            beta[j] = new_beta_j;

            // Track maximum change
            let change = (new_beta_j - old_beta_j).abs();
            max_change = max_change.max(change);
        }

        // Check convergence
        if max_change < tol {
            converged = true;
            return Ok((iter + 1, converged));
        }
    }

    Ok((max_iter, converged))
}

/// OLS fit for lambda = 0 (special case of lasso).
fn lasso_ols_fit(x: &Matrix, y: &[f64], options: &LassoFitOptions) -> Result<LassoFit> {
    // Use QR decomposition for OLS on original (non-standardized) data
    let (q, r) = x.qr();

    // Solve R * beta = Q^T * y
    let n = x.rows;
    let p = x.cols;
    let mut qty = vec![0.0; p];

    for i in 0..p {
        for k in 0..n {
            qty[i] += q.get(k, i) * y[k];
        }
    }

    let mut beta = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = qty[i];
        for j in (i + 1)..p {
            sum -= r.get(i, j) * beta[j];
        }
        beta[i] = sum / r.get(i, i);
    }

    // Extract intercept and slope coefficients directly (no unstandardization needed)
    // OLS on original data gives coefficients on original scale
    let (intercept, beta_orig) = if options.intercept {
        // beta[0] is intercept, beta[1..] are slopes
        let slopes: Vec<f64> = beta[1..].to_vec();
        (beta[0], slopes)
    } else {
        // No intercept, all coefficients are slopes
        (0.0, beta)
    };

    // Compute fitted values and residuals
    let fitted = predict(x, intercept, &beta_orig);
    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, yh)| yi - yh)
        .collect();

    // Count non-zero coefficients (beta_orig already excludes intercept col coefficient)
    let n_nonzero = beta_orig.iter().filter(|&&b| b.abs() > 0.0).count();

    // Compute model fit statistics
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
    let r_squared = if ss_tot > 1e-10 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    // Adjusted R²
    let eff_df = n_nonzero as f64;
    let adj_r_squared = if ss_tot > 1e-10 && n > eff_df as usize {
        1.0 - (1.0 - r_squared) * ((n - 1) as f64 / (n as f64 - eff_df))
    } else {
        r_squared
    };

    let mse = ss_res / (n as f64 - p as f64);
    let rmse = mse.sqrt();
    let mae: f64 = residuals.iter().map(|r| r.abs()).sum::<f64>() / n as f64;

    Ok(LassoFit {
        lambda: 0.0,
        intercept,
        coefficients: beta_orig,
        fitted_values: fitted,
        residuals,
        n_nonzero,
        iterations: 1,
        converged: true,
        r_squared,
        adj_r_squared,
        mse,
        rmse,
        mae,
    })
}

/// Makes predictions using a lasso regression fit.
///
/// # Arguments
///
/// * `fit` - The lasso regression fit result
/// * `x_new` - New data matrix (n_new × p)
///
/// # Returns
///
/// Predictions for each row in x_new.
pub fn predict_lasso(fit: &LassoFit, x_new: &Matrix) -> Vec<f64> {
    predict(x_new, fit.intercept, &fit.coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_threshold() {
        assert_eq!(soft_threshold(5.0, 2.0), 3.0);
        assert_eq!(soft_threshold(-5.0, 2.0), -3.0);
        assert_eq!(soft_threshold(1.0, 2.0), 0.0);
        assert_eq!(soft_threshold(-1.0, 2.0), 0.0);
        assert_eq!(soft_threshold(2.0, 2.0), 0.0);
        assert_eq!(soft_threshold(-2.0, 2.0), 0.0);
        assert_eq!(soft_threshold(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_lasso_fit_simple() {
        // Simple test: y = 2*x with perfect linear relationship
        let x_data = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
        let x = Matrix::new(4, 2, x_data);
        let y = vec![2.0, 4.0, 6.0, 8.0];

        let options = LassoFitOptions {
            lambda: 0.01, // Very small lambda for near-OLS solution
            intercept: true,
            standardize: true, // Standardize for better convergence
            ..Default::default()
        };

        let fit = lasso_fit(&x, &y, &options).unwrap();

        // With small lambda, should get a good fit
        assert!(fit.converged);
        assert!(fit.n_nonzero > 0);

        // Predictions should be close to actual values
        for i in 0..4 {
            assert!((fit.fitted_values[i] - y[i]).abs() < 0.5);
        }
    }

    #[test]
    fn test_lasso_with_large_lambda() {
        let x_data = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let x = Matrix::new(3, 2, x_data);
        let y = vec![2.0, 4.0, 6.0];

        let options = LassoFitOptions {
            lambda: 100.0,
            intercept: true,
            standardize: false,
            ..Default::default()
        };

        let fit = lasso_fit(&x, &y, &options).unwrap();

        // With large lambda, all coefficients should be zero
        // Only intercept should be non-zero (equal to mean of y)
        assert_eq!(fit.n_nonzero, 0);
        // coefficients[0] is the first (and only) slope coefficient
        assert!((fit.coefficients[0]).abs() < 1e-10);
    }

    #[test]
    fn test_lasso_zero_lambda_is_ols() {
        let x_data = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let x = Matrix::new(3, 2, x_data);
        let y = vec![2.0, 4.0, 6.0];

        let options = LassoFitOptions {
            lambda: 0.0,
            intercept: true,
            standardize: false,
            ..Default::default()
        };

        let fit = lasso_fit(&x, &y, &options).unwrap();

        // Should be close to perfect fit
        assert!((fit.fitted_values[0] - 2.0).abs() < 1e-6);
        assert!((fit.fitted_values[1] - 4.0).abs() < 1e-6);
        assert!((fit.fitted_values[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_predict_lasso() {
        let x_data = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let x = Matrix::new(3, 2, x_data);
        let y = vec![2.0, 4.0, 6.0];

        let options = LassoFitOptions {
            lambda: 0.1,
            intercept: true,
            standardize: false,
            ..Default::default()
        };

        let fit = lasso_fit(&x, &y, &options).unwrap();
        let preds = predict_lasso(&fit, &x);

        // Predictions on training data should equal fitted values
        for i in 0..3 {
            assert!((preds[i] - fit.fitted_values[i]).abs() < 1e-10);
        }
    }
}
