//! Data preprocessing for regularized regression.
//!
//! This module provides standardization utilities that match glmnet output behavior:
//!
//! - Predictors are centered and scaled (if enabled)
//! - The intercept column is not penalized, so it's handled specially
//! - Coefficients can be unstandardized back to the original scale
//! - Observation weights are supported for weighted regression
//!
//! # Standardization Convention
//!
//! The scaling factor used is `sqrt(sum(x²) / n)`, which gives unit variance
//! under the 1/n convention (matching the glmnet paper).
//!
//! # Weighted Standardization
//!
//! When weights are provided, they are first normalized to sum to 1:
//! `weights_normalized = w / sum(w)`. Then weighted means and variances are computed.

use crate::linalg::Matrix;

/// Information stored during standardization, used to unstandardize coefficients.
///
/// This struct captures all the information needed to transform coefficients
/// from the standardized space back to the original data scale.
///
/// # Fields
///
/// * `x_mean` - Mean of each predictor column (length p)
/// * `x_scale` - Scale factor for each predictor column (length p)
/// * `y_mean` - Mean of response variable
/// * `y_scale` - Scale factor for response (optional, used for lambda path)
/// * `intercept` - Whether an intercept term was included
/// * `standardized_x` - Whether X was standardized
/// * `standardized_y` - Whether y was standardized
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::preprocess::StandardizationInfo;
/// let info = StandardizationInfo {
///     x_mean: vec![0.0, 5.0],
///     x_scale: vec![1.0, 2.0],
///     column_squared_norms: vec![1.0, 1.0],
///     y_mean: 10.0,
///     y_scale: Some(3.0),
///     y_scale_before_sqrt_weights_normalized: Some(3.0),
///     intercept: true,
///     standardized_x: true,
///     standardized_y: false,
/// };
///
/// assert_eq!(info.x_mean.len(), 2);
/// assert!(info.intercept);
/// ```
#[derive(Clone, Debug)]
pub struct StandardizationInfo {
    /// Mean of each predictor column
    pub x_mean: Vec<f64>,
    /// Scale factor for each predictor column
    pub x_scale: Vec<f64>,
    /// Squared norm of each predictor column after standardization.
    /// This is used in the coordinate descent update denominator.
    /// - With intercept and standardize: column_squared_norms\[j\] = 1.0 (unit norm after centering)
    /// - Without intercept and standardize: column_squared_norms\[j\] = 1.0 + x_squared_mean/x_centered_variance (glmnet formula)
    /// - Without standardize: column_squared_norms\[j\] = ||x_j||^2 (actual squared norm)
    pub column_squared_norms: Vec<f64>,
    /// Mean of response variable
    pub y_mean: f64,
    /// Scale factor for response (for lambda path construction)
    /// This is the norm AFTER sqrt_weights_normalized transformation and centering: sqrt(sum((sqrt_weights_normalized*(y-ym))^2))
    pub y_scale: Option<f64>,
    /// Scale factor for response BEFORE sqrt_weights_normalized transformation: sqrt(sum((y-ym)^2))
    /// This is used for lambda scaling between original and standardized data
    pub y_scale_before_sqrt_weights_normalized: Option<f64>,
    /// Whether an intercept was included
    pub intercept: bool,
    /// Whether X was standardized
    pub standardized_x: bool,
    /// Whether y was standardized
    pub standardized_y: bool,
}

/// Options for standardization.
///
/// # Fields
///
/// * `intercept` - Whether to include/center an intercept (default: true)
/// * `standardize_x` - Whether to standardize predictors (default: true)
/// * `standardize_y` - Whether to standardize response (default: false)
/// * `weights` - Optional observation weights (default: None)
///   If provided, weights are normalized to sum to 1 before use.
///
/// # Note
///
/// Setting `standardize_y` to `true` is mainly useful when you want to match
/// glmnet's lambda sequence exactly. For single-lambda fits, you typically
/// don't need to standardize y.
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::preprocess::StandardizeOptions;
/// let opts = StandardizeOptions {
///     intercept: true,
///     standardize_x: true,
///     standardize_y: false,
///     weights: None,
/// };
///
/// assert!(opts.intercept);
/// assert!(opts.standardize_x);
/// ```
#[derive(Clone, Debug)]
pub struct StandardizeOptions {
    /// Whether to include an intercept (and center X)
    pub intercept: bool,
    /// Whether to standardize predictor columns
    pub standardize_x: bool,
    /// Whether to standardize the response variable
    pub standardize_y: bool,
    /// Optional observation weights (normalized to sum to 1)
    pub weights: Option<Vec<f64>>,
}

impl Default for StandardizeOptions {
    fn default() -> Self {
        StandardizeOptions {
            intercept: true,
            standardize_x: true,
            standardize_y: false,
            weights: None,
        }
    }
}

/// Standardizes X and y for regularized regression (glmnet-compatible).
///
/// This function performs the same standardization as glmnet with
/// `standardize=TRUE`. The first column of X is assumed to be the intercept
/// (all ones) and is NOT standardized.
///
/// # Arguments
///
/// * `x` - Design matrix (n × p). First column should be intercept if `intercept=true`.
/// * `y` - Response vector (n elements)
/// * `options` - Standardization options (including optional observation weights)
///
/// # Returns
///
/// A tuple `(x_standardized, y_standardized, info)` where:
/// - `x_standardized` is the standardized design matrix
/// - `y_standardized` is the (optionally) standardized response
/// - `info` contains the standardization parameters for unstandardization
///
/// # Standardization Details
///
/// ## Unweighted case:
/// For the intercept column (first column, if present):
/// - Not centered (stays as ones)
/// - Not scaled
///
/// For other columns (if `standardize_x=true`):
/// - Centered: `x_centered = x - mean(x)`
/// - Scaled: `x_scaled = x_centered / sqrt(sum(x²))`
///
/// For y (if `standardize_y=true`):
/// - Centered: `y_centered = y - mean(y)`
/// - Scaled: `y_scaled = y_centered / sqrt(sum(y²))`
///
/// ## Weighted case:
/// Weights are first normalized: `weights_normalized = w / sum(w)`, then `sqrt_weights_normalized = sqrt(weights_normalized)`
/// - Weighted mean: `ym = sum(w * y) / sum(w) = sum(weights_normalized * y)`
/// - Weighted variance: `sum(w * (y - ym)^2) / sum(w)`
/// - Data transformed by `sqrt_weights_normalized`: `y_new = sqrt_weights_normalized * (y - ym)`, then scaled
#[allow(clippy::needless_range_loop)]
pub fn standardize_xy(
    x: &Matrix,
    y: &[f64],
    options: &StandardizeOptions,
) -> (Matrix, Vec<f64>, StandardizationInfo) {
    let n = x.rows;
    let p = x.cols;

    // Validate weights if provided
    if let Some(ref w) = options.weights {
        if w.len() != n {
            return (
                Matrix::new(n, p, vec![0.0; n * p]),
                vec![0.0; n],
                StandardizationInfo {
                    x_mean: vec![0.0; p],
                    x_scale: vec![1.0; p],
                    column_squared_norms: vec![0.0; p],
                    y_mean: 0.0,
                    y_scale: None,
                    y_scale_before_sqrt_weights_normalized: None,
                    intercept: options.intercept,
                    standardized_x: options.standardize_x,
                    standardized_y: options.standardize_y,
                },
            );
        }
        if w.iter().any(|&wi| wi < 0.0) {
            panic!("Weights must be non-negative");
        }
    }

    // Prepare normalized weights and sqrt(weights)
    // w = w / sum(w) then sqrt_weights_normalized = sqrt(w)
    let (weights_normalized, sqrt_weights_normalized): (Vec<f64>, Vec<f64>) = if let Some(ref w) = options.weights {
        let w_sum: f64 = w.iter().sum();
        if w_sum > 0.0 {
            let weights_normalized_vec: Vec<f64> = w.iter().map(|&wi| wi / w_sum).collect();
            let sqrt_weights_normalized_vec: Vec<f64> = weights_normalized_vec.iter().map(|&wi| wi.sqrt()).collect();
            (weights_normalized_vec, sqrt_weights_normalized_vec)
        } else {
            (vec![0.0; n], vec![0.0; n])
        }
    } else {
        // No weights: use uniform weights
        let w_uniform = vec![1.0 / n as f64; n];
        let sqrt_weights_normalized_uniform = vec![1.0 / (n as f64).sqrt(); n];
        (w_uniform, sqrt_weights_normalized_uniform)
    };

    let mut x_standardized = x.clone();
    let mut y_standardized = y.to_vec();

    let mut x_mean = vec![0.0; p];
    let mut x_scale = vec![1.0; p];
    let mut column_squared_norms = vec![0.0; p];  // Column squared norms for coordinate descent

    let y_mean = if options.intercept && !y.is_empty() {
        // Weighted mean: ym = sum(w * y)
        weights_normalized.iter().zip(y.iter()).map(|(&w, &yi)| w * yi).sum()
    } else {
        0.0
    };

    // GLMNET: y is ALWAYS scaled to unit norm!
    // This is critical for correct lambda_max computation
    let (y_scale, y_scale_before_sqrt_weights_normalized) = if options.intercept {
        // WITH INTERCEPT: Center y, then scale to unit norm
        // First compute y_scale_before_sqrt_weights_normalized (centered but not sqrt_weights_normalized-transformed)
        let y_centered: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();
        let y_ss_before_sqrt_weights_normalized: f64 = y_centered.iter().map(|&yi| yi * yi).sum();
        let y_scale_before_sqrt_weights_normalized_val = y_ss_before_sqrt_weights_normalized.sqrt();

        // Center y: y_new = sqrt_weights_normalized * (y - ym)
        for (yi, &sqrt_weight) in y_standardized.iter_mut().zip(&sqrt_weights_normalized) {
            *yi = sqrt_weight * (*yi - y_mean);
        }

        // Scale to unit norm (GLMNET always does this!)
        let y_ss: f64 = y_standardized.iter().map(|&yi| yi * yi).sum();
        let y_scale_val = y_ss.sqrt();
        if y_scale_val > 0.0 {
            for yi in y_standardized.iter_mut() {
                *yi /= y_scale_val;
            }
        }
        (Some(y_scale_val), Some(y_scale_before_sqrt_weights_normalized_val))
    } else {
        // WITHOUT INTERCEPT: Don't center y, but DO scale to unit norm (GLMNET output behavior!)
        // y_new = sqrt_weights_normalized * y, then y = y / ||y||
        for (yi, &sqrt_weight) in y_standardized.iter_mut().zip(&sqrt_weights_normalized) {
            *yi *= sqrt_weight;
        }
        let y_ss: f64 = y_standardized.iter().map(|&yi| yi * yi).sum();
        let y_scale_val = y_ss.sqrt();
        if y_scale_val > 0.0 {
            for yi in y_standardized.iter_mut() {
                *yi /= y_scale_val;
            }
        }
        (Some(y_scale_val), Some(y_scale_val))  // y_scale_before_sqrt_weights_normalized = y_scale when no centering
    };

    // Standardize X columns
    // If intercept is present, first column is NOT standardized (it's the intercept column)
    let first_penalized_column_index = if options.intercept { 1 } else { 0 };

    if options.intercept {
        // WITH INTERCEPT (intercept column not standardized)
        for j in first_penalized_column_index..p {
            // Compute weighted column mean and center
            let col_mean: f64 = (0..n)
                .map(|i| x_standardized.get(i, j) * weights_normalized[i])
                .sum();
            x_mean[j] = col_mean;

            // Center the column and apply sqrt_weights_normalized transformation
            // x_new = sqrt_weights_normalized * (x - xm)
            for i in 0..n {
                let val = sqrt_weights_normalized[i] * (x_standardized.get(i, j) - col_mean);
                x_standardized.set(i, j, val);
            }

            // Compute squared norm
            let col_squared_norm_val: f64 = (0..n)
                .map(|i| {
                    let val = x_standardized.get(i, j);
                    val * val
                })
                .sum();

            if options.standardize_x {
                // Scale to unit norm
                let col_scale = col_squared_norm_val.sqrt();
                if col_scale > 0.0 {
                    for i in 0..n {
                        let val = x_standardized.get(i, j) / col_scale;
                        x_standardized.set(i, j, val);
                    }
                    x_scale[j] = col_scale;
                    column_squared_norms[j] = 1.0;  // Unit norm
                }
            } else {
                // No standardization - column_squared_norms stays as the actual squared norm
                column_squared_norms[j] = col_squared_norm_val;
                x_scale[j] = 1.0;
            }
        }
    } else {
        // WITHOUT INTERCEPT (no centering)
        for j in first_penalized_column_index..p {
            x_mean[j] = 0.0;  // No centering

            // Apply sqrt_weights_normalized transformation
            for i in 0..n {
                let val = sqrt_weights_normalized[i] * x_standardized.get(i, j);
                x_standardized.set(i, j, val);
            }

            // Compute squared norm after sqrt_weights_normalized transformation
            let col_squared_norm_val: f64 = (0..n)
                .map(|i| {
                    let val = x_standardized.get(i, j);
                    val * val
                })
                .sum();

            if options.standardize_x {
                // GLMNET special formula for no-intercept case:
                // x_squared_mean = dot_product(sqrt_weights_normalized, x)^2  (squared mean)
                // x_centered_variance = col_squared_norm - x_squared_mean  (variance-like quantity)
                // xs = sqrt(x_centered_variance)
                // column_squared_norms_final = 1.0 + x_squared_mean / x_centered_variance
                let x_squared_mean: f64 = (0..n)
                    .map(|i| sqrt_weights_normalized[i] * x_standardized.get(i, j))
                    .sum::<f64>().powi(2);
                let x_centered_variance = col_squared_norm_val - x_squared_mean;

                if x_centered_variance > 0.0 {
                    let col_scale = x_centered_variance.sqrt();
                    // Scale by col_scale (NOT by sqrt(col_squared_norm_val))
                    for i in 0..n {
                        let val = x_standardized.get(i, j) / col_scale;
                        x_standardized.set(i, j, val);
                    }
                    x_scale[j] = col_scale;
                    column_squared_norms[j] = 1.0 + x_squared_mean / x_centered_variance;  // GLMNET formula
                } else {
                    column_squared_norms[j] = 1.0;
                    x_scale[j] = 1.0;
                }
            } else {
                // No standardization
                column_squared_norms[j] = col_squared_norm_val;
                x_scale[j] = 1.0;
            }
        }
    }

    // If intercept column exists, set its scale to 1.0 (not penalized)
    if options.intercept && p > 0 {
        x_scale[0] = 1.0;
        x_mean[0] = 0.0; // Intercept column has no "mean" to subtract
        column_squared_norms[0] = 1.0;  // Intercept column is not penalized
    }

    let info = StandardizationInfo {
        x_mean,
        x_scale,
        column_squared_norms,
        y_mean,
        y_scale,
        y_scale_before_sqrt_weights_normalized,
        intercept: options.intercept,
        standardized_x: options.standardize_x,
        standardized_y: options.standardize_y,
    };

    (x_standardized, y_standardized, info)
}

/// Unstandardizes coefficients from the standardized space back to original scale.
///
/// This reverses the standardization transformation to get coefficients that
/// can be applied to the original (unscaled) data.
///
/// # Arguments
///
/// * `coefficients_standardized` - Coefficients in standardized space (length p)
/// * `info` - Standardization information from [`standardize_xy`]
///
/// # Returns
///
/// A tuple `(beta0, beta_slopes)` where:
/// - `beta0` is the intercept on the original scale
/// - `beta_slopes` are the slope coefficients only (excluding intercept column coefficient)
///
/// # Unstandardization Formula
///
/// For non-intercept coefficients:
/// ```text
/// β_original[j] = (y_scale * β_std[j]) / x_scale[j]
/// ```
///
/// For the intercept:
/// ```text
/// β₀ = y_mean - Σⱼ x_mean[j] * β_original[j]
/// ```
///
/// If y was not standardized, `y_scale = 1`.
/// If X was not standardized, `x_scale[j] = 1`.
///
/// # Note
///
/// If `intercept=true` in the info, `coefficients_standardized[0]` is assumed to be the intercept
/// coefficient (which is already 0 in the standardized space since X was centered).
/// The returned `beta_slopes` will NOT include this zeroed coefficient - only actual
/// slope coefficients are returned.
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::preprocess::{unstandardize_coefficients, StandardizationInfo};
/// let info = StandardizationInfo {
///     x_mean: vec![0.0, 5.0],
///     x_scale: vec![1.0, 2.0],
///     column_squared_norms: vec![1.0, 1.0],
///     y_mean: 10.0,
///     y_scale: Some(3.0),
///     y_scale_before_sqrt_weights_normalized: Some(3.0),
///     intercept: true,
///     standardized_x: true,
///     standardized_y: false,
/// };
///
/// // Standardized coefficients: [intercept=0, slope1=2.0]
/// let coefficients_standardized = vec![0.0, 2.0];
/// let (beta0, beta_slopes) = unstandardize_coefficients(&coefficients_standardized, &info);
///
/// // slope_original = (y_scale * slope_std) / x_scale[1]
/// //                 = (3.0 * 2.0) / 2.0 = 3.0
/// assert!((beta_slopes[0] - 3.0).abs() < 0.01);
/// ```
#[allow(clippy::needless_range_loop)]
pub fn unstandardize_coefficients(coefficients_standardized: &[f64], info: &StandardizationInfo) -> (f64, Vec<f64>) {
    let p = coefficients_standardized.len();
    let y_scale = info.y_scale.unwrap_or(1.0);

    // Determine where slope coefficients start in coefficients_standardized
    let start_idx = if info.intercept { 1 } else { 0 };
    let n_slopes = p - start_idx;

    // Unstandardize slope coefficients only (exclude intercept column coefficient)
    // NOTE: X is ALWAYS standardized for the solver, so we always apply the unstandardization formula.
    // The user's `standardize_x` option doesn't affect the internal computation, only the
    // interpretation of results.
    let mut beta_slopes = vec![0.0; n_slopes];
    for j in start_idx..p {
        let slope_idx = j - start_idx;
        // Standard unstandardization: beta_original = (y_scale * coefficients_standardized) / x_scale
        // This converts from the standardized space back to original data scale
        beta_slopes[slope_idx] = (y_scale * coefficients_standardized[j]) / info.x_scale[j];
    }

    // Compute intercept on original scale
    // beta0 = y_mean - sum(x_mean[j] * beta_slopes[j-1]) for j in 1..p
    let beta0 = if info.intercept {
        let mut sum = 0.0;
        for j in 1..p {
            sum += info.x_mean[j] * beta_slopes[j - 1];
        }
        info.y_mean - sum
    } else {
        0.0
    };

    (beta0, beta_slopes)
}

/// Computes predictions using unstandardized coefficients.
///
/// # Arguments
///
/// * `x_new` - New data matrix (n_new × p, with intercept column if applicable)
/// * `beta0` - Intercept on original scale
/// * `beta` - Slope coefficients on original scale (does NOT include intercept column coefficient)
///
/// # Returns
///
/// Predictions for each row in x_new.
///
/// # Note
///
/// If `x_new` has an intercept column (first column of all ones), `beta` should have
/// `p - 1` elements corresponding to the non-intercept columns. If `x_new` has no
/// intercept column, `beta` should have `p` elements.
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::preprocess::predict;
/// # use linreg_core::linalg::Matrix;
/// // X matrix with intercept: [[1, 2], [1, 3], [1, 4]]
/// let x_new = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
/// let beta0 = 1.0;
/// let beta = vec![2.0];  // One slope coefficient
///
/// // predictions[i] = 1.0 + 2.0 * x[i,1]
/// let preds = predict(&x_new, beta0, &beta);
/// assert_eq!(preds, vec![5.0, 7.0, 9.0]);
/// ```
#[allow(clippy::needless_range_loop)]
pub fn predict(x_new: &Matrix, beta0: f64, beta: &[f64]) -> Vec<f64> {
    let n = x_new.rows;
    let p = x_new.cols;

    let mut predictions = vec![0.0; n];

    // Determine if there's an intercept column based on beta length
    // If beta has one fewer element than columns, first column is intercept
    let has_intercept_col = beta.len() == p - 1;
    let first_penalized_column_index = if has_intercept_col { 1 } else { 0 };

    for i in 0..n {
        let mut sum = beta0;
        for (j, &beta_j) in beta.iter().enumerate() {
            let col = first_penalized_column_index + j;
            if col < p {
                sum += x_new.get(i, col) * beta_j;
            }
        }
        predictions[i] = sum;
    }

    predictions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standardize_xy_with_intercept() {
        // Simple test data
        let x_data = vec![1.0, 2.0, 3.0, 1.0, 4.0, 6.0, 1.0, 6.0, 9.0];
        let x = Matrix::new(3, 3, x_data);
        let y = vec![3.0, 5.0, 7.0];

        let options = StandardizeOptions {
            intercept: true,
            standardize_x: true,
            standardize_y: false,  // Note: y is still scaled to unit norm by glmnet convention
            weights: None,
        };

        let (x_standardized, y_standardized, info) = standardize_xy(&x, &y, &options);

        // First column (intercept) should be unchanged
        assert_eq!(x_standardized.get(0, 0), 1.0);
        assert_eq!(x_standardized.get(1, 0), 1.0);
        assert_eq!(x_standardized.get(2, 0), 1.0);

        // GLMNET: y is ALWAYS scaled to unit norm
        // y_centered = y - y_mean = [-2, 0, 2]
        // sqrt_weights_normalized-transform: y_sqrt_weights = sqrt_weights_normalized * y_centered = [-2/sqrt(3), 0, 2/sqrt(3)]
        // Scale to unit norm: y_standardized = y_sqrt_weights / ||y_sqrt_weights|| = [-1/sqrt(2), 0, 1/sqrt(2)]
        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        assert!((y_standardized[0] - (-inv_sqrt2)).abs() < 1e-10);
        assert!((y_standardized[1] - 0.0).abs() < 1e-10);
        assert!((y_standardized[2] - inv_sqrt2).abs() < 1e-10);

        // x_mean should capture the column means
        assert_eq!(info.x_mean[0], 0.0); // Intercept column
        assert!((info.x_mean[1] - 4.0).abs() < 1e-10);
        assert!((info.x_mean[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_unstandardize_coefficients() {
        // Create a simple standardization scenario
        let x_mean = vec![0.0, 4.0, 6.0];
        let x_scale = vec![1.0, 2.0, 3.0];
        let column_squared_norms = vec![1.0, 1.0, 1.0];  // Unit norm after standardization
        let y_mean = 5.0;
        let y_scale = Some(2.0);

        let info = StandardizationInfo {
            x_mean: x_mean.clone(),
            x_scale: x_scale.clone(),
            column_squared_norms,
            y_mean,
            y_scale,
            y_scale_before_sqrt_weights_normalized: None,
            intercept: true,
            standardized_x: true,
            standardized_y: true,
        };

        // Coefficients in standardized space: [intercept=0, beta1=1, beta2=2]
        let coefficients_standardized = vec![0.0, 1.0, 2.0];

        let (beta0, beta_slopes) = unstandardize_coefficients(&coefficients_standardized, &info);

        // Check unstandardization - beta_slopes now only contains slope coefficients
        // beta_slopes[0] = (y_scale * coefficients_standardized[1]) / x_scale[1] = (2 * 1) / 2 = 1
        assert!((beta_slopes[0] - 1.0).abs() < 1e-10);
        // beta_slopes[1] = (y_scale * coefficients_standardized[2]) / x_scale[2] = (2 * 2) / 3 = 4/3
        assert!((beta_slopes[1] - 4.0 / 3.0).abs() < 1e-10);

        // beta0 = y_mean - sum(x_mean[j] * beta_slopes[j-1])
        //      = 5 - (4 * 1 + 6 * 4/3) = 5 - 4 - 8 = -7
        assert!((beta0 - (-7.0)).abs() < 1e-10);

        // Verify beta_slopes has the correct length (only slopes, not intercept col coef)
        assert_eq!(beta_slopes.len(), 2);
    }

    #[test]
    fn test_predict() {
        // X has intercept column (first col all 1s) plus 2 predictors
        let x_data = vec![1.0, 2.0, 3.0, 1.0, 4.0, 6.0];
        let x = Matrix::new(2, 3, x_data);

        // beta0 = 1, beta = [2.0, 3.0] (slope coefficients only, no intercept col coef)
        let beta0 = 1.0;
        let beta = vec![2.0, 3.0];

        let preds = predict(&x, beta0, &beta);

        // pred[0] = 1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        assert!((preds[0] - 14.0).abs() < 1e-10);
        // pred[1] = 1 + 2*4 + 3*6 = 1 + 8 + 18 = 27
        assert!((preds[1] - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_standardize_xy() {
        // Simple test data
        let x_data = vec![1.0, 2.0, 3.0, 1.0, 4.0, 6.0, 1.0, 6.0, 9.0];
        let x = Matrix::new(3, 3, x_data);
        let y = vec![3.0, 5.0, 7.0];

        // Weights: give more weight to the middle observation
        let weights = vec![1.0, 2.0, 1.0];

        let options = StandardizeOptions {
            intercept: true,
            standardize_x: true,
            standardize_y: false,  // Note: y is still scaled to unit norm by glmnet convention
            weights: Some(weights),
        };

        let (x_standardized, y_standardized, info) = standardize_xy(&x, &y, &options);

        // First column (intercept) should be unchanged
        assert_eq!(x_standardized.get(0, 0), 1.0);
        assert_eq!(x_standardized.get(1, 0), 1.0);
        assert_eq!(x_standardized.get(2, 0), 1.0);

        // y_mean should be weighted mean
        // weights normalized: [1/4, 2/4, 1/4] = [0.25, 0.5, 0.25]
        // weighted mean: 0.25*3 + 0.5*5 + 0.25*7 = 0.75 + 2.5 + 1.75 = 5.0
        assert!((info.y_mean - 5.0).abs() < 1e-10);

        // GLMNET: y is ALWAYS scaled to unit norm
        // y_centered = y - y_mean = [-2, 0, 2]
        // sqrt_weights_normalized = sqrt([0.25, 0.5, 0.25]) = [0.5, ~0.707, 0.5]
        // y_sqrt_weights = sqrt_weights_normalized * y_centered = [-1, 0, 1]
        // sum(y_sqrt_weights^2) = 2, so y_scale = sqrt(2)
        // y_standardized = y_sqrt_weights / y_scale = [-1/sqrt(2), 0, 1/sqrt(2)]
        let expected_0 = -1.0 / (2.0_f64).sqrt();
        assert!((y_standardized[0] - expected_0).abs() < 1e-10);
        assert!((y_standardized[1] - 0.0).abs() < 1e-10);
        assert!((y_standardized[2] + expected_0).abs() < 1e-10);  // Should be 1/sqrt(2)
    }

    #[test]
    fn test_weighted_standardize_uniform_weights() {
        // Test that uniform weights give same result as no weights
        let x_data = vec![1.0, 2.0, 3.0, 1.0, 4.0, 6.0];
        let x = Matrix::new(2, 3, x_data);
        let y = vec![3.0, 5.0];

        // Uniform weights (should be equivalent to no weights after normalization)
        let weights = vec![1.0, 1.0];

        let options_with_weights = StandardizeOptions {
            intercept: true,
            standardize_x: true,
            standardize_y: false,
            weights: Some(weights),
        };

        let options_no_weights = StandardizeOptions {
            intercept: true,
            standardize_x: true,
            standardize_y: false,
            weights: None,
        };

        let (_x_standardized_w, y_standardized_w, info_w) = standardize_xy(&x, &y, &options_with_weights);
        let (_x_standardized, y_standardized, info) = standardize_xy(&x, &y, &options_no_weights);

        // Results should be the same
        assert_eq!(info_w.y_mean, info.y_mean);
        for i in 0..2 {
            assert!((y_standardized_w[i] - y_standardized[i]).abs() < 1e-10);
        }
    }
}
