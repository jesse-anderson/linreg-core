//! Data preprocessing for regularized regression.
//!
//! This module provides standardization utilities that match glmnet's behavior:
//!
//! - Predictors are centered and scaled (if enabled)
//! - The intercept column is not penalized, so it's handled specially
//! - Coefficients can be unstandardized back to the original scale
//!
//! # Standardization Convention
//!
//! The scaling factor used is `sqrt(sum(x²) / n)`, which gives unit variance
//! under the 1/n convention (matching glmnet).

use crate::linalg::{vec_mean, Matrix};

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
#[derive(Clone, Debug)]
pub struct StandardizationInfo {
    /// Mean of each predictor column
    pub x_mean: Vec<f64>,
    /// Scale factor for each predictor column
    pub x_scale: Vec<f64>,
    /// Mean of response variable
    pub y_mean: f64,
    /// Scale factor for response (for lambda path construction)
    pub y_scale: Option<f64>,
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
///
/// # Note
///
/// Setting `standardize_y` to `true` is mainly useful when you want to match
/// glmnet's lambda sequence exactly. For single-lambda fits, you typically
/// don't need to standardize y.
#[derive(Clone, Debug)]
pub struct StandardizeOptions {
    /// Whether to include an intercept (and center X)
    pub intercept: bool,
    /// Whether to standardize predictor columns
    pub standardize_x: bool,
    /// Whether to standardize the response variable
    pub standardize_y: bool,
}

impl Default for StandardizeOptions {
    fn default() -> Self {
        StandardizeOptions {
            intercept: true,
            standardize_x: true,
            standardize_y: false,
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
/// * `options` - Standardization options
///
/// # Returns
///
/// A tuple `(x_std, y_std, info)` where:
/// - `x_std` is the standardized design matrix
/// - `y_std` is the (optionally) standardized response
/// - `info` contains the standardization parameters for unstandardization
///
/// # Standardization Details
///
/// For the intercept column (first column, if present):
/// - Not centered (stays as ones)
/// - Not scaled
///
/// For other columns (if `standardize_x=true`):
/// - Centered: `x_centered = x - mean(x)`
/// - Scaled: `x_scaled = x_centered / sqrt(sum(x²) / n)`
///
/// For y (if `standardize_y=true`):
/// - Centered: `y_centered = y - mean(y)`
/// - Scaled: `y_scaled = y_centered / sqrt(sum(y²) / n)`
pub fn standardize_xy(
    x: &Matrix,
    y: &[f64],
    options: &StandardizeOptions,
) -> (Matrix, Vec<f64>, StandardizationInfo) {
    let n = x.rows;
    let p = x.cols;

    let mut x_std = x.clone();
    let mut y_std = y.to_vec();

    let mut x_mean = vec![0.0; p];
    let mut x_scale = vec![1.0; p];

    let y_mean = if options.intercept && !y.is_empty() {
        vec_mean(y)
    } else {
        0.0
    };

    // Always center y when there's an intercept (required for coordinate descent)
    // Additionally scale y if standardize_y is requested
    let y_scale = if options.intercept {
        // Center y
        for yi in y_std.iter_mut() {
            *yi -= y_mean;
        }

        if options.standardize_y {
            // Also scale y
            let y_var = y_std.iter().map(|&yi| yi * yi).sum::<f64>() / n as f64;
            let y_scale_val = y_var.sqrt();
            if y_scale_val > 0.0 {
                for yi in y_std.iter_mut() {
                    *yi /= y_scale_val;
                }
            }
            Some(y_scale_val)
        } else {
            None
        }
    } else if options.standardize_y {
        // No intercept but standardize_y requested - center and scale
        let y_mean_local = vec_mean(y);
        for yi in y_std.iter_mut() {
            *yi -= y_mean_local;
        }
        let y_var = y_std.iter().map(|&yi| yi * yi).sum::<f64>() / n as f64;
        let y_scale_val = y_var.sqrt();
        if y_scale_val > 0.0 {
            for yi in y_std.iter_mut() {
                *yi /= y_scale_val;
            }
        }
        Some(y_scale_val)
    } else {
        None
    };

    // Standardize X columns
    // If intercept is present, first column is NOT standardized
    let start_col = if options.intercept { 1 } else { 0 };

    for j in start_col..p {
        // Compute column mean
        let mut col_mean = 0.0;
        for i in 0..n {
            col_mean += x_std.get(i, j);
        }
        col_mean /= n as f64;
        x_mean[j] = col_mean;

        // Always center the column (required for coordinate descent to work correctly)
        for i in 0..n {
            let val = x_std.get(i, j) - col_mean;
            x_std.set(i, j, val);
        }

        // Always compute and apply scaling (coordinate descent assumes unit variance)
        // The standardize_x option only affects how coefficients are returned
        let mut col_scale_sq = 0.0;
        for i in 0..n {
            let val = x_std.get(i, j);
            col_scale_sq += val * val;
        }
        let col_scale = (col_scale_sq / n as f64).sqrt();

        if col_scale > 0.0 {
            // Always scale internally for correct coordinate descent behavior
            for i in 0..n {
                let val = x_std.get(i, j) / col_scale;
                x_std.set(i, j, val);
            }

            // Always store actual scale factor for correct unstandardization
            // (coefficients are always returned on original scale, matching glmnet)
            x_scale[j] = col_scale;
        }
    }

    // If intercept column exists, set its scale to 1.0 (not penalized)
    if options.intercept && p > 0 {
        x_scale[0] = 1.0;
        x_mean[0] = 0.0; // Intercept column has no "mean" to subtract
    }

    let info = StandardizationInfo {
        x_mean,
        x_scale,
        y_mean,
        y_scale,
        intercept: options.intercept,
        standardized_x: options.standardize_x,
        standardized_y: options.standardize_y,
    };

    (x_std, y_std, info)
}

/// Unstandardizes coefficients from the standardized space back to original scale.
///
/// This reverses the standardization transformation to get coefficients that
/// can be applied to the original (unscaled) data.
///
/// # Arguments
///
/// * `beta_std` - Coefficients in standardized space (length p)
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
/// If `intercept=true` in the info, `beta_std[0]` is assumed to be the intercept
/// coefficient (which is already 0 in the standardized space since X was centered).
/// The returned `beta_slopes` will NOT include this zeroed coefficient - only actual
/// slope coefficients are returned.
pub fn unstandardize_coefficients(beta_std: &[f64], info: &StandardizationInfo) -> (f64, Vec<f64>) {
    let p = beta_std.len();
    let y_scale = info.y_scale.unwrap_or(1.0);

    // Determine where slope coefficients start in beta_std
    let start_idx = if info.intercept { 1 } else { 0 };
    let n_slopes = p - start_idx;

    // Unstandardize slope coefficients only (exclude intercept column coefficient)
    let mut beta_slopes = vec![0.0; n_slopes];
    for j in start_idx..p {
        let slope_idx = j - start_idx;
        beta_slopes[slope_idx] = (y_scale * beta_std[j]) / info.x_scale[j];
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
pub fn predict(x_new: &Matrix, beta0: f64, beta: &[f64]) -> Vec<f64> {
    let n = x_new.rows;
    let p = x_new.cols;

    let mut predictions = vec![0.0; n];

    // Determine if there's an intercept column based on beta length
    // If beta has one fewer element than columns, first column is intercept
    let has_intercept_col = beta.len() == p - 1;
    let start_col = if has_intercept_col { 1 } else { 0 };

    for i in 0..n {
        let mut sum = beta0;
        for (j, &beta_j) in beta.iter().enumerate() {
            let col = start_col + j;
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
            standardize_y: false,
        };

        let (x_std, y_std, info) = standardize_xy(&x, &y, &options);

        // First column (intercept) should be unchanged
        assert_eq!(x_std.get(0, 0), 1.0);
        assert_eq!(x_std.get(1, 0), 1.0);
        assert_eq!(x_std.get(2, 0), 1.0);

        // y is now centered when intercept=true (required for coordinate descent)
        let y_mean = 5.0; // mean of [3, 5, 7]
        for i in 0..3 {
            assert!((y_std[i] - (y[i] - y_mean)).abs() < 1e-10);
        }

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
        let y_mean = 5.0;
        let y_scale = Some(2.0);

        let info = StandardizationInfo {
            x_mean: x_mean.clone(),
            x_scale: x_scale.clone(),
            y_mean,
            y_scale,
            intercept: true,
            standardized_x: true,
            standardized_y: true,
        };

        // Coefficients in standardized space: [intercept=0, beta1=1, beta2=2]
        let beta_std = vec![0.0, 1.0, 2.0];

        let (beta0, beta_slopes) = unstandardize_coefficients(&beta_std, &info);

        // Check unstandardization - beta_slopes now only contains slope coefficients
        // beta_slopes[0] = (y_scale * beta_std[1]) / x_scale[1] = (2 * 1) / 2 = 1
        assert!((beta_slopes[0] - 1.0).abs() < 1e-10);
        // beta_slopes[1] = (y_scale * beta_std[2]) / x_scale[2] = (2 * 2) / 3 = 4/3
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
}
