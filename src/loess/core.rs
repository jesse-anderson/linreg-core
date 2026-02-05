//! Core LOESS fitting logic

#![allow(clippy::needless_range_loop)]

use crate::{error::Error, linalg::Matrix};
use super::{
    types::{LoessFit, LoessOptions},
    normalize::normalize_predictors,
    predict::fit_at_point_impl,
    robust::compute_biweight_weights,
    weights::MIN_NEIGHBORS_QUADRATIC,
};

/// Fit LOESS model
///
/// Performs Locally Estimated Scatterplot Smoothing by fitting weighted
/// polynomial regressions at each data point.
///
/// # Algorithm
///
/// For each observation i:
/// 1. Find k nearest neighbors where k = max(2, floor(span * n))
/// 2. Compute tricube weights based on distance to neighbors
/// 3. Build local polynomial design matrix (constant, linear, or quadratic)
/// 4. Fit weighted least squares
/// 5. Store fitted value at the query point
///
/// If robust_iterations > 0, the algorithm iterates:
/// - Compute residuals from current fit
/// - Compute biweight robustness weights from residuals
/// - Refit using tricube weights × robustness weights
///
/// # Arguments
///
/// * `y` - Response variable (n observations)
/// * `x` - Predictor variables (p vectors, each of length n)
/// * `options` - LOESS fit options
///
/// # Returns
///
/// `LoessFit` result containing fitted values and fit parameters
///
/// # Errors
///
/// - Returns `Error::InsufficientData` if n is too small
/// - Returns `Error::InvalidInput` if dimensions don't match
/// - Returns `Error::InvalidParameter` if span or degree are invalid
/// - Returns `Error::SingularMatrix` if local fit fails
///
/// # Example
///
/// ```
/// use linreg_core::loess::{loess_fit, LoessOptions};
///
/// let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let y = vec![1.0, 3.5, 4.8, 6.2, 8.5, 11.0, 13.2, 14.8, 17.5, 19.0, 22.0];
///
/// let options = LoessOptions::default();
/// let result = loess_fit(&y, &[x], &options)?;
///
/// // Fitted values should have same length as input
/// assert_eq!(result.fitted.len(), y.len());
/// // Each fitted value should be finite
/// assert!(result.fitted.iter().all(|v| v.is_finite()));
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn loess_fit(
    y: &[f64],
    x: &[Vec<f64>],
    options: &LoessOptions,
) -> Result<LoessFit, Error> {
    // 1. Validate inputs
    let n = y.len();
    let p = x.len();

    let min_required = if options.degree == 2 {
        MIN_NEIGHBORS_QUADRATIC
    } else {
        2
    };

    if n < min_required {
        return Err(Error::InsufficientData {
            required: min_required,
            available: n,
        });
    }

    if p == 0 {
        return Err(Error::InvalidInput(
            "At least one predictor variable is required".to_string(),
        ));
    }

    // Check all x vectors have correct length
    for (i, x_var) in x.iter().enumerate() {
        if x_var.len() != n {
            return Err(Error::InvalidInput(format!(
                "x[{}] has {} elements, expected {}",
                i,
                x_var.len(),
                n
            )));
        }
    }

    // Validate span
    if options.span <= 0.0 || options.span > 1.0 {
        return Err(Error::InvalidInput(format!(
            "Span must be in (0, 1], got {}",
            options.span
        )));
    }

    // Validate degree
    if options.degree > 2 {
        return Err(Error::InvalidInput(
            "Degree must be 0 (constant), 1 (linear), or 2 (quadratic)".to_string(),
        ));
    }

    // 2. Build predictor matrix (row-major order)
    let mut x_data = Vec::with_capacity(n * p);
    for i in 0..n {
        for j in 0..p {
            x_data.push(x[j][i]);
        }
    }
    let x_matrix = Matrix::new(n, p, x_data);

    // 3. Normalize predictors
    let (x_normalized, _normalization_info) = normalize_predictors(&x_matrix);

    // 4. Initialize robustness weights (all 1.0 for first iteration)
    let mut robust_weights = vec![1.0; n];
    let mut fitted = vec![0.0; n];

    // 5. Perform robustness iterations
    for iteration in 0..=options.robust_iterations {
        // For each observation, compute fitted value
        for i in 0..n {
            // Get query point (normalized)
            let mut query = Vec::with_capacity(p);
            for j in 0..p {
                query.push(x_normalized.get(i, j));
            }

            let robustness_weights = if iteration > 0 {
                Some(robust_weights.as_slice())
            } else {
                None
            };

            let fitted_value = fit_at_point_impl(&query, &x_normalized, y, options, robustness_weights)?;
            fitted[i] = fitted_value;
        }

        // Compute robustness weights for next iteration (if not the last)
        if iteration < options.robust_iterations {
            // Compute residuals
            let residuals: Vec<f64> = y.iter().zip(fitted.iter()).map(|(yi, fitted)| yi - fitted).collect();

            // Compute mean absolute deviation (scale estimate) - following reference
            let mean_abs_deviation: f64 = residuals.iter().map(|r| r.abs()).sum::<f64>() / n as f64;

            // Compute MAD (median absolute deviation) and consistent scale (6 * MAD)
            let mut abs_residuals: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
            abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median_index = n / 2;
            let mad = if n.is_multiple_of(2) {
                (abs_residuals[median_index - 1] + abs_residuals[median_index]) / 2.0
            } else {
                abs_residuals[median_index]
            };
            // consistent_mad = 6 * MAD (scaled MAD that estimates σ for normal distributions)
            let consistent_mad = 6.0 * mad;

            // Check if we should stop iterating ( if(consistent_mad < 1e-7 * mean_abs_deviation) break)
            // This means robustness won't help - residuals are too uniform
            // eprintln!("Iteration {}: consistent_mad={}, mean_abs_deviation={}, 1e-7*mean_abs_deviation={}, consistent_mad>=1e-7*mean_abs_deviation: {}",
            //     iteration, consistent_mad, mean_abs_deviation, 1e-7 * mean_abs_deviation, consistent_mad >= 1e-7 * mean_abs_deviation);
            if consistent_mad >= 1e-7 * mean_abs_deviation {
                // Scale is sufficient - compute biweight weights
                robust_weights = compute_biweight_weights(&residuals);
            } else {
                // eprintln!("  Scale too small, keeping robust_weights unchanged (all 1.0)");
            }
        }
    }

    // 6. Return LoessFit
    Ok(LoessFit {
        fitted,
        predictions: None,
        span: options.span,
        degree: options.degree,
        robust_iterations: options.robust_iterations,
        surface: options.surface,
    })
}
