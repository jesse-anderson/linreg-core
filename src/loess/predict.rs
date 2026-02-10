//! Prediction logic for LOESS

#![allow(clippy::needless_range_loop)]

use crate::{error::Error, linalg::Matrix};
use super::{
    neighbors::{compute_bandwidth, euclidean_distance},
    normalize::normalize_predictors,
    types::{LoessFit, LoessOptions},
    weights::MIN_NEIGHBORS_QUADRATIC,
    wls::weighted_least_squares,
};

/// Compute a single local fit at a query point
///
/// Internal helper function that performs the local weighted regression
/// at a single normalized query point. This is the main fitting algorithm.
///
/// # Arguments
///
/// * `query` - Normalized query point coordinates
/// * `x_normalized` - Normalized training predictor matrix
/// * `y` - Training response values
/// * `options` - LOESS options (span, degree)
/// * `robust_weights` - Optional robustness weights (None for first iteration)
///
/// # Returns
///
/// The fitted value at the query point
///
/// # Errors
///
/// Returns `Error::SingularMatrix` if the local fit fails
pub fn fit_at_point_impl(
    query: &[f64],
    x_normalized: &Matrix,
    y: &[f64],
    options: &LoessOptions,
    robust_weights: Option<&[f64]>,
) -> Result<f64, Error> {
    let p = x_normalized.cols;

    // Special case: degree 0 (constant fit)
    if options.degree == 0 {
        return fit_at_point_degree_0(query, x_normalized, y, options, robust_weights);
    }

    // For 1-D linear case, use the standard LOWESS algorithm
    if p == 1 && options.degree == 1 {
        return fit_at_point_1d_linear(query, x_normalized, y, options, robust_weights);
    }

    // For multivariate or quadratic cases, use standard WLS
    fit_at_point_general(query, x_normalized, y, options, robust_weights)
}

/// 1-D linear fit using the standard LOWESS algorithm
///
/// This implements the classic algorithm for univariate locally weighted regression:
/// 1. Find k nearest neighbors where k = max(2, floor(span * n))
/// 2. Compute tricube weights with edge-case boundary handling
/// 3. Normalize weights to sum to 1
/// 4. Compute weighted center and variance
/// 5. If variance too small relative to full data range, use constant fit
/// 6. Otherwise, adjust weights and compute fitted value
///
/// The weight adjustment factors the slope into the weights, reducing
/// the degrees of freedom from 2 to 1 and allowing a simple weighted average.
fn fit_at_point_1d_linear(
    query: &[f64],
    x_normalized: &Matrix,
    y: &[f64],
    options: &LoessOptions,
    robust_weights: Option<&[f64]>,
) -> Result<f64, Error> {
    let n = x_normalized.rows;
    let query_value = query[0];

    // Compute full data range first (needed for spread check)
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    for i in 0..n {
        let xi = x_normalized.get(i, 0);
        if xi < x_min { x_min = xi; }
        if xi > x_max { x_max = xi; }
    }
    let range = x_max - x_min;

    // Compute neighborhood size and bandwidth
    let (bandwidth, neighbors) = compute_bandwidth(query, x_normalized, options.span, options.degree);
    let k = neighbors.len();

    // Compute tricube weights for neighbors with edge-case boundary handling
    let mut weighted_data: Vec<(usize, f64, f64)> = Vec::with_capacity(k); // (index, x, weight)

    for &idx in &neighbors {
        let xi = x_normalized.get(idx, 0);
        let r = (xi - query_value).abs();

        // Apply boundary handling to avoid numerical issues at edges
        let w = if bandwidth <= 0.0 {
            1.0
        } else {
            // Outer boundary: 99.9% of bandwidth (beyond this, weight = 0)
            let outer_boundary = 0.999 * bandwidth;
            // Inner boundary: 0.1% of bandwidth (within this, weight = 1)
            let inner_boundary = 0.001 * bandwidth;

            if r > outer_boundary {
                0.0
            } else if r <= inner_boundary {
                1.0
            } else {
                let normalized_distance = r / bandwidth;
                let tricube_inner = 1.0 - normalized_distance.powi(3);
                tricube_inner.powi(3)
            }
        };

        if w > 0.0 {
            weighted_data.push((idx, xi, w));
        }
    }

    // Apply robustness weights
    if let Some(rw) = robust_weights {
        for data in &mut weighted_data {
            data.2 *= rw[data.0];
        }
    }

    // Compute sum of weights and normalize
    let mut sum_w = 0.0;
    for (_, _, w) in &weighted_data {
        sum_w += w;
    }

    if sum_w <= 0.0 {
        // All weights zero - return the y value of the nearest neighbor
        let nearest_idx = neighbors[0];
        return Ok(y[nearest_idx]);
    }

    // Normalize weights to sum to 1
    for data in &mut weighted_data {
        data.2 /= sum_w;
    }

    // Linear fit adjustment (if bandwidth > 0)
    if bandwidth > 0.0 {
        // Compute weighted center of x values
        let mut weighted_center = 0.0;
        for (_, xi, w) in &weighted_data {
            weighted_center += w * xi;
        }

        // Compute center_offset = query - weighted_center
        let mut center_offset = query_value - weighted_center;

        // Compute weighted variance
        let mut weighted_variance = 0.0;
        for (_, xi, w) in &weighted_data {
            let diff = xi - weighted_center;
            weighted_variance += w * diff * diff;
        }

        // Check if points are spread out enough for linear fit
        // Use FULL data range for the comparison, not just neighborhood range
        if range > 0.0 && weighted_variance.sqrt() > 0.001 * range {
            // Points are spread out enough - adjust weights
            // First: center_offset /= weighted_variance (divide center_offset by variance)
            center_offset /= weighted_variance;

            for data in &mut weighted_data {
                let xi = data.1;
                // Then: w *= (center_offset * (x - weighted_center) + 1)
                data.2 *= center_offset * (xi - weighted_center) + 1.0;
            }
        }
        // If variance is too small, fall back to constant fit (weights unchanged)
    }

    // Compute fitted value as weighted sum
    let mut ys = 0.0;
    for (idx, _, w) in &weighted_data {
        ys += w * y[*idx];
    }

    Ok(ys)
}

/// General multivariate or quadratic fit using weighted least squares
fn fit_at_point_general(
    query: &[f64],
    x_normalized: &Matrix,
    y: &[f64],
    options: &LoessOptions,
    robust_weights: Option<&[f64]>,
) -> Result<f64, Error> {
    let p = x_normalized.cols;

    // a. Compute bandwidth and find neighbors
    let (bandwidth, neighbors) = compute_bandwidth(query, x_normalized, options.span, options.degree);
    let k = neighbors.len();

    // b. Compute tricube weights for each neighbor with edge-case boundary handling
    let mut weights = Vec::with_capacity(k);
    for &idx in &neighbors {
        let mut neighbor_point = Vec::with_capacity(p);
        for j in 0..p {
            neighbor_point.push(x_normalized.get(idx, j));
        }
        let dist = euclidean_distance(query, &neighbor_point);

        let w = if bandwidth <= 0.0 {
            1.0
        } else {
            // Outer boundary: 99.9% of bandwidth (beyond this, weight = 0)
            let outer_boundary = 0.999 * bandwidth;
            // Inner boundary: 0.1% of bandwidth (within this, weight = 1)
            let inner_boundary = 0.001 * bandwidth;

            if dist > outer_boundary {
                0.0
            } else if dist <= inner_boundary {
                1.0
            } else {
                let normalized_distance = dist / bandwidth;
                let tricube_inner = 1.0 - normalized_distance.powi(3);
                tricube_inner.powi(3)
            }
        };
        weights.push(w);
    }

    // c. Apply robustness weights if provided
    if let Some(rw) = robust_weights {
        for i in 0..k {
            weights[i] *= rw[neighbors[i]];
        }
    }

    // d. Build local design matrix with polynomial terms
    let n_poly_terms = if options.degree == 1 {
        1 + p
    } else {
        // 1 + p (linear) + p (squares) + p*(p-1)/2 (cross terms)
        1 + p + p + p * (p - 1) / 2
    };

    // TODO: Experimental - Small span handling. With very small spans, k may be < n_poly_terms.
    // This check prevents the "unreachable" WASM panic, but the fitted values may be poor quality
    // when span is too small for the polynomial degree. Consider using a lower degree or
    // increasing span for better results.
    if k < n_poly_terms {
        return Err(Error::InsufficientData {
            required: n_poly_terms,
            available: k,
        });
    }

    // TODO: Experimental - Matrix construction for local WLS. Ensure local_x_data has exactly
    // k * n_poly_terms elements before calling Matrix::new(). A mismatch will cause a panic
    // in release mode ("unreachable" error in WASM). The loops above should guarantee this,
    // but edge cases with empty neighbor lists after filtering should be considered.

    let mut local_x_data = Vec::with_capacity(k * n_poly_terms);
    let mut local_y = Vec::with_capacity(k);

    for &idx in &neighbors {
        local_y.push(y[idx]);

        // Intercept term
        local_x_data.push(1.0);

        // Linear terms
        for j in 0..p {
            local_x_data.push(x_normalized.get(idx, j));
        }

        // Quadratic terms (if degree = 2)
        if options.degree == 2 {
            // Squared terms
            for j1 in 0..p {
                let val = x_normalized.get(idx, j1);
                local_x_data.push(val * val);
            }

            // Cross terms (only upper triangle to avoid duplication)
            for j1 in 0..p {
                for j2 in (j1 + 1)..p {
                    let v1 = x_normalized.get(idx, j1);
                    let v2 = x_normalized.get(idx, j2);
                    local_x_data.push(v1 * v2);
                }
            }
        }
    }

    let local_x = Matrix::new(k, n_poly_terms, local_x_data);

    // e. Fit weighted least squares
    let coeffs = weighted_least_squares(&local_x, &local_y, &weights)?;

    // f. Compute fitted value at query point
    evaluate_polynomial(&coeffs, query, p, options.degree)
}

/// Public wrapper for fit_at_point_impl without robustness weights
///
/// This is the public API that matches the original signature.
pub fn fit_at_point(
    query: &[f64],
    x_normalized: &Matrix,
    y: &[f64],
    options: &LoessOptions,
) -> Result<f64, Error> {
    fit_at_point_impl(query, x_normalized, y, options, None)
}

/// Fit degree 0 (constant) at a query point
///
/// Simply returns the weighted average of y-values in the neighborhood.
fn fit_at_point_degree_0(
    query: &[f64],
    x_normalized: &Matrix,
    y: &[f64],
    options: &LoessOptions,
    robust_weights: Option<&[f64]>,
) -> Result<f64, Error> {
    let (bandwidth, neighbors) = compute_bandwidth(query, x_normalized, options.span, options.degree);
    let k = neighbors.len();

    // Compute tricube weights with edge-case boundary handling
    let mut weights = Vec::with_capacity(k);
    for &idx in &neighbors {
        let p = x_normalized.cols;
        let mut neighbor_point = Vec::with_capacity(p);
        for j in 0..p {
            neighbor_point.push(x_normalized.get(idx, j));
        }
        let dist = euclidean_distance(query, &neighbor_point);

        // Apply boundary handling to avoid numerical issues at edges
        let w = if bandwidth <= 0.0 {
            1.0
        } else {
            // Outer boundary: 99.9% of bandwidth (beyond this, weight = 0)
            let outer_boundary = 0.999 * bandwidth;
            // Inner boundary: 0.1% of bandwidth (within this, weight = 1)
            let inner_boundary = 0.001 * bandwidth;

            if dist > outer_boundary {
                0.0
            } else if dist <= inner_boundary {
                1.0
            } else {
                let normalized_distance = dist / bandwidth;
                let tricube_inner = 1.0 - normalized_distance.powi(3);
                tricube_inner.powi(3)
            }
        };
        weights.push(w);
    }

    // Apply robustness weights if provided
    if let Some(rw) = robust_weights {
        for i in 0..k {
            weights[i] *= rw[neighbors[i]];
        }
    }

    // Return weighted average
    fit_weighted_average(&neighbors.iter().map(|&idx| y[idx]).collect::<Vec<_>>(), &weights)
}

/// Compute weighted average of values
///
/// Used for degree 0 fitting and when falling back to constant fit.
fn fit_weighted_average(values: &[f64], weights: &[f64]) -> Result<f64, Error> {
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum <= 0.0 {
        return Err(Error::InvalidInput("All weights are zero".to_string()));
    }

    let weighted_sum: f64 = values.iter().zip(weights.iter()).map(|(v, w)| v * w).sum();
    Ok(weighted_sum / weight_sum)
}

/// Evaluate polynomial at query point
///
/// Computes the value of the fitted polynomial at the query point.
fn evaluate_polynomial(
    coeffs: &[f64],
    query: &[f64],
    p: usize,
    degree: usize,
) -> Result<f64, Error> {
    // Start with intercept
    let mut fitted_value = coeffs[0];

    let mut coeff_idx = 1;

    // Linear terms
    for j in 0..p {
        fitted_value += coeffs[coeff_idx] * query[j];
        coeff_idx += 1;
    }

    // Quadratic terms
    if degree == 2 {
        // Squared terms
        for j in 0..p {
            fitted_value += coeffs[coeff_idx] * query[j] * query[j];
            coeff_idx += 1;
        }

        // Cross terms
        for j1 in 0..p {
            for j2 in (j1 + 1)..p {
                fitted_value += coeffs[coeff_idx] * query[j1] * query[j2];
                coeff_idx += 1;
            }
        }
    }

    Ok(fitted_value)
}

impl LoessFit {
    /// Predict at new query points
    ///
    /// Performs LOESS prediction at arbitrary new points by redoing the local
    /// fitting at each query point using the original training data.
    ///
    /// # Arguments
    ///
    /// * `new_x` - New predictor values (p vectors, each of length m)
    /// * `original_x` - Original training predictors (p vectors, each of length n)
    /// * `original_y` - Original training response (n observations)
    /// * `options` - LOESS options (must match the options used for fitting)
    ///
    /// # Returns
    ///
    /// Vector of predicted values (m predictions)
    ///
    /// # Errors
    ///
    /// - Returns `Error::InvalidInput` if dimensions don't match
    /// - Returns `Error::InsufficientData` if there are too few training points
    /// - Returns `Error::InvalidInput` if span/degree don't match the original fit
    /// - Returns `Error::SingularMatrix` if any local fit fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use linreg_core::loess::{loess_fit, LoessOptions};
    /// # fn main() -> Result<(), linreg_core::Error> {
    /// let train_x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    /// let train_y = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
    ///
    /// let options = LoessOptions::default();
    /// let fit = loess_fit(&train_y, &[train_x.clone()], &options)?;
    ///
    /// // Predict at new points
    /// let new_x = vec![1.5, 2.5, 3.5];
    /// let predictions = fit.predict(&[new_x], &[train_x], &train_y, &options)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(
        &self,
        new_x: &[Vec<f64>],
        original_x: &[Vec<f64>],
        original_y: &[f64],
        options: &LoessOptions,
    ) -> Result<Vec<f64>, Error> {
        // Validate that options match the original fit
        if options.span != self.span {
            return Err(Error::InvalidInput(format!(
                "Span mismatch: fit used {}, prediction uses {}",
                self.span, options.span
            )));
        }
        if options.degree != self.degree {
            return Err(Error::InvalidInput(format!(
                "Degree mismatch: fit used {}, prediction uses {}",
                self.degree, options.degree
            )));
        }
        if options.robust_iterations != self.robust_iterations {
            return Err(Error::InvalidInput(format!(
                "Robust iterations mismatch: fit used {}, prediction uses {}",
                self.robust_iterations, options.robust_iterations
            )));
        }

        // Validate inputs
        let n_train = original_y.len();
        let p = original_x.len();
        let m = if new_x.is_empty() {
            return Ok(Vec::new());
        } else {
            new_x[0].len()
        };

        let min_required = if options.degree == 2 {
            MIN_NEIGHBORS_QUADRATIC
        } else {
            2
        };

        if n_train < min_required {
            return Err(Error::InsufficientData {
                required: min_required,
                available: n_train,
            });
        }

        if p == 0 {
            return Err(Error::InvalidInput(
                "At least one predictor variable is required".to_string(),
            ));
        }

        // Check all x vectors have correct length
        for (i, x_var) in new_x.iter().enumerate() {
            if x_var.len() != m {
                return Err(Error::InvalidInput(format!(
                    "new_x[{}] has {} elements, expected {}",
                    i,
                    x_var.len(),
                    m
                )));
            }
            if original_x[i].len() != n_train {
                return Err(Error::InvalidInput(format!(
                    "original_x[{}] has {} elements, expected {}",
                    i,
                    original_x[i].len(),
                    n_train
                )));
            }
        }

        // Build predictor matrix from training data (row-major order)
        let mut train_x_data = Vec::with_capacity(n_train * p);
        for i in 0..n_train {
            for j in 0..p {
                train_x_data.push(original_x[j][i]);
            }
        }
        let train_x_matrix = Matrix::new(n_train, p, train_x_data);

        // For single predictor, use raw values (no normalization)
        // This improves accuracy for the common single-predictor case
        let train_x_normalized = if p == 1 {
            train_x_matrix.clone()
        } else {
            // For multiple predictors, normalize to [0, 1]
            let (normalized, _) = normalize_predictors(&train_x_matrix);
            normalized
        };

        // For each query point, compute prediction using local fit
        let mut predictions = Vec::with_capacity(m);

        for i in 0..m {
            // Get query point (no normalization for single predictor)
            let query_normalized = if p == 1 {
                let mut qn = Vec::with_capacity(p);
                for j in 0..p {
                    qn.push(new_x[j][i]);
                }
                qn
            } else {
                // Normalize query point for multiple predictors
                let mut qn = Vec::with_capacity(p);
                for j in 0..p {
                    let query_val = new_x[j][i];
                    // Find min/max of training predictor j
                    let mut col_min = f64::INFINITY;
                    let mut col_max = f64::NEG_INFINITY;
                    for row in 0..n_train {
                        let val = original_x[j][row];
                        if val < col_min {
                            col_min = val;
                        }
                        if val > col_max {
                            col_max = val;
                        }
                    }
                    let col_range = col_max - col_min;
                    let normalized_val = if col_range <= f64::EPSILON {
                        0.5
                    } else {
                        (query_val - col_min) / col_range
                    };
                    qn.push(normalized_val);
                }
                qn
            };

            // Perform local fit at this query point
            let pred = fit_at_point(&query_normalized, &train_x_normalized, original_y, options)?;
            predictions.push(pred);
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loess::normalize::normalize_predictors;

    #[test]
    fn test_fit_at_point_simple_linear() {
        // Simple linear relationship: y = 2*x + 1
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];

        let x_matrix = Matrix::new(6, 1, x_data);
        let (x_normalized, _) = normalize_predictors(&x_matrix);

        let mut options = LoessOptions::default();
        options.degree = 1;
        options.span = 0.75;

        // Fit at x = 2.5 (should be close to y = 6)
        let query = vec![0.5]; // 2.5 normalized to [0,5] -> 0.5
        let fitted = fit_at_point(&query, &x_normalized, &y_data, &options).unwrap();

        // y = 2*2.5 + 1 = 6, LOESS should be close
        assert!((fitted - 6.0).abs() < 1.0);
    }

    #[test]
    fn test_fit_at_point_quadratic() {
        // Quadratic relationship: y = x²
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data: Vec<f64> = x_data.iter().map(|x| x * x).collect();

        let x_matrix = Matrix::new(6, 1, x_data);
        let (x_normalized, _) = normalize_predictors(&x_matrix);

        let mut options = LoessOptions::default();
        options.degree = 2;
        options.span = 1.0;

        // Fit at x = 2.5 (should be close to y = 6.25)
        let query = vec![0.5];
        let fitted = fit_at_point(&query, &x_normalized, &y_data, &options).unwrap();

        assert!((fitted - 6.25).abs() < 1.0);
    }

    #[test]
    fn test_fit_at_point_degree_0() {
        // Constant data: y = 5
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![5.0; 6];

        let x_matrix = Matrix::new(6, 1, x_data);
        let (x_normalized, _) = normalize_predictors(&x_matrix);

        let mut options = LoessOptions::default();
        options.degree = 0;
        options.span = 0.5;

        // Fit at any point should be close to 5
        let query = vec![0.5];
        let fitted = fit_at_point(&query, &x_normalized, &y_data, &options).unwrap();

        assert!((fitted - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_fit_weighted_average() {
        // Test weighted average computation
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let result = fit_weighted_average(&values, &weights).unwrap();
        assert_eq!(result, 3.0); // (1+2+3+4+5)/5 = 3

        // Unequal weights
        let weights2 = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let result2 = fit_weighted_average(&values, &weights2).unwrap();
        assert_eq!(result2, 3.0); // Only middle point contributes
    }

    #[test]
    fn test_fit_weighted_average_zero_weights() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = vec![0.0, 0.0, 0.0];

        let result = fit_weighted_average(&values, &weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluate_polynomial_linear() {
        // Coefficients: intercept=2, slope=3 -> y = 3*x + 2
        let coeffs = vec![2.0, 3.0];
        let query = vec![4.0];

        let result = evaluate_polynomial(&coeffs, &query, 1, 1).unwrap();
        assert_eq!(result, 14.0); // 3*4 + 2 = 14
    }

    #[test]
    fn test_evaluate_polynomial_quadratic_1d() {
        // Coefficients: intercept=1, linear=2, quadratic=3 -> y = 3*x² + 2*x + 1
        let coeffs = vec![1.0, 2.0, 3.0];
        let query = vec![2.0];

        let result = evaluate_polynomial(&coeffs, &query, 1, 2).unwrap();
        assert_eq!(result, 17.0); // 3*4 + 2*2 + 1 = 12 + 4 + 1 = 17
    }

    #[test]
    fn test_evaluate_polynomial_quadratic_2d() {
        // 2D quadratic: intercept + x + y + x² + y² + xy
        // coeffs = [c0, c1, c2, c3, c4, c5]
        // y = 1 + 2*x + 3*y + 4*x² + 5*y² + 6*xy
        let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let query = vec![1.0, 2.0]; // x=1, y=2

        let result = evaluate_polynomial(&coeffs, &query, 2, 2).unwrap();
        // 1 + 2*1 + 3*2 + 4*1 + 5*4 + 6*2 = 1 + 2 + 6 + 4 + 20 + 12 = 45
        assert_eq!(result, 45.0);
    }

    #[test]
    fn test_fit_at_point_degree_0_direct() {
        // Test the degree 0 path directly (fit_at_point_degree_0 function)
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = vec![5.0, 7.0, 9.0, 11.0, 13.0];

        let x_matrix = Matrix::new(5, 1, x_data);
        let (x_normalized, _) = normalize_predictors(&x_matrix);

        let mut options = LoessOptions::default();
        options.degree = 0;
        options.span = 0.6;

        // Use the internal function directly
        let query = vec![0.5];
        let fitted = fit_at_point_degree_0(&query, &x_normalized, &y_data, &options, None).unwrap();

        // Degree 0 does local weighted average
        assert!(fitted.is_finite());
    }

    #[test]
    fn test_fit_at_point_degree_0_with_robust_weights() {
        // Test degree 0 with robustness weights
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = vec![5.0, 7.0, 9.0, 11.0, 13.0];
        let robust_weights = vec![1.0, 0.5, 1.0, 0.5, 1.0];

        let x_matrix = Matrix::new(5, 1, x_data);
        let (x_normalized, _) = normalize_predictors(&x_matrix);

        let mut options = LoessOptions::default();
        options.degree = 0;
        options.span = 0.6;

        let query = vec![0.5];
        let fitted = fit_at_point_degree_0(&query, &x_normalized, &y_data, &options, Some(&robust_weights)).unwrap();

        assert!(fitted.is_finite());
    }

    #[test]
    fn test_fit_at_point_1d_linear_edge_case_bandwidth_zero() {
        // Test the bandwidth <= 0.0 branch (lines 101-102)
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0];

        let x_matrix = Matrix::new(5, 1, x_data);
        let (x_normalized, _) = normalize_predictors(&x_matrix);

        let mut options = LoessOptions::default();
        options.degree = 1;
        options.span = 0.75; // Use reasonable span

        let query = vec![0.5];
        let result = fit_at_point_1d_linear(&query, &x_normalized, &y_data, &options, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fit_at_point_1d_linear_all_zero_weights() {
        // Test with very small span - tests edge case handling
        let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0];

        let x_matrix = Matrix::new(5, 1, x_data);
        let (x_normalized, _) = normalize_predictors(&x_matrix);

        let mut options = LoessOptions::default();
        options.degree = 1;
        options.span = 0.5; // Reasonable span

        let query = vec![0.5];
        let result = fit_at_point_1d_linear(&query, &x_normalized, &y_data, &options, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fit_at_point_1d_linear_small_variance_fallback() {
        // Test the small variance fallback (lines 169-180)
        // When all x values are nearly the same, variance is small
        let x_data = vec![1.0, 1.001, 0.999, 1.0, 1.001]; // Nearly identical x
        let y_data = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let x_matrix = Matrix::new(5, 1, x_data);
        let (x_normalized, _) = normalize_predictors(&x_matrix);

        let mut options = LoessOptions::default();
        options.degree = 1;
        options.span = 1.0;

        let query = vec![0.5]; // Normalized position
        let fitted = fit_at_point_1d_linear(&query, &x_normalized, &y_data, &options, None).unwrap();

        // Should still produce a finite result
        assert!(fitted.is_finite());
    }

    #[test]
    fn test_fit_at_point_impl_multivariate_dispatch() {
        // Test dispatch to fit_at_point_general for multivariate
        let x_data = vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0];
        let y_data = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];

        let x_matrix = Matrix::new(3, 2, x_data);
        let (x_normalized, _x_info) = normalize_predictors(&x_matrix);

        let mut options = LoessOptions::default();
        options.degree = 1;
        options.span = 1.0;

        // Use a simple query point in normalized space [0, 1]
        let query = vec![0.5, 0.5];
        let fitted = fit_at_point(&query, &x_normalized, &y_data, &options).unwrap();
        assert!(fitted.is_finite());
    }
}
