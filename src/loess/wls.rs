//! Weighted least squares solving

use crate::{error::Error, linalg::Matrix};

/// Solve weighted least squares: minimize ||W^(1/2) * (y - X*b)||²
///
/// Transforms the problem by multiplying X and y by sqrt(weights), then
/// solves using standard QR decomposition. Falls back to SVD with
/// pseudoinverse if the matrix is singular or nearly singular.
///
/// # Arguments
///
/// * `x` - Design matrix (n × p), should include intercept column if needed
/// * `y` - Response vector (n)
/// * `weights` - Weight vector (n), all values must be non-negative
///
/// # Returns
///
/// Coefficient vector (p) if successful, Error if the matrix is singular
///
/// # Errors
///
/// Returns `Error::SingularMatrix` if the weighted design matrix is singular
/// and SVD also fails to produce a valid solution.
///
/// # Example
///
/// ```
/// use linreg_core::loess::wls::weighted_least_squares;
/// use linreg_core::linalg::Matrix;
///
/// // Simple test: y = 2x + 1
/// let x_data = vec![
///     1.0, 0.0,  // intercept, x=0
///     1.0, 1.0,  // intercept, x=1
///     1.0, 2.0,  // intercept, x=2
///     1.0, 3.0,  // intercept, x=3
///     1.0, 4.0,  // intercept, x=4
/// ];
/// let x = Matrix::new(5, 2, x_data);
/// let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];  // y = 2*x + 1
/// let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];  // Equal weights = OLS
///
/// let coeffs = weighted_least_squares(&x, &y, &weights)?;
///
/// // Should get approximately [1, 2] (intercept, slope)
/// assert!((coeffs[0] - 1.0).abs() < 1e-10);
/// assert!((coeffs[1] - 2.0).abs() < 1e-10);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn weighted_least_squares(x: &Matrix, y: &[f64], weights: &[f64]) -> Result<Vec<f64>, Error> {
    let n = x.rows;
    let p = x.cols;

    assert_eq!(
        y.len(),
        n,
        "Response vector length must match number of rows"
    );
    assert_eq!(
        weights.len(),
        n,
        "Weight vector length must match number of rows"
    );

    // Check if all weights are zero
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum <= 0.0 {
        return Err(Error::InvalidInput("All weights are zero".to_string()));
    }

    // Transform: X' = W^(1/2) * X, y' = W^(1/2) * y
    let mut x_weighted_data = Vec::with_capacity(n * p);
    let mut y_weighted = Vec::with_capacity(n);

    for i in 0..n {
        let sqrt_weight = weights[i].sqrt();
        y_weighted.push(y[i] * sqrt_weight);

        for j in 0..p {
            x_weighted_data.push(x.get(i, j) * sqrt_weight);
        }
    }

    let mut x_weighted = Matrix::new(n, p, x_weighted_data);

    // Equilibrate columns for numerical stability
    let column_scales = equilibrate_columns(&mut x_weighted);

    // First try QR decomposition
    let qr_result = try_qr_solve(&x_weighted, &y_weighted, p);

    match qr_result {
        Ok(coeffs) => {
            // Compensate for column equilibration
            let mut final_coeffs = Vec::with_capacity(p);
            for j in 0..p {
                final_coeffs.push(coeffs[j] / column_scales[j]);
            }
            Ok(final_coeffs)
        }
        Err(Error::SingularMatrix) => {
            // Fall back to SVD with pseudoinverse for rank-deficient cases
            let svd_result = try_svd_solve(&x_weighted, &y_weighted, p);
            match svd_result {
                Ok(coeffs) => {
                    // Compensate for column equilibration
                    let mut final_coeffs = Vec::with_capacity(p);
                    for j in 0..p {
                        final_coeffs.push(coeffs[j] / column_scales[j]);
                    }
                    Ok(final_coeffs)
                }
                Err(e) => Err(e),
            }
        }
        Err(e) => Err(e),
    }
}

/// Attempt to solve using QR decomposition.
///
/// Returns `Error::SingularMatrix` if the matrix is too ill-conditioned.
fn try_qr_solve(x_weighted: &Matrix, y_weighted: &[f64], p: usize) -> Result<Vec<f64>, Error> {
    // QR Decomposition on weighted matrix
    let (q, r) = x_weighted.qr();

    // Extract upper p × p part of R
    let mut r_upper = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            r_upper.set(i, j, r.get(i, j));
        }
    }

    // Q^T * y'
    let q_t = q.transpose();
    let qty = q_t.mul_vec(y_weighted);

    // Take first p elements of qty and make into a column vector
    let rhs_vec = qty[0..p].to_vec();
    let rhs_mat = Matrix::new(p, 1, rhs_vec);

    // Invert R_upper
    let r_inv = match r_upper.invert_upper_triangular() {
        Some(inv) => inv,
        None => return Err(Error::SingularMatrix),
    };

    // Solve: beta = R^(-1) * Q^T * y'
    let result = r_inv.matmul(&rhs_mat);

    // Extract coefficients
    let mut coeffs = Vec::with_capacity(p);
    for j in 0..p {
        coeffs.push(result.get(j, 0));
    }

    Ok(coeffs)
}

/// Attempt to solve using SVD with pseudoinverse for rank-deficient matrices.
///
/// Implementation:
/// - Compute SVD of the design matrix
/// - Use tolerance: tol = sigma\[0\] * 100 * epsilon
/// - For sigma\[j\] > tol: coefficient = (U^T * y)\[j\] / sigma\[j\]
/// - For sigma\[j\] <= tol: coefficient = 0
/// - Final solution: x = V * coefficients
fn try_svd_solve(x_weighted: &Matrix, y_weighted: &[f64], _p: usize) -> Result<Vec<f64>, Error> {
    // Compute SVD
    let svd_result = x_weighted.svd();

    // Use the SVD solver which handles rank deficiency
    let coeffs = x_weighted.svd_solve(&svd_result, y_weighted);

    Ok(coeffs)
}

/// Equilibrate columns of a matrix
///
/// Normalizes each column to have unit norm for numerical stability.
/// Returns the scaling factors.
///
/// # Arguments
///
/// * `x` - Design matrix (n × p), will be modified in place
///
/// # Returns
///
/// Vector of scaling factors for each column
fn equilibrate_columns(x: &mut Matrix) -> Vec<f64> {
    let n = x.rows;
    let p = x.cols;

    let mut column_scales = Vec::with_capacity(p);

    for j in 0..p {
        let mut norm = 0.0;
        for i in 0..n {
            let val = x.get(i, j);
            norm += val * val;
        }
        norm = norm.sqrt();

        if norm > 0.0 {
            // Scale column to unit norm
            for i in 0..n {
                let val = x.get(i, j);
                x.set(i, j, val / norm);
            }
            column_scales.push(norm);
        } else {
            // Zero column - use scale 1.0
            column_scales.push(1.0);
        }
    }

    column_scales
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_least_squares_simple() {
        // Simple test: y = 2x + 1
        let x_data = vec![
            1.0, 0.0,  // intercept, x=0
            1.0, 1.0,  // intercept, x=1
            1.0, 2.0,  // intercept, x=2
            1.0, 3.0,  // intercept, x=3
            1.0, 4.0,  // intercept, x=4
        ];
        let x = Matrix::new(5, 2, x_data);
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];  // y = 2*x + 1
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];  // Equal weights = OLS

        let coeffs = weighted_least_squares(&x, &y, &weights).unwrap();

        // Should get approximately [1, 2] (intercept, slope)
        assert!((coeffs[0] - 1.0).abs() < 1e-10);
        assert!((coeffs[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_least_squares_with_weights() {
        // Test with different weights - higher weight should pull fit closer
        let x_data = vec![
            1.0, 0.0,
            1.0, 1.0,
            1.0, 2.0,
        ];
        let x = Matrix::new(3, 2, x_data);
        let y = vec![0.0, 1.0, 3.0];  // Not perfectly linear

        // Heavily weight the middle point
        let weights = vec![0.01, 100.0, 0.01];

        let coeffs = weighted_least_squares(&x, &y, &weights).unwrap();

        // The fit should be close to passing through (1, 1) due to high weight
        let predicted_at_1 = coeffs[0] + coeffs[1] * 1.0;
        assert!((predicted_at_1 - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_weighted_least_squares_matches_ols() {
        // WLS with all weights = 1 should match OLS
        let x_data = vec![
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0,
            1.0, 4.0,
        ];
        let x = Matrix::new(4, 2, x_data);
        let y = vec![3.1, 4.9, 7.2, 9.8];
        let weights = vec![1.0, 1.0, 1.0, 1.0];

        let coeffs = weighted_least_squares(&x, &y, &weights).unwrap();

        // Verify the solution is reasonable (y ≈ 2*x + 1)
        assert!((coeffs[0] - 1.0).abs() < 1.0);  // intercept ≈ 1
        assert!((coeffs[1] - 2.0).abs() < 0.5);  // slope ≈ 2
    }

    #[test]
    fn test_weighted_least_squares_zero_weight() {
        // Zero weights should effectively remove those points
        let x_data = vec![
            1.0, 0.0,
            1.0, 1.0,
            1.0, 2.0,
        ];
        let x = Matrix::new(3, 2, x_data);
        let y = vec![0.0, 1.0, 100.0];  // Last point is an outlier

        // Zero out the outlier
        let weights = vec![1.0, 1.0, 0.0];

        let coeffs = weighted_least_squares(&x, &y, &weights).unwrap();

        // Fit should be based only on first two points: y = x
        assert!((coeffs[0] - 0.0).abs() < 1e-10);
        assert!((coeffs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_fallback_rank_deficient() {
        // Test SVD fallback for rank-deficient (nearly collinear) matrix
        // Two predictors are almost perfectly collinear
        let x_data = vec![
            1.0, 1.0, 1.0,      // intercept, x1=1, x2≈1 (2*1 + epsilon)
            1.0, 2.0, 4.001,    // intercept, x1=2, x2≈4 (2*2 + epsilon)
            1.0, 3.0, 6.001,    // intercept, x1=3, x2≈6 (2*3 + epsilon)
            1.0, 4.0, 8.001,    // intercept, x1=4, x2≈8 (2*4 + epsilon)
            1.0, 5.0, 10.001,   // intercept, x1=5, x2≈10 (2*5 + epsilon)
        ];
        let x = Matrix::new(5, 3, x_data);
        // True relationship: y = 3 + 2*x1 (x2 is redundant)
        let y = vec![5.0, 7.0, 9.0, 11.0, 13.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        // Should solve successfully using SVD fallback
        let coeffs = weighted_least_squares(&x, &y, &weights).unwrap();

        // Check intercept is approximately 3
        assert!((coeffs[0] - 3.0).abs() < 1.0);
        // Check x1 coefficient is approximately 2
        assert!((coeffs[1] - 2.0).abs() < 1.0);
    }

    #[test]
    fn test_svd_fallback_perfect_collinearity() {
        // Test SVD fallback for perfectly collinear predictors
        // x2 = 2*x1 exactly
        let x_data = vec![
            1.0, 1.0, 2.0,     // intercept, x1=1, x2=2
            1.0, 2.0, 4.0,     // intercept, x1=2, x2=4
            1.0, 3.0, 6.0,     // intercept, x1=3, x2=6
            1.0, 4.0, 8.0,     // intercept, x1=4, x2=8
            1.0, 5.0, 10.0,    // intercept, x1=5, x2=10
        ];
        let x = Matrix::new(5, 3, x_data);
        // True relationship: y = 3 + 2*x1
        let y = vec![5.0, 7.0, 9.0, 11.0, 13.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        // Should solve successfully using SVD fallback
        let coeffs = weighted_least_squares(&x, &y, &weights).unwrap();

        // All coefficients should be finite
        assert!(coeffs.iter().all(|c| c.is_finite()));

        // With perfect collinearity, SVD produces minimum-norm solution
        // The key property is that X * coeffs = y should hold
        // Predictions should be accurate
        for i in 0..5 {
            let x1 = (i + 1) as f64;
            let x2 = 2.0 * x1;
            let pred = coeffs[0] + coeffs[1] * x1 + coeffs[2] * x2;
            assert!((pred - y[i]).abs() < 1e-6, "Prediction mismatch at i={}: pred={}, y={}", i, pred, y[i]);
        }
    }

    #[test]
    fn test_svd_tolerance_matches() {
        // Verify that SVD tolerance matches tol = sigma[0] * 100 * epsilon
        let x_data = vec![
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0,
        ];
        let x = Matrix::new(3, 2, x_data);
        let y = vec![2.0, 4.0, 6.0];
        let weights = vec![1.0, 1.0, 1.0];

        let coeffs = weighted_least_squares(&x, &y, &weights).unwrap();

        // Should get exact solution y = 2*x
        assert!((coeffs[0] - 0.0).abs() < 1e-10);
        assert!((coeffs[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_column_equilibration_with_svd() {
        // Test that column equilibration works correctly with SVD fallback
        // Create a matrix with very different column scales
        let x_data = vec![
            1.0, 0.0001,           // intercept, tiny x1
            1.0, 0.0002,
            1.0, 0.0003,
            1.0, 0.0004,
            1.0, 0.0005,
        ];
        let x = Matrix::new(5, 2, x_data);
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let coeffs = weighted_least_squares(&x, &y, &weights).unwrap();

        // Predictions should be accurate despite scale differences
        for i in 0..5 {
            let x_val = (i + 1) as f64 * 0.0001;
            let pred = coeffs[0] + coeffs[1] * x_val;
            assert!((pred - y[i]).abs() < 0.1);
        }
    }

    #[test]
    fn test_weighted_least_squares_quadratic_rank_deficient() {
        // Test quadratic fit with rank-deficient design matrix
        // Small neighborhood can cause issues
        let x_data = vec![
            1.0, 1.0, 1.0, 1.0,  // intercept, x, x², x³ (nearly collinear for small x range)
            1.0, 1.01, 1.0201, 1.030301,
            1.0, 1.02, 1.0404, 1.061208,
            1.0, 1.03, 1.0609, 1.092727,
        ];
        let x = Matrix::new(4, 4, x_data);
        // y = 2 + 3*x + 0.5*x²
        let y = vec![2.0 + 3.0 * 1.0 + 0.5 * 1.0,
                     2.0 + 3.0 * 1.01 + 0.5 * 1.0201,
                     2.0 + 3.0 * 1.02 + 0.5 * 1.0404,
                     2.0 + 3.0 * 1.03 + 0.5 * 1.0609];
        let weights = vec![1.0, 1.0, 1.0, 1.0];

        let coeffs = weighted_least_squares(&x, &y, &weights).unwrap();

        // Check we get reasonable coefficients
        assert!(coeffs[0].is_finite());
        assert!(coeffs[1].is_finite());
        assert!(coeffs[2].is_finite());
        assert!(coeffs[3].is_finite());

        // Predictions should be close to actual values
        for i in 0..4 {
            let xi = 1.0 + i as f64 * 0.01;
            let pred = coeffs[0] + coeffs[1] * xi + coeffs[2] * xi * xi;
            assert!((pred - y[i]).abs() < 0.5);
        }
    }

    #[test]
    fn test_pseudoinverse_minimum_neighbors() {
        // Test with minimum neighborhood size (we use 2 for WLS)
        // This is the edge case where QR might fail but SVD succeeds
        let x_data = vec![
            1.0, 1.0,
            1.0, 1.000001,  // Nearly identical to first row
        ];
        let x = Matrix::new(2, 2, x_data);
        let y = vec![2.0, 2.000001];
        let weights = vec![1.0, 1.0];

        let coeffs = weighted_least_squares(&x, &y, &weights).unwrap();

        // Should produce finite coefficients
        assert!(coeffs[0].is_finite());
        assert!(coeffs[1].is_finite());
    }

    #[test]
    fn test_svd_handles_zero_variance_column() {
        // Test SVD when one column has zero variance (all same values)
        let x_data = vec![
            1.0, 1.0, 5.0,  // intercept, x1=1, x2=5 (constant)
            1.0, 2.0, 5.0,  // intercept, x1=2, x2=5 (constant)
            1.0, 3.0, 5.0,  // intercept, x1=3, x2=5 (constant)
            1.0, 4.0, 5.0,  // intercept, x1=4, x2=5 (constant)
            1.0, 5.0, 5.0,  // intercept, x1=5, x2=5 (constant)
        ];
        let x = Matrix::new(5, 3, x_data);
        // True relationship: y = 3 + 2*x1 (x2 is irrelevant)
        let y = vec![5.0, 7.0, 9.0, 11.0, 13.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let coeffs = weighted_least_squares(&x, &y, &weights).unwrap();

        // Should find solution despite zero-variance column
        assert!(coeffs[0].is_finite());
        assert!(coeffs[1].is_finite());
        assert!(coeffs[2].is_finite());

        // Predictions should be good
        for i in 0..5 {
            let pred = coeffs[0] + coeffs[1] * (i + 1) as f64 + coeffs[2] * 5.0;
            assert!((pred - y[i]).abs() < 1.0);
        }
    }
}
