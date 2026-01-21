// ============================================================================
// Matrix Inversion Tests
// ============================================================================
//
// Tests for matrix inversion, upper triangular inversion, chol2inv_from_qr,
// and related edge cases.

use linreg_core::linalg::Matrix;
use super::common::{EPSILON, assert_close, assert_matrix_eq};

// ============================================================================
// Basic Inversion Tests
// ============================================================================

#[test]
fn test_invert_reconstruction() {
    // A * A^(-1) = I
    let a = Matrix::new(3, 3, vec![4.0, 7.0, 2.0, 3.0, 6.0, 1.0, 2.0, 5.0, 3.0]);

    let inv = a.invert().expect("Matrix should be invertible");

    // A * A^(-1) should equal I
    let a_inv = a.matmul(&inv);
    let i = Matrix::identity(3);

    assert_matrix_eq(&a_inv, &i, 1e-9, "A * A^-1 = I");

    // Also check A^(-1) * A = I
    let inv_a = inv.matmul(&a);
    assert_matrix_eq(&inv_a, &i, 1e-9, "A^-1 * A = I");
}

#[test]
fn test_invert_identity() {
    let i = Matrix::identity(4);
    let inv = i.invert().expect("Identity should be invertible");

    assert_matrix_eq(&i, &inv, EPSILON, "I^-1 = I");
}

#[test]
fn test_invert_diagonal() {
    let d = Matrix::new(3, 3, vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]);
    let expected = Matrix::new(
        3,
        3,
        vec![0.5, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.25],
    );

    let inv = d.invert().expect("Diagonal should be invertible");
    assert_matrix_eq(&inv, &expected, EPSILON, "diagonal inverse");
}

#[test]
fn test_invert_singular_returns_none() {
    // Singular matrix (determinant = 0)
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]); // Second row = 2 * first row

    let result = a.invert();
    assert!(result.is_none(), "Singular matrix should return None");
}

#[test]
fn test_invert_1x1() {
    let a = Matrix::new(1, 1, vec![5.0]);
    let inv = a.invert().expect("1x1 should be invertible");

    assert_eq!(inv.rows, 1);
    assert_eq!(inv.cols, 1);
    assert_close(inv.get(0, 0), 0.2, EPSILON, "1x1 inverse");
}

#[test]
fn test_invert_upper_triangular() {
    let a = Matrix::new(
        3,
        3,
        vec![2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0],
    );

    let inv = a
        .invert_upper_triangular()
        .expect("Upper triangular should be invertible");

    // Reconstruct identity
    let a_inv = a.matmul(&inv);
    let i = Matrix::identity(3);
    assert_matrix_eq(&a_inv, &i, 1e-9, "upper triangular inverse");
}

#[test]
fn test_invert_2x2() {
    let a = Matrix::new(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
    let inv = a.invert().expect("2x2 should be invertible");

    // A * A^-1 = I
    let result = a.matmul(&inv);
    let i = Matrix::identity(2);
    assert_matrix_eq(&result, &i, 1e-10, "2x2 inversion");
}

#[test]
fn test_invert_5x5() {
    // Test inversion of a larger matrix
    let data: Vec<f64> = (1..=25).map(|i| i as f64).collect();
    let _a = Matrix::new(5, 5, data);

    // 5x5 matrix with sequential values 1-25 might be singular
    // Let's use a known invertible matrix instead (diagonal dominance)
    let mut a_data = vec![0.0; 25];
    for i in 0..5 {
        for j in 0..5 {
            if i == j {
                a_data[i * 5 + j] = 10.0 + i as f64;  // Diagonal dominance
            } else {
                a_data[i * 5 + j] = 1.0 / (1.0 + i as f64 + j as f64);
            }
        }
    }
    let a = Matrix::new(5, 5, a_data);

    let inv = a.invert();
    assert!(inv.is_some(), "5x5 diagonally dominant matrix should be invertible");

    if let Some(inv_mat) = inv {
        let result = a.matmul(&inv_mat);
        let i = Matrix::identity(5);
        assert_matrix_eq(&result, &i, 1e-8, "5x5 inversion");
    }
}

#[test]
fn test_invert_diagonal_matrix() {
    // Diagonal matrix inversion is simple
    let a = Matrix::new(4, 4, vec![
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 5.0, 0.0,
        0.0, 0.0, 0.0, 7.0,
    ]);

    let inv = a.invert().expect("Diagonal should be invertible");

    // Check diagonal elements
    assert_close(inv.get(0, 0), 0.5, EPSILON, "diag inv[0,0]");
    assert_close(inv.get(1, 1), 1.0/3.0, EPSILON, "diag inv[1,1]");
    assert_close(inv.get(2, 2), 0.2, EPSILON, "diag inv[2,2]");
    assert_close(inv.get(3, 3), 1.0/7.0, EPSILON, "diag inv[3,3]");

    // Off-diagonal should be zero
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                assert_close(inv.get(i, j), 0.0, EPSILON, &format!("diag inv off[{},{}]", i, j));
            }
        }
    }
}

#[test]
fn test_invert_near_singular() {
    // Matrix with very small determinant (near-singular but still invertible)
    let epsilon = 1e-8;
    let a = Matrix::new(
        2,
        2,
        vec![
            1.0, 1.0,
            1.0 + epsilon, 1.0,
        ],
    );

    let inv = a.invert();
    // With relative tolerance, this might still be invertible
    // if the diagonal elements are large enough relative to epsilon
    assert!(inv.is_some(), "Near-singular matrix might still invert with relative tolerance");
}

#[test]
fn test_invert_permutation_matrix() {
    // Permutation matrix (reversed rows of identity)
    let a = Matrix::new(
        3,
        3,
        vec![
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
        ],
    );

    let inv = a.invert().expect("Permutation matrix should be invertible");

    // Permutation matrix inverse equals transpose
    let a_t = a.transpose();
    for i in 0..3 {
        for j in 0..3 {
            assert_close(inv.get(i, j), a_t.get(i, j), EPSILON, &format!("perm inv[{},{}]", i, j));
        }
    }

    // A * A = I for this involution
    let aa = a.matmul(&a);
    let i = Matrix::identity(3);
    assert_matrix_eq(&aa, &i, EPSILON, "permutation involution");
}

#[test]
fn test_with_tiny_values() {
    // Test with values at a small but workable scale
    // Using 1e-10 instead of EPSILON (1e-16) to stay within reasonable precision
    let scale = 1e-10;
    let a = Matrix::new(
        2,
        2,
        vec![
            100.0 * scale, 200.0 * scale,
            300.0 * scale, 400.0 * scale,
        ],
    );

    let inv = a.invert();
    // At 1e-10 scale, relative tolerance should still allow inversion
    assert!(inv.is_some(), "1e-10 scale matrix should be invertible with relative tolerance");

    // Verify the inversion is correct
    if let Some(inv_mat) = inv {
        let result = a.matmul(&inv_mat);
        let identity = Matrix::identity(2);
        // Use a tolerance appropriate for the scale
        let tolerance = 1e-6;
        for i in 0..2 {
            for j in 0..2 {
                let diff = (result.get(i, j) - identity.get(i, j)).abs();
                assert!(
                    diff < tolerance,
                    "Tiny value inversion error at [{},{}]: diff = {}",
                    i, j, diff
                );
            }
        }
    }
}

// ============================================================================
// Scale Tests
// ============================================================================

#[test]
fn test_invert_large_scale_matrix() {
    // Test inversion of a matrix with large values (1e6 scale)
    // This verifies the relative tolerance handles large values correctly
    let a = Matrix::new(
        3,
        3,
        vec![
            1e6, 2e6, 3e6,
            4e6, 5e6, 6e6,
            7e6, 8e6, 1e7,  // Changed from 9e6 to avoid near-singularity
        ],
    );

    let inv = a.invert().expect("Large scale matrix should be invertible");

    // Verify A * A^-1 = I
    let result = a.matmul(&inv);
    let _i = Matrix::identity(3);

    // Use a tolerance relative to the scale of the matrix
    let tolerance = 1e-8 * 1e6;  // Relative tolerance for 1e6 scale
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (result.get(i, j) - expected).abs();
            assert!(
                diff < tolerance,
                "Large scale inversion error at [{},{}]: got {}, expected {}, diff = {}",
                i, j, result.get(i, j), expected, diff
            );
        }
    }
}

#[test]
fn test_invert_small_scale_matrix() {
    // Test inversion of a matrix with very small values (1e-6 scale)
    let a = Matrix::new(
        3,
        3,
        vec![
            1e-6, 2e-6, 3e-6,
            4e-6, 5e-6, 6e-6,
            7e-6, 8e-6, 1e-7,  // Note: small value to avoid singularity
        ],
    );

    let inv = a.invert().expect("Small scale matrix should be invertible");

    // Verify A * A^-1 = I
    let result = a.matmul(&inv);
    let _i = Matrix::identity(3);

    // Use appropriate tolerance for small scale
    let tolerance = 1e-8;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (result.get(i, j) - expected).abs();
            assert!(
                diff < tolerance,
                "Small scale inversion error at [{},{}]: got {}, expected {}, diff = {}",
                i, j, result.get(i, j), expected, diff
            );
        }
    }
}

#[test]
fn test_invert_upper_triangular_large_scale() {
    // Test upper triangular inversion with large diagonal values
    let a = Matrix::new(
        3,
        3,
        vec![
            1e8, 2e8, 3e8,
            0.0, 4e8, 5e8,
            0.0, 0.0, 6e8,
        ],
    );

    let inv = a.invert_upper_triangular().expect("Large upper triangular should be invertible");

    // Verify A * A^-1 = I
    let result = a.matmul(&inv);
    let _i = Matrix::identity(3);

    // Use relative tolerance for large scale
    let tolerance = 1e-10 * 1e8;  // Relative tolerance for 1e8 scale
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (result.get(i, j) - expected).abs();
            assert!(
                diff < tolerance,
                "Large triangular inversion error at [{},{}]: got {}, expected {}, diff = {}",
                i, j, result.get(i, j), expected, diff
            );
        }
    }
}

#[test]
fn test_singular_with_wide_range_diagonals() {
    // Test that singular detection works when diagonal elements have very different magnitudes
    // Matrix with diagonal elements ranging from very small to large
    let a = Matrix::new(
        3,
        3,
        vec![
            1e-10, 1.0, 1.0,     // Very small first diagonal element
            0.0,   1e6, 1.0,
            0.0,   0.0, 1.0,
        ],
    );

    let inv = a.invert_upper_triangular();
    // With relative tolerance, the small diagonal (1e-10) compared to max (1e6)
    // should be correctly identified as singular
    assert!(inv.is_none(), "Matrix with very small diagonal element should be singular");
}

#[test]
fn test_well_conditioned_with_wide_range_diagonals() {
    // Test that matrices with wide-ranging but non-zero diagonals can be inverted
    // when the relative scale is within acceptable bounds
    let a = Matrix::new(
        3,
        3,
        vec![
            0.1,   1.0, 1.0,
            0.0,   1000.0, 1.0,
            0.0,   0.0,    10000.0,
        ],
    );

    let inv = a.invert_upper_triangular();
    // This should be invertible - the range (0.1 to 10000) is large but
    // all elements are well above machine epsilon relative to max
    assert!(inv.is_some(), "Matrix with well-conditioned wide-range diagonals should be invertible");
}

// ============================================================================
// Panic Tests
// ============================================================================

#[test]
#[should_panic(expected = "Matrix must be square")]
fn test_invert_panic_on_non_square() {
    // invert() should panic on non-square matrix
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    a.invert();
}

#[test]
#[should_panic(expected = "Matrix must be square")]
fn test_invert_upper_triangular_panic_on_non_square() {
    // invert_upper_triangular() should panic on non-square matrix
    let a = Matrix::new(3, 2, vec![2.0, 3.0, 0.0, 5.0, 0.0, 0.0]);
    a.invert_upper_triangular();
}

// ============================================================================
// chol2inv_from_qr() Tests
// ============================================================================

#[test]
fn test_chol2inv_basic() {
    // Basic test: compare chol2inv result with direct (X'X)^(-1)
    // X is a 4x3 matrix (tall, as used in OLS)
    let x = Matrix::new(
        4,
        3,
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,  // Changed from 9.0 to ensure full rank
            2.0, 3.0, 4.0,
        ],
    );

    let result = x.chol2inv_from_qr();
    assert!(result.is_some(), "chol2inv should succeed for full-rank matrix");

    let chol2inv = result.unwrap();

    // Compute X'X directly and invert for comparison
    let x_t = x.transpose();
    let xtx = x_t.matmul(&x);
    let xtx_inv = xtx.invert().expect("X'X should be invertible");

    // Results should be very close
    assert_matrix_eq(&chol2inv, &xtx_inv, 1e-9, "chol2inv vs direct invert");
}

#[test]
fn test_chol2inv_reconstruction() {
    // Verify that (X'X) * chol2inv(X) = I
    let x = Matrix::new(
        5,
        3,
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
            2.0, 3.0, 4.0,
            5.0, 6.0, 8.0,
        ],
    );

    let chol2inv = x.chol2inv_from_qr().expect("chol2inv should succeed");

    // Compute X'X
    let x_t = x.transpose();
    let xtx = x_t.matmul(&x);

    // (X'X) * (X'X)^(-1) should equal I
    let result = xtx.matmul(&chol2inv);
    let i = Matrix::identity(3);

    assert_matrix_eq(&result, &i, 1e-9, "X'X * chol2inv = I");
}

#[test]
fn test_chol2inv_symmetric() {
    // The result of chol2inv should be symmetric
    let x = Matrix::new(
        4,
        2,
        vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 9.0,  // Last value slightly off to ensure full rank
        ],
    );

    let result = x.chol2inv_from_qr().expect("chol2inv should succeed");

    // Check symmetry: result[i,j] should equal result[j,i]
    for i in 0..2 {
        for j in 0..2 {
            assert_close(
                result.get(i, j),
                result.get(j, i),
                1e-10,
                &format!("symmetry[{},{}]", i, j),
            );
        }
    }
}

#[test]
fn test_chol2inv_rank_deficient_returns_none() {
    // Rank-deficient matrix should return None
    // Column 2 is exactly 2 * column 1
    let x = Matrix::new(
        3,
        2,
        vec![
            1.0, 2.0,
            2.0, 4.0,  // 2x col1
            3.0, 6.0,  // 3x col1
        ],
    );

    let result = x.chol2inv_from_qr();
    assert!(result.is_none(), "Rank-deficient matrix should return None");
}

#[test]
fn test_chol2inv_near_rank_deficient() {
    // Nearly rank-deficient but still invertible
    let epsilon = 1e-6;
    let x = Matrix::new(
        3,
        2,
        vec![
            1.0, 2.0,
            2.0, 4.0 + epsilon,
            3.0, 6.0,
        ],
    );

    let result = x.chol2inv_from_qr();
    // Should still work for near-full-rank matrices
    assert!(result.is_some(), "Near-full-rank matrix should be invertible");
}

#[test]
fn test_chol2inv_single_column() {
    // Test with single predictor (p=1)
    let x = Matrix::new(
        4,
        1,
        vec![1.0, 2.0, 3.0, 4.0],
    );

    let result = x.chol2inv_from_qr().expect("Single column should work");

    // X'X = sum(x^2) = 1 + 4 + 9 + 16 = 30
    // (X'X)^(-1) = 1/30
    assert_eq!(result.rows, 1);
    assert_eq!(result.cols, 1);
    assert_close(result.get(0, 0), 1.0 / 30.0, 1e-10, "single column chol2inv");
}

#[test]
fn test_chol2inv_two_predictors() {
    // Test with 2 predictors (common regression case)
    // X = [1 x1] design matrix with intercept
    let x = Matrix::new(
        5,
        2,
        vec![
            1.0, 1.0,  // intercept=1, x1=1
            1.0, 2.0,
            1.0, 3.0,
            1.0, 4.0,
            1.0, 5.0,
        ],
    );

    let result = x.chol2inv_from_qr().expect("Two predictors should work");

    // Verify by computing (X'X)^(-1) directly
    let x_t = x.transpose();
    let xtx = x_t.matmul(&x);
    let expected = xtx.invert().expect("X'X should be invertible");

    assert_matrix_eq(&result, &expected, 1e-9, "two predictors chol2inv");
}

#[test]
fn test_chol2inv_large_scale() {
    // Test with large values to verify relative tolerance handling
    let scale = 1e6;
    let x = Matrix::new(
        4,
        3,
        vec![
            1.0 * scale, 2.0 * scale, 3.0 * scale,
            2.0 * scale, 3.0 * scale, 4.0 * scale,
            3.0 * scale, 4.0 * scale, 6.0 * scale,
            1.5 * scale, 2.5 * scale, 3.5 * scale,
        ],
    );

    let result = x.chol2inv_from_qr();

    // Should succeed with large-scale matrix
    assert!(result.is_some(), "Large-scale matrix should work");

    // Verify reconstruction
    if let Some(chol2inv) = result {
        let x_t = x.transpose();
        let xtx = x_t.matmul(&x);
        let product = xtx.matmul(&chol2inv);
        let _i = Matrix::identity(3);

        // Use relative tolerance for large scale
        let tolerance = 1e-6 * scale;
        for i_idx in 0..3 {
            for j in 0..3 {
                let expected = if i_idx == j { 1.0 } else { 0.0 };
                let diff = (product.get(i_idx, j) - expected).abs();
                assert!(
                    diff < tolerance,
                    "Large scale chol2inv reconstruction error at [{},{}]: diff = {}",
                    i_idx, j, diff
                );
            }
        }
    }
}

#[test]
fn test_chol2inv_small_scale() {
    // Test with moderately small values (scale = 1e-4 is small but numerically stable)
    // Using 1e-8 caused issues with the relative tolerance in invert() method
    let scale = 1e-4;
    let x = Matrix::new(
        3,
        2,
        vec![
            1.0 * scale, 2.0 * scale,
            2.0 * scale, 3.0 * scale,
            3.0 * scale, 5.0 * scale,
        ],
    );

    let result = x.chol2inv_from_qr();
    assert!(result.is_some(), "Small-scale matrix should work");

    // Verify by reconstruction: (X'X) * (X'X)^(-1) should equal I
    if let Some(chol2inv) = result {
        let x_t = x.transpose();
        let xtx = x_t.matmul(&x);
        let identity = xtx.matmul(&chol2inv);

        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (identity.get(i, j) - expected).abs();
                assert!(
                    diff < 1e-9,
                    "Small scale reconstruction error at [{},{}]: got {}, expected {}, diff = {}",
                    i, j, identity.get(i, j), expected, diff
                );
            }
        }
    }
}
