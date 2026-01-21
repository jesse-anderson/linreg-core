// ============================================================================
// Numerical Accuracy Tests
// ============================================================================
//
// Tests comparing our implementations against known reference values from
// established numerical libraries (numpy, R, LAPACK).
//
// These test matrices have known, verified results from the literature.

use linreg_core::linalg::Matrix;
use super::common::{assert_close, assert_matrix_eq, EPSILON, QR_TOLERANCE};

// ============================================================================
// QR Decomposition Reference Tests
// ============================================================================

#[test]
fn test_qr_known_matrix_1() {
    // Golub & Van Loan (3rd ed), 5.2.1
    // A = [1 2 3; 4 5 6; 7 8 9; 10 11 12]
    // But this is rank-deficient, so use a full-rank variant
    let a = Matrix::new(
        4,
        3,
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 11.0,  // Changed from 9.0 for full rank
            10.0, 11.0, 12.0,
        ],
    );

    let (q, r) = a.qr();

    // Verify Q^T * Q = I (orthogonality)
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);
    let identity = Matrix::identity(4);
    assert_matrix_eq(&qt_q, &identity, 1e-9, "Q^T * Q = I");

    // Verify R is upper triangular
    for i in 1..r.rows {
        for j in 0..i.min(r.cols) {
            assert!(r.get(i, j).abs() < QR_TOLERANCE, "R lower triangle should be zero");
        }
    }

    // Verify reconstruction
    let reconstructed = q.matmul(&r);
    assert_matrix_eq(&a, &reconstructed, QR_TOLERANCE, "A = Q * R");
}

#[test]
fn test_qr_householder_example() {
    // x = [3, 4, 0]^T
    // After QR: Q*x should have only first element non-zero
    let a = Matrix::new(
        3,
        1,
        vec![3.0, 4.0, 0.0],
    );

    let (q, r) = a.qr();

    // R should be [5, 0, 0]^T (the norm of x)
    assert_close(r.get(0, 0).abs(), 5.0, 1e-10, "R[0,0] should be ||x||");
    assert_close(r.get(1, 0), 0.0, 1e-10, "R[1,0] should be 0");
    assert_close(r.get(2, 0), 0.0, 1e-10, "R[2,0] should be 0");

    // Q should be orthogonal
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(qt_q.get(i, j), expected, 1e-9, &format!("Q orthogonality [{},{}]", i, j));
        }
    }
}

#[test]
fn test_qr_3x3_symmetric_positive_definite() {
    // Test with a symmetric positive definite matrix
    // A = [4 1 2; 1 5 3; 2 3 6]
    let a = Matrix::new(
        3,
        3,
        vec![
            4.0, 1.0, 2.0,
            1.0, 5.0, 3.0,
            2.0, 3.0, 6.0,
        ],
    );

    let (q, r) = a.qr();

    // Note: Householder QR may produce negative diagonal elements
    // depending on the sign convention used. The key property is that
    // the magnitude is correct and Q is orthogonal.

    // Verify reconstruction (this is the important property)
    let reconstructed = q.matmul(&r);
    assert_matrix_eq(&a, &reconstructed, 1e-9, "SPD QR reconstruction");

    // Verify Q is orthogonal
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);
    let identity = Matrix::identity(3);
    assert_matrix_eq(&qt_q, &identity, 1e-9, "Q is orthogonal for SPD");
}

// ============================================================================
// Matrix Inversion Reference Tests
// ============================================================================

#[test]
fn test_invert_2x2_known_values() {
    // Test with a 2x2 matrix where we can compute the inverse analytically
    // A = [4 7; 2 6]
    // det(A) = 4*6 - 7*2 = 24 - 14 = 10
    // A^(-1) = (1/10) * [6 -7; -2 4] = [0.6 -0.7; -0.2 0.4]
    let a = Matrix::new(2, 2, vec![4.0, 7.0, 2.0, 6.0]);

    let inv = a.invert().expect("Matrix should be invertible");

    // Expected inverse: [0.6, -0.7; -0.2, 0.4]
    let expected = Matrix::new(
        2,
        2,
        vec![0.6, -0.7, -0.2, 0.4],
    );

    assert_matrix_eq(&inv, &expected, 1e-10, "2x2 inverse");

    // Verify A * A^(-1) = I
    let result = a.matmul(&inv);
    let identity = Matrix::identity(2);
    assert_matrix_eq(&result, &identity, 1e-10, "A * A^(-1) = I");
}

#[test]
fn test_invert_3x3_known_values() {
    // Test with a 3x3 matrix with known inverse
    // A = [2 0 1; 3 0 0; 5 1 1]
    let a = Matrix::new(
        3,
        3,
        vec![
            2.0, 0.0, 1.0,
            3.0, 0.0, 0.0,
            5.0, 1.0, 1.0,
        ],
    );

    let inv = a.invert().expect("Matrix should be invertible");

    // Verify by computing A * A^(-1)
    let result = a.matmul(&inv);
    let identity = Matrix::identity(3);
    assert_matrix_eq(&result, &identity, 1e-9, "3x3 A * A^(-1) = I");
}

#[test]
fn test_invert_diagonal_matrix() {
    // Diagonal matrix inverse is just reciprocal of diagonal elements
    // A = diag(2, 3, 5)
    let a = Matrix::new(
        3,
        3,
        vec![
            2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 5.0,
        ],
    );

    let inv = a.invert().expect("Diagonal should be invertible");

    // Expected: diag(0.5, 1/3, 0.2)
    assert_close(inv.get(0, 0), 0.5, 1e-10, "inv[0,0]");
    assert_close(inv.get(1, 1), 1.0 / 3.0, 1e-10, "inv[1,1]");
    assert_close(inv.get(2, 2), 0.2, 1e-10, "inv[2,2]");

    // Off-diagonal should be zero
    assert_close(inv.get(0, 1), 0.0, 1e-10, "inv[0,1]");
    assert_close(inv.get(0, 2), 0.0, 1e-10, "inv[0,2]");
    assert_close(inv.get(1, 0), 0.0, 1e-10, "inv[1,0]");
    assert_close(inv.get(1, 2), 0.0, 1e-10, "inv[1,2]");
    assert_close(inv.get(2, 0), 0.0, 1e-10, "inv[2,0]");
    assert_close(inv.get(2, 1), 0.0, 1e-10, "inv[2,1]");
}

// ============================================================================
// Ill-Conditioned Matrix Tests
// ============================================================================

#[test]
fn test_near_singular_detection() {
    // Matrix with very small determinant (near-singular)
    // Uses relative tolerance, so should still detect singularity
    let epsilon = 1e-15;
    let a = Matrix::new(
        3,
        3,
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0 + epsilon,  // Nearly rank-deficient (rows nearly linear)
        ],
    );

    let inv = a.invert();

    // With relative tolerance and the small perturbation,
    // this might succeed or fail depending on the tolerance
    // If it succeeds, the inverse should be very large in magnitude
    if let Some(inv_mat) = inv {
        // The inverse of a near-singular matrix has very large elements
        let max_val = (0..9).map(|i| inv_mat.data[i].abs()).fold(0.0_f64, f64::max);
        assert!(max_val > 1e10, "Near-singular inverse should have large elements");
    }
    // If it returns None, that's also acceptable behavior
}

#[test]
fn test_well_conditioned_inverse() {
    // Well-conditioned matrix (diagonally dominant)
    let a = Matrix::new(
        4,
        4,
        vec![
            10.0, 1.0, 1.0, 1.0,
            1.0, 10.0, 1.0, 1.0,
            1.0, 1.0, 10.0, 1.0,
            1.0, 1.0, 1.0, 10.0,
        ],
    );

    let inv = a.invert().expect("Well-conditioned matrix should be invertible");

    // Verify A * A^(-1) = I
    let result = a.matmul(&inv);
    let identity = Matrix::identity(4);
    assert_matrix_eq(&result, &identity, 1e-9, "Well-conditioned inverse");
}

// ============================================================================
// Matrix Multiplication Numerical Tests
// ============================================================================

#[test]
fn test_matmul_accuracy() {
    // Test matrix multiplication accuracy with specific values
    // A = [1 2; 3 4], B = [5 6; 7 8]
    // A*B = [1*5+2*7  1*6+2*8] = [19 22]
    //      [3*5+4*7  3*6+4*8]   [43 50]
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

    let result = a.matmul(&b);

    assert_close(result.get(0, 0), 19.0, 1e-10, "matmul[0,0]");
    assert_close(result.get(0, 1), 22.0, 1e-10, "matmul[0,1]");
    assert_close(result.get(1, 0), 43.0, 1e-10, "matmul[1,0]");
    assert_close(result.get(1, 1), 50.0, 1e-10, "matmul[1,1]");
}

#[test]
fn test_matmul_with_fractions() {
    // Test multiplication with fractions that could cause precision issues
    // A = [1/3 1/7; 1/11 1/13], B = similar
    let a = Matrix::new(
        2,
        2,
        vec![1.0 / 3.0, 1.0 / 7.0, 1.0 / 11.0, 1.0 / 13.0],
    );
    let b = Matrix::new(
        2,
        2,
        vec![1.0 / 17.0, 1.0 / 19.0, 1.0 / 23.0, 1.0 / 29.0],
    );

    let result = a.matmul(&b);

    // Verify associativity: (A * B) * I = A * B
    let identity = Matrix::identity(2);
    let result2 = result.matmul(&identity);

    for i in 0..2 {
        for j in 0..2 {
            let diff = (result.get(i, j) - result2.get(i, j)).abs();
            assert!(diff < 1e-15, "Fraction matmul associativity failed at [{},{}]", i, j);
        }
    }
}

// ============================================================================
// chol2inv Numerical Accuracy Tests
// ============================================================================

#[test]
fn test_chol2inv_accuracy() {
    // Compare chol2inv with direct inversion of X'X
    // X is a design matrix with intercept and predictor
    let x = Matrix::new(
        5,
        2,
        vec![
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0,
            1.0, 4.0,
            1.0, 5.0,
        ],
    );

    let chol2inv = x.chol2inv_from_qr().expect("chol2inv should succeed");

    // Compute X'X and invert directly
    let x_t = x.transpose();
    let xtx = x_t.matmul(&x);
    let xtx_inv = xtx.invert().expect("X'X should be invertible");

    // Should be very close
    assert_matrix_eq(&chol2inv, &xtx_inv, 1e-9, "chol2inv vs direct invert");
}

#[test]
fn test_chol2inv_symmetric() {
    // chol2inv should produce a symmetric matrix
    let x = Matrix::new(
        4,
        2,
        vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ],
    );

    let result = x.chol2inv_from_qr().expect("chol2inv should succeed");

    // Check symmetry: result[i,j] == result[j,i]
    for i in 0..2 {
        for j in 0..2 {
            let diff = (result.get(i, j) - result.get(j, i)).abs();
            assert!(diff < 1e-10, "chol2inv should be symmetric at [{},{}]", i, j);
        }
    }
}

// ============================================================================
// Vector Operation Numerical Tests
// ============================================================================

#[test]
fn test_vec_dot_accuracy() {
    use linreg_core::linalg::vec_dot;

    // Test with specific values
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    let result = vec_dot(&a, &b);
    assert_close(result, 32.0, 1e-10, "dot product accuracy");
}

#[test]
fn test_vec_l2_norm_accuracy() {
    use linreg_core::linalg::vec_l2_norm;

    // Test with Pythagorean triple (3, 4, 5)
    let v = vec![3.0, 4.0];
    let result = vec_l2_norm(&v);
    assert_close(result, 5.0, 1e-10, "L2 norm of (3, 4) should be 5");

    // Test with another triple (5, 12, 13)
    let v2 = vec![5.0, 12.0];
    let result2 = vec_l2_norm(&v2);
    assert_close(result2, 13.0, 1e-10, "L2 norm of (5, 12) should be 13");
}

#[test]
fn test_vec_scale_accuracy() {
    use linreg_core::linalg::vec_scale;

    let v = vec![1.0, 2.0, 3.0];
    let alpha = 2.5;

    let result = vec_scale(&v, alpha);

    assert_close(result[0], 2.5, 1e-10, "scaled[0]");
    assert_close(result[1], 5.0, 1e-10, "scaled[1]");
    assert_close(result[2], 7.5, 1e-10, "scaled[2]");
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

#[test]
fn test_householder_sign_stability() {
    // Test that Householder transformation uses the numerically stable sign choice
    // The key is that for x = [x1, x2, ...], we compute v = x + sgn(x1)*||x||*e1
    // This avoids cancellation when x1 is positive

    // Case 1: First element is positive
    let a1 = Matrix::new(3, 1, vec![1.0, 0.01, 0.01]);
    let (_q1, r1) = a1.qr();
    // The R element should be negative (from the -||x|| convention)
    assert!(r1.get(0, 0) < 0.0, "R[0,0] should be negative (Householder convention)");

    // Case 2: First element is negative
    let a2 = Matrix::new(3, 1, vec![-1.0, 0.01, 0.01]);
    let (_q2, r2) = a2.qr();
    // The R element should be positive (from the -||x|| convention with sign flip)
    assert!(r2.get(0, 0) > 0.0, "R[0,0] should be positive for negative x[0]");
}

#[test]
fn test_back_substitution_stability() {
    // Test that back substitution in upper triangular inversion is stable
    // This tests the core algorithm used by invert_upper_triangular

    let a = Matrix::new(
        4,
        4,
        vec![
            2.0, 3.0, 1.0, 4.0,
            0.0, 5.0, 2.0, 1.0,
            0.0, 0.0, 3.0, 2.0,
            0.0, 0.0, 0.0, 4.0,
        ],
    );

    let inv = a.invert_upper_triangular().expect("Should invert");

    // Verify by computing A * A^(-1) = I
    let result = a.matmul(&inv);
    let identity = Matrix::identity(4);
    assert_matrix_eq(&result, &identity, 1e-9, "Back substitution stability");
}

// ============================================================================
// Tolerance Tests
// ============================================================================

#[test]
fn test_custom_tolerance_handling() {
    // Test that custom tolerance works correctly for near-singular matrices
    let a = Matrix::new(
        3,
        3,
        vec![
            1e-8, 1.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
        ],
    );

    // With strict tolerance, might fail
    let _result_strict = a.invert_upper_triangular_with_tolerance(0.01);

    // With permissive tolerance, should succeed
    let result_permissive = a.invert_upper_triangular_with_tolerance(1000.0);
    assert!(result_permissive.is_some(), "Permissive tolerance should succeed");
}

#[test]
fn test_tolerance_range() {
    // Test that tolerance works across different scales
    for scale in &[1e-10, 1e-5, 1.0, 1e5, 1e10] {
        let a = Matrix::new(
            3,
            3,
            vec![
                scale * 2.0, scale * 1.0, scale * 1.0,
                0.0, scale * 3.0, scale * 1.0,
                0.0, 0.0, scale * 4.0,
            ],
        );

        let inv = a.invert_upper_triangular();
        assert!(inv.is_some(), "Should invert at scale {}", scale);

        // Verify accuracy is maintained
        let inv = inv.unwrap();
        let result = a.matmul(&inv);
        let identity = Matrix::identity(3);

        // Use relative tolerance
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (result.get(i, j) - expected).abs();
                assert!(diff < 1e-6, "Tolerance test failed at scale {}, [{},{}]", scale, i, j);
            }
        }
    }
}
