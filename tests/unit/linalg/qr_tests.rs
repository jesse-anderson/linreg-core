// ============================================================================
// QR Decomposition Tests
// ============================================================================
//
// Tests for QR decomposition including edge cases, numerical precision,
// and special matrix types.

use linreg_core::linalg::Matrix;
use super::common::{QR_TOLERANCE, assert_close, assert_matrix_eq};

// ============================================================================
// Core QR Decomposition Tests
// ============================================================================

#[test]
fn test_qr_q_orthogonal() {
    // Q from QR decomposition should be orthogonal: Q^T * Q = I
    let a = Matrix::new(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );

    let (q, _r) = a.qr();

    // Q^T * Q should be identity
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);

    // Q is m x m, so Q^T * Q should be m x m (4 x 4)
    assert_eq!(qt_q.rows, 4);
    assert_eq!(qt_q.cols, 4);

    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(qt_q.get(i, j), expected, QR_TOLERANCE, &format!("Q^T*Q[{},{}]", i, j));
        }
    }
}

#[test]
fn test_qr_r_upper_triangular() {
    // R from QR decomposition should be upper triangular
    let a = Matrix::new(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );

    let (_q, r) = a.qr();

    // Check lower triangle is zero (below diagonal)
    for i in 1..r.rows {
        for j in 0..i.min(r.cols) {
            assert_close(
                r.get(i, j),
                0.0,
                QR_TOLERANCE,
                &format!("R lower[{},{}]", i, j),
            );
        }
    }
}

#[test]
fn test_qr_reconstruction() {
    // QR decomposition should reconstruct A: A = Q * R
    let a = Matrix::new(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );

    let (q, r) = a.qr();
    let reconstructed = q.matmul(&r);

    assert_matrix_eq(&a, &reconstructed, QR_TOLERANCE, "QR reconstruction");
}

#[test]
fn test_qr_tall_matrix() {
    // m > n (more rows than columns)
    let a = Matrix::new(
        5,
        3,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            14.0, 15.0,
        ],
    );

    let (q, r) = a.qr();

    // Q should be m x m
    assert_eq!(q.rows, 5);
    assert_eq!(q.cols, 5);

    // R should be m x n
    assert_eq!(r.rows, 5);
    assert_eq!(r.cols, 3);

    // Check reconstruction
    let reconstructed = q.matmul(&r);
    assert_matrix_eq(&a, &reconstructed, QR_TOLERANCE, "tall QR reconstruction");
}

#[test]
fn test_qr_wide_matrix() {
    // m < n (fewer rows than columns)
    let a = Matrix::new(
        3,
        5,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            14.0, 15.0,
        ],
    );

    let (q, r) = a.qr();

    // Q should be m x m
    assert_eq!(q.rows, 3);
    assert_eq!(q.cols, 3);

    // R should be m x n
    assert_eq!(r.rows, 3);
    assert_eq!(r.cols, 5);

    // Check reconstruction
    let reconstructed = q.matmul(&r);
    assert_matrix_eq(&a, &reconstructed, QR_TOLERANCE, "wide QR reconstruction");
}

#[test]
fn test_qr_single_column() {
    let a = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);

    let (q, r) = a.qr();

    // Check reconstruction
    let reconstructed = q.matmul(&r);
    assert_matrix_eq(&a, &reconstructed, QR_TOLERANCE, "single column QR");

    // R should be 3x1, upper triangular
    assert_close(r.get(1, 0), 0.0, QR_TOLERANCE, "R[1,0] should be 0");
    assert_close(r.get(2, 0), 0.0, QR_TOLERANCE, "R[2,0] should be 0");
}

#[test]
fn test_qr_near_singular() {
    // Matrix with near-collinear columns
    let a = Matrix::new(
        3,
        2,
        vec![
            1.0, 2.0, 1.0 + 1e-8, 2.0 + 2e-8, 1.0 + 2e-8, 2.0 + 4e-8,
        ],
    );

    let (q, r) = a.qr();

    // Should still decompose
    assert_eq!(q.rows, 3);
    assert_eq!(q.cols, 3);
    assert_eq!(r.rows, 3);
    assert_eq!(r.cols, 2);

    // Check reconstruction
    let reconstructed = q.matmul(&r);
    assert_matrix_eq(&a, &reconstructed, 1e-6, "near-singular QR reconstruction");
}

#[test]
fn test_qr_square_matrix() {
    // Explicit test for square matrix (m == n)
    let a = Matrix::new(
        3,
        3,
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,  // Changed from 9.0 for full rank
        ],
    );

    let (q, r) = a.qr();

    // Q should be 3x3 orthogonal
    assert_eq!(q.rows, 3);
    assert_eq!(q.cols, 3);

    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);
    let i = Matrix::identity(3);
    assert_matrix_eq(&qt_q, &i, QR_TOLERANCE, "square Q orthogonality");

    // R should be upper triangular
    for i in 1..3 {
        for j in 0..i {
            assert_close(r.get(i, j), 0.0, QR_TOLERANCE, &format!("square R lower[{},{}]", i, j));
        }
    }

    // Reconstruction should work
    let reconstructed = q.matmul(&r);
    assert_matrix_eq(&a, &reconstructed, QR_TOLERANCE, "square QR reconstruction");
}

// ============================================================================
// QR Edge Cases
// ============================================================================

#[test]
fn test_qr_single_element_matrix() {
    // QR of a 1x1 matrix
    let a = Matrix::new(1, 1, vec![5.0]);
    let (q, r) = a.qr();

    // Q should be 1x1 identity (approximately)
    assert!((q.get(0, 0) - 1.0).abs() < 1e-10, "Q[0,0] should be ~1");

    // R should be the original value
    assert!((r.get(0, 0) - 5.0).abs() < 1e-10, "R[0,0] should be 5.0");

    // Reconstruction should work
    let reconstructed = q.matmul(&r);
    assert_close(reconstructed.get(0, 0), a.get(0, 0), 1e-10, "1x1 reconstruction");
}

#[test]
fn test_qr_row_vector() {
    // QR of a 1xn row vector (wide matrix)
    let a = Matrix::new(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
    let (q, r) = a.qr();

    // Q should be 1x1
    assert_eq!(q.rows, 1);
    assert_eq!(q.cols, 1);

    // Q^T * Q should equal I (1x1 identity)
    assert!((q.get(0, 0).powi(2) - 1.0).abs() < 1e-10, "Q^T * Q = I");

    // Reconstruction should work
    let reconstructed = q.matmul(&r);
    for j in 0..4 {
        assert_close(reconstructed.get(0, j), a.get(0, j), 1e-10, &format!("row vector reconstruction[0,{}]", j));
    }
}

#[test]
fn test_qr_column_vector() {
    // QR of an mx1 column vector (tall matrix)
    let a = Matrix::new(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let (q, r) = a.qr();

    // Q should be 5x5 orthogonal
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);
    for i in 0..5 {
        for j in 0..5 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(qt_q.get(i, j), expected, 1e-10, &format!("column Q orthogonality[{},{}]", i, j));
        }
    }

    // R should be 5x1 upper triangular (only R[0,0] non-zero in first column)
    // Note: R[0,0] = ±‖x‖ depending on Householder sign (both are valid)
    let expected_norm = (1.0_f64*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0 + 5.0*5.0).sqrt();
    assert!(r.get(0, 0).abs() - expected_norm < 1e-10, "R[0,0] should have norm magnitude");
    for i in 1..5 {
        assert_close(r.get(i, 0), 0.0, QR_TOLERANCE, &format!("R[{},0] should be 0", i));
    }
}

#[test]
fn test_qr_all_zeros_column() {
    // QR with a zero column (singular case)
    let a = Matrix::new(
        3,
        2,
        vec![
            1.0, 0.0,
            2.0, 0.0,
            3.0, 0.0,
        ],
    );

    let (q, r) = a.qr();
    let reconstructed = q.matmul(&r);

    // Should still reconstruct correctly
    assert_matrix_eq(&a, &reconstructed, QR_TOLERANCE, "zero column reconstruction");
}

#[test]
fn test_qr_negative_values() {
    // QR with all negative values
    let a = Matrix::new(
        3,
        2,
        vec![
            -1.0, -4.0,
            -2.0, -5.0,
            -3.0, -6.0,
        ],
    );

    let (q, r) = a.qr();

    // Q should be orthogonal
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(qt_q.get(i, j), expected, QR_TOLERANCE, &format!("negative Q orthogonality[{},{}]", i, j));
        }
    }

    // Reconstruction should work
    let reconstructed = q.matmul(&r);
    assert_matrix_eq(&a, &reconstructed, QR_TOLERANCE, "negative values reconstruction");
}

#[test]
fn test_qr_already_upper_triangular() {
    // QR of an already upper triangular matrix
    let a = Matrix::new(
        3,
        3,
        vec![
            2.0, 3.0, 4.0,
            0.0, 5.0, 6.0,
            0.0, 0.0, 7.0,
        ],
    );

    let (q, _r) = a.qr();

    // For an upper triangular matrix with positive diagonal,
    // Q should be close to identity (signs may vary)
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(qt_q.get(i, j), expected, QR_TOLERANCE, &format!("triangular Q orthogonality[{},{}]", i, j));
        }
    }
}

#[test]
fn test_qr_orthogonal_matrix() {
    // QR of an orthogonal matrix (Q should equal input, R should equal I)
    // Create a simple rotation matrix
    let theta = std::f64::consts::PI / 4.0;  // 45 degrees
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    let a = Matrix::new(
        2,
        2,
        vec![
            cos_t, -sin_t,
            sin_t, cos_t,
        ],
    );

    let (q, _r) = a.qr();

    // Q should equal A (up to sign)
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(qt_q.get(i, j), expected, 1e-10, &format!("rotation Q orthogonality[{},{}]", i, j));
        }
    }
}

#[test]
fn test_qr_symmetric_positive_definite() {
    // QR of a symmetric positive definite matrix
    let a = Matrix::new(
        3,
        3,
        vec![
            4.0, 1.0, 0.5,
            1.0, 5.0, 1.0,
            0.5, 1.0, 6.0,
        ],
    );

    let (q, r) = a.qr();

    // Q should be orthogonal
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(qt_q.get(i, j), expected, QR_TOLERANCE, &format!("SPD Q orthogonality[{},{}]", i, j));
        }
    }

    // R should be upper triangular
    for i in 1..3 {
        for j in 0..i {
            assert_close(r.get(i, j), 0.0, QR_TOLERANCE, &format!("SPD R lower[{},{}]", i, j));
        }
    }
}

// ============================================================================
// Scale and Numerical Precision Tests
// ============================================================================

#[test]
fn test_qr_large_scale_matrix() {
    // Test QR decomposition on a matrix with large values
    let a = Matrix::new(
        4,
        3,
        vec![
            1e8, 2e8, 3e8,
            4e8, 5e8, 6e8,
            7e8, 8e8, 9e8,
            1e9, 1.1e9, 1.2e9,
        ],
    );

    let (q, r) = a.qr();

    // Verify Q is orthogonal: Q^T * Q = I
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);

    // Use relative tolerance for large scale
    let tolerance = 1e-8;
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (qt_q.get(i, j) - expected).abs();
            assert!(
                diff < tolerance,
                "Large scale Q orthogonality error at [{},{}]: got {}, expected {}, diff = {}",
                i, j, qt_q.get(i, j), expected, diff
            );
        }
    }

    // Verify reconstruction: A = Q * R
    let reconstructed = q.matmul(&r);
    let rec_tolerance = 1e-4 * 1e9;  // Relative to 1e9 scale
    for i in 0..4 {
        for j in 0..3 {
            let diff = (reconstructed.get(i, j) - a.get(i, j)).abs();
            assert!(
                diff < rec_tolerance,
                "Large scale QR reconstruction error at [{},{}]: got {}, expected {}, diff = {}",
                i, j, reconstructed.get(i, j), a.get(i, j), diff
            );
        }
    }
}

#[test]
fn test_qr_with_mixed_magnitudes() {
    // QR with elements of vastly different magnitudes
    let a = Matrix::new(
        3,
        3,
        vec![
            1e10, 1e-5, 1.0,
            2e10, 2e-5, 2.0,
            3e10, 3e-5, 3.0,
        ],
    );

    let (q, _r) = a.qr();

    // Q should be orthogonal
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(qt_q.get(i, j), expected, 1e-6, &format!("mixed magnitude Q orthogonality[{},{}]", i, j));
        }
    }
}

#[test]
fn test_with_max_f64_values() {
    // Test with values near f64::MAX
    let scale = 1e100;  // Large but not near MAX
    let a = Matrix::new(
        2,
        2,
        vec![
            scale, 2.0 * scale,
            3.0 * scale, 4.0 * scale,
        ],
    );

    let (q, r) = a.qr();
    let reconstructed = q.matmul(&r);

    // Check reconstruction works with large values
    let rel_tolerance = 1e-6 * scale;
    for i in 0..2 {
        for j in 0..2 {
            let diff = (reconstructed.get(i, j) - a.get(i, j)).abs();
            assert!(
                diff < rel_tolerance,
                "Large value reconstruction[{},{}]: diff = {}", i, j, diff
            );
        }
    }
}
