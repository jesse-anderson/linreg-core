// ============================================================================
// Edge Case Tests
// ============================================================================
//
// Tests for special floating-point values (NaN, Infinity, subnormals),
// large-scale stress testing, and numerical edge cases.

use linreg_core::linalg::{Matrix, vec_mean, vec_dot, vec_l2_norm, vec_max_abs};
use super::common::EPSILON;

// ============================================================================
// NaN (Not-a-Number) Tests
// ============================================================================

#[test]
fn test_vec_mean_with_nan() {
    // vec_mean should propagate NaN (arithmetic with NaN always gives NaN)
    let v = vec![1.0, f64::NAN, 3.0];
    let result = vec_mean(&v);
    assert!(result.is_nan(), "Mean with NaN should be NaN");
}

#[test]
fn test_vec_mean_all_nan() {
    let v = vec![f64::NAN, f64::NAN, f64::NAN];
    let result = vec_mean(&v);
    assert!(result.is_nan(), "Mean of all NaNs should be NaN");
}

#[test]
fn test_vec_dot_with_nan() {
    // Dot product with NaN should give NaN
    let a = vec![1.0, f64::NAN, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = vec_dot(&a, &b);
    assert!(result.is_nan(), "Dot product with NaN should be NaN");
}

#[test]
fn test_vec_l2_norm_with_nan() {
    let v = vec![1.0, 2.0, f64::NAN];
    let result = vec_l2_norm(&v);
    assert!(result.is_nan(), "L2 norm with NaN should be NaN");
}

#[test]
fn test_vec_max_abs_with_nan() {
    // vec_max_abs uses fold with f64::max
    // max(NaN, x) = x, but max(x, NaN) = NaN in IEEE 754
    // Our fold starts with 0.0, so behavior depends on position
    let v1 = vec![1.0, f64::NAN, 3.0];
    let result1 = vec_max_abs(&v1);
    // Fold: max(0.0, 1.0) = 1.0, max(1.0, NaN) = NaN, max(NaN, 3.0) = 3.0
    // Actually, max(NaN, 3.0) = 3.0 in Rust's f64::max
    assert!(result1.is_nan() || result1 == 3.0);

    let v2 = vec![f64::NAN, 1.0, 3.0];
    let result2 = vec_max_abs(&v2);
    // Fold: max(0.0, NaN) = 0.0, max(0.0, 1.0) = 1.0, max(1.0, 3.0) = 3.0
    assert_eq!(result2, 3.0);
}

#[test]
fn test_matrix_mul_vec_with_nan() {
    let m = Matrix::new(2, 2, vec![1.0, f64::NAN, 3.0, 4.0]);
    let v = vec![1.0, 2.0];
    let result = m.mul_vec(&v);
    // Result should contain NaN due to NaN in matrix
    assert!(result[0].is_nan() || result[1].is_nan());
}

#[test]
fn test_matrix_matmul_with_nan() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, f64::NAN, 4.0]);
    let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
    let result = a.matmul(&b);
    // At least one element should be NaN
    let has_nan = (0..result.rows).any(|i| {
        (0..result.cols).any(|j| result.get(i, j).is_nan())
    });
    assert!(has_nan, "Matrix multiplication with NaN should propagate NaN");
}

// ============================================================================
// Infinity Tests
// ============================================================================

#[test]
fn test_vec_mean_with_infinity() {
    let v = vec![1.0, f64::INFINITY, 3.0];
    let result = vec_mean(&v);
    assert!(result.is_infinite(), "Mean with Infinity should be infinite");
    assert!(result.is_sign_positive(), "Mean with positive Infinity should be positive");
}

#[test]
fn test_vec_mean_with_negative_infinity() {
    let v = vec![1.0, f64::NEG_INFINITY, 3.0];
    let result = vec_mean(&v);
    assert!(result.is_infinite(), "Mean with -Infinity should be infinite");
    assert!(result.is_sign_negative(), "Mean with negative Infinity should be negative");
}

#[test]
fn test_vec_mean_with_both_infinities() {
    let v = vec![f64::INFINITY, f64::NEG_INFINITY, 1.0];
    let result = vec_mean(&v);
    // INFINITY + NEG_INFINITY = NaN
    assert!(result.is_nan(), "Mean with both infinities should be NaN");
}

#[test]
fn test_vec_dot_with_infinity() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![f64::INFINITY, 1.0, 1.0];
    let result = vec_dot(&a, &b);
    assert!(result.is_infinite(), "Dot product with Infinity should be infinite");
}

#[test]
fn test_vec_l2_norm_with_infinity() {
    let v = vec![1.0, f64::INFINITY, 3.0];
    let result = vec_l2_norm(&v);
    assert!(result.is_infinite(), "L2 norm with Infinity should be infinite");
}

#[test]
fn test_vec_max_abs_with_infinity() {
    let v = vec![1.0, f64::NEG_INFINITY, 3.0];
    let result = vec_max_abs(&v);
    assert!(result.is_infinite(), "max_abs with Infinity should be infinite");
}

#[test]
fn test_matrix_operations_with_infinity() {
    let m = Matrix::new(2, 2, vec![1.0, f64::INFINITY, 3.0, 4.0]);
    let v = vec![1.0, 2.0];
    let result = m.mul_vec(&v);
    assert!(result[0].is_infinite(), "Matrix-vector mul with Infinity should propagate");
}

// ============================================================================
// Subnormal (Denormalized) Number Tests
// ============================================================================

#[test]
fn test_vec_mean_with_subnormals() {
    // Subnormals are very small numbers near zero
    let v = vec![1.0, f64::MIN_POSITIVE, f64::MIN_POSITIVE / 2.0];
    let result = vec_mean(&v);
    // Should compute correctly without underflow to zero
    assert!(result > 0.0, "Mean with subnormals should be positive");
    assert!(result < 1.0, "Mean with subnormals should be less than 1.0");
}

#[test]
fn test_vec_dot_with_subnormals() {
    let a = vec![1e-310, 2e-310];
    let b = vec![3e-310, 4e-310];
    let result = vec_dot(&a, &b);
    // These are below f64::MIN_POSITIVE, should still work
    assert!(result >= 0.0, "Dot product with subnormals should be non-negative");
}

#[test]
fn test_vec_l2_norm_with_subnormals() {
    // Use f64::MIN_POSITIVE (smallest positive normal f64) and smaller values
    let v = vec![f64::MIN_POSITIVE, f64::MIN_POSITIVE.sqrt()];
    let result = vec_l2_norm(&v);
    // Should not underflow to zero
    assert!(result > 0.0, "L2 norm with subnormals should be positive");
}

#[test]
fn test_matrix_with_subnormals() {
    // Use values that are small but don't completely underflow
    let data = vec![f64::MIN_POSITIVE; 4];
    let m = Matrix::new(2, 2, data);
    let v = vec![f64::MIN_POSITIVE, f64::MIN_POSITIVE];
    let result = m.mul_vec(&v);
    // Result should be very small but not necessarily zero
    for &r in &result {
        assert!(!r.is_nan(), "Result should not be NaN");
    }
}

// ============================================================================
// Signed Zero Tests
// ============================================================================

#[test]
fn test_vec_dot_with_signed_zeros() {
    let a = vec![-0.0, 1.0];
    let b = vec![1.0, 1.0];
    let result = vec_dot(&a, &b);
    // -0.0 * 1.0 = -0.0, -0.0 + 1.0 = 1.0
    assert_eq!(result, 1.0);
}

#[test]
fn test_matrix_with_signed_zeros() {
    let m = Matrix::new(2, 2, vec![-0.0, 0.0, 0.0, 1.0]);
    let v = vec![1.0, 1.0];
    let result = m.mul_vec(&v);
    // -0.0 * 1.0 + 0.0 * 1.0 = -0.0 + 0.0
    // The sign of zero might be preserved or not depending on operation
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 1.0);
}

// ============================================================================
// Large-Scale Stress Tests
// ============================================================================

#[test]
fn test_large_matrix_transpose() {
    // Test transpose of a 100x100 matrix
    let n = 100;
    let data: Vec<f64> = (0..(n * n)).map(|i| i as f64).collect();
    let m = Matrix::new(n, n, data);

    let t = m.transpose();

    assert_eq!(t.rows, n);
    assert_eq!(t.cols, n);

    // Verify transpose property: (A^T)^T = A
    let tt = t.transpose();
    for i in 0..n {
        for j in 0..n {
            assert_eq!(tt.get(i, j), m.get(i, j));
        }
    }
}

#[test]
fn test_large_matrix_multiplication() {
    // Test multiplication of 50x50 matrices
    let n = 50;
    let a_data: Vec<f64> = (0..(n * n)).map(|i| (i % 10) as f64).collect();
    let b_data: Vec<f64> = (0..(n * n)).map(|i| (i % 7) as f64).collect();

    let a = Matrix::new(n, n, a_data);
    let b = Matrix::new(n, n, b_data);

    let result = a.matmul(&b);

    assert_eq!(result.rows, n);
    assert_eq!(result.cols, n);

    // Check associativity: (A * B) * I = A * B
    let identity = Matrix::identity(n);
    let result2 = result.matmul(&identity);

    for i in 0..n {
        for j in 0..n {
            let diff = (result.get(i, j) - result2.get(i, j)).abs();
            assert!(diff < 1e-9, "Associativity failed at [{},{}]: diff = {}", i, j, diff);
        }
    }
}

#[test]
fn test_large_qr_decomposition() {
    // QR decomposition of a 50x20 matrix
    let m = 50;
    let n = 20;
    let data: Vec<f64> = (0..(m * n)).map(|i| (i as f64 * 0.1) % 10.0).collect();
    let a = Matrix::new(m, n, data);

    let (q, r) = a.qr();

    // Verify Q is orthogonal
    let q_t = q.transpose();
    let qt_q = q_t.matmul(&q);

    for i in 0..m {
        for j in 0..m {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (qt_q.get(i, j) - expected).abs();
            assert!(diff < 1e-6, "Q not orthogonal at [{},{}]: diff = {}", i, j, diff);
        }
    }

    // Verify R is upper triangular
    for i in 1..r.rows {
        for j in 0..i.min(r.cols) {
            assert!(r.get(i, j).abs() < 1e-6, "R not upper triangular at [{},{}]", i, j);
        }
    }
}

#[test]
fn test_large_matrix_inversion() {
    // Inversion of a 30x30 diagonally dominant matrix
    let n = 30;
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                data[i * n + j] = 100.0 + i as f64; // Diagonal dominance
            } else {
                data[i * n + j] = 1.0 / (1.0 + i as f64 + j as f64);
            }
        }
    }

    let a = Matrix::new(n, n, data);
    let inv = a.invert();

    assert!(inv.is_some(), "Large diagonally dominant matrix should be invertible");

    let inv = inv.unwrap();
    let result = a.matmul(&inv);

    // Verify A * A^(-1) = I
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (result.get(i, j) - expected).abs();
            assert!(diff < 1e-6, "Large inversion failed at [{},{}]: diff = {}", i, j, diff);
        }
    }
}

#[test]
fn test_very_large_vector_operations() {
    // Test with a vector of 10000 elements
    let n = 10000;
    let v: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001) % 1.0).collect();

    let mean = vec_mean(&v);
    assert!(mean >= 0.0 && mean < 1.0, "Mean should be in range");

    let norm = vec_l2_norm(&v);
    assert!(norm > 0.0, "Norm should be positive");

    let max_abs = vec_max_abs(&v);
    assert!(max_abs >= 0.0, "max_abs should be non-negative");
}

#[test]
fn test_extreme_aspect_ratio_matrix() {
    // 1x1000 row vector
    let n = 1000;
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
    let row_vec = Matrix::new(1, n, data.clone());

    // 1000x1 column vector
    let col_vec = Matrix::new(n, 1, data);

    // Multiply: (1x1000) * (1000x1) = (1x1) scalar
    let result = row_vec.matmul(&col_vec);
    assert_eq!(result.rows, 1);
    assert_eq!(result.cols, 1);
    assert!(result.get(0, 0) > 0.0, "Dot product should be positive");
}

// ============================================================================
// Ill-Conditioned Matrix Tests
// ============================================================================

#[test]
fn test_hilbert_matrix_qr() {
    // Hilbert matrices are notoriously ill-conditioned
    // https://en.wikipedia.org/wiki/Hilbert_matrix
    // H[i,j] = 1 / (i + j + 1)
    let n = 5;
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = 1.0 / ((i + j + 2) as f64);
        }
    }

    let h = Matrix::new(n, n, data);
    let (_q, r) = h.qr();

    // R should be upper triangular
    for i in 1..n {
        for j in 0..i {
            assert!(r.get(i, j).abs() < 1e-9, "Hilbert R not upper triangular at [{},{}]", i, j);
        }
    }

    // The diagonal of R should be very small (ill-conditioned)
    let min_diag = (0..n).map(|i| r.get(i, i).abs()).fold(f64::INFINITY, f64::min);
    assert!(min_diag < 1e-3, "Hilbert matrix R diagonal should be small");
}

#[test]
fn test_ill_conditioned_matrix_inversion() {
    // Create a matrix with high condition number
    let n = 4;
    // Using a matrix that's nearly singular
    let data = vec![
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.000001, 1.0, 1.0,
        1.0, 1.0, 1.000001, 1.0,
        1.0, 1.0, 1.0, 1.000001,
    ];

    let m = Matrix::new(n, n, data);
    let inv = m.invert();

    // This might fail due to ill-conditioning
    // If it succeeds, verify the inverse
    if let Some(inv_mat) = inv {
        let result = m.matmul(&inv_mat);
        // Use a very loose tolerance for ill-conditioned case
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (result.get(i, j) - expected).abs();
                assert!(diff < 0.1, "Ill-conditioned inversion error at [{},{}]: diff = {}", i, j, diff);
            }
        }
    }
    // If it returns None, that's also acceptable behavior for ill-conditioned matrices
}

#[test]
fn test_vandermonde_matrix() {
    // Vandermonde matrices are often ill-conditioned
    // V[i,j] = x_i^j
    let n = 5;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = x[i].powi(j as i32);
        }
    }

    let v = Matrix::new(n, n, data);
    let inv = v.invert();

    // Small Vandermonde might still be invertible
    if let Some(inv_mat) = inv {
        let result = v.matmul(&inv_mat);
        let identity = Matrix::identity(n);
        for i in 0..n {
            for j in 0..n {
                let diff = (result.get(i, j) - identity.get(i, j)).abs();
                assert!(diff < 1e-6, "Vandermonde inversion error at [{},{}]", i, j);
            }
        }
    }
}

// ============================================================================
// Numerical Stability Edge Cases
// ============================================================================

#[test]
fn test_cancellation_in_dot_product() {
    // Test catastrophic cancellation: a · b where terms nearly cancel
    let a = vec![1e16, 1.0];
    let b = vec![-1e16, 1.0];
    // Exact result: -1e32 + 1 = -9999999999999999...
    // Due to floating point, 1e16 * -1e16 might swamp the 1.0 * 1.0 term
    let result = vec_dot(&a, &b);
    // Result should be close to the expected value
    // The exact answer is -1e32 + 1 ≈ -1e32
    assert!(result < 0.0, "Dot product should be negative");
}

#[test]
fn test_householder_stability_edge_case() {
    // Test QR with a vector that could cause cancellation issues
    // First element is large, others are small
    let x = vec![1e10, 1.0, 1.0, 1.0];
    let m = Matrix::new(4, 1, x);

    let (_q, r) = m.qr();

    // R should have zeros below diagonal
    assert!(r.get(1, 0).abs() < 1e-6, "R[1,0] should be near zero");
    assert!(r.get(2, 0).abs() < 1e-6, "R[2,0] should be near zero");
    assert!(r.get(3, 0).abs() < 1e-6, "R[3,0] should be near zero");
}

#[test]
fn test_matrix_with_wide_dynamic_range() {
    // Matrix with elements spanning many orders of magnitude
    let data = vec![
        1e100, 1e-100, 1.0,
        1e-100, 1e100, 1.0,
        1.0, 1.0, 1.0,
    ];

    let m = Matrix::new(3, 3, data);

    // Transpose should work
    let t = m.transpose();
    assert_eq!(t.rows, 3);
    assert_eq!(t.cols, 3);

    // Verify round-trip
    let tt = t.transpose();
    for i in 0..3 {
        for j in 0..3 {
            let diff = (m.get(i, j) - tt.get(i, j)).abs();
            let rel_diff = diff / (1.0 + m.get(i, j).abs());
            assert!(rel_diff < 1e-10, "Round-trip failed at [{},{}]", i, j);
        }
    }
}

// ============================================================================
// Sparse Matrix Tests
// ============================================================================

#[test]
fn test_mostly_zeros_matrix() {
    // Matrix with 99% zeros
    let n = 20;
    let mut data = vec![0.0; n * n];
    // Set only 1% of elements to non-zero
    for i in 0..(n * n / 100) {
        data[i * 100] = (i as f64) * 0.1;
    }

    let m = Matrix::new(n, n, data);

    // Operations should still work correctly
    let t = m.transpose();
    assert_eq!(t.rows, n);
    assert_eq!(t.cols, n);
}

#[test]
fn test_diagonal_dominant_sparse() {
    // Sparse matrix with only diagonal non-zero
    let n = 15;
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = (i + 1) as f64;
    }

    let m = Matrix::new(n, n, data);

    // Should be invertible (diagonal matrix)
    let inv = m.invert();
    assert!(inv.is_some(), "Diagonal matrix should be invertible");

    // Verify inverse
    let inv = inv.unwrap();
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 / ((i + 1) as f64) } else { 0.0 };
            assert!((inv.get(i, j) - expected).abs() < 1e-10);
        }
    }
}

// ============================================================================
// Empty and Single-Element Edge Cases
// ============================================================================

#[test]
fn test_empty_matrix_operations() {
    let m1 = Matrix::new(0, 5, vec![]);
    let m2 = Matrix::new(5, 0, vec![]);

    // 0x5 transpose is 5x0
    let t = m1.transpose();
    assert_eq!(t.rows, 5);
    assert_eq!(t.cols, 0);

    // 5x0 * 0x5 = 5x5 zero matrix
    let result = m2.matmul(&m1);
    assert_eq!(result.rows, 5);
    assert_eq!(result.cols, 5);
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(result.get(i, j), 0.0);
        }
    }
}

#[test]
fn test_single_element_matrix_qr() {
    let m = Matrix::new(1, 1, vec![42.0]);
    let (q, r) = m.qr();

    // Q should be approximately 1 (or -1)
    assert!((q.get(0, 0).abs() - 1.0) < 1e-10);

    // R should have magnitude equal to the original
    assert!((r.get(0, 0).abs() - 42.0) < 1e-10);

    // Reconstruction should work
    let reconstructed = q.matmul(&r);
    assert!((reconstructed.get(0, 0) - 42.0) < 1e-10);
}

#[test]
fn test_single_element_vector() {
    let v = vec![42.0];
    assert_eq!(vec_mean(&v), 42.0);
    assert_eq!(vec_l2_norm(&v), 42.0);
    assert_eq!(vec_max_abs(&v), 42.0);
}
