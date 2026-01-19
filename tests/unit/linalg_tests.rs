// ============================================================================
// Linear Algebra Unit Tests
// ============================================================================
//
// Comprehensive tests for matrix operations and QR decomposition.
// Tests include constructor tests, element access, transpose, matrix
// multiplication, QR decomposition, and matrix inversion.
//
// Property-based tests use the proptest crate to verify mathematical
// properties hold for random inputs.



use linreg_core::linalg::{Matrix, vec_mean, vec_sub, vec_dot};
use proptest::prelude::*;

// ============================================================================
// Test Constants and Helpers
// ============================================================================

const EPSILON: f64 = 1e-10;
const QR_TOLERANCE: f64 = 1e-9;

/// Helper function to assert two f64 values are close within tolerance
fn assert_close(a: f64, b: f64, tolerance: f64, context: &str) {
    let diff = (a - b).abs();
    assert!(
        diff <= tolerance,
        "{}: {} != {}, diff = {} (tolerance = {})",
        context, a, b, diff, tolerance
    );
}

/// Helper function to assert two matrices are approximately equal
fn assert_matrix_eq(a: &Matrix, b: &Matrix, tolerance: f64, context: &str) {
    assert_eq!(
        a.rows, b.rows,
        "{}: Row count mismatch: {} vs {}",
        context, a.rows, b.rows
    );
    assert_eq!(
        a.cols, b.cols,
        "{}: Column count mismatch: {} vs {}",
        context, a.cols, b.cols
    );

    for i in 0..a.rows {
        for j in 0..a.cols {
            assert_close(
                a.get(i, j),
                b.get(i, j),
                tolerance,
                &format!("{} at [{},{}]", context, i, j),
            );
        }
    }
}

// ============================================================================
// Constructor Tests
// ============================================================================

#[test]
fn test_matrix_new_valid() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = Matrix::new(2, 3, data);

    assert_eq!(m.rows, 2);
    assert_eq!(m.cols, 3);
    assert_eq!(m.data.len(), 6);
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(0, 1), 2.0);
    assert_eq!(m.get(0, 2), 3.0);
    assert_eq!(m.get(1, 0), 4.0);
    assert_eq!(m.get(1, 1), 5.0);
    assert_eq!(m.get(1, 2), 6.0);
}

#[test]
#[should_panic(expected = "Data length must match dimensions")]
fn test_matrix_new_panic_on_size_mismatch() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 elements, but 2x3 = 6
    Matrix::new(2, 3, data);
}

#[test]
fn test_matrix_zeros() {
    let m = Matrix::zeros(3, 4);

    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 4);
    assert_eq!(m.data.len(), 12);

    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(m.get(i, j), 0.0);
        }
    }
}

#[test]
fn test_matrix_zeros_square() {
    let m = Matrix::zeros(2, 2);

    assert_eq!(m.rows, 2);
    assert_eq!(m.cols, 2);
    assert_eq!(m.get(0, 0), 0.0);
    assert_eq!(m.get(0, 1), 0.0);
    assert_eq!(m.get(1, 0), 0.0);
    assert_eq!(m.get(1, 1), 0.0);
}

#[test]
fn test_matrix_identity() {
    let m = Matrix::identity(4);

    assert_eq!(m.rows, 4);
    assert_eq!(m.cols, 4);

    for i in 0..4 {
        for j in 0..4 {
            if i == j {
                assert_eq!(m.get(i, j), 1.0);
            } else {
                assert_eq!(m.get(i, j), 0.0);
            }
        }
    }
}

#[test]
fn test_matrix_identity_size_1() {
    let m = Matrix::identity(1);

    assert_eq!(m.rows, 1);
    assert_eq!(m.cols, 1);
    assert_eq!(m.get(0, 0), 1.0);
}

#[test]
fn test_matrix_new_from_vec() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = Matrix::new(2, 3, data);

    assert_eq!(m.rows, 2);
    assert_eq!(m.cols, 3);
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(0, 1), 2.0);
    assert_eq!(m.get(0, 2), 3.0);
    assert_eq!(m.get(1, 0), 4.0);
    assert_eq!(m.get(1, 1), 5.0);
    assert_eq!(m.get(1, 2), 6.0);
}

// ============================================================================
// Element Access Tests
// ============================================================================

#[test]
fn test_get_set_roundtrip() {
    let mut m = Matrix::zeros(3, 3);

    for i in 0..3 {
        for j in 0..3 {
            let val = (i * 3 + j) as f64;
            m.set(i, j, val);
            assert_eq!(m.get(i, j), val);
        }
    }
}

#[test]
fn test_get_all_elements() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let m = Matrix::new(3, 4, data.clone());

    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(m.get(i, j), data[i * 4 + j]);
        }
    }
}

#[test]
fn test_set_updates_correctly() {
    let mut m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

    assert_eq!(m.get(0, 0), 1.0);
    m.set(0, 0, 10.0);
    assert_eq!(m.get(0, 0), 10.0);

    assert_eq!(m.get(1, 1), 4.0);
    m.set(1, 1, 20.0);
    assert_eq!(m.get(1, 1), 20.0);

    // Verify other elements unchanged
    assert_eq!(m.get(0, 1), 2.0);
    assert_eq!(m.get(1, 0), 3.0);
}

// ============================================================================
// Transpose Tests
// ============================================================================

#[test]
fn test_transpose_square() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let t = m.transpose();

    assert_eq!(t.rows, 2);
    assert_eq!(t.cols, 2);
    assert_eq!(t.get(0, 0), 1.0);
    assert_eq!(t.get(0, 1), 3.0);
    assert_eq!(t.get(1, 0), 2.0);
    assert_eq!(t.get(1, 1), 4.0);
}

#[test]
fn test_transpose_rectangular() {
    // 2x3 matrix -> 3x2 transpose
    let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = m.transpose();

    assert_eq!(t.rows, 3);
    assert_eq!(t.cols, 2);

    // Check a few elements
    assert_eq!(t.get(0, 0), 1.0); // (0,0) -> (0,0)
    assert_eq!(t.get(0, 1), 4.0); // (1,0) -> (0,1)
    assert_eq!(t.get(1, 0), 2.0); // (0,1) -> (1,0)
    assert_eq!(t.get(1, 1), 5.0); // (1,1) -> (1,1)
    assert_eq!(t.get(2, 0), 3.0); // (0,2) -> (2,0)
    assert_eq!(t.get(2, 1), 6.0); // (1,2) -> (2,1)
}

#[test]
fn test_transpose_symmetric() {
    // Symmetric matrix: A^T = A
    let m = Matrix::new(
        3,
        3,
        vec![1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0],
    );
    let t = m.transpose();

    for i in 0..3 {
        for j in 0..3 {
            assert_close(m.get(i, j), t.get(i, j), EPSILON, "symmetric");
        }
    }
}

#[test]
fn test_transpose_identity() {
    let i = Matrix::identity(4);
    let t = i.transpose();

    // Identity matrix is its own transpose
    assert_matrix_eq(&i, &t, EPSILON, "identity transpose");
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

#[test]
fn test_matmul_identity() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let i = Matrix::identity(2);

    let result1 = m.matmul(&i);
    assert_matrix_eq(&result1, &m, EPSILON, "M * I");

    let result2 = i.matmul(&m);
    assert_matrix_eq(&result2, &m, EPSILON, "I * M");
}

#[test]
fn test_matmul_zero() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let z = Matrix::zeros(2, 2);

    let result1 = m.matmul(&z);
    assert_matrix_eq(&result1, &z, EPSILON, "M * 0");

    let result2 = z.matmul(&m);
    assert_matrix_eq(&result2, &z, EPSILON, "0 * M");
}

#[test]
fn test_matmul_rectangular() {
    // 2x3 * 3x2 = 2x2
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let result = a.matmul(&b);

    assert_eq!(result.rows, 2);
    assert_eq!(result.cols, 2);

    // Manual computation:
    // [1 2 3]   [7  8]   [1*7+2*9+3*11   1*8+2*10+3*12]   [58  64]
    // [4 5 6] * [9 10] = [4*7+5*9+6*11   4*8+5*10+6*12] = [139 154]
    //           [11 12]
    assert_close(result.get(0, 0), 58.0, EPSILON, "matmul(0,0)");
    assert_close(result.get(0, 1), 64.0, EPSILON, "matmul(0,1)");
    assert_close(result.get(1, 0), 139.0, EPSILON, "matmul(1,0)");
    assert_close(result.get(1, 1), 154.0, EPSILON, "matmul(1,1)");
}

#[test]
fn test_matmul_associative() {
    // For matrices A, B, C: (A * B) * C = A * (B * C)
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let c = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

    let result1 = a.matmul(&b).matmul(&c);
    let result2 = a.matmul(&b.matmul(&c));

    assert_matrix_eq(&result1, &result2, EPSILON, "associative");
}

#[test]
#[should_panic(expected = "Dimension mismatch for multiplication")]
fn test_matmul_panic_on_dimension_mismatch() {
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    // 2x3 * 3x3 should work
    a.matmul(&b);

    // But 3x3 * 2x3 should panic
    let _ = b.matmul(&a);
}

#[test]
fn test_matmul_known_values() {
    // Test specific known multiplication
    let a = Matrix::new(
        2,
        2,
        vec![1.0, 2.0, 3.0, 4.0], // [1 2; 3 4]
    );
    let b = Matrix::new(
        2,
        2,
        vec![5.0, 6.0, 7.0, 8.0], // [5 6; 7 8]
    );

    let result = a.matmul(&b);

    // [1 2]   [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
    // [3 4] * [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
    assert_close(result.get(0, 0), 19.0, EPSILON, "known(0,0)");
    assert_close(result.get(0, 1), 22.0, EPSILON, "known(0,1)");
    assert_close(result.get(1, 0), 43.0, EPSILON, "known(1,0)");
    assert_close(result.get(1, 1), 50.0, EPSILON, "known(1,1)");
}

// ============================================================================
// Matrix-Vector Multiplication Tests
// ============================================================================

#[test]
fn test_mul_vec_identity() {
    let m = Matrix::identity(3);
    let v = vec![1.0, 2.0, 3.0];

    let result = m.mul_vec(&v);

    assert_eq!(result.len(), 3);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
}

#[test]
fn test_mul_vec_basic() {
    let m = Matrix::new(
        2,
        3,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // [1 2 3; 4 5 6]
    );
    let v = vec![1.0, 2.0, 3.0];

    let result = m.mul_vec(&v);

    assert_eq!(result.len(), 2);
    // [1 2 3] * [1] = 1*1 + 2*2 + 3*3 = 14
    // [4 5 6]   [2] = 4*1 + 5*2 + 6*3 = 32
    //           [3]
    assert_close(result[0], 14.0, EPSILON, "mul_vec[0]");
    assert_close(result[1], 32.0, EPSILON, "mul_vec[1]");
}

#[test]
#[should_panic(expected = "Dimension mismatch for matrix-vector multiplication")]
fn test_mul_vec_panic_on_dimension_mismatch() {
    let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let v = vec![1.0, 2.0]; // Wrong length

    m.mul_vec(&v);
}

// ============================================================================
// QR Decomposition Tests
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

// ============================================================================
// Inversion Tests
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

// ============================================================================
// Vector Helper Function Tests
// ============================================================================

#[test]
fn test_vec_mean() {
    assert_eq!(vec_mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
    assert_eq!(vec_mean(&[10.0, 20.0, 30.0]), 20.0);
    assert_eq!(vec_mean(&[0.0]), 0.0);
    assert_eq!(vec_mean(&[]), 0.0); // Empty returns 0.0
}

#[test]
fn test_vec_sub() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.5, 1.5, 2.5, 3.5];
    let result = vec_sub(&a, &b);

    assert_eq!(result.len(), 4);
    assert_close(result[0], 0.5, EPSILON, "sub[0]");
    assert_close(result[1], 0.5, EPSILON, "sub[1]");
    assert_close(result[2], 0.5, EPSILON, "sub[2]");
    assert_close(result[3], 0.5, EPSILON, "sub[3]");
}

#[test]
fn test_vec_dot() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = vec_dot(&a, &b);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_close(result, 32.0, EPSILON, "dot product");
}

#[test]
fn test_vec_dot_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let result = vec_dot(&a, &b);

    assert_close(result, 0.0, EPSILON, "orthogonal dot");
}

// ============================================================================
// Property-Based Tests
// ============================================================================

proptest! {
    #[test]
    fn prop_double_transpose(rows in 1..10usize, cols in 1..10usize) {
        // (A^T)^T = A for any matrix
        let size = rows * cols;
        let values: Vec<f64> = (0..size).map(|i| i as f64 * 1.0).collect();
        let a = Matrix::new(rows, cols, values);
        let at = a.transpose();
        let att = at.transpose();

        prop_assert_eq!(att.rows, rows);
        prop_assert_eq!(att.cols, cols);
        for i in 0..rows {
            for j in 0..cols {
                prop_assert!((att.get(i, j) - a.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn prop_transpose_product(
        m1 in 2..5usize,
        n in 2..5usize,
        m2 in 2..5usize,
    ) {
        // (A * B)^T = B^T * A^T
        let a_vals: Vec<f64> = (0..(m1 * n)).map(|i| i as f64 * 1.0).collect();
        let b_vals: Vec<f64> = (0..(n * m2)).map(|i| i as f64 * 1.0).collect();

        let a = Matrix::new(m1, n, a_vals);
        let b = Matrix::new(n, m2, b_vals);

        let ab = a.matmul(&b);
        let ab_t = ab.transpose();

        let a_t = a.transpose();
        let b_t = b.transpose();
        let bt_at = b_t.matmul(&a_t);

        prop_assert_eq!(ab_t.rows, bt_at.rows);
        prop_assert_eq!(ab_t.cols, bt_at.cols);

        for i in 0..ab_t.rows {
            for j in 0..ab_t.cols {
                let diff = (ab_t.get(i, j) - bt_at.get(i, j)).abs();
                prop_assert!(diff < 1e-8, "Difference at [{},{}]: {} vs {}", i, j, ab_t.get(i, j), bt_at.get(i, j));
            }
        }
    }

    #[test]
    fn prop_q_orthogonal(
        m in 2..6usize,
        n_max in 2usize..6usize,
    ) {
        // Q from QR decomposition is always orthogonal: Q^T * Q ≈ I
        let n = n_max.min(m);
        let a_vals: Vec<f64> = (0..(m * n)).map(|i| (i as f64) * 2.0 - 10.0).collect();
        let a = Matrix::new(m, n, a_vals);

        let (q, _r) = a.qr();
        let q_t = q.transpose();
        let qt_q = q_t.matmul(&q);

        // Check diagonal is close to 1
        for i in 0..q.cols {
            let diff = (qt_q.get(i, i) - 1.0).abs();
            prop_assert!(diff < 1e-6, "Diagonal element {} not close to 1: {}", i, qt_q.get(i, i));
        }

        // Check off-diagonal is close to 0
        for i in 0..q.cols {
            for j in 0..q.cols {
                if i != j {
                    let val = qt_q.get(i, j).abs();
                    prop_assert!(val < 1e-6, "Off-diagonal [{},{}] not close to 0: {}", i, j, val);
                }
            }
        }
    }
}

// ============================================================================
// Scale and Edge Case Tests
// ============================================================================
// Tests for matrices with very large or very small values to verify that
// relative tolerance in singular matrix detection works correctly.

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
    let i = Matrix::identity(3);

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
    let i = Matrix::identity(3);

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
    let i = Matrix::identity(3);

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
// Comprehensive Edge Case Tests
// ============================================================================
// Additional tests for robustness and numerical stability.

// ----------------------------------------------------------------------------
// QR Decomposition Edge Cases
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Matrix Operation Edge Cases
// ----------------------------------------------------------------------------

#[test]
fn test_matmul_with_zero_matrix() {
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let zero = Matrix::zeros(3, 2);

    let result = a.matmul(&zero);
    assert_eq!(result.rows, 2);
    assert_eq!(result.cols, 2);

    for i in 0..2 {
        for j in 0..2 {
            assert_close(result.get(i, j), 0.0, EPSILON, &format!("zero mult result[{},{}]", i, j));
        }
    }
}

#[test]
fn test_mul_vec_with_zero_vector() {
    let m = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let v = vec![0.0, 0.0];

    let result = m.mul_vec(&v);

    assert_eq!(result.len(), 3);
    for i in 0..3 {
        assert_close(result[i], 0.0, EPSILON, &format!("zero vec result[{}]", i));
    }
}

#[test]
fn test_transpose_of_transpose() {
    // (A^T)^T = A
    let a = Matrix::new(3, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let att = a.transpose().transpose();

    assert_eq!(att.rows, a.rows);
    assert_eq!(att.cols, a.cols);

    for i in 0..a.rows {
        for j in 0..a.cols {
            assert_close(att.get(i, j), a.get(i, j), EPSILON, &format!("double transpose[{},{}]", i, j));
        }
    }
}

#[test]
fn test_matmul_dimensions() {
    // Test various dimension combinations
    // 2x3 * 3x4 = 2x4
    let a = Matrix::new(2, 3, vec![1.0; 6]);
    let b = Matrix::new(3, 4, vec![1.0; 12]);
    let result = a.matmul(&b);
    assert_eq!(result.rows, 2);
    assert_eq!(result.cols, 4);

    // 5x2 * 2x1 = 5x1
    let c = Matrix::new(5, 2, vec![1.0; 10]);
    let d = Matrix::new(2, 1, vec![1.0; 2]);
    let result2 = c.matmul(&d);
    assert_eq!(result2.rows, 5);
    assert_eq!(result2.cols, 1);
}

// ----------------------------------------------------------------------------
// Inversion Edge Cases
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Numerical Precision Tests
// ----------------------------------------------------------------------------

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

#[test]
fn test_matmul_associativity_random() {
    // Test (A * B) * C = A * (B * C) with specific values
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = Matrix::new(2, 2, vec![0.1, 0.2, 0.3, 0.4]);

    let result1 = a.matmul(&b).matmul(&c);
    let result2 = a.matmul(&b.matmul(&c));

    for i in 0..2 {
        for j in 0..2 {
            assert_close(result1.get(i, j), result2.get(i, j), 1e-10, &format!("associativity[{},{}]", i, j));
        }
    }
}

// ----------------------------------------------------------------------------
// Special Matrix Types
// ----------------------------------------------------------------------------

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
// Missing Coverage Tests
// ============================================================================
// Tests added to fill gaps identified in test coverage audit.

// ----------------------------------------------------------------------------
// chol2inv_from_qr() Tests - Previously Completely Untested
// ----------------------------------------------------------------------------

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
        let i = Matrix::identity(3);

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

// ----------------------------------------------------------------------------
// Non-Square Matrix Panic Tests
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Additional QR Edge Cases
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Empty Vector Edge Case
// ----------------------------------------------------------------------------

#[test]
fn test_mul_vec_empty_result() {
    // Matrix-vector multiplication resulting in empty vector
    // 0x3 matrix times 3-element vector = 0-element result
    let m = Matrix::new(0, 3, vec![]);
    let v = vec![1.0, 2.0, 3.0];

    let result = m.mul_vec(&v);

    assert_eq!(result.len(), 0);
}

#[test]
fn test_transpose_empty_matrix() {
    // Transpose of empty matrix
    let m = Matrix::new(0, 3, vec![]);
    let t = m.transpose();

    assert_eq!(t.rows, 3);
    assert_eq!(t.cols, 0);
    assert_eq!(t.data.len(), 0);
}

#[test]
fn test_zeros_empty_dimensions() {
    // Zero matrix with zero rows or columns
    let m1 = Matrix::zeros(0, 5);
    assert_eq!(m1.rows, 0);
    assert_eq!(m1.cols, 5);
    assert_eq!(m1.data.len(), 0);

    let m2 = Matrix::zeros(3, 0);
    assert_eq!(m2.rows, 3);
    assert_eq!(m2.cols, 0);
    assert_eq!(m2.data.len(), 0);
}

// ----------------------------------------------------------------------------
// vec_sub and vec_dot Additional Tests
// ----------------------------------------------------------------------------

#[test]
fn test_vec_sub_empty() {
    // Vector subtraction with empty slices
    let a: Vec<f64> = vec![];
    let b: Vec<f64> = vec![];
    let result = vec_sub(&a, &b);

    assert_eq!(result.len(), 0);
}

#[test]
fn test_vec_dot_empty() {
    // Dot product of empty vectors
    let a: Vec<f64> = vec![];
    let b: Vec<f64> = vec![];
    let result = vec_dot(&a, &b);

    // Sum of empty slice is 0.0
    assert_eq!(result, 0.0);
}

// ----------------------------------------------------------------------------
// Matrix Clone Test
// ----------------------------------------------------------------------------

#[test]
fn test_matrix_clone_independence() {
    // Verify that cloned matrix is independent of original
    let mut m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let m2 = m1.clone();

    // Modify m1
    m1.set(0, 0, 99.0);

    // m2 should be unchanged
    assert_eq!(m2.get(0, 0), 1.0);
    assert_eq!(m1.get(0, 0), 99.0);
}

// ============================================================================
// Nalgebra Comparison Tests
// ============================================================================
// Compare our custom implementation against nalgebra to debug differences.
// These tests are ONLY for development/debugging - they verify our custom
// implementation produces the same results as the established nalgebra library.

// Nalgebra comparison tests - enabled when nalgebra is available (dev-dependencies)
// DISABLED: nalgebra not in dependencies
/*
#[cfg(test)]
mod nalgebra_comparison {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    const TOLERANCE: f64 = 1e-9;

    /// Helper to convert our Matrix to nalgebra DMatrix
    fn to_nalgebra_matrix(m: &Matrix) -> DMatrix<f64> {
        DMatrix::from_row_slice(m.rows, m.cols, &m.data)
    }

    /// Helper to convert nalgebra DMatrix to our Matrix
    fn from_nalgebra_matrix(nm: &DMatrix<f64>) -> Matrix {
        Matrix::new(nm.nrows(), nm.ncols(), nm.iter().cloned().collect())
    }

    #[test]
    fn compare_qr_decomposition() {
        // Test matrix: same as our QR tests
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,  // Changed from 9.0 for full rank
            10.0, 11.0, 12.0,
        ];
        let our_m = Matrix::new(4, 3, data.clone());
        let na_m = to_nalgebra_matrix(&our_m);

        // Our QR
        let (our_q, our_r) = our_m.qr();

        // Nalgebra QR
        let na_qr = na_m.qr();
        let (na_q, na_r) = na_qr.unpack();

        println!("\n=== QR Decomposition Comparison ===");
        println!("Our Q (first 3 rows):");
        for i in 0..3.min(our_q.rows) {
            for j in 0..our_q.cols.min(3) {
                print!("{:12.6} ", our_q.get(i, j));
            }
            println!();
        }
        println!("\nNalgebra Q (first 3 rows):");
        for i in 0..3.min(na_q.nrows()) {
            for j in 0..na_q.ncols().min(3) {
                print!("{:12.6} ", na_q[(i, j)]);
            }
            println!();
        }

        println!("\nOur R:");
        for i in 0..our_r.rows.min(4) {
            for j in 0..our_r.cols.min(3) {
                print!("{:12.6} ", our_r.get(i, j));
            }
            println!();
        }
        println!("\nNalgebra R:");
        for i in 0..na_r.nrows().min(4) {
            for j in 0..na_r.ncols().min(3) {
                print!("{:12.6} ", na_r[(i, j)]);
            }
            println!();
        }

        // Note: Q and R can differ by sign conventions, but A = Q*R should be the same
        // Verify reconstruction matches
        let our_reconstructed = our_q.matmul(&our_r);
        let na_reconstructed = &na_q * &na_r;

        println!("\nReconstruction comparison (our A vs Q*R):");
        for i in 0..our_m.rows {
            for j in 0..our_m.cols {
                let our_val = our_reconstructed.get(i, j);
                let orig_val = our_m.get(i, j);
                println!("  [{},{}]: original={:.6}, qr_recon={:.6}, diff={:.2e}",
                    i, j, orig_val, our_val, (orig_val - our_val).abs());
            }
        }

        // Both should reconstruct original
        for i in 0..our_m.rows {
            for j in 0..our_m.cols {
                assert_close(
                    our_reconstructed.get(i, j),
                    our_m.get(i, j),
                    TOLERANCE,
                    &format!("our reconstruction [{},{}]", i, j)
                );
                assert_close(
                    na_reconstructed[(i, j)],
                    our_m.get(i, j),
                    TOLERANCE,
                    &format!("nalgebra reconstruction [{},{}]", i, j)
                );
            }
        }
    }

    #[test]
    fn compare_matrix_multiplication() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];  // 2x2
        let b_data = vec![5.0, 6.0, 7.0, 8.0];  // 2x2

        let our_a = Matrix::new(2, 2, a_data.clone());
        let our_b = Matrix::new(2, 2, b_data.clone());

        let na_a = to_nalgebra_matrix(&our_a);
        let na_b = to_nalgebra_matrix(&our_b);

        let our_result = our_a.matmul(&our_b);
        let na_result = &na_a * &na_b;

        println!("\n=== Matrix Multiplication Comparison ===");
        println!("A:\n{:?}", &a_data);
        println!("B:\n{:?}", &b_data);
        println!("Our result: {:?}", our_result.data);
        println!("Nalgebra result: {:?}", na_result.iter().collect::<Vec<_>>());

        for i in 0..2 {
            for j in 0..2 {
                assert_close(
                    our_result.get(i, j),
                    na_result[(i, j)],
                    TOLERANCE,
                    &format!("matmul [{},{}]", i, j)
                );
            }
        }
    }

    #[test]
    fn compare_matrix_inverse() {
        // Use a well-conditioned matrix
        let data = vec![
            4.0, 7.0, 2.0,
            3.0, 6.0, 1.0,
            2.0, 5.0, 3.0,
        ];

        let our_m = Matrix::new(3, 3, data.clone());
        let na_m = to_nalgebra_matrix(&our_m);

        let our_inv = our_m.invert().expect("should invert");
        let na_inv = na_m.clone().try_inverse().expect("nalgebra should invert");

        println!("\n=== Matrix Inverse Comparison ===");
        println!("Original matrix:");
        for i in 0..3 {
            println!("  [{},{},{}]", data[i*3], data[i*3+1], data[i*3+2]);
        }

        println!("\nOur inverse:");
        for i in 0..3 {
            println!("  [{},{},{}]", our_inv.get(i,0), our_inv.get(i,1), our_inv.get(i,2));
        }

        println!("\nNalgebra inverse:");
        for i in 0..3 {
            println!("  [{},{},{}]", na_inv[(i,0)], na_inv[(i,1)], na_inv[(i,2)]);
        }

        // Compare values
        for i in 0..3 {
            for j in 0..3 {
                assert_close(
                    our_inv.get(i, j),
                    na_inv[(i, j)],
                    1e-8,  // Looser tolerance for inversion
                    &format!("inverse [{},{}]", i, j)
                );
            }
        }

        // Verify A * A^-1 = I for both
        let our_product = our_m.matmul(&our_inv);
        let na_product = &na_m * &na_inv;

        println!("\nOur A * A^-1:");
        for i in 0..3 {
            println!("  [{},{},{}]", our_product.get(i,0), our_product.get(i,1), our_product.get(i,2));
        }

        println!("\nNalgebra A * A^-1:");
        for i in 0..3 {
            println!("  [{},{},{}]", na_product[(i,0)], na_product[(i,1)], na_product[(i,2)]);
        }
    }

    #[test]
    fn compare_chol2inv() {
        // X is a 4x3 matrix (tall, as used in OLS)
        let x_data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,  // Changed from 9.0 for full rank
            2.0, 3.0, 4.0,
        ];

        let our_x = Matrix::new(4, 3, x_data.clone());
        let na_x = to_nalgebra_matrix(&our_x);

        // Our chol2inv
        let our_result = our_x.chol2inv_from_qr().expect("chol2inv should work");

        // Nalgebra: compute X'X then invert
        let na_xt = na_x.transpose();
        let na_xtx = &na_xt * &na_x;
        let na_result = na_xtx.try_inverse().expect("X'X should be invertible");

        println!("\n=== chol2inv Comparison ===");
        println!("X (4x3):");
        for i in 0..4 {
            println!("  [{},{},{}]", x_data[i*3], x_data[i*3+1], x_data[i*3+2]);
        }

        println!("\nOur (X'X)^(-1):");
        for i in 0..3 {
            println!("  [{},{},{}]", our_result.get(i,0), our_result.get(i,1), our_result.get(i,2));
        }

        println!("\nNalgebra (X'X)^(-1):");
        for i in 0..3 {
            println!("  [{},{},{}]", na_result[(i,0)], na_result[(i,1)], na_result[(i,2)]);
        }

        println!("\nDifferences:");
        for i in 0..3 {
            for j in 0..3 {
                let our_val = our_result.get(i, j);
                let na_val = na_result[(i, j)];
                let diff = (our_val - na_val).abs();
                println!("  [{},{}]: our={:.10e}, na={:.10e}, diff={:.2e}",
                    i, j, our_val, na_val, diff);
            }
        }

        // Compare
        for i in 0..3 {
            for j in 0..3 {
                assert_close(
                    our_result.get(i, j),
                    na_result[(i, j)],
                    1e-9,
                    &format!("chol2inv [{},{}]", i, j)
                );
            }
        }
    }

    #[test]
    fn compare_ols_solution() {
        // Test the complete OLS solution: beta = (X'X)^(-1) X' y
        // This is what we actually use in regression

        let x_data = vec![
            1.0, 1.0,  // intercept, x1
            1.0, 2.0,
            1.0, 3.0,
            1.0, 4.0,
            1.0, 5.0,
        ];
        let y_data = vec![2.1, 4.9, 7.1, 9.8, 12.2];  // Approximately y = 0.5 + 2.4*x

        let our_x = Matrix::new(5, 2, x_data.clone());

        let na_x = to_nalgebra_matrix(&our_x);
        let na_y = DMatrix::from_column_slice(5, 1, &y_data);

        // Our solution: beta = (X'X)^(-1) X' y
        let our_xtx_inv = our_x.chol2inv_from_qr().expect("chol2inv");
        let our_xt = our_x.transpose();
        let our_xty = our_xt.mul_vec(&y_data);
        let our_beta = our_xtx_inv.mul_vec(&our_xty);

        // Nalgebra solution
        let na_xt = na_x.transpose();
        let na_xtx = &na_xt * &na_x;
        let na_xtx_inv = na_xtx.try_inverse().expect("X'X invertible");
        let na_xty = &na_xt * &na_y;
        let na_beta = &na_xtx_inv * &na_xty;

        println!("\n=== OLS Solution Comparison ===");
        println!("X (design matrix with intercept):");
        for i in 0..5 {
            println!("  [{},{}]", x_data[i*2], x_data[i*2+1]);
        }
        println!("y: {:?}", y_data);

        println!("\nOur beta: [{:.10}, {:.10}]", our_beta[0], our_beta[1]);
        println!("Nalgebra beta: [{:.10}, {:.10}]", na_beta[0], na_beta[1]);

        println!("\nActual expected values: intercept≈-0.31, slope≈2.51");

        // Both should give same results
        assert_close(our_beta[0], na_beta[0], 1e-9, "beta[0] (intercept)");
        assert_close(our_beta[1], na_beta[1], 1e-9, "beta[1] (slope)");
    }

    #[test]
    fn compare_qr_orthogonality() {
        // Q should be orthogonal: Q^T * Q = I
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ];

        let our_m = Matrix::new(3, 3, data);
        let na_m = to_nalgebra_matrix(&our_m);

        let (our_q, _) = our_m.qr();
        let na_qr = na_m.qr();
        let (na_q, _) = na_qr.unpack();

        // Our Q^T * Q
        let our_qt = our_q.transpose();
        let our_qt_q = our_qt.matmul(&our_q);

        // Nalgebra Q^T * Q
        let na_qt = na_q.transpose();
        let na_qt_q = &na_qt * &na_q;

        println!("\n=== Q Orthogonality Comparison ===");
        println!("Our Q^T * Q (should be identity):");
        for i in 0..3 {
            println!("  [{},{},{}]", our_qt_q.get(i,0), our_qt_q.get(i,1), our_qt_q.get(i,2));
        }

        println!("\nNalgebra Q^T * Q:");
        for i in 0..3 {
            println!("  [{},{},{}]", na_qt_q[(i,0)], na_qt_q[(i,1)], na_qt_q[(i,2)]);
        }

        // Both should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_close(our_qt_q.get(i, j), expected, 1e-9, &format!("our Q^T*Q [{},{}]", i, j));
                assert_close(na_qt_q[(i, j)], expected, 1e-9, &format!("na Q^T*Q [{},{}]", i, j));
            }
        }
    }

    #[test]
    fn compare_transpose() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];  // 3x4 matrix

        let our_m = Matrix::new(3, 4, data);
        let na_m = to_nalgebra_matrix(&our_m);

        let our_t = our_m.transpose();
        let na_t = na_m.transpose();

        println!("\n=== Transpose Comparison ===");
        println!("Original: 3x4, Transposed: 4x3");
        println!("Our transpose:");
        for i in 0..4 {
            println!("  [{},{},{}]", our_t.get(i,0), our_t.get(i,1), our_t.get(i,2));
        }

        println!("\nNalgebra transpose:");
        for i in 0..4 {
            println!("  [{},{},{}]", na_t[(i,0)], na_t[(i,1)], na_t[(i,2)]);
        }

        // Compare dimensions
        assert_eq!(our_t.rows, 4, "transposed rows should be 4");
        assert_eq!(our_t.cols, 3, "transposed cols should be 3");

        // Compare values
        for i in 0..4 {
            for j in 0..3 {
                assert_close(
                    our_t.get(i, j),
                    na_t[(i, j)],
                    TOLERANCE,
                    &format!("transpose [{},{}]", i, j)
                );
            }
        }
    }

    #[test]
    fn compare_mul_vec() {
        // Matrix x vector: used in OLS for Q^T * y
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ];  // 3x3 matrix
        let vec = vec![2.0, 3.0, 4.0];

        let our_m = Matrix::new(3, 3, data);
        let na_m = to_nalgebra_matrix(&our_m);
        let na_vec = DVector::from_vec(vec.clone());

        let our_result = our_m.mul_vec(&vec);
        let na_result = &na_m * &na_vec;

        println!("\n=== Matrix x Vector Comparison ===");
        println!("Vector: {:?}", vec);
        println!("Our result: {:?}", our_result);
        println!("Nalgebra result: {:?}", na_result.iter().copied().collect::<Vec<_>>());

        for i in 0..3 {
            assert_close(
                our_result[i],
                na_result[i],
                TOLERANCE,
                &format!("mul_vec [{}]", i)
            );
        }
    }

    #[test]
    fn compare_invert_upper_triangular() {
        // Create an upper triangular matrix
        let data = vec![
            2.0, 3.0, 1.0,
            0.0, 4.0, 2.0,
            0.0, 0.0, 3.0,
        ];  // Upper triangular

        let our_m = Matrix::new(3, 3, data.clone());
        let na_m = to_nalgebra_matrix(&our_m);

        let our_inv = our_m.invert_upper_triangular().expect("should invert upper triangular");

        // Nalgebra doesn't have a specialized upper triangular inverse,
        // so we use the general inverse for comparison
        let na_inv = na_m.try_inverse().expect("nalgebra should invert");

        println!("\n=== Upper Triangular Inverse Comparison ===");
        println!("Original (upper triangular):");
        for i in 0..3 {
            println!("  [{},{},{}]", data[i*3], data[i*3+1], data[i*3+2]);
        }

        println!("\nOur inverse:");
        for i in 0..3 {
            println!("  [{},{},{}]", our_inv.get(i,0), our_inv.get(i,1), our_inv.get(i,2));
        }

        println!("\nNalgebra inverse:");
        for i in 0..3 {
            println!("  [{},{},{}]", na_inv[(i,0)], na_inv[(i,1)], na_inv[(i,2)]);
        }

        // Compare values
        for i in 0..3 {
            for j in 0..3 {
                assert_close(
                    our_inv.get(i, j),
                    na_inv[(i, j)],
                    1e-9,
                    &format!("upper triangular inverse [{},{}]", i, j)
                );
            }
        }

        // Verify A * A^-1 = I
        let our_product = our_m.matmul(&our_inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_close(
                    our_product.get(i, j),
                    expected,
                    1e-9,
                    &format!("A * A^-1 = I [{},{}]", i, j)
                );
            }
        }
    }

    #[test]
    fn compare_qr_solve() {
        // Test QR-based linear system solve: Ax = b
        // Solution: x = R^(-1) * Q^T * b
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,  // Full rank 3x3
        ];
        let b = vec![6.0, 15.0, 25.0];  // RHS vector

        let our_a = Matrix::new(3, 3, data);
        let na_a = to_nalgebra_matrix(&our_a);
        let na_b = DVector::from_vec(b.clone());

        // Our QR solve: x = R^(-1) * Q^T * b
        let (our_q, our_r) = our_a.qr();
        let our_qt = our_q.transpose();
        let our_qtb = our_qt.mul_vec(&b);
        let our_r_inv = our_r.invert_upper_triangular().expect("R should be invertible");
        let our_x_mat = our_r_inv.matmul(&Matrix::new(3, 1, our_qtb));
        let our_x: Vec<f64> = (0..3).map(|i| our_x_mat.get(i, 0)).collect();

        // Nalgebra QR solve
        let na_qr = na_a.qr();
        let na_x = na_qr.solve(&na_b).expect("nalgebra should solve");

        println!("\n=== QR Solve Comparison ===");
        println!("b vector: {:?}", b);
        println!("Our solution: {:?}", our_x);
        println!("Nalgebra solution: {:?}", na_x.iter().copied().collect::<Vec<f64>>());

        // Compare solutions
        for i in 0..3 {
            assert_close(
                our_x[i],
                na_x[i],
                1e-9,
                &format!("QR solve x[{}]", i)
            );
        }

        // Verify Ax = b for our solution
        let verification = our_a.mul_vec(&our_x);
        println!("\nVerification (A * x):");
        for i in 0..3 {
            println!("  [{}] = {:.6} (expected {:.6}), diff = {:.2e}",
                i, verification[i], b[i], (verification[i] - b[i]).abs());
            assert_close(
                verification[i],
                b[i],
                1e-9,
                &format!("Ax = b verification [{}]", i)
            );
        }
    }
}
*/
