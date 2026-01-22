// ============================================================================
// Basic Matrix Tests
// ============================================================================
//
// Tests for matrix constructors, element access, transpose operations,
// and basic matrix properties.

use super::common::{assert_close, assert_matrix_eq, EPSILON};
use linreg_core::linalg::Matrix;

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
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
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
    let m = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0]);
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

#[test]
fn test_transpose_of_transpose() {
    // (A^T)^T = A
    let a = Matrix::new(
        3,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );

    let att = a.transpose().transpose();

    assert_eq!(att.rows, a.rows);
    assert_eq!(att.cols, a.cols);

    for i in 0..a.rows {
        for j in 0..a.cols {
            assert_close(
                att.get(i, j),
                a.get(i, j),
                EPSILON,
                &format!("double transpose[{},{}]", i, j),
            );
        }
    }
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

// ============================================================================
// Clone Tests
// ============================================================================

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
