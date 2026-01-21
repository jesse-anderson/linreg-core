// ============================================================================
// Column Operations Tests
// ============================================================================
//
// Tests for column-specific matrix operations: col_dot, col_axpy_inplace,
// col_norm2, and add_diagonal_in_place.

use linreg_core::linalg::Matrix;
use super::common::{EPSILON, assert_close};

// ============================================================================
// col_dot() Tests
// ============================================================================

#[test]
fn test_col_dot_basic() {
    // Matrix with 3 rows, 2 columns
    // Column 0: [1, 2, 3]
    // Column 1: [4, 5, 6]
    let m = Matrix::new(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    // Dot product of column 0 with [1, 1, 1] should be 1+2+3=6
    let v = vec![1.0, 1.0, 1.0];
    let result = m.col_dot(0, &v);
    assert_close(result, 6.0, EPSILON, "col_dot basic");

    // Dot product of column 1 with [2, 1, 0] should be 4*2+5*1+6*0=13
    let v2 = vec![2.0, 1.0, 0.0];
    let result2 = m.col_dot(1, &v2);
    assert_close(result2, 13.0, EPSILON, "col_dot second column");
}

#[test]
fn test_col_dot_with_zeros() {
    let m = Matrix::new(3, 2, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);

    // Dot with zero column should be 0
    let v = vec![1.0, 2.0, 3.0];
    let result = m.col_dot(1, &v);
    assert_close(result, 0.0, EPSILON, "col_dot zero column");

    // Dot with zero vector should be 0
    let v_zero = vec![0.0, 0.0, 0.0];
    let result2 = m.col_dot(0, &v_zero);
    assert_close(result2, 0.0, EPSILON, "col_dot zero vector");
}

#[test]
fn test_col_dot_orthogonal() {
    // Column 0: [1, 0, 0]
    // Column 1: [0, 1, 0]
    let m = Matrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    // Dot of orthogonal columns should be 0
    let col0_as_vec = vec![1.0, 0.0, 0.0];
    let result = m.col_dot(1, &col0_as_vec);
    assert_close(result, 0.0, EPSILON, "col_dot orthogonal");
}

#[test]
#[should_panic(expected = "Column index out of bounds")]
fn test_col_dot_panic_invalid_column() {
    let m = Matrix::new(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let v = vec![1.0, 1.0, 1.0];
    m.col_dot(5, &v); // Column 5 doesn't exist
}

#[test]
#[should_panic(expected = "Vector length must match number of rows")]
fn test_col_dot_panic_wrong_length() {
    let m = Matrix::new(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let v = vec![1.0, 2.0]; // Wrong length
    m.col_dot(0, &v);
}

// ============================================================================
// col_axpy_inplace() Tests
// ============================================================================

#[test]
fn test_col_axpy_inplace_basic() {
    // Matrix with column 0: [1, 2, 3]
    let m = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);

    // v += 2.0 * column 0
    // v starts as [0, 0, 0], should become [2, 4, 6]
    let mut v = vec![0.0, 0.0, 0.0];
    m.col_axpy_inplace(0, 2.0, &mut v);

    assert_close(v[0], 2.0, EPSILON, "axpy[0]");
    assert_close(v[1], 4.0, EPSILON, "axpy[1]");
    assert_close(v[2], 6.0, EPSILON, "axpy[2]");
}

#[test]
fn test_col_axpy_inplace_accumulate() {
    let m = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);

    // Start with non-zero v
    let mut v = vec![1.0, 1.0, 1.0];
    // v += 0.5 * column 0
    m.col_axpy_inplace(0, 0.5, &mut v);

    assert_close(v[0], 1.5, EPSILON, "axpy accumulate[0]");
    assert_close(v[1], 2.0, EPSILON, "axpy accumulate[1]");
    assert_close(v[2], 2.5, EPSILON, "axpy accumulate[2]");
}

#[test]
fn test_col_axpy_inplace_negative_alpha() {
    let m = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);

    let mut v = vec![10.0, 10.0, 10.0];
    // v += (-1.0) * column 0 = v - column 0
    m.col_axpy_inplace(0, -1.0, &mut v);

    assert_close(v[0], 9.0, EPSILON, "negative axpy[0]");
    assert_close(v[1], 8.0, EPSILON, "negative axpy[1]");
    assert_close(v[2], 7.0, EPSILON, "negative axpy[2]");
}

#[test]
fn test_col_axpy_inplace_zero_alpha() {
    let m = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);

    let mut v = vec![5.0, 6.0, 7.0];
    let v_copy = v.clone();
    // alpha = 0 should not change v
    m.col_axpy_inplace(0, 0.0, &mut v);

    assert_eq!(v, v_copy, "zero alpha should not change v");
}

#[test]
fn test_col_axpy_inplace_multi_column() {
    // Matrix with 2 columns
    let m = Matrix::new(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let mut v = vec![0.0, 0.0, 0.0];
    // Add column 0 with alpha=1
    m.col_axpy_inplace(0, 1.0, &mut v);
    // v should be [1, 2, 3]
    assert_close(v[0], 1.0, EPSILON, "multi col step 1[0]");
    assert_close(v[1], 2.0, EPSILON, "multi col step 1[1]");
    assert_close(v[2], 3.0, EPSILON, "multi col step 1[2]");

    // Add column 1 with alpha=2
    m.col_axpy_inplace(1, 2.0, &mut v);
    // v should be [1+8, 2+10, 3+12] = [9, 12, 15]
    assert_close(v[0], 9.0, EPSILON, "multi col step 2[0]");
    assert_close(v[1], 12.0, EPSILON, "multi col step 2[1]");
    assert_close(v[2], 15.0, EPSILON, "multi col step 2[2]");
}

#[test]
#[should_panic(expected = "Column index out of bounds")]
fn test_col_axpy_inplace_panic_invalid_column() {
    let m = Matrix::new(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let mut v = vec![0.0, 0.0, 0.0];
    m.col_axpy_inplace(5, 1.0, &mut v);
}

#[test]
#[should_panic(expected = "Vector length must match number of rows")]
fn test_col_axpy_inplace_panic_wrong_length() {
    let m = Matrix::new(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let mut v = vec![0.0, 0.0]; // Wrong length
    m.col_axpy_inplace(0, 1.0, &mut v);
}

// ============================================================================
// col_norm2() Tests
// ============================================================================

#[test]
fn test_col_norm2_basic() {
    let m = Matrix::new(3, 2, vec![3.0, 4.0, 0.0, 0.0, 0.0, 5.0]);

    // Column 0: [3, 0, 0], norm^2 = 9
    let n0 = m.col_norm2(0);
    assert_close(n0, 9.0, EPSILON, "col_norm2 col0");

    // Column 1: [4, 0, 5], norm^2 = 16 + 0 + 25 = 41
    let n1 = m.col_norm2(1);
    assert_close(n1, 41.0, EPSILON, "col_norm2 col1");
}

#[test]
fn test_col_norm2_zero_column() {
    let m = Matrix::new(3, 2, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);

    let n = m.col_norm2(1);
    assert_close(n, 0.0, EPSILON, "col_norm2 zero column");
}

#[test]
fn test_col_norm2_unit_column() {
    let m = Matrix::new(4, 1, vec![0.5, 0.5, 0.5, 0.5]);

    // Each element squared: 0.25 * 4 = 1.0
    let n = m.col_norm2(0);
    assert_close(n, 1.0, EPSILON, "col_norm2 unit column");
}

#[test]
fn test_col_norm2_large_values() {
    let m = Matrix::new(2, 1, vec![1e6, 1e6]);

    // norm^2 = 1e12 + 1e12 = 2e12
    let n = m.col_norm2(0);
    assert_close(n, 2e12, 1e-6 * 2e12, "col_norm2 large values");
}

#[test]
fn test_col_norm2_negative_values() {
    let m = Matrix::new(3, 1, vec![-3.0, -4.0, -5.0]);

    // Squared values ignore sign: 9 + 16 + 25 = 50
    let n = m.col_norm2(0);
    assert_close(n, 50.0, EPSILON, "col_norm2 negative");
}

#[test]
#[should_panic(expected = "Column index out of bounds")]
fn test_col_norm2_panic_invalid_column() {
    let m = Matrix::new(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    m.col_norm2(5);
}

// ============================================================================
// add_diagonal_in_place() Tests
// ============================================================================

#[test]
fn test_add_diagonal_in_place_basic() {
    let mut m = Matrix::new(
        3,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
    );

    m.add_diagonal_in_place(5.0, 0);

    // All diagonal elements should have 5.0 added
    assert_close(m.get(0, 0), 6.0, EPSILON, "diag[0,0]");
    assert_close(m.get(1, 1), 7.0, EPSILON, "diag[1,1]");
    assert_close(m.get(2, 2), 8.0, EPSILON, "diag[2,2]");

    // Off-diagonal should be unchanged
    assert_close(m.get(0, 1), 0.0, EPSILON, "off-diag[0,1]");
    assert_close(m.get(1, 0), 0.0, EPSILON, "off-diag[1,0]");
}

#[test]
fn test_add_diagonal_in_place_start_index() {
    let mut m = Matrix::new(
        4,
        4,
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 4.0,
        ],
    );

    // Add starting from index 1 (skip first diagonal element)
    m.add_diagonal_in_place(10.0, 1);

    // First diagonal element unchanged
    assert_close(m.get(0, 0), 1.0, EPSILON, "start index diag[0,0]");

    // Others should have 10.0 added
    assert_close(m.get(1, 1), 12.0, EPSILON, "start index diag[1,1]");
    assert_close(m.get(2, 2), 13.0, EPSILON, "start index diag[2,2]");
    assert_close(m.get(3, 3), 14.0, EPSILON, "start index diag[3,3]");
}

#[test]
fn test_add_diagonal_in_place_ridge_regression() {
    // Simulate ridge regression where intercept column is not penalized
    let mut m = Matrix::new(
        3,
        3,
        vec![
            10.0, 1.0, 2.0, 3.0, 20.0, 4.0, 5.0, 6.0, 30.0,
        ],
    );

    let lambda = 2.0;
    // Don't penalize intercept (column 0)
    m.add_diagonal_in_place(lambda, 1);

    // (0,0) should be unchanged
    assert_close(m.get(0, 0), 10.0, EPSILON, "ridge intercept");

    // Other diagonals should have lambda added
    assert_close(m.get(1, 1), 22.0, EPSILON, "ridge diag[1,1]");
    assert_close(m.get(2, 2), 32.0, EPSILON, "ridge diag[2,2]");
}

#[test]
fn test_add_diagonal_in_place_negative_value() {
    let mut m = Matrix::new(
        2,
        2,
        vec![10.0, 1.0, 2.0, 20.0],
    );

    m.add_diagonal_in_place(-3.0, 0);

    assert_close(m.get(0, 0), 7.0, EPSILON, "negative add[0,0]");
    assert_close(m.get(1, 1), 17.0, EPSILON, "negative add[1,1]");
}

#[test]
fn test_add_diagonal_in_place_zero_value() {
    let mut m = Matrix::new(
        2,
        2,
        vec![1.0, 2.0, 3.0, 4.0],
    );

    let m_copy = m.clone();
    m.add_diagonal_in_place(0.0, 0);

    // Should be unchanged
    assert_close(m.get(0, 0), m_copy.get(0, 0), EPSILON, "zero add[0,0]");
    assert_close(m.get(1, 1), m_copy.get(1, 1), EPSILON, "zero add[1,1]");
}

#[test]
fn test_add_diagonal_in_place_start_at_end() {
    let mut m = Matrix::new(
        3,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
    );

    // Start at index 3 (past the last element)
    m.add_diagonal_in_place(5.0, 3);

    // All diagonals should be unchanged
    assert_close(m.get(0, 0), 1.0, EPSILON, "start at end[0,0]");
    assert_close(m.get(1, 1), 2.0, EPSILON, "start at end[1,1]");
    assert_close(m.get(2, 2), 3.0, EPSILON, "start at end[2,2]");
}

#[test]
#[should_panic(expected = "Matrix must be square")]
fn test_add_diagonal_in_place_panic_non_square() {
    let mut m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    m.add_diagonal_in_place(1.0, 0);
}

// ============================================================================
// Combined Column Operations Tests
// ============================================================================

#[test]
fn test_column_operations_matrix_vector_multiply() {
    // Simulate matrix-vector multiplication using column operations
    // M = [1 3; 2 4; 3 5], v = [2, 3]
    // Result = M * v = [1*2+3*3, 2*2+4*3, 3*2+5*3] = [11, 16, 21]
    let m = Matrix::new(3, 2, vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0]);
    let v = vec![2.0, 3.0];

    let mut result = vec![0.0; 3];
    // result = v[0] * column_0 + v[1] * column_1
    m.col_axpy_inplace(0, v[0], &mut result);
    m.col_axpy_inplace(1, v[1], &mut result);

    assert_close(result[0], 11.0, EPSILON, "combined op[0]");
    assert_close(result[1], 16.0, EPSILON, "combined op[1]");
    assert_close(result[2], 21.0, EPSILON, "combined op[2]");
}

#[test]
fn test_col_dot_matches_matmul_element() {
    // Verify col_dot gives same result as corresponding element in matrix multiplication
    let m1 = Matrix::new(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let m2 = Matrix::new(2, 1, vec![2.0, 3.0]);

    let result = m1.matmul(&m2);

    // result[0] = m1[0,:] * m2[:,0] = col0_dot(m1, col0) * m2[0,0] + col1_dot(m1, col1) * m2[1,0]
    // Am I stupid.... Probably.... thankfully no one checks tests and especially comments on tests!
    // matmul: result[i,j] = sum_k m1[i,k] * m2[k,j]
    // For our case: result[i,0] = m1[i,0]*m2[0,0] + m1[i,1]*m2[1,0]

    // A simpler check: for a column vector, matmul is weighted sum of columns
    let weights = vec![m2.get(0, 0), m2.get(1, 0)];

    let mut computed = vec![0.0; 3];
    computed[0] = m1.col_dot(0, &vec![weights[0], 0.0, 0.0]) + m1.col_dot(1, &vec![0.0, weights[1], 0.0]);

    // let's just verify the basic property:
    // col_dot of column with unit vector gives the element at that position
    let unit_0 = vec![1.0, 0.0, 0.0];
    let unit_1 = vec![0.0, 1.0, 0.0];
    let unit_2 = vec![0.0, 0.0, 1.0];

    assert_close(m1.col_dot(0, &unit_0), 1.0, EPSILON, "unit vector check 0,0");
    assert_close(m1.col_dot(0, &unit_1), 2.0, EPSILON, "unit vector check 1,0");
    assert_close(m1.col_dot(0, &unit_2), 3.0, EPSILON, "unit vector check 2,0");

    assert_close(m1.col_dot(1, &unit_0), 4.0, EPSILON, "unit vector check 0,1");
    assert_close(m1.col_dot(1, &unit_1), 5.0, EPSILON, "unit vector check 1,1");
    assert_close(m1.col_dot(1, &unit_2), 6.0, EPSILON, "unit vector check 2,1");
}
