// ============================================================================
// Matrix Operations Tests
// ============================================================================
//
// Tests for matrix multiplication, matrix-vector multiplication,
// and vector helper functions.

use linreg_core::linalg::{Matrix, vec_mean, vec_sub, vec_dot};
use super::common::{EPSILON, assert_close, assert_matrix_eq};

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
fn test_mul_vec_empty_result() {
    // Matrix-vector multiplication resulting in empty vector
    // 0x3 matrix times 3-element vector = 0-element result
    let m = Matrix::new(0, 3, vec![]);
    let v = vec![1.0, 2.0, 3.0];

    let result = m.mul_vec(&v);

    assert_eq!(result.len(), 0);
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
fn test_vec_sub_empty() {
    // Vector subtraction with empty slices
    let a: Vec<f64> = vec![];
    let b: Vec<f64> = vec![];
    let result = vec_sub(&a, &b);

    assert_eq!(result.len(), 0);
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

#[test]
fn test_vec_dot_empty() {
    // Dot product of empty vectors
    let a: Vec<f64> = vec![];
    let b: Vec<f64> = vec![];
    let result = vec_dot(&a, &b);

    // Sum of empty slice is 0.0
    assert_eq!(result, 0.0);
}

#[test]
#[should_panic(expected = "slice lengths must match")]
fn test_vec_sub_panic_on_length_mismatch() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0]; // Different length
    vec_sub(&a, &b);
}

#[test]
#[should_panic(expected = "slice lengths must match")]
fn test_vec_sub_panic_on_length_mismatch_reversed() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0]; // Different length (b longer)
    vec_sub(&a, &b);
}

#[test]
#[should_panic(expected = "slice lengths must match")]
fn test_vec_dot_panic_on_length_mismatch() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0]; // Different length
    vec_dot(&a, &b);
}

#[test]
#[should_panic(expected = "slice lengths must match")]
fn test_vec_dot_panic_on_length_mismatch_reversed() {
    let a = vec![1.0];
    let b = vec![1.0, 2.0, 3.0]; // Different length (b longer)
    vec_dot(&a, &b);
}
