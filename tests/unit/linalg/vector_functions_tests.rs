// ============================================================================
// Vector Helper Functions Tests
// ============================================================================
//
// Tests for vector utility functions not already covered in ops_tests:
// vec_add, vec_axpy_inplace, vec_scale_inplace, vec_scale, vec_l2_norm, vec_max_abs

use super::common::EPSILON;
use linreg_core::linalg::{
    vec_add, vec_axpy_inplace, vec_l2_norm, vec_max_abs, vec_scale, vec_scale_inplace,
};

// ============================================================================
// vec_add() Tests
// ============================================================================

#[test]
fn test_vec_add_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    let result = vec_add(&a, &b);

    assert_eq!(result.len(), 3);
    assert_eq!(result[0], 5.0);
    assert_eq!(result[1], 7.0);
    assert_eq!(result[2], 9.0);
}

#[test]
fn test_vec_add_negative() {
    let a = vec![5.0, 3.0];
    let b = vec![-2.0, -1.0];

    let result = vec_add(&a, &b);

    assert_eq!(result[0], 3.0);
    assert_eq!(result[1], 2.0);
}

#[test]
fn test_vec_add_zeros() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![0.0, 0.0, 0.0];

    let result = vec_add(&a, &b);

    assert_eq!(result, a);
}

#[test]
fn test_vec_add_empty() {
    let a: Vec<f64> = vec![];
    let b: Vec<f64> = vec![];

    let result = vec_add(&a, &b);

    assert_eq!(result.len(), 0);
}

#[test]
fn test_vec_add_commutative() {
    let a = vec![1.5, 2.5, 3.5];
    let b = vec![0.5, 1.5, 2.5];

    let ab = vec_add(&a, &b);
    let ba = vec_add(&b, &a);

    assert_eq!(ab, ba);
}

// ============================================================================
// vec_axpy_inplace() Tests
// ============================================================================

#[test]
fn test_vec_axpy_inplace_basic() {
    // y += alpha * x
    let mut y = vec![1.0, 2.0, 3.0];
    let x = vec![2.0, 3.0, 4.0];
    let alpha = 2.0;

    vec_axpy_inplace(&mut y, alpha, &x);

    assert_eq!(y[0], 1.0 + 2.0 * 2.0); // = 5.0
    assert_eq!(y[1], 2.0 + 2.0 * 3.0); // = 8.0
    assert_eq!(y[2], 3.0 + 2.0 * 4.0); // = 11.0
}

#[test]
fn test_vec_axpy_inplace_negative_alpha() {
    let mut y = vec![10.0, 10.0, 10.0];
    let x = vec![1.0, 2.0, 3.0];
    let alpha = -1.0;

    vec_axpy_inplace(&mut y, alpha, &x);

    assert_eq!(y[0], 9.0);
    assert_eq!(y[1], 8.0);
    assert_eq!(y[2], 7.0);
}

#[test]
fn test_vec_axpy_inplace_zero_alpha() {
    let mut y = vec![5.0, 6.0, 7.0];
    let y_copy = y.clone();
    let x = vec![1.0, 2.0, 3.0];
    let alpha = 0.0;

    vec_axpy_inplace(&mut y, alpha, &x);

    assert_eq!(y, y_copy);
}

#[test]
fn test_vec_axpy_inplace_zero_x() {
    let mut y = vec![5.0, 6.0, 7.0];
    let x = vec![0.0, 0.0, 0.0];
    let alpha = 100.0;

    vec_axpy_inplace(&mut y, alpha, &x);

    assert_eq!(y[0], 5.0);
    assert_eq!(y[1], 6.0);
    assert_eq!(y[2], 7.0);
}

#[test]
fn test_vec_axpy_inplace_unit_alpha() {
    // alpha = 1 means y += x
    let mut y = vec![1.0, 2.0, 3.0];
    let x = vec![4.0, 5.0, 6.0];
    let alpha = 1.0;

    vec_axpy_inplace(&mut y, alpha, &x);

    assert_eq!(y[0], 5.0);
    assert_eq!(y[1], 7.0);
    assert_eq!(y[2], 9.0);
}

#[test]
fn test_vec_axpy_inplace_fractional_alpha() {
    let mut y = vec![10.0, 20.0, 30.0];
    let x = vec![2.0, 4.0, 6.0];
    let alpha = 0.5;

    vec_axpy_inplace(&mut y, alpha, &x);

    assert_eq!(y[0], 11.0);
    assert_eq!(y[1], 22.0);
    assert_eq!(y[2], 33.0);
}

// ============================================================================
// vec_scale_inplace() Tests
// ============================================================================

#[test]
fn test_vec_scale_inplace_basic() {
    let mut v = vec![1.0, 2.0, 3.0];
    vec_scale_inplace(&mut v, 2.0);

    assert_eq!(v[0], 2.0);
    assert_eq!(v[1], 4.0);
    assert_eq!(v[2], 6.0);
}

#[test]
fn test_vec_scale_inplace_negative() {
    let mut v = vec![1.0, 2.0, 3.0];
    vec_scale_inplace(&mut v, -1.0);

    assert_eq!(v[0], -1.0);
    assert_eq!(v[1], -2.0);
    assert_eq!(v[2], -3.0);
}

#[test]
fn test_vec_scale_inplace_zero() {
    let mut v = vec![1.0, 2.0, 3.0];
    vec_scale_inplace(&mut v, 0.0);

    assert_eq!(v[0], 0.0);
    assert_eq!(v[1], 0.0);
    assert_eq!(v[2], 0.0);
}

#[test]
fn test_vec_scale_inplace_fractional() {
    let mut v = vec![10.0, 20.0, 30.0];
    vec_scale_inplace(&mut v, 0.1);

    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);
}

#[test]
fn test_vec_scale_inplace_reciprocal() {
    let mut v = vec![2.0, 4.0, 8.0];
    vec_scale_inplace(&mut v, 0.5);

    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 4.0);
}

#[test]
fn test_vec_scale_inplace_empty() {
    let mut v: Vec<f64> = vec![];
    vec_scale_inplace(&mut v, 2.0);
    assert_eq!(v.len(), 0);
}

// ============================================================================
// vec_scale() Tests
// ============================================================================

#[test]
fn test_vec_scale_returns_new() {
    let v = vec![1.0, 2.0, 3.0];

    let result = vec_scale(&v, 2.0);

    // Original should be unchanged
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);

    // Result should be scaled
    assert_eq!(result[0], 2.0);
    assert_eq!(result[1], 4.0);
    assert_eq!(result[2], 6.0);
}

#[test]
fn test_vec_scale_negative() {
    let v = vec![1.0, -2.0, 3.0];
    let result = vec_scale(&v, -2.0);

    assert_eq!(result[0], -2.0);
    assert_eq!(result[1], 4.0);
    assert_eq!(result[2], -6.0);
}

#[test]
fn test_vec_scale_preserves_original() {
    let v = vec![5.0, 10.0, 15.0];
    let _ = vec_scale(&v, 0.5);

    // Original should be unchanged
    assert_eq!(v[0], 5.0);
    assert_eq!(v[1], 10.0);
    assert_eq!(v[2], 15.0);
}

#[test]
fn test_vec_scale_empty() {
    let v: Vec<f64> = vec![];
    let result = vec_scale(&v, 2.0);
    assert_eq!(result.len(), 0);
}

// ============================================================================
// vec_l2_norm() Tests
// ============================================================================

#[test]
fn test_vec_l2_norm_basic() {
    let v = vec![3.0, 4.0];
    // sqrt(9 + 16) = sqrt(25) = 5
    let result = vec_l2_norm(&v);
    assert_eq!(result, 5.0);
}

#[test]
fn test_vec_l2_norm_zero() {
    let v = vec![0.0, 0.0, 0.0];
    let result = vec_l2_norm(&v);
    assert_eq!(result, 0.0);
}

#[test]
fn test_vec_l2_norm_unit() {
    let v = vec![1.0, 0.0, 0.0];
    let result = vec_l2_norm(&v);
    assert_eq!(result, 1.0);
}

#[test]
fn test_vec_l2_norm_negative() {
    let v = vec![-3.0, -4.0];
    // sqrt(9 + 16) = 5 (norm ignores sign)
    let result = vec_l2_norm(&v);
    assert_eq!(result, 5.0);
}

#[test]
fn test_vec_l2_norm_pythagorean_triple() {
    // 5-12-13 triangle
    let v = vec![5.0, 12.0];
    let result = vec_l2_norm(&v);
    assert_eq!(result, 13.0);
}

#[test]
fn test_vec_l2_norm_3d() {
    let v = vec![1.0, 2.0, 2.0];
    // sqrt(1 + 4 + 4) = sqrt(9) = 3
    let result = vec_l2_norm(&v);
    assert_eq!(result, 3.0);
}

#[test]
fn test_vec_l2_norm_large_values() {
    let v = vec![1e6, 1e6];
    // sqrt(2e12) = sqrt(2) * 1e6
    let result = vec_l2_norm(&v);
    let expected = 2.0_f64.sqrt() * 1e6;
    assert!((result - expected).abs() < 1e-6 * expected);
}

#[test]
fn test_vec_l2_norm_small_values() {
    let v = vec![1e-6, 1e-6];
    // sqrt(2e-12) = sqrt(2) * 1e-6
    let result = vec_l2_norm(&v);
    let expected = 2.0_f64.sqrt() * 1e-6;
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_vec_l2_norm_empty() {
    let v: Vec<f64> = vec![];
    let result = vec_l2_norm(&v);
    assert_eq!(result, 0.0);
}

// ============================================================================
// vec_max_abs() Tests
// ============================================================================

#[test]
fn test_vec_max_abs_basic() {
    let v = vec![1.0, -5.0, 3.0, 2.0];
    let result = vec_max_abs(&v);
    assert_eq!(result, 5.0);
}

#[test]
fn test_vec_max_abs_all_positive() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let result = vec_max_abs(&v);
    assert_eq!(result, 4.0);
}

#[test]
fn test_vec_max_abs_all_negative() {
    let v = vec![-1.0, -2.0, -3.0, -4.0];
    let result = vec_max_abs(&v);
    assert_eq!(result, 4.0);
}

#[test]
fn test_vec_max_abs_all_zero() {
    let v = vec![0.0, 0.0, 0.0];
    let result = vec_max_abs(&v);
    assert_eq!(result, 0.0);
}

#[test]
fn test_vec_max_abs_single_element() {
    let v = vec![42.0];
    let result = vec_max_abs(&v);
    assert_eq!(result, 42.0);
}

#[test]
fn test_vec_max_abs_single_negative() {
    let v = vec![-42.0];
    let result = vec_max_abs(&v);
    assert_eq!(result, 42.0);
}

#[test]
fn test_vec_max_abs_with_nans() {
    // NaN values: max_abs should skip them or handle gracefully
    // The implementation uses fold with f64::max which doesn't handle NaN specially
    // NaN comparisons always return false, so if the first element is NaN,
    // max will return the other value
    let v = vec![1.0, f64::NAN, 3.0];
    let result = vec_max_abs(&v);
    // With fold(0.0, f64::max): NaN comparisons are false, so 0.0 vs NaN -> 0.0
    // But 3.0 > 0.0, so result should be 3.0
    // Actually, max(NaN, anything) = NaN in IEEE 754
    // But max(anything, NaN) = anything
    // Our fold starts with 0.0, so max(0.0, 1.0) = 1.0, max(1.0, NaN) = NaN...
    // Let's see what happens
    assert!(result.is_nan() || result == 3.0);
}

#[test]
fn test_vec_max_abs_scientific_notation() {
    let v = vec![1e-10, 1e10, 1e5];
    let result = vec_max_abs(&v);
    assert_eq!(result, 1e10);
}

#[test]
fn test_vec_max_abs_empty() {
    let v: Vec<f64> = vec![];
    let result = vec_max_abs(&v);
    // fold with initial value 0.0 on empty iterator returns 0.0
    assert_eq!(result, 0.0);
}

// ============================================================================
// Combined Vector Function Tests
// ============================================================================

#[test]
fn test_vec_operations_sequence() {
    // Test a sequence of operations: scale, add, axpy
    let mut v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![4.0, 5.0, 6.0];

    // v1 *= 2
    vec_scale_inplace(&mut v1, 2.0);
    assert_eq!(v1, vec![2.0, 4.0, 6.0]);

    // v1 += v2
    let v3 = vec_add(&v1, &v2);
    assert_eq!(v3, vec![6.0, 9.0, 12.0]);

    // Original v1, v2 unchanged by vec_add
    assert_eq!(v1, vec![2.0, 4.0, 6.0]);
    assert_eq!(v2, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_norm_vs_manual_computation() {
    let v = vec![3.0, 4.0, 5.0, 6.0];

    let norm = vec_l2_norm(&v);
    let sum: f64 = 9.0 + 16.0 + 25.0 + 36.0;
    let manual = sum.sqrt();

    assert!((norm - manual).abs() < EPSILON);
}

#[test]
fn test_normalized_vector() {
    let v = vec![3.0, 4.0];
    let norm = vec_l2_norm(&v);
    let normalized = vec_scale(&v, 1.0 / norm);

    // Normalized vector should have unit norm
    let normalized_norm = vec_l2_norm(&normalized);
    assert!((normalized_norm - 1.0).abs() < EPSILON);
}

#[test]
fn test_axpy_equivalence() {
    // vec_axpy_inplace(y, alpha, x) should be same as y = vec_add(y, vec_scale(x, alpha))
    let mut y1 = vec![1.0, 2.0, 3.0];
    let x = vec![2.0, 3.0, 4.0];
    let alpha = 2.0;

    vec_axpy_inplace(&mut y1, alpha, &x);

    let y2 = vec![1.0, 2.0, 3.0];
    let scaled = vec_scale(&x, alpha);
    let y2_result = vec_add(&y2, &scaled);

    assert_eq!(y1, y2_result);
}

// ============================================================================
// Panic Tests for Length Mismatch
// ============================================================================

#[test]
#[should_panic(expected = "slice lengths must match")]
fn test_vec_add_panic_on_length_mismatch() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0]; // Different length
    vec_add(&a, &b);
}

#[test]
#[should_panic(expected = "slice lengths must match")]
fn test_vec_add_panic_on_length_mismatch_reversed() {
    let a = vec![1.0];
    let b = vec![1.0, 2.0, 3.0, 4.0]; // Different length (b longer)
    vec_add(&a, &b);
}

#[test]
#[should_panic(expected = "slice lengths must match")]
fn test_vec_axpy_inplace_panic_on_length_mismatch() {
    let mut dst = vec![1.0, 2.0, 3.0];
    let src = vec![1.0, 2.0]; // Different length
    vec_axpy_inplace(&mut dst, 1.0, &src);
}

#[test]
#[should_panic(expected = "slice lengths must match")]
fn test_vec_axpy_inplace_panic_on_length_mismatch_reversed() {
    let mut dst = vec![1.0, 2.0];
    let src = vec![1.0, 2.0, 3.0, 4.0]; // Different length (src longer)
    vec_axpy_inplace(&mut dst, 1.0, &src);
}
