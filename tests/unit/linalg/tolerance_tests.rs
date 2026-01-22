// ============================================================================
// Tolerance Variants Tests
// ============================================================================
//
// Tests for functions with custom tolerance parameters:
// - invert_upper_triangular_with_tolerance()
// - chol2inv_from_qr_with_tolerance()

use super::common::{assert_close, assert_matrix_eq, EPSILON};
use linreg_core::linalg::Matrix;

// ============================================================================
// invert_upper_triangular_with_tolerance() Tests
// ============================================================================

#[test]
fn test_invert_upper_triangular_with_tolerance_basic() {
    let a = Matrix::new(3, 3, vec![2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0]);

    let inv = a
        .invert_upper_triangular_with_tolerance(1.0)
        .expect("Should be invertible with tolerance 1.0");

    // Verify A * A^-1 = I
    let a_inv = a.matmul(&inv);
    let i = Matrix::identity(3);
    assert_matrix_eq(&a_inv, &i, 1e-9, "tolerance 1.0");
}

#[test]
fn test_invert_upper_triangular_with_tolerance_standard() {
    // Test that default and tolerance 1.0 give same result
    let a = Matrix::new(3, 3, vec![2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0]);

    let inv_default = a.invert_upper_triangular().expect("Default should work");
    let inv_tol = a
        .invert_upper_triangular_with_tolerance(1.0)
        .expect("Tolerance 1.0 should work");

    // Results should be identical
    for i in 0..3 {
        for j in 0..3 {
            assert_close(
                inv_default.get(i, j),
                inv_tol.get(i, j),
                EPSILON,
                &format!("default vs tol[{},{}]", i, j),
            );
        }
    }
}

#[test]
fn test_invert_upper_triangular_with_permissive_tolerance() {
    // Test that we can control tolerance behavior
    // Note: Due to SINGULAR_TOLERANCE floor of 1e-10, elements smaller
    // than 1e-10 are always rejected. This test validates the behavior
    // around that boundary.

    // Element below SINGULAR_TOLERANCE floor - always rejected
    let a_below = Matrix::new(3, 3, vec![1e-12, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    let result_below = a_below.invert_upper_triangular_with_tolerance(1e-10);
    assert!(
        result_below.is_none(),
        "Below-floor element should be rejected even with custom mult"
    );

    // Element above SINGULAR_TOLERANCE floor - should be accepted
    let a_above = Matrix::new(3, 3, vec![1e-8, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    let result_above = a_above.invert_upper_triangular_with_tolerance(1e-10);
    assert!(
        result_above.is_some(),
        "Above-floor element should be accepted"
    );
}

#[test]
fn test_invert_upper_triangular_with_strict_tolerance() {
    let a = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0]);

    // Strict tolerance should still work for well-conditioned matrix
    let inv = a
        .invert_upper_triangular_with_tolerance(0.1)
        .expect("Strict tolerance should work for well-conditioned matrix");

    // Verify it's actually the inverse
    let a_inv = a.matmul(&inv);
    let i = Matrix::identity(3);
    assert_matrix_eq(&a_inv, &i, 1e-9, "strict tolerance");
}

#[test]
fn test_invert_upper_triangular_with_tolerance_near_singular() {
    // Matrix with moderately small diagonal elements
    let a = Matrix::new(3, 3, vec![0.001, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);

    // Might fail with strict tolerance
    let result_strict = a.invert_upper_triangular_with_tolerance(0.01);

    // Should succeed with permissive tolerance
    let result_permissive = a
        .invert_upper_triangular_with_tolerance(1000.0)
        .expect("Permissive tolerance should succeed");

    // Verify the permissive result is a valid inverse
    let a_inv = a.matmul(&result_permissive);
    let i = Matrix::identity(3);
    // Use looser tolerance for near-singular case
    assert_matrix_eq(&a_inv, &i, 1e-3, "permissive near-singular");
}

#[test]
fn test_invert_upper_triangular_with_tolerance_large_scale() {
    // Large diagonal values
    let a = Matrix::new(3, 3, vec![1e8, 2e8, 3e8, 0.0, 4e8, 5e8, 0.0, 0.0, 6e8]);

    let inv = a
        .invert_upper_triangular_with_tolerance(1.0)
        .expect("Large scale should invert");

    // Verify reconstruction
    let a_inv = a.matmul(&inv);
    let i = Matrix::identity(3);

    // Use relative tolerance for large scale
    let tolerance = 1e-10 * 1e8;
    for i_idx in 0..3 {
        for j in 0..3 {
            let expected = if i_idx == j { 1.0 } else { 0.0 };
            let diff = (a_inv.get(i_idx, j) - expected).abs();
            assert!(
                diff < tolerance,
                "Large scale tolerance error at [{},{}]: diff = {}",
                i_idx,
                j,
                diff
            );
        }
    }
}

#[test]
fn test_invert_upper_triangular_with_tolerance_zero_diagonal() {
    // Exactly zero diagonal should always fail regardless of tolerance
    let a = Matrix::new(3, 3, vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);

    // Even with very permissive tolerance, zero diagonal should fail
    // (division by zero would occur)
    let result = a.invert_upper_triangular_with_tolerance(1e100);
    assert!(
        result.is_none(),
        "Zero diagonal should always fail to invert"
    );
}

#[test]
fn test_invert_upper_triangular_tolerance_multiplier_effect() {
    let a = Matrix::new(3, 3, vec![1e-8, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);

    // Small multiplier might fail
    let result_small = a.invert_upper_triangular_with_tolerance(1.0);

    // Large multiplier should succeed
    let result_large = a
        .invert_upper_triangular_with_tolerance(1e6)
        .expect("Large multiplier should succeed");

    // Verify it's a valid inverse
    let a_inv = a.matmul(&result_large);
    let i = Matrix::identity(3);
    assert_matrix_eq(&a_inv, &i, 1e-4, "large multiplier result");
}

// ============================================================================
// chol2inv_from_qr_with_tolerance() Tests
// ============================================================================

#[test]
fn test_chol2inv_with_tolerance_basic() {
    let x = Matrix::new(
        4,
        3,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 3.0, 4.0],
    );

    let result = x
        .chol2inv_from_qr_with_tolerance(1.0)
        .expect("Should work with tolerance 1.0");

    // Verify by comparing with direct (X'X)^(-1)
    let x_t = x.transpose();
    let xtx = x_t.matmul(&x);
    let xtx_inv = xtx.invert().expect("X'X should be invertible");

    assert_matrix_eq(&result, &xtx_inv, 1e-9, "chol2inv with tolerance");
}

#[test]
fn test_chol2inv_with_tolerance_matches_default() {
    let x = Matrix::new(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]);

    let result_default = x.chol2inv_from_qr().expect("Default should work");
    let result_tol = x
        .chol2inv_from_qr_with_tolerance(1.0)
        .expect("Tolerance 1.0 should work");

    assert_matrix_eq(&result_default, &result_tol, EPSILON, "default vs tol 1.0");
}

#[test]
fn test_chol2inv_with_tolerance_near_rank_deficient() {
    // Nearly rank-deficient matrix
    let epsilon = 1e-8;
    let x = Matrix::new(3, 2, vec![1.0, 2.0, 2.0, 4.0 + epsilon, 3.0, 6.0]);

    // Default tolerance might fail or succeed depending on the values
    let result_default = x.chol2inv_from_qr();

    // Permissive tolerance should always succeed
    let result_permissive = x
        .chol2inv_from_qr_with_tolerance(1000.0)
        .expect("Permissive tolerance should succeed");

    // Verify the result is valid (if default also succeeded, compare)
    if let Some(default_result) = result_default {
        // Both succeeded - compare
        assert_matrix_eq(
            &default_result,
            &result_permissive,
            1e-5,
            "default vs permissive",
        );
    } else {
        // Only permissive succeeded - verify it produces valid inverse
        let x_t = x.transpose();
        let xtx = x_t.matmul(&x);
        let product = xtx.matmul(&result_permissive);
        let i = Matrix::identity(2);
        assert_matrix_eq(&product, &i, 1e-3, "permissive chol2inv reconstruction");
    }
}

#[test]
fn test_chol2inv_with_tolerance_permissive_vs_strict() {
    // Matrix with some collinearity
    let x = Matrix::new(
        4,
        3,
        vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0],
    );

    // This is exactly rank 1 (columns 2 and 3 are multiples of column 1)
    // Should fail even with permissive tolerance for truly singular
    let result_strict = x.chol2inv_from_qr_with_tolerance(0.1);
    let result_permissive = x.chol2inv_from_qr_with_tolerance(1000.0);

    // Both should fail for truly singular matrix
    assert!(result_strict.is_none(), "Strict should fail on singular");
    // Permissive might also fail or return garbage, but shouldn't crash
    assert!(
        result_permissive.is_none(),
        "Permissive should also fail on truly singular"
    );
}

#[test]
fn test_chol2inv_with_tolerance_large_scale() {
    let scale = 1e6;
    let x = Matrix::new(
        4,
        3,
        vec![
            1.0 * scale,
            2.0 * scale,
            3.0 * scale,
            2.0 * scale,
            3.0 * scale,
            4.0 * scale,
            3.0 * scale,
            4.0 * scale,
            6.0 * scale,
            1.5 * scale,
            2.5 * scale,
            3.5 * scale,
        ],
    );

    let result = x
        .chol2inv_from_qr_with_tolerance(1.0)
        .expect("Large scale should work");

    // Verify reconstruction
    let x_t = x.transpose();
    let xtx = x_t.matmul(&x);
    let product = xtx.matmul(&result);
    let i = Matrix::identity(3);

    // Use relative tolerance for large scale
    let tolerance = 1e-6 * scale;
    for i_idx in 0..3 {
        for j in 0..3 {
            let expected = if i_idx == j { 1.0 } else { 0.0 };
            let diff = (product.get(i_idx, j) - expected).abs();
            assert!(
                diff < tolerance,
                "Large scale chol2inv error at [{},{}]: diff = {}",
                i_idx,
                j,
                diff
            );
        }
    }
}

#[test]
fn test_chol2inv_with_tolerance_small_scale() {
    let scale = 1e-4;
    let x = Matrix::new(
        3,
        2,
        vec![
            1.0 * scale,
            2.0 * scale,
            2.0 * scale,
            3.0 * scale,
            3.0 * scale,
            5.0 * scale,
        ],
    );

    let result = x
        .chol2inv_from_qr_with_tolerance(1.0)
        .expect("Small scale should work");

    // Verify by reconstruction
    let x_t = x.transpose();
    let xtx = x_t.matmul(&x);
    let identity = xtx.matmul(&result);

    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (identity.get(i, j) - expected).abs();
            assert!(
                diff < 1e-9,
                "Small scale error at [{},{}]: diff = {}",
                i,
                j,
                diff
            );
        }
    }
}

#[test]
fn test_chol2inv_tolerance_levels_comparison() {
    // Create a matrix that's moderately ill-conditioned
    let x = Matrix::new(4, 2, vec![1.0, 2.0, 2.0, 4.01, 3.0, 6.0, 4.0, 8.0]);

    // Try different tolerance levels
    let tol_1 = x.chol2inv_from_qr_with_tolerance(0.1);
    let tol_10 = x.chol2inv_from_qr_with_tolerance(10.0);
    let tol_100 = x.chol2inv_from_qr_with_tolerance(100.0);

    // All should succeed for this matrix
    assert!(tol_1.is_some(), "tol 0.1 should succeed");
    assert!(tol_10.is_some(), "tol 10.0 should succeed");
    assert!(tol_100.is_some(), "tol 100.0 should succeed");

    // Results should be very similar for well-conditioned enough matrix
    let r1 = tol_1.unwrap();
    let r10 = tol_10.unwrap();
    let r100 = tol_100.unwrap();

    for i in 0..2 {
        for j in 0..2 {
            // All should produce similar results
            let diff_1_10 = (r1.get(i, j) - r10.get(i, j)).abs();
            let diff_10_100 = (r10.get(i, j) - r100.get(i, j)).abs();
            assert!(
                diff_1_10 < 1e-6,
                "tol 0.1 vs 10.0 diff at [{},{}]: {}",
                i,
                j,
                diff_1_10
            );
            assert!(
                diff_10_100 < 1e-6,
                "tol 10.0 vs 100.0 diff at [{},{}]: {}",
                i,
                j,
                diff_10_100
            );
        }
    }
}
