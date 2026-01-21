// ============================================================================
// Linear Algebra Test Helpers
// ============================================================================
//
// Shared constants and helper functions for all linalg tests.

use linreg_core::linalg::Matrix;

/// Tolerance for general floating-point comparisons
pub const EPSILON: f64 = 1e-10;

/// Tolerance for QR decomposition comparisons (slightly looser)
pub const QR_TOLERANCE: f64 = 1e-9;

/// Helper function to assert two f64 values are close within tolerance
pub fn assert_close(a: f64, b: f64, tolerance: f64, context: &str) {
    let diff = (a - b).abs();
    assert!(
        diff <= tolerance,
        "{}: {} != {}, diff = {} (tolerance = {})",
        context, a, b, diff, tolerance
    );
}

/// Helper function to assert two matrices are approximately equal
pub fn assert_matrix_eq(a: &Matrix, b: &Matrix, tolerance: f64, context: &str) {
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
