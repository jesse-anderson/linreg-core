//! Weight functions for LOESS
//!
//! Contains the tricube weight function and related utilities.

/// Small epsilon for floating point comparisons
pub const EPSILON: f64 = 1e-7;

/// Outer boundary factor for weight computation (99.9% of bandwidth)
pub const OUTER_BOUNDARY_FACTOR: f64 = 0.999;

/// Inner boundary factor for weight computation (0.1% of bandwidth)
pub const INNER_BOUNDARY_FACTOR: f64 = 0.001;

/// Minimum neighborhood size for degree 2 (quadratic) fits
pub const MIN_NEIGHBORS_QUADRATIC: usize = 3;

/// Tricube weight function
///
/// Computes weight for a normalized distance u using the tricube kernel:
/// `w(u) = (1 - |u|³)³` for `|u| < 1`, and `w(u) = 0` for `|u| >= 1`.
///
/// The tricube kernel is a symmetric, non-negative weighting function that
/// gives maximum weight (1.0) at u=0 and smoothly decreases to 0 at |u|=1.
///
/// # Arguments
///
/// * `u` - Normalized distance (typically 0 to 1, but can be negative)
///
/// # Returns
///
/// Weight value in the range [0, 1]. Returns 0 for |u| >= 1.
///
/// # Examples
///
/// ```
/// use linreg_core::loess::tricube_weight;
///
/// assert_eq!(tricube_weight(0.0), 1.0);
/// assert!((tricube_weight(0.5) - 0.669921875).abs() < 1e-10);
/// assert_eq!(tricube_weight(1.0), 0.0);
/// assert_eq!(tricube_weight(-0.5), tricube_weight(0.5)); // symmetric
/// ```
#[inline]
pub fn tricube_weight(u: f64) -> f64 {
    let abs_u = u.abs();
    if abs_u >= 1.0 {
        0.0
    } else {
        let inner = 1.0 - abs_u.powi(3);
        inner.powi(3)
    }
}

/// Compute tricube weight with boundary handling
///
/// Similar to `tricube_weight` but with special handling for numerical stability
/// near zero. Uses inner/outer boundary thresholds.
///
/// # Arguments
///
/// * `r` - Raw distance
/// * `h` - Bandwidth
///
/// # Returns
///
/// Weight value in the range [0, 1].
#[inline]
pub fn tricube_weight_with_bounds(r: f64, h: f64) -> f64 {
    if h <= 0.0 {
        return 1.0; // All points are at the same location
    }

    let outer_boundary = OUTER_BOUNDARY_FACTOR * h;
    let inner_boundary = INNER_BOUNDARY_FACTOR * h;

    if r > outer_boundary {
        return 0.0;
    }

    if r <= inner_boundary {
        // Near zero distance - return 1.0 for numerical stability
        1.0
    } else {
        // Apply tricube formula
        let u = r / h;
        let inner = 1.0 - u.powi(3);
        inner.powi(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tricube_weight_zero() {
        // At distance 0, weight should be 1.0
        assert_eq!(tricube_weight(0.0), 1.0);
    }

    #[test]
    fn test_tricube_weight_half() {
        // w(0.5) = (1 - 0.5³)³ = (1 - 0.125)³ = 0.875³ = 0.669921875
        let result = tricube_weight(0.5);
        let expected = 0.669921875;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_tricube_weight_boundary() {
        // At |u| >= 1, weight should be 0
        assert_eq!(tricube_weight(1.0), 0.0);
        assert_eq!(tricube_weight(-1.0), 0.0);
        assert_eq!(tricube_weight(1.5), 0.0);
        assert_eq!(tricube_weight(-2.0), 0.0);
    }

    #[test]
    fn test_tricube_weight_symmetry() {
        // Function should be symmetric: w(-u) = w(u)
        assert_eq!(tricube_weight(-0.3), tricube_weight(0.3));
        assert_eq!(tricube_weight(-0.7), tricube_weight(0.7));
        assert_eq!(tricube_weight(-0.5), tricube_weight(0.5));
    }

    #[test]
    fn test_tricube_weight_range() {
        // All weights should be in [0, 1]
        for i in 0..=100 {
            let u = i as f64 / 100.0; // 0.0 to 1.0
            let w = tricube_weight(u);
            assert!(w >= 0.0 && w <= 1.0, "Weight at {} = {} is out of [0,1]", u, w);
        }
    }

    #[test]
    fn test_tricube_weight_monotonic() {
        // Weight should decrease as distance increases from 0 to 1
        let mut prev_weight = tricube_weight(0.0);
        for i in 1..=100 {
            let u = i as f64 / 100.0;
            let weight = tricube_weight(u);
            assert!(
                weight <= prev_weight,
                "Weight at {} = {} should be <= {}",
                u,
                weight,
                prev_weight
            );
            prev_weight = weight;
        }
    }

    #[test]
    fn test_tricube_weight_with_bounds() {
        // Test the boundary handling version
        // At very small distance, should return 1.0
        assert_eq!(tricube_weight_with_bounds(0.0, 1.0), 1.0);

        // At distance just above inner boundary, should use tricube formula
        let h = 1.0;
        let inner_boundary = INNER_BOUNDARY_FACTOR * h;
        let r = inner_boundary * 1.1; // Just above inner boundary
        let w = tricube_weight_with_bounds(r, h);
        assert!(w > 0.0 && w < 1.0);

        // At distance just below outer boundary, should use tricube formula
        let outer_boundary = OUTER_BOUNDARY_FACTOR * h;
        let r = outer_boundary * 0.9; // Just below outer boundary
        let w = tricube_weight_with_bounds(r, h);
        assert!(w > 0.0 && w < 1.0);

        // At distance above outer boundary, should return 0
        let r = outer_boundary * 1.1; // Just above outer boundary
        let w = tricube_weight_with_bounds(r, h);
        assert_eq!(w, 0.0);

        // Zero bandwidth should return 1.0
        let w = tricube_weight_with_bounds(0.5, 0.0);
        assert_eq!(w, 1.0);
    }
}
