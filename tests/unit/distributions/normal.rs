// ============================================================================
// Normal Distribution Functions Unit Tests
// ============================================================================
//
// Tests for:
// - normal_cdf(): Abramowitz & Stegun approximation
// - normal_cdf_cephes(): Cephes-style high-precision CDF
// - normal_sf_cephes(): Survival function (upper tail)
// - normal_inverse_cdf(): Probit function (quantile)
// - cephes_erf(): Error function
// - cephes_erfc(): Complementary error function
// - cephes_erfce(): Exponentially scaled erfc

use linreg_core::distributions::{
    cephes_erf, cephes_erfc, cephes_erfce, normal_cdf, normal_cdf_cephes,
    normal_inverse_cdf, normal_sf_cephes,
};

// ============================================================================
// Test Utilities
// ============================================================================

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn phi(z: f64) -> f64 {
    // Standard normal PDF: φ(z) = exp(-z²/2) / √(2π)
    (-0.5 * z * z).exp() / 2.5066282746310005
}

// ============================================================================
// normal_cdf() Tests - Abramowitz & Stegun approximation
// ============================================================================

#[test]
fn test_normal_cdf_known_values() {
    // Standard normal CDF: Φ(z)
    // Known values from standard normal tables
    let cases = [
        (0.0, 0.5),
        (0.5, 0.691462461274013),
        (1.0, 0.841344746068543),
        (1.96, 0.975002104851780),  // Close to 0.975
        (2.0, 0.977249868051821),
        (2.58, 0.995060),
        (3.0, 0.998650101968370),
        (-0.5, 0.308537538725987),
        (-1.0, 0.158655253931457),
        (-1.96, 0.024997895148220),
        (-2.0, 0.022750131948179),
        (-3.0, 0.001349898031630),
    ];

    for (z, expected) in cases {
        let result = normal_cdf(z);
        assert!(
            approx_eq(result, expected, 1e-5),
            "normal_cdf({}) = {}, expected {}",
            z,
            result,
            expected
        );
    }
}

#[test]
fn test_normal_cdf_symmetry() {
    // Φ(-z) = 1 - Φ(z)
    for z in [0.5, 1.0, 1.96, 2.0, 3.0, 5.0] {
        let p_pos = normal_cdf(z);
        let p_neg = normal_cdf(-z);
        let sum = p_pos + p_neg;
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "normal_cdf symmetry failed at z={}: Φ(z)={}, Φ(-z)={}, sum={}",
            z,
            p_pos,
            p_neg,
            sum
        );
    }
}

#[test]
fn test_normal_cdf_monotonicity() {
    let zs = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0];
    let mut last = 0.0;

    for &z in &zs {
        let p = normal_cdf(z);
        assert!(
            p >= 0.0 && p <= 1.0,
            "normal_cdf out of bounds: z={}, p={}",
            z,
            p
        );
        assert!(
            p >= last,
            "normal_cdf not monotone: z={}, p={}, last={}",
            z,
            p,
            last
        );
        last = p;
    }
}

// ============================================================================
// normal_cdf_cephes() Tests - High-precision Cephes implementation
// ============================================================================

#[test]
fn test_normal_cdf_cephes_known_values() {
    // Reference values with high precision
    let cases = [
        (0.0, 0.5),
        (1.0, 0.8413447460685429),
        (-1.0, 0.15865525393145705),
        (2.0, 0.9772498680518208),
        (-2.0, 0.022750131948179207),
        (3.5, 0.9997673709209645),
        (-3.5, 0.00023262907903552504),
        (6.0, 0.9999999990134124),
        (-6.0, 0.0000000009865876450376981),
    ];

    for (z, expected) in cases {
        let result = normal_cdf_cephes(z);
        assert!(
            approx_eq(result, expected, 1e-14),
            "normal_cdf_cephes({}) = {}, expected {}",
            z,
            result,
            expected
        );
    }
}

#[test]
fn test_normal_cdf_cephes_symmetry() {
    let zs = [-10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0];

    for &z in &zs {
        let p = normal_cdf_cephes(z);
        assert!(
            p >= 0.0 && p <= 1.0 && p.is_finite(),
            "CDF out of bounds: z={}, p={}",
            z,
            p
        );

        // Symmetry: Phi(-z) = 1 - Phi(z)
        let p_neg = normal_cdf_cephes(-z);
        assert!(
            (p_neg - (1.0 - p)).abs() < 1e-14,
            "CDF symmetry failed: z={}, Phi(z)={}, Phi(-z)={}, sum={}",
            z,
            p,
            p_neg,
            p + p_neg
        );
    }

    // Exact-ish: Phi(0)=0.5
    assert!((normal_cdf_cephes(0.0) - 0.5).abs() < 1e-15);
}

#[test]
fn test_normal_cdf_cephes_bounds_and_monotone() {
    let zs = [-10.0, -8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0];
    let mut last = 0.0;

    for &z in &zs {
        let p = normal_cdf_cephes(z);
        assert!(p.is_finite());
        assert!(p >= 0.0 && p <= 1.0, "Phi out of bounds at z={}: {}", z, p);
        assert!(
            p >= last,
            "Phi not monotone: z={}, p={}, last={}",
            z,
            p,
            last
        );
        last = p;
    }
}

#[test]
fn test_normal_cdf_cephes_infinity() {
    assert_eq!(normal_cdf_cephes(f64::INFINITY), 1.0);
    assert_eq!(normal_cdf_cephes(f64::NEG_INFINITY), 0.0);
    assert!(normal_cdf_cephes(f64::NAN).is_nan());
}

// ============================================================================
// normal_sf_cephes() Tests - Survival function
// ============================================================================

#[test]
fn test_normal_sf_cephes_complement() {
    // Q(z) = 1 - Phi(z) = Phi(-z) for z >= 0
    for &z in &[-10.0, -6.0, -2.0, 0.0, 2.0, 6.0, 10.0] {
        let p = normal_cdf_cephes(z);
        let q = normal_sf_cephes(z);
        assert!(
            (p + q - 1.0).abs() < 1e-14,
            "z={}, p={}, q={}, sum={}",
            z,
            p,
            q,
            p + q
        );
    }
}

#[test]
fn test_normal_sf_cephes_known_values() {
    let cases = [
        (0.0, 0.5),
        (1.0, 0.15865525393145705),
        (1.96, 0.024997895148220),
        (2.0, 0.022750131948179),
        (3.0, 0.001349898031630),
    ];

    for (z, expected) in cases {
        let result = normal_sf_cephes(z);
        assert!(
            approx_eq(result, expected, 1e-12),
            "normal_sf_cephes({}) = {}, expected {}",
            z,
            result,
            expected
        );
    }
}

#[test]
fn test_normal_sf_cephes_infinity() {
    assert_eq!(normal_sf_cephes(f64::INFINITY), 0.0);
    assert_eq!(normal_sf_cephes(f64::NEG_INFINITY), 1.0);
    assert!(normal_sf_cephes(f64::NAN).is_nan());
}

// ============================================================================
// normal_inverse_cdf() Tests - Probit function
// ============================================================================

#[test]
fn test_normal_inverse_cdf_standard_quantiles() {
    // Standard normal quantiles (z-scores for common probabilities)
    let cases = [
        (0.5, 0.0),
        (0.025, -1.959963984540054),
        (0.975, 1.959963984540054),
        (0.05, -1.644853626951472),
        (0.95, 1.644853626951472),
        (0.10, -1.281551565544600),
        (0.90, 1.281551565544600),
        (0.01, -2.326347874040841),
        (0.99, 2.326347874040841),
    ];

    for (p, expected) in cases {
        let result = normal_inverse_cdf(p);
        assert!(
            approx_eq(result, expected, 1e-12),
            "normal_inverse_cdf({}) = {}, expected {}",
            p,
            result,
            expected
        );
    }
}

#[test]
fn test_normal_inverse_cdf_roundtrip() {
    // Inv(CDF(z)) ≈ z
    let ps = [
        1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9,
        0.99, 1.0 - 1e-3, 1.0 - 1e-4, 1.0 - 1e-6, 1.0 - 1e-8, 1.0 - 1e-10,
        1.0 - 1e-12,
    ];

    for &p in &ps {
        let z = normal_inverse_cdf(p);
        let p2 = normal_cdf_cephes(z);
        let diff = (p2 - p).abs();

        assert!(
            diff <= 1e-11,
            "roundtrip failed: p={}, z={}, p2={}, diff={}",
            p,
            z,
            p2,
            diff
        );
    }
}

#[test]
fn test_normal_inverse_cdf_monotonicity_and_symmetry() {
    let ps = [
        1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.5,
        0.75, 0.9, 0.95, 0.99, 1.0 - 1e-3, 1.0 - 1e-6, 1.0 - 1e-8, 1.0 - 1e-10,
        1.0 - 1e-12,
    ];

    let mut last = f64::NEG_INFINITY;
    for &p in &ps {
        let z = normal_inverse_cdf(p);
        assert!(z.is_finite(), "inv({}) not finite: {}", p, z);
        assert!(
            z > last,
            "monotonicity failed: inv({})={} <= last={}",
            p,
            z,
            last
        );
        last = z;

        // Symmetry: inv(1-p) ≈ -inv(p)
        if p <= 0.5 {
            let z2 = normal_inverse_cdf(1.0 - p);
            let sum = z + z2;
            let pdf = phi(z.abs());
            let tol = if pdf > 0.0 {
                (1e-16 / pdf).max(1e-12)
            } else {
                1e-5
            };

            assert!(
                sum.abs() < tol,
                "symmetry failed: p={}, inv(p)={}, inv(1-p)={}, sum={}, tol={}",
                p,
                z,
                z2,
                sum,
                tol
            );
        }
    }

    // Endpoint behavior
    assert!(normal_inverse_cdf(0.0).is_infinite()
        && normal_inverse_cdf(0.0).is_sign_negative());
    assert!(normal_inverse_cdf(1.0).is_infinite()
        && normal_inverse_cdf(1.0).is_sign_positive());
}

#[test]
fn test_normal_inverse_cdf_domain() {
    assert!(normal_inverse_cdf(-0.1).is_infinite() && normal_inverse_cdf(-0.1).is_sign_negative());
    assert!(normal_inverse_cdf(1.1).is_infinite() && normal_inverse_cdf(1.1).is_sign_positive());
    assert!(normal_inverse_cdf(f64::NAN).is_nan());
}

// ============================================================================
// cephes_erf() Tests - Error function
// ============================================================================

#[test]
fn test_cephes_erf_known_values() {
    // erf(x) = (2/√π) ∫[0,x] exp(-t²) dt
    let cases = [
        (0.0, 0.0),
        (0.5, 0.5204998778130465),
        (1.0, 0.8427007929497149),
        (1.5, 0.9661051464753107),
        (2.0, 0.9953222650189527),
        (3.0, 0.9999779095030014),
    ];

    for (x, expected) in cases {
        let result = cephes_erf(x);
        assert!(
            approx_eq(result, expected, 1e-12),
            "cephes_erf({}) = {}, expected {}",
            x,
            result,
            expected
        );
    }
}

#[test]
fn test_cephes_erf_symmetry() {
    // erf(-x) = -erf(x)
    for x in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0] {
        let erf_pos = cephes_erf(x);
        let erf_neg = cephes_erf(-x);
        assert!(
            (erf_neg + erf_pos).abs() < 1e-14,
            "erf symmetry failed at x={}: erf(x)={}, erf(-x)={}",
            x,
            erf_pos,
            erf_neg
        );
    }
}

#[test]
fn test_cephes_erf_bounds() {
    // -1 <= erf(x) <= 1 for all real x
    for x in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let result = cephes_erf(x);
        assert!(result >= -1.0 && result <= 1.0);
    }

    // erf(∞) = 1, erf(-∞) = -1
    assert_eq!(cephes_erf(f64::INFINITY), 1.0);
    assert_eq!(cephes_erf(f64::NEG_INFINITY), -1.0);
}

// ============================================================================
// cephes_erfc() Tests - Complementary error function
// ============================================================================

#[test]
fn test_cephes_erfc_known_values() {
    // erfc(x) = 1 - erf(x)
    let cases = [
        (0.0, 1.0),
        (0.5, 0.47950012218695346),
        (1.0, 0.15729920705028513),
        (1.5, 0.03389485352468927),
        (2.0, 0.004677734981047266),
        (3.0, 0.00002209049699858544),
    ];

    for (x, expected) in cases {
        let result = cephes_erfc(x);
        assert!(
            approx_eq(result, expected, 1e-12),
            "cephes_erfc({}) = {}, expected {}",
            x,
            result,
            expected
        );
    }
}

#[test]
fn test_cephes_erfc_identity() {
    // erf(x) + erfc(x) = 1
    for x in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0] {
        let erf_v = cephes_erf(x);
        let erfc_v = cephes_erfc(x);
        assert!(
            (erf_v + erfc_v - 1.0).abs() < 1e-14,
            "erf+erfc identity failed at x={}: erf={}, erfc={}, sum={}",
            x,
            erf_v,
            erfc_v,
            erf_v + erfc_v
        );
    }
}

#[test]
fn test_cephes_erfc_bounds() {
    // 0 <= erfc(x) <= 2 for all real x
    assert_eq!(cephes_erfc(0.0), 1.0);
    assert!(cephes_erfc(1.0) > 0.0 && cephes_erfc(1.0) < 1.0);
    assert!(cephes_erfc(-1.0) > 1.0 && cephes_erfc(-1.0) < 2.0);

    // erfc(∞) = 0, erfc(-∞) = 2
    assert_eq!(cephes_erfc(f64::INFINITY), 0.0);
    assert_eq!(cephes_erfc(f64::NEG_INFINITY), 2.0);
}

// ============================================================================
// cephes_erfce() Tests - Exponentially scaled erfc
// ============================================================================

#[test]
fn test_cephes_erfce_definition() {
    // erfce(x) = exp(x²) * erfc(x)
    let cases = [
        (0.5, 0.6156903441929259),
        (1.0, 0.4275835761558070),
        (1.5, 0.3215854164543175),
        (2.0, 0.25539567631050574),
        (3.0, 0.17900115118138995),
    ];

    for (x, expected) in cases {
        let result = cephes_erfce(x);
        assert!(
            approx_eq(result, expected, 1e-12),
            "cephes_erfce({}) = {}, expected {}",
            x,
            result,
            expected
        );

        // Also verify the definition
        let erfc_v = cephes_erfc(x);
        let expected_from_def = (x * x).exp() * erfc_v;
        assert!(
            approx_eq(result, expected_from_def, 1e-12),
            "cephes_erfce({}) definition failed",
            x
        );
    }
}

#[test]
fn test_cephes_erfce_large_values() {
    // For large x, erfce(x) should use the stable approximation
    for x in [8.0, 10.0, 20.0, 26.0] {
        let result = cephes_erfce(x);
        assert!(
            result > 0.0 && result.is_finite(),
            "cephes_erfce({}) = {}",
            x,
            result
        );

        // erfce should be bounded
        assert!(result < 1.0, "cephes_erfce({}) = {} should be < 1", x, result);
    }
}

#[test]
fn test_cephes_erfce_zero() {
    // erfce(0) = exp(0) * erfc(0) = 1 * 1 = 1
    let result = cephes_erfce(0.0);
    assert!(approx_eq(result, 1.0, 1e-15));
}

// ============================================================================
// Combined identity tests
// ============================================================================

#[test]
fn test_normal_cdf_roundtrip_consistency() {
    // Both normal_cdf variants should give similar results
    for z in [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0] {
        let p_a_s = normal_cdf(z);
        let p_cephes = normal_cdf_cephes(z);

        // The A&S approximation is slightly less accurate but close
        assert!(
            (p_a_s - p_cephes).abs() < 1e-4,
            "normal_cdf variants differ at z={}: A&S={}, Cephes={}",
            z,
            p_a_s,
            p_cephes
        );
    }
}

#[test]
#[ignore]  // Algorithm mismatch: normal_cdf (A&S) vs cephes_erf (Cephes)
fn test_erf_normal_cdf_relation() {
    // Φ(z) = 0.5 * (1 + erf(z/√2))
    //
    // NOTE: This test is ignored because:
    // - normal_cdf() uses Abramowitz & Stegun approximation (~1e-5 accuracy in tails)
    // - cephes_erf() uses Cephes-style implementation (~1e-15 accuracy)
    // - The two algorithms are independently implemented, not derived from each other
    //
    // For production use, prefer normal_cdf_cephes() which matches cephes_erf().
    for z in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
        let erf_val = cephes_erf(z / 2.0_f64.sqrt());
        let phi_from_erf = 0.5 * (1.0 + erf_val);
        let phi_direct = normal_cdf(z);

        assert!(
            (phi_from_erf - phi_direct).abs() < 1e-12,
            "erf relation failed at z={}",
            z
        );
    }
}
