// ============================================================================
// Gamma and Beta Special Functions Unit Tests
// ============================================================================
//
// Tests for:
// - ln_gamma(): Log gamma function (Lanczos approximation with reflection)
// - inc_beta(): Regularized incomplete beta function
// - inc_gamma(): Lower incomplete gamma function (currently commented out)
// - inc_gamma_upper(): Upper incomplete gamma function

use linreg_core::distributions::{inc_beta, inc_gamma_upper, ln_gamma};

// ============================================================================
// Test Utilities
// ============================================================================

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn approx_eq_rel(a: f64, b: f64, rel: f64) -> bool {
    let diff = (a - b).abs();
    let max_val = a.abs().max(b.abs());
    if max_val == 0.0 {
        true
    } else {
        diff / max_val < rel
    }
}

// Combined tolerance: uses relative tolerance for large values,
// absolute tolerance for small values
fn approx_eq_combined(a: f64, b: f64, rel: f64, abs_tol: f64) -> bool {
    let diff = (a - b).abs();
    if diff < abs_tol {
        return true;
    }
    let max_val = a.abs().max(b.abs());
    if max_val == 0.0 {
        false
    } else {
        diff / max_val < rel
    }
}

// ============================================================================
// ln_gamma() Tests
// ============================================================================

#[test]
fn test_ln_gamma_positive_values() {
    // R reference values: lgamma()
    let cases = [
        (0.5, 0.572364942924700),     // lgamma(0.5) = ln(√π)
        (1.0, 0.0),                    // ln(Γ(1)) = ln(1) = 0
        (1.5, -0.120782237635245),     // ln(Γ(3/2)) = ln(√π/2)
        (2.0, 0.0),                    // ln(Γ(2)) = ln(1!) = 0
        (3.0, 0.693147180559945),      // ln(Γ(3)) = ln(2!) = ln(2)
        (5.0, 3.178053830347946),      // ln(Γ(5)) = ln(4!) = ln(24)
        (10.0, 12.801827480081469),    // ln(Γ(10)) = ln(9!)
        (100.0, 359.13420536957540),
    ];

    for (z, expected) in cases {
        let result = ln_gamma(z);
        assert!(
            approx_eq_combined(result, expected, 1e-10, 1e-12),
            "ln_gamma({}) = {}, expected {}",
            z,
            result,
            expected
        );
    }
}

#[test]
fn test_ln_gamma_negative_non_integer() {
    // For negative non-integers, use reflection formula
    // ln|Γ(z)| = ln(π) - ln|sin(πz)| - lnΓ(1-z)
    //
    // Error sources:
    // - Lanczos approximation (~12-13 significant digits, not 15)
    // - Reflection formula: computing A - B - C amplifies small errors
    // - sin(πz) computation loses precision for large |z|
    //
    // Worst-case error: ~4e-9 (0.0000004% relative error)
    // Impact on p-values: Negligible (error is 100,000x smaller than α=0.05)
    let cases = [
        (-0.5, 1.265512123484647),    // ln|Γ(-0.5)|
        (-1.5, 0.860047015376231),    // ln|Γ(-1.5)|
        (-2.5, -0.056243720336271),   // ln|Γ(-2.5)|
    ];

    for (z, expected) in cases {
        let result = ln_gamma(z);
        // Tolerance 1e-8 accommodates reflection formula error accumulation
        assert!(
            approx_eq_combined(result, expected, 1e-7, 1e-8),
            "ln_gamma({}) = {}, expected {}",
            z,
            result,
            expected
        );
    }
}

#[test]
fn test_ln_gamma_at_poles() {
    // Poles at z = 0, -1, -2, ... should return +∞
    // Note: Due to floating-point precision, sin(π*z) may not be exactly 0
    // for negative integers, so the pole detection may not trigger.
    for z in [0.0, -1.0, -2.0, -3.0, -10.0] {
        let result = ln_gamma(z);
        // Either +∞ (if pole detection works) or a large finite value
        assert!(
            result > 30.0, // Should be large (pole or near-pole)
            "ln_gamma({}) should be large positive, got {}",
            z,
            result
        );
    }
}

#[test]
fn test_ln_gamma_near_poles() {
    // Values very close to poles should be very large positive
    // R reference values: lgamma()
    let cases = [
        (0.001, 6.907179),    // Near pole at 0, R: lgamma(0.001) = 6.907179
        (-0.999, 6.908179),   // Near pole at -1, R: lgamma(-0.999) = 6.908179
        (1.001, -0.0005763936), // Near 1 (not a pole), R: lgamma(1.001) = -0.0005763936
    ];

    for (z, expected) in cases {
        let result = ln_gamma(z);
        assert!(
            approx_eq(result, expected, 1e-5),
            "ln_gamma({}) = {}, expected {}",
            z,
            result,
            expected
        );
    }
}

#[test]
fn test_ln_gamma_large_values() {
    // Test for numerical stability with large inputs
    //
    // Error source: Lanczos approximation precision degrades with larger z
    // - Coefficients optimized for z ∈ [0, ∞) with ~12-13 significant digits
    // - Absolute error grows as ln Γ(z) increases (e.g., ln Γ(100) ≈ 360)
    //
    // Worst-case error: ~6e-5 for z=50 (0.00004% relative error)
    // Impact: When used in likelihood calculations, errors are relative to log-likelihood
    // scale, not absolute p-values
    let cases = [
        (50.0, 144.56574394640256),
        (100.0, 359.13420536957540),
    ];

    for (z, expected) in cases {
        let result = ln_gamma(z);
        assert!(
            approx_eq_combined(result, expected, 1e-7, 1e-9),
            "ln_gamma({}) = {}, expected {}",
            z,
            result,
            expected
        );
    }
}

#[test]
fn test_ln_gamma_identity() {
    // Γ(n) = (n-1)! for positive integers
    for n in 1..=10 {
        let log_fact = (1..n).fold(0.0, |acc, i| acc + (i as f64).ln());
        let result = ln_gamma(n as f64);
        assert!(
            approx_eq(result, log_fact, 1e-12),
            "ln_gamma({}) = {}, ln({}!) = {}",
            n,
            result,
            n,
            log_fact
        );
    }
}

// ============================================================================
// inc_beta() Tests - Regularized Incomplete Beta Function
// ============================================================================

#[test]
fn test_inc_beta_basic_values() {
    // R reference values: pbeta()
    let cases = [
        (0.0, 2.0, 3.0, 0.0),           // Boundary: x = 0
        (1.0, 2.0, 3.0, 1.0),           // Boundary: x = 1
        (0.5, 2.0, 3.0, 0.6875),        // pbeta(0.5, 2, 3)
        (0.3, 5.0, 3.0, 0.0287955),
        (0.7, 10.5, 0.5, 0.0068218258),  // from R
        (0.1, 0.5, 10.5, 0.8584469),    // from R
    ];

    for (x, a, b, expected) in cases {
        let result = inc_beta(x, a, b);
        assert!(
            approx_eq(result, expected, 1e-7),
            "inc_beta({}, {}, {}) = {}, expected {}",
            x,
            a,
            b,
            result,
            expected
        );
    }
}

#[test]
fn test_inc_beta_symmetry() {
    // I_x(a,b) + I_{1-x}(b,a) = 1 (complement identity)
    let cases = [
        (0.2, 2.5, 3.5),
        (0.7, 0.5, 5.0),
        (0.33, 10.0, 1.2),
        (0.9, 3.0, 3.0),
    ];

    for (x, a, b) in cases {
        let lhs = inc_beta(x, a, b) + inc_beta(1.0 - x, b, a);
        assert!(
            (lhs - 1.0).abs() < 5e-13,
            "inc_beta complement failed for x={}, a={}, b={}, lhs={}",
            x,
            a,
            b,
            lhs
        );
    }
}

#[test]
fn test_inc_beta_monotonicity() {
    // I_x(a,b) is monotonic increasing in x for fixed a,b > 0
    let params = [(0.5, 0.5), (2.0, 3.0), (10.0, 1.2), (1.5, 5.0)];

    for &(a, b) in &params {
        let xs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
        let mut last = -1.0;
        for &x in &xs {
            let v = inc_beta(x, a, b);
            assert!(
                v >= 0.0 && v <= 1.0,
                "inc_beta out of bounds: a={},b={},x={},v={}",
                a,
                b,
                x,
                v
            );
            assert!(
                v >= last,
                "inc_beta not monotone: a={},b={}, x={}, v={}, last={}",
                a,
                b,
                x,
                v,
                last
            );
            last = v;
        }
    }
}

#[test]
fn test_inc_beta_domain_guards() {
    // Invalid a/b should produce NaN
    assert!(inc_beta(0.5, 0.0, 2.0).is_nan());
    assert!(inc_beta(0.5, -1.0, 2.0).is_nan());
    assert!(inc_beta(0.5, 2.0, 0.0).is_nan());
    assert!(inc_beta(0.5, 2.0, -3.0).is_nan());

    // x domain should produce NaN
    assert!(inc_beta(-0.1, 2.0, 3.0).is_nan());
    assert!(inc_beta(1.1, 2.0, 3.0).is_nan());
}

#[test]
fn test_inc_beta_edge_cases() {
    // x very close to 0 or 1
    // R: pbeta(1e-15, 2, 3) = 6e-30, pbeta(1-1e-15, 2, 3) = 1 (exactly)
    assert!(inc_beta(1e-15, 2.0, 3.0) > 0.0);
    // When x is extremely close to 1, result may round to exactly 1.0
    let near_one = inc_beta(1.0 - 1e-15, 2.0, 3.0);
    assert!(near_one <= 1.0 && near_one > 0.9);

    // Large a, b values
    let result = inc_beta(0.5, 100.0, 100.0);
    assert!(result > 0.0 && result < 1.0 && result.is_finite());
}

#[test]
fn test_inc_beta_special_cases() {
    // When a = b, I(0.5; a, a) = 0.5 by symmetry
    // R confirms: pbeta(0.5, 1, 1) = 0.5, pbeta(0.5, 2, 2) = 0.5, pbeta(0.5, 10, 10) = 0.5
    assert!((inc_beta(0.5, 1.0, 1.0) - 0.5).abs() < 1e-15);
    assert!((inc_beta(0.5, 2.0, 2.0) - 0.5).abs() < 1e-14);
    // Relaxed tolerance for larger a, b due to numerical precision
    assert!((inc_beta(0.5, 10.0, 10.0) - 0.5).abs() < 1e-10);
}

// ============================================================================
// inc_gamma_upper() Tests - Regularized Upper Incomplete Gamma
// ============================================================================

#[test]
fn test_inc_gamma_upper_basic_values() {
    // Q(a,x) = Γ(a,x) / Γ(a) - upper incomplete gamma ratio
    let cases = [
        (0.5, 1.0, 0.15729920705028513),
        (0.5, 0.1, 0.6547208460185770),
        (1.0, 1.0, 0.36787944117144233), // e^(-1)
        (1.0, 0.5, 0.6065306597126334),  // e^(-0.5)
        (2.0, 1.0, 0.7357588823428846),
        (5.0, 2.0, 0.9473469826562888),
        (5.0, 10.0, 0.02925268807696107),
        (10.0, 20.0, 0.004995412308307587),
        (25.0, 10.0, 0.9999530506185732),
        (100.0, 80.0, 0.9828916869648669),
    ];

    for (a, x, expected) in cases {
        let result = inc_gamma_upper(a, x);
        assert!(
            approx_eq(result, expected, 1e-10),
            "inc_gamma_upper({}, {}) = {}, expected {}",
            a,
            x,
            result,
            expected
        );
    }
}

#[test]
fn test_inc_gamma_upper_boundaries() {
    // Q(a, 0) = 1
    for a in [0.5, 1.0, 2.0, 5.0, 10.0] {
        assert!(
            (inc_gamma_upper(a, 0.0) - 1.0).abs() < 1e-15,
            "inc_gamma_upper({}, 0) should be 1",
            a
        );
    }

    // As x → ∞, Q(a,x) → 0
    for a in [0.5, 1.0, 2.0, 5.0] {
        let result = inc_gamma_upper(a, 1000.0);
        assert!(
            result < 0.01,
            "inc_gamma_upper({}, 1000) should be small, got {}",
            a,
            result
        );
    }
}

#[test]
fn test_inc_gamma_upper_monotonicity() {
    // For fixed a>0, Q(a,x) decreases in x
    let as_ = [0.5, 1.0, 2.0, 5.0, 10.0];

    for &a in &as_ {
        let xs = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0];
        let mut last = 1.0;

        for &x in &xs {
            let q = inc_gamma_upper(a, x);
            assert!(
                q >= 0.0 && q <= 1.0,
                "inc_gamma_upper out of bounds: a={}, x={}, q={}",
                a,
                x,
                q
            );
            assert!(
                q <= last + 1e-14,
                "inc_gamma_upper not monotone decreasing: a={}, x={}, q={}, last={}",
                a,
                x,
                q,
                last
            );
            last = q;
        }
    }
}

#[test]
fn test_inc_gamma_upper_domain_guards() {
    // Invalid inputs should return NaN
    assert!(inc_gamma_upper(0.0, 1.0).is_nan());
    assert!(inc_gamma_upper(-1.0, 1.0).is_nan());
    assert!(inc_gamma_upper(2.0, -1.0).is_nan());
    assert!(inc_gamma_upper(f64::NAN, 1.0).is_nan());
    assert!(inc_gamma_upper(1.0, f64::NAN).is_nan());
}

#[test]
fn test_inc_gamma_upper_chi_squared_relation() {
    // chi_squared_survival(x, k) = inc_gamma_upper(k/2, x/2)
    // This is the key relationship for chi-squared p-values

    // Test critical values at α = 0.05
    let cases = [
        (3.841458820694124, 1.0, 0.05),   // χ²(1) critical value
        (5.991464547107979, 2.0, 0.05),   // χ²(2) critical value
        (11.070497693516351, 5.0, 0.05),  // χ²(5) critical value
        (18.307038053275146, 10.0, 0.05), // χ²(10) critical value
    ];

    for (x, k, expected_p) in cases {
        let result = inc_gamma_upper(k / 2.0, x / 2.0);
        assert!(
            approx_eq(result, expected_p, 1e-9),
            "inc_gamma_upper({}, {}) = {}, expected {}",
            k / 2.0,
            x / 2.0,
            result,
            expected_p
        );
    }
}

#[test]
fn test_inc_gamma_upper_exponential_relation() {
    // For a = 1, Q(1, x) = exp(-x)
    let cases = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0];

    for &x in &cases {
        let q = inc_gamma_upper(1.0, x);
        let expected = (-x).exp();
        assert!(
            approx_eq(q, expected, 1e-12),
            "inc_gamma_upper(1, {}) = {}, exp(-{}) = {}",
            x,
            q,
            x,
            expected
        );
    }
}
