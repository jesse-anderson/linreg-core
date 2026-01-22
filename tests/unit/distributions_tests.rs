// ============================================================================
// Statistical Distributions Unit Tests
// ============================================================================
//
// Tests compare against a mixture of:
// - R reference values (pt/pf/pchisq/qt where applicable),
// - known high-precision constants,
// - and identity/monotonicity properties for sanity and regression safety.

use linreg_core::core::{f_p_value, two_tailed_p_value};
use linreg_core::distributions::{
    cephes_erf, cephes_erfc, cephes_erfce, chi_squared_survival, fisher_snedecor_cdf, inc_beta,
    inc_gamma_upper, ln_gamma, normal_cdf_cephes, normal_inverse_cdf, normal_sf_cephes,
    student_t_cdf, student_t_inverse_cdf,
};

#[allow(dead_code)]
const TOL: f64 = 1e-8; // General tolerance (reserved for future use)
const TOL_LOOSE: f64 = 1e-6; // For iterative methods

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn approx_eq_rel_abs(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    let diff = (a - b).abs();
    diff <= abs.max(rel * a.abs().max(b.abs()))
}

// ========================================================================
// Log Gamma Tests (R: lgamma)
// ========================================================================
#[test]
fn test_ln_gamma() {
    let cases = [
        (0.5, 0.572364942924700),
        (1.0, 0.000000000000000),
        (1.5, -0.120782237635245),
        (2.0, 0.000000000000000),
        (5.0, 3.178053830347946),
        (10.0, 12.801827480081469),
        (11.5, 16.292000476567242),
        (100.0, 359.134205369575398),
    ];

    for (z, expected) in cases {
        let result = ln_gamma(z);
        assert!(
            approx_eq(result, expected, TOL_LOOSE),
            "ln_gamma({}) = {}, expected {}",
            z,
            result,
            expected
        );
    }
}

// ========================================================================
// Incomplete Beta Tests (R: pbeta)
// ========================================================================
#[test]
fn test_inc_beta() {
    let cases = [
        (0.0, 2.0, 3.0, 0.0),
        (0.5, 2.0, 3.0, 0.6875),
        (1.0, 2.0, 3.0, 1.0),
        (0.3, 5.0, 3.0, 0.0287955),
        (0.7, 10.5, 0.5, 0.006821825813971),
        (0.1, 0.5, 10.5, 0.858446908187113),
    ];

    for (x, a, b, expected) in cases {
        let result = inc_beta(x, a, b);
        assert!(
            approx_eq(result, expected, TOL_LOOSE),
            "inc_beta({}, {}, {}) = {}, expected {}",
            x,
            a,
            b,
            result,
            expected
        );
    }
}

// ========================================================================
// Student's t CDF Tests (R: pt)
// ========================================================================
#[test]
fn test_student_t_cdf() {
    let cases = [
        (0.0, 10.0, 0.5),
        (1.0, 10.0, 0.829553433848970),
        (-1.0, 10.0, 0.170446566151030),
        (2.0, 21.0, 0.970699993975645),
        (-2.0, 21.0, 0.029300006024355),
        (1.67, 21.0, 0.945120486597924),
        (-1.67, 21.0, 0.054879513402076),
        (0.5, 5.0, 0.680850564179535),
        (10.0, 21.0, 0.999999999031288),
        (-10.0, 21.0, 0.000000000968712),
    ];

    for (t, df, expected) in cases {
        let result = student_t_cdf(t, df);
        assert!(
            approx_eq(result, expected, TOL_LOOSE),
            "student_t_cdf({}, {}) = {}, expected {}",
            t,
            df,
            result,
            expected
        );
    }
}

// ========================================================================
// Two-tailed p-value Tests (R: 2 * pt(-|t|, df))
// ========================================================================
#[test]
fn test_two_tailed_p_value() {
    let cases = [
        (0.0, 10.0, 1.0),
        (1.0, 10.0, 0.340893132302060),
        (-1.0, 10.0, 0.340893132302060),
        (2.0, 21.0, 0.058600012048710),
        (-2.0, 21.0, 0.058600012048710),
        (1.67, 21.0, 0.109759026804152),
        (-1.67, 21.0, 0.109759026804152),
        (0.5, 5.0, 0.638298871640929),
        (10.0, 21.0, 0.000000001937424),
    ];

    for (t, df, expected) in cases {
        let result = two_tailed_p_value(t, df);
        assert!(
            approx_eq(result, expected, TOL_LOOSE),
            "two_tailed_p_value({}, {}) = {}, expected {}",
            t,
            df,
            result,
            expected
        );
    }
}

// ========================================================================
// F-distribution CDF Tests (R: pf)
// ========================================================================
#[test]
fn test_fisher_snedecor_cdf() {
    let cases = [
        (0.0, 3.0, 21.0, 0.0),
        (1.0, 3.0, 21.0, 0.587715322376118),
        (2.0, 3.0, 21.0, 0.855144210917774),
        (5.0, 3.0, 21.0, 0.990993683132052),
        (10.0, 3.0, 21.0, 0.999731241389630),
        (100.0, 3.0, 21.0, 0.999999999998652),
    ];

    for (f, df1, df2, expected) in cases {
        let result = fisher_snedecor_cdf(f, df1, df2);
        assert!(
            approx_eq(result, expected, TOL_LOOSE),
            "fisher_snedecor_cdf({}, {}, {}) = {}, expected {}",
            f,
            df1,
            df2,
            result,
            expected
        );
    }
}

// ========================================================================
// F p-value Tests (R: pf(f, df1, df2, lower.tail=FALSE))
// ========================================================================
#[test]
fn test_f_p_value() {
    let cases = [
        (0.0, 3.0, 21.0, 1.0),
        (1.0, 3.0, 21.0, 0.412284677623881656849),
        (2.0, 3.0, 21.0, 0.144855789082225933084),
        (5.0, 3.0, 21.0, 0.00900631686794783198335),
        (10.0, 3.0, 21.0, 0.000268758610370509090203),
    ];

    for (f, df1, df2, expected) in cases {
        let result = f_p_value(f, df1, df2);
        assert!(
            approx_eq(result, expected, TOL_LOOSE),
            "f_p_value({}, {}, {}) = {}, expected {}",
            f,
            df1,
            df2,
            result,
            expected
        );
    }
}

// ========================================================================
// Chi-Squared Survival Tests (R: pchisq(x, df, lower.tail=FALSE))
// ========================================================================
#[test]
fn test_chi_squared_survival() {
    let cases = [
        (0.5, 1.0, 0.479500122186953),
        (1.0, 1.0, 0.317310507862914),
        (3.84, 1.0, 0.050043521248705), // Close to 0.05 alpha for df=1
        (2.0, 2.0, 0.367879441171442),
        (5.99, 2.0, 0.050036687364377), // Close to 0.05 alpha for df=2
        (10.0, 5.0, 0.075235246132234),
        (20.0, 10.0, 0.029252688076961),
    ];

    for (x, k, expected) in cases {
        let result = chi_squared_survival(x, k);
        assert!(
            approx_eq(result, expected, TOL_LOOSE),
            "chi_squared_survival({}, {}) = {}, expected {}",
            x,
            k,
            result,
            expected
        );
    }
}

// ========================================================================
// Inverse t-distribution Tests (R: qt)
// ========================================================================
#[test]
fn test_student_t_inverse_cdf() {
    let cases = [
        (0.025, 21.0, -2.079613844727680),
        (0.05, 21.0, -1.720742902811878),
        (0.1, 21.0, -1.323187873865172),
        (0.5, 21.0, 0.0),
        (0.9, 21.0, 1.323187873865172),
        (0.95, 21.0, 1.720742902811878),
        (0.975, 21.0, 2.079613844727680),
    ];

    for (p, df, expected) in cases {
        let result = student_t_inverse_cdf(p, df);
        assert!(
            approx_eq(result, expected, TOL_LOOSE),
            "student_t_inverse_cdf({}, {}) = {}, expected {}",
            p,
            df,
            result,
            expected
        );
    }
}

// ========================================================================
// Normal Inverse CDF Tests
// ========================================================================
#[test]
fn test_normal_inverse_cdf() {
    // Standard normal quantiles
    let cases = [
        (0.5, 0.0),
        (0.025, -1.959963984540054),
        (0.975, 1.959963984540054),
        (0.05, -1.644853626951472),
        (0.95, 1.644853626951472),
    ];

    for (p, expected) in cases {
        let result = normal_inverse_cdf(p);
        assert!(
            approx_eq(result, expected, TOL_LOOSE),
            "normal_inverse_cdf({}) = {}, expected {}",
            p,
            result,
            expected
        );
    }
}

// ========================================================================
// Edge Case Tests
// ========================================================================
#[test]
fn test_edge_cases() {
    // t-CDF at extreme values
    assert!(student_t_cdf(100.0, 21.0) > 0.9999);
    assert!(student_t_cdf(-100.0, 21.0) < 0.0001);

    // F-CDF at zero
    assert_eq!(fisher_snedecor_cdf(0.0, 3.0, 21.0), 0.0);

    // Inc beta at boundaries
    assert_eq!(inc_beta(0.0, 2.0, 3.0), 0.0);
    assert_eq!(inc_beta(1.0, 2.0, 3.0), 1.0);
}

// ========================================================================
// P-value Sanity Tests
// ========================================================================
#[test]
fn test_p_value_bounds() {
    // P-values must be in [0, 1]
    for t in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
        for df in [1.0, 5.0, 10.0, 21.0, 100.0] {
            let p = two_tailed_p_value(t, df);
            assert!(
                p >= 0.0 && p <= 1.0,
                "two_tailed_p_value({}, {}) = {} is out of bounds [0,1]",
                t,
                df,
                p
            );
        }
    }
}

// ========================================================================
// Incomplete Gamma Upper Tests (High-precision reference)
// Q(a,x) = Γ(a,x) / Γ(a)
// ========================================================================
#[test]
fn test_inc_gamma_upper() {
    let cases = [
        // (a, x, expected_Q)
        (0.5, 1.0, 0.15729920705028513),
        (0.5, 0.1, 0.6547208460185770),
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

// ========================================================================
// Chi-squared survival tests (critical values: survival ~ 0.05)
// ========================================================================
#[test]
fn test_chi_squared_survival_critical_values() {
    let cases = [
        // (x, k, expected_survival)
        (3.841458820694124, 1.0, 0.05),
        (5.991464547107979, 2.0, 0.05),
        (11.070497693516351, 5.0, 0.05),
        (18.307038053275146, 10.0, 0.05),
    ];

    for (x, k, expected) in cases {
        let result = chi_squared_survival(x, k);
        assert!(
            approx_eq(result, expected, 1e-8),
            "chi_squared_survival({}, {}) = {}, expected {}",
            x,
            k,
            result,
            expected
        );
    }
}

// ========================================================================
// Normal CDF (Cephes) tests against known Φ(z) values
// ========================================================================
#[test]
fn test_normal_cdf_cephes_known_values() {
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

// ========================================================================
// Cephes erf/erfc/erfce sanity tests against high-precision values
// ========================================================================
#[test]
fn test_cephes_erf_erfc_erfce() {
    let cases = [
        // x, erf(x), erfc(x), erfce(x)=exp(x^2)*erfc(x)
        (
            0.5,
            0.5204998778130465,
            0.47950012218695346,
            0.6156903441929259,
        ),
        (
            1.0,
            0.8427007929497149,
            0.15729920705028513,
            0.4275835761558070,
        ),
        (
            1.5,
            0.9661051464753107,
            0.03389485352468927,
            0.3215854164543175,
        ),
        (
            2.0,
            0.9953222650189527,
            0.004677734981047266,
            0.25539567631050574,
        ),
        (
            3.0,
            0.9999779095030014,
            0.00002209049699858544,
            0.17900115118138995,
        ),
    ];

    for (x, erf_expected, erfc_expected, erfce_expected) in cases {
        let erf_v = cephes_erf(x);
        let erfc_v = cephes_erfc(x);
        let erfce_v = cephes_erfce(x);

        assert!(
            approx_eq(erf_v, erf_expected, 1e-12),
            "cephes_erf({}) = {}, expected {}",
            x,
            erf_v,
            erf_expected
        );
        assert!(
            approx_eq(erfc_v, erfc_expected, 1e-12),
            "cephes_erfc({}) = {}, expected {}",
            x,
            erfc_v,
            erfc_expected
        );
        assert!(
            approx_eq(erfce_v, erfce_expected, 1e-12),
            "cephes_erfce({}) = {}, expected {}",
            x,
            erfce_v,
            erfce_expected
        );
    }
}

// ========================================================================
// inc_beta domain behavior tests
// ========================================================================
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
fn test_inc_beta_complement_identity() {
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
fn test_student_t_symmetry() {
    let cases = [(0.1, 1.0), (1.5, 2.0), (2.0, 10.0), (5.0, 30.0)];

    for (t, df) in cases {
        let p_pos = student_t_cdf(t, df);
        let p_neg = student_t_cdf(-t, df);
        assert!(
            (p_neg - (1.0 - p_pos)).abs() < 5e-13,
            "t symmetry failed: t={}, df={}, p_pos={}, p_neg={}",
            t,
            df,
            p_pos,
            p_neg
        );
    }

    // t=0 should always be 0.5
    assert!((student_t_cdf(0.0, 1.0) - 0.5).abs() < 1e-15);
    assert!((student_t_cdf(0.0, 10.0) - 0.5).abs() < 1e-15);
}

#[test]
fn test_normal_inverse_cdf_roundtrip() {
    let ps = [
        1e-12,
        1e-10,
        1e-8,
        1e-6,
        1e-4,
        1e-3,
        0.01,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        0.99,
        1.0 - 1e-3,
        1.0 - 1e-4,
        1.0 - 1e-6,
        1.0 - 1e-8,
        1.0 - 1e-10,
        1.0 - 1e-12,
    ];

    const ABS_TOL_P: f64 = 1e-11;

    for &p in &ps {
        let z = normal_inverse_cdf(p);
        let p2 = normal_cdf_cephes(z);
        let diff = (p2 - p).abs();

        assert!(
            diff <= ABS_TOL_P,
            "roundtrip failed: p={}, z={}, p2={}, diff={}",
            p,
            z,
            p2,
            diff
        );
    }
}

#[test]
fn test_normal_inverse_cdf_z_roundtrip() {
    let ps = [
        1e-12,
        1e-10,
        1e-8,
        1e-6,
        1e-4,
        1e-3,
        0.01,
        0.1,
        0.5,
        0.9,
        0.99,
        1.0 - 1e-6,
        1.0 - 1e-10,
        1.0 - 1e-12,
    ];

    for &p in &ps {
        let z = normal_inverse_cdf(p);
        let p2 = normal_cdf_cephes(z);
        let z2 = normal_inverse_cdf(p2);
        let dz = (z2 - z).abs();

        let az = z.abs();
        let dz_tol = if az <= 3.0 {
            1e-10
        } else if az <= 4.0 {
            5e-9
        } else if az <= 8.0 {
            2e-8
        } else {
            1e-6
        };

        assert!(
            dz <= dz_tol,
            "z-roundtrip failed: p={}, z={}, p2={}, z2={}, dz={}, tol={}",
            p,
            z,
            p2,
            z2,
            dz,
            dz_tol
        );
    }
}

#[test]
fn test_distribution_domain_guards() {
    assert!(student_t_cdf(1.0, 0.0).is_nan());
    assert!(student_t_cdf(f64::NAN, 10.0).is_nan());
    assert!(student_t_cdf(1.0, f64::INFINITY).is_nan());

    assert!(fisher_snedecor_cdf(1.0, 0.0, 10.0).is_nan());
    assert!(fisher_snedecor_cdf(1.0, 10.0, -2.0).is_nan());

    assert!(chi_squared_survival(1.0, 0.0).is_nan());
    assert!(chi_squared_survival(f64::NAN, 2.0).is_nan());

    assert!(inc_gamma_upper(0.0, 1.0).is_nan());
    assert!(inc_gamma_upper(2.0, -1.0).is_nan());
}

#[test]
fn test_normal_inverse_cdf_monotonicity_and_symmetry() {
    // Monotonic: if p1 < p2 then inv(p1) < inv(p2)
    let ps = [
        1e-12,
        1e-10,
        1e-8,
        1e-6,
        1e-4,
        1e-3,
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        0.95,
        0.99,
        1.0 - 1e-3,
        1.0 - 1e-6,
        1.0 - 1e-8,
        1.0 - 1e-10,
        1.0 - 1e-12,
    ];

    // Helper: standard normal pdf φ(z)
    fn phi(z: f64) -> f64 {
        // φ(z) = exp(-z^2/2) / sqrt(2π)
        (-0.5 * z * z).exp() / 2.5066282746310005
    }

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

        // Symmetry: inv(1-p) ≈ -inv(p).
        //
        // In extreme tails, tiny probability-space errors in CDF/SF evaluation
        // are amplified by 1/φ(z). We scale the allowed z-error accordingly.
        if p <= 0.5 {
            let z2 = normal_inverse_cdf(1.0 - p);
            let sum = z + z2;

            // Condition number scale: 1/φ(|z|).
            // Use |z| from the smaller tail side (z should be <= 0 for p<=0.5).
            let pdf = phi(z.abs());

            // Base absolute tolerance in probability-space we are willing to tolerate
            // from floating point / approximation / single Newton step.
            // This roughly translates to a z-space tolerance of dp / φ(z).
            //
            // Tuned to be strict but non-flaky across platforms.
            let dp_abs = 1e-16; // "near double precision" probability error floor

            // Convert dp tolerance to z tolerance; add a small fixed floor too.
            let tol = if pdf > 0.0 {
                (dp_abs / pdf).max(1e-12)
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
    assert!(normal_inverse_cdf(0.0).is_infinite() && normal_inverse_cdf(0.0).is_sign_negative());
    assert!(normal_inverse_cdf(1.0).is_infinite() && normal_inverse_cdf(1.0).is_sign_positive());
}

#[test]
fn test_normal_cdf_cephes_symmetry_and_bounds() {
    let zs = [
        -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0,
        10.0,
    ];

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
fn test_cephes_erf_erfc_identities() {
    let xs = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0];

    for &x in &xs {
        let erf_v = cephes_erf(x);
        let erfc_v = cephes_erfc(x);

        // erf + erfc = 1 (for all real x; numerically very tight for x>=0)
        assert!(
            (erf_v + erfc_v - 1.0).abs() < 1e-14,
            "erf+erfc identity failed at x={}: erf={}, erfc={}, sum={}",
            x,
            erf_v,
            erfc_v,
            erf_v + erfc_v
        );

        // Symmetry: erf(-x) = -erf(x), erfc(-x) = 2 - erfc(x)
        let erf_neg = cephes_erf(-x);
        let erfc_neg = cephes_erfc(-x);

        assert!(
            (erf_neg + erf_v).abs() < 1e-14,
            "erf symmetry failed at x={}: erf(x)={}, erf(-x)={}",
            x,
            erf_v,
            erf_neg
        );

        assert!(
            (erfc_neg - (2.0 - erfc_v)).abs() < 1e-14,
            "erfc symmetry failed at x={}: erfc(x)={}, erfc(-x)={}",
            x,
            erfc_v,
            erfc_neg
        );
    }
}

#[test]
fn test_inc_beta_complement_and_monotonicity() {
    let params = [(0.5, 0.5), (2.0, 3.0), (10.0, 1.2), (1.5, 5.0), (5.0, 5.0)];

    for &(a, b) in &params {
        // Complement identity: I_x(a,b) + I_{1-x}(b,a) = 1
        for &x in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let lhs = inc_beta(x, a, b) + inc_beta(1.0 - x, b, a);
            assert!(
                (lhs - 1.0).abs() < 5e-13,
                "inc_beta complement failed: a={}, b={}, x={}, lhs={}",
                a,
                b,
                x,
                lhs
            );
        }

        // Monotonic in x
        let xs = [
            1e-12,
            1e-6,
            1e-3,
            0.01,
            0.1,
            0.5,
            0.9,
            0.99,
            1.0 - 1e-6,
            1.0 - 1e-12,
        ];
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
fn test_inc_gamma_upper_monotonicity() {
    // For fixed a>0, Q(a,x) decreases in x.
    let as_ = [0.5, 1.0, 2.0, 5.0, 10.0];

    for &a in &as_ {
        let xs = [
            0.0, 1e-12, 1e-6, 1e-3, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0,
        ];
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
fn test_normal_cdf_symmetry() {
    for &z in &[0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0] {
        let p = normal_cdf_cephes(z);
        let pn = normal_cdf_cephes(-z);
        assert!(
            (p + pn - 1.0).abs() < 1e-14,
            "Phi symmetry failed at z={}: p={}, pn={}, sum={}",
            z,
            p,
            pn,
            p + pn
        );
    }
}

#[test]
fn test_normal_cdf_bounds_and_monotone() {
    let zs = [
        -10.0, -8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0,
    ];
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
fn test_normal_cdf_sf_complement() {
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
fn test_normal_cdf_inv_roundtrip_in_z() {
    fn phi(z: f64) -> f64 {
        (-0.5 * z * z).exp() / 2.5066282746310005
    }

    for &z in &[-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0] {
        let p = normal_cdf_cephes(z);
        let z2 = normal_inverse_cdf(p);
        let dz = (z2 - z).abs();

        // Allow "dp noise" of about 1e-16 and convert to z-space using 1/phi(z)
        let pdf = phi(z.abs());
        let tol = if pdf > 0.0 {
            (1e-16 / pdf).max(1e-12)
        } else {
            1e-6
        };

        assert!(
            dz < tol,
            "z={}, p={}, z2={}, dz={}, tol={}",
            z,
            p,
            z2,
            dz,
            tol
        );
    }
}
