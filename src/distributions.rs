//! Custom statistical special functions and distribution utilities (CDF/SF/quantiles),
//! primarily to avoid pulling in `statrs` for regression diagnostics.
//!
//! Includes: ln-gamma, regularized incomplete beta/gamma, Student-t CDF + inverse CDF,
//! F CDF, chi-squared survival, and normal CDF/SF + inverse CDF (including a Cephes-style path).

use std::f64::consts::PI;

// ============================================================================
// Core Math Functions
// ============================================================================
//
// Fundamental operations for gamma and beta functions, which form the
// building blocks for statistical distributions.

/// Computes ln|Γ(z)| (the natural log of the absolute value of the gamma function).
///
/// For z > 0 this equals ln Γ(z). For z <= 0 (non-integers), uses the reflection
/// formula to keep the result real-valued. Poles at z = 0, -1, -2, ... return +∞.
///
/// Uses the Lanczos approximation for numerical stability.
/// Handles negative inputs via the reflection formula.
///
/// # Arguments
///
/// * `z` - Input value
///
/// # References
///
/// Numerical Recipes
///
/// # Example
///
/// ```
/// # use linreg_core::distributions::ln_gamma;
/// let log_gamma = ln_gamma(5.0);
/// assert!((log_gamma - 3.1780538303479458).abs() < 1e-10);
/// ```
pub fn ln_gamma(z: f64) -> f64 {
    if z <= 0.0 {
        // Reflection formula:
        //   Γ(z)Γ(1-z) = π / sin(πz)
        //
        // For a real-valued log across negative non-integers, compute ln|Γ(z)|:
        //   ln|Γ(z)| = ln(π) - ln|sin(πz)| - lnΓ(1-z)
        //
        // Poles remain at z = 0, -1, -2, ...
        let s = (PI * z).sin();
        if s.abs() < 1e-300 {
            return f64::INFINITY; // pole / singularity
        }
        return PI.ln() - s.abs().ln() - ln_gamma(1.0 - z);
    }

    let c = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let mut sum = 1.000000000190015;
    for (i, &val) in c.iter().enumerate() {
        sum += val / (z + i as f64 + 1.0);
    }

    let tmp = z + 5.5;
    let tmp = (z + 0.5) * tmp.ln() - tmp;

    tmp + (2.5066282746310005 * sum / z).ln()
}

/// Computes the regularized incomplete beta function Iₓ(a, b).
///
/// Evaluates the continued fraction representation for numerical stability.
/// Uses a symmetry relation to optimize convergence.
///
/// # Arguments
///
/// * `x` - Point at which to evaluate (0 ≤ x ≤ 1)
/// * `a` - First shape parameter (a > 0)
/// * `b` - Second shape parameter (b > 0)
///
/// # Example
///
/// ```
/// # use linreg_core::distributions::inc_beta;
/// let b = inc_beta(0.5, 2.0, 3.0);
/// assert!(b > 0.0 && b < 1.0);
/// ```
#[allow(clippy::manual_range_contains)]
pub fn inc_beta(x: f64, a: f64, b: f64) -> f64 {
    if x < 0.0 || x > 1.0 {
        return f64::NAN;
    }
    // Domain guard: regularized incomplete beta requires a > 0 and b > 0.
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    // Symmetry relation to optimize convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - inc_beta(1.0 - x, b, a);
    }

    let lbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - lbeta).exp();

    // Lentz's method (modified) for a continued-fraction factor used by the
    // regularized incomplete beta. We accumulate a multiplicative sequence of
    // Lentz "delta" updates into `f`, then combine with the standard front-factor.
    //
    // NOTE: The return form here uses `front / (a * f)` which corresponds to the
    // particular continued-fraction normalization implied by the `aa`/`bb` terms below.

    let max_iter = 300;
    let epsilon = 1e-14;
    let tiny = 1e-30;

    let mut f = 1.0; // b0 = 1
    let mut c = f;
    let mut d = 0.0; // D0 = 0

    for m in 0..max_iter {
        let m_f = m as f64;

        // Odd step: d_{2m+1} (aa)
        // d1, d3, ...
        let aa = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        let mut delta = evaluate_fraction_step(aa, &mut d, &mut c, tiny);
        f *= delta;
        if (delta - 1.0).abs() < epsilon {
            break;
        }

        // Even step: d_{2m+2} (bb)
        // d2, d4, ...
        let bb =
            (m_f + 1.0) * (b - m_f - 1.0) * x / ((a + 2.0 * m_f + 1.0) * (a + 2.0 * m_f + 2.0));
        delta = evaluate_fraction_step(bb, &mut d, &mut c, tiny);
        f *= delta;
        if (delta - 1.0).abs() < epsilon {
            break;
        }
    }

    // Result is front * (1/f) / a
    front / (f * a)
}

fn evaluate_fraction_step(val: f64, d: &mut f64, c: &mut f64, tiny: f64) -> f64 {
    if d.abs() < tiny {
        *d = tiny;
    }
    *d = 1.0 + val * *d;
    if d.abs() < tiny {
        *d = tiny;
    }
    *d = 1.0 / *d;

    if c.abs() < tiny {
        *c = tiny;
    }
    *c = 1.0 + val / *c;
    if c.abs() < tiny {
        *c = tiny;
    }

    *c * *d
}

// ============================================================================
// Incomplete Gamma Functions
// ============================================================================
//
// Regularized incomplete gamma functions used in chi-squared and
// gamma distribution calculations.

/// Series approximation for the lower incomplete gamma P(a, x).
///
/// Converges quickly when x < a + 1.
///
/// # Arguments
///
/// * `a` - Shape parameter
/// * `x` - Upper limit of integration
fn inc_gamma_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    let gln = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    let max_iter = 300;
    let epsilon = 1e-14;

    for _ in 0..max_iter {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * epsilon {
            return sum * (-x + a * x.ln() - gln).exp();
        }
    }

    // Fallback if not converged (though should converge for x < a+1)
    sum * (-x + a * x.ln() - gln).exp()
}

/// Continued fraction approximation for the upper incomplete gamma Q(a, x).
///
/// Converges quickly when x ≥ a + 1.
///
/// # Arguments
///
/// * `a` - Shape parameter
/// * `x` - Upper limit of integration
fn inc_gamma_cf(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let tiny = 1e-30;

    let max_iter = 300;
    let epsilon = 1e-14;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < epsilon {
            break;
        }
    }

    (-x + a * x.ln() - gln).exp() * h
}

// NOTE: Currently unused but kept as reference implementation.
// Uncomment if needed for future statistical calculations.
/*
/// Computes the regularized lower incomplete gamma function P(a, x).
///
/// Automatically selects the series approximation or continued fraction
/// method based on the input values for optimal convergence.
///
/// # Arguments
///
/// * `a` - Shape parameter
/// * `x` - Upper limit of integration
pub fn inc_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 { return 0.0; }
    if x < a + 1.0 {
        inc_gamma_series(a, x)
    } else {
        1.0 - inc_gamma_cf(a, x)
    }
}
*/

/// Computes the regularized upper incomplete gamma function Q(a, x).
///
/// Automatically selects the series approximation or continued fraction
/// method based on the input values for optimal convergence.
///
/// # Arguments
///
/// * `a` - Shape parameter
/// * `x` - Upper limit of integration
///
/// # Example
///
/// ```
/// # use linreg_core::distributions::inc_gamma_upper;
/// let q = inc_gamma_upper(2.0, 3.0);
/// assert!(q > 0.0 && q < 1.0);
/// ```
pub fn inc_gamma_upper(a: f64, x: f64) -> f64 {
    if !a.is_finite() || !x.is_finite() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }

    if x < a + 1.0 {
        1.0 - inc_gamma_series(a, x)
    } else {
        inc_gamma_cf(a, x)
    }
}

// ============================================================================
// Statistical Distributions
// ============================================================================
//
// Cumulative distribution functions and inverse CDFs for common
// statistical distributions used in regression analysis.

/// Computes the Student's t-distribution cumulative distribution function.
///
/// Returns P(T ≤ t) for a t-distribution with the given degrees of freedom.
///
/// # Arguments
///
/// * `t` - t-statistic value
/// * `df` - Degrees of freedom
///
/// # Example
///
/// ```
/// # use linreg_core::distributions::student_t_cdf;
/// let p = student_t_cdf(1.96, 20.0);
/// assert!(p > 0.95); // approximately 0.968
/// ```
pub fn student_t_cdf(t: f64, df: f64) -> f64 {
    if !df.is_finite() || df <= 0.0 || !t.is_finite() {
        return f64::NAN;
    }
    let x = df / (df + t * t);
    let p = 0.5 * inc_beta(x, 0.5 * df, 0.5);

    if t >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Computes the Fisher-Snedecor (F) distribution cumulative distribution function.
///
/// Returns P(F ≤ f) for an F-distribution with the given degrees of freedom.
///
/// # Arguments
///
/// * `f` - F-statistic value
/// * `d1` - Numerator degrees of freedom
/// * `d2` - Denominator degrees of freedom
///
/// # Example
///
/// ```
/// # use linreg_core::distributions::fisher_snedecor_cdf;
/// let p = fisher_snedecor_cdf(3.0, 5.0, 10.0);
/// assert!(p > 0.0 && p < 1.0);
/// ```
pub fn fisher_snedecor_cdf(f: f64, d1: f64, d2: f64) -> f64 {
    if !f.is_finite() || !d1.is_finite() || !d2.is_finite() || d1 <= 0.0 || d2 <= 0.0 {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 0.0;
    }
    let x = (d1 * f) / (d1 * f + d2);
    inc_beta(x, 0.5 * d1, 0.5 * d2)
}

/// Computes the chi-squared survival function (p-value).
///
/// Returns P(X > x) for a chi-squared distribution with k degrees of freedom.
/// Equivalent to the upper incomplete gamma Q(k/2, x/2).
///
/// # Arguments
///
/// * `x` - Chi-squared statistic value
/// * `k` - Degrees of freedom
///
/// # Example
///
/// ```
/// # use linreg_core::distributions::chi_squared_survival;
/// let p = chi_squared_survival(5.0, 2.0);
/// assert!(p > 0.0 && p < 1.0);
/// ```
pub fn chi_squared_survival(x: f64, k: f64) -> f64 {
    if !x.is_finite() || !k.is_finite() || k <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    inc_gamma_upper(k / 2.0, x / 2.0)
}

/// Computes the standard normal cumulative distribution function.
///
/// Returns P(Z ≤ z) for a standard normal distribution using the error
/// function approximation (Abramowitz and Stegun 7.1.26).
///
/// The relationship is: Φ(z) = 0.5 * (1 + erf(z/√2))
/// Note: This is a fast approximation and is not explicitly clamped to [0, 1].
/// For high-precision tails or strict bounds, prefer `normal_cdf_cephes`.
///
/// # Arguments
///
/// * `z` - z-score value
///
/// # Returns
///
/// The probability that a standard normal random variable is less than or equal to z.
///
/// # Example
///
/// ```
/// # use linreg_core::distributions::normal_cdf;
/// let p = normal_cdf(1.96);
/// assert!((p - 0.975).abs() < 0.001);
/// ```
pub fn normal_cdf(z: f64) -> f64 {
    // Abramowitz and Stegun approximation 7.1.26 for erf(x)
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    // For normal CDF, we need erf(z/sqrt(2)), not erf(z)
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x_abs = z.abs() / 2.0_f64.sqrt(); // Divide by sqrt(2) for normal CDF

    let t = 1.0 / (1.0 + P * x_abs);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x_abs * x_abs).exp();

    // Φ(z) = 0.5 * (1 + erf(z/√2)) = 0.5 + 0.5 * sign * erf_approx
    0.5 + 0.5 * sign * y
}

/// Computes the standard normal cumulative distribution function using a Cephes-style algorithm.
///
/// Returns P(Z ≤ z) for a standard normal distribution using a high-precision
/// rational-approximation approach derived from Cephes `ndtr`.
///
/// Note: R's `pnorm` uses its own long-standing rational approximations (see R's `pnorm.c`);
/// this function is not claiming to be the *same* algorithm, but it is intended to match
/// reference implementations to near machine precision.
///
/// # Arguments
///
/// * `z` - z-score value
///
/// # Returns
///
/// The probability that a standard normal random variable is less than or equal to z.
///
/// # Accuracy
///
/// Relative error: arithmetic domain # trials peak rms
/// IEEE -13,0 30000 1.3e-15 2.2e-16
///
/// # References
///
/// Cephes Math Library (cprob/ndtr.c) by Stephen L. Moshier
/// R's pnorm function (uses same underlying algorithm)
///
/// # Example
///
/// ```
/// # use linreg_core::distributions::normal_cdf_cephes;
/// let p = normal_cdf_cephes(1.96);
/// assert!((p - 0.975).abs() < 0.001);
/// ```
#[allow(clippy::manual_clamp)]
pub fn normal_cdf_cephes(z: f64) -> f64 {
    use std::f64::consts::FRAC_1_SQRT_2;
    const SQRTH: f64 = FRAC_1_SQRT_2; // 1/sqrt(2)

    if !z.is_finite() {
        return if z.is_nan() {
            f64::NAN
        } else if z.is_sign_negative() {
            0.0
        } else {
            1.0
        };
    }

    let x = z * SQRTH;
    let ax = x.abs();

    // Small |x|: erf is accurate and cheap
    let p = if ax < 1.0 {
        0.5 + 0.5 * cephes_erf(x)
    } else {
        // Tail: Phi(z) = 1 - 0.5*erfc(z/sqrt(2)) for z>=0
        // and Phi(z) = 0.5*erfc(|z|/sqrt(2)) for z<0.
        //
        // Using erfce: erfc(t) = exp(-t^2) * erfce(t)
        // Here t = |z|/sqrt(2) = ax, so exp(-t^2) = exp(-ax^2) = exp(-z^2/2).
        let tail = 0.5 * (-(ax * ax)).exp() * cephes_erfce(ax);
        if z >= 0.0 {
            1.0 - tail
        } else {
            tail
        }
    };

    // --------------------------------------------------------------------
    // Belt-and-suspenders clamp: right here, right at the end,
    // right before returning, so it applies to both branches.
    // --------------------------------------------------------------------
    p.max(0.0).min(1.0)
}

#[allow(clippy::manual_clamp)]
pub fn normal_sf_cephes(z: f64) -> f64 {
    use std::f64::consts::FRAC_1_SQRT_2;
    const SQRTH: f64 = FRAC_1_SQRT_2; // 1/sqrt(2)

    if !z.is_finite() {
        return if z.is_nan() {
            f64::NAN
        } else if z.is_sign_negative() {
            1.0
        } else {
            0.0
        };
    }

    // For z < 0: Q(z) = Phi(-z)
    if z < 0.0 {
        return normal_cdf_cephes(-z);
    }

    // z >= 0:
    let x = z * SQRTH;
    let ax = x.abs();

    let q = if ax < 1.0 {
        0.5 * cephes_erfc(x)
    } else {
        0.5 * (-(ax * ax)).exp() * cephes_erfce(ax)
    };

    q.max(0.0).min(1.0)
}

/// Computes the error function erf(x) using Cephes rational approximation.
///
/// For |x| < 1: erf(x) = x * P4(x²)/Q5(x²)
/// For |x| >= 1: erf(x) = 1 - erfc(x)
///
/// Accuracy: Relative error < 3.7e-16 (IEEE double precision)
#[allow(clippy::excessive_precision)]
pub fn cephes_erf(x: f64) -> f64 {
    const T: [f64; 5] = [
        9.60497373987051638749e0,
        9.00260197203842689217e1,
        2.23200534594684319226e3,
        7.00332514112805075473e3,
        5.55923013010394962768e4,
    ];

    const U: [f64; 5] = [
        // 1.0 is implicit for p1evl
        3.35617141647503099647e1,
        5.21357949780152679795e2,
        4.59432382970980127987e3,
        2.26290000613890934246e4,
        4.92673942608635921086e4,
    ];

    if x.abs() > 1.0 {
        return 1.0 - cephes_erfc(x);
    }

    let z = x * x;
    x * polevl(z, &T) / p1evl(z, &U)
}

/// Computes the complementary error function erfc(x).
///
/// erfc(x) = 1 - erf(x) = (2/√π) ∫[x,∞] exp(-t²) dt
///
/// Uses rational approximations with exp(-x²) computed via expx2
/// to avoid error amplification for large |x|.
#[allow(clippy::excessive_precision)]
pub fn cephes_erfc(a: f64) -> f64 {
    const P: [f64; 9] = [
        2.46196981473530512524e-10,
        5.64189564831068821977e-1,
        7.46321056442269912687e0,
        4.86371970985681366614e1,
        1.96520832956077098242e2,
        5.26445194995477358631e2,
        9.34528527171957607540e2,
        1.02755188689515710272e3,
        5.57535335369399327526e2,
    ];

    const Q: [f64; 8] = [
        // 1.0 is implicit for p1evl
        1.32281951154744992508e1,
        8.67072140885989742329e1,
        3.54937778887819891062e2,
        9.75708501743205489753e2,
        1.82390916687909736289e3,
        2.24633760818710981792e3,
        1.65666309194161350182e3,
        5.57535340817727675546e2,
    ];

    const R: [f64; 6] = [
        5.64189583547755073984e-1,
        1.27536670759978104416e0,
        5.01905042251180477414e0,
        6.16021097993053585195e0,
        7.40974269950448939160e0,
        2.97886665372100240670e0,
    ];

    const S: [f64; 6] = [
        // 1.0 is implicit for p1evl
        2.26052863220117276590e0,
        9.39603524938001434673e0,
        1.20489539808096656605e1,
        1.70814450747565897222e1,
        9.60896809063285878198e0,
        3.36907645100081516050e0,
    ];

    const MAXLOG: f64 = 7.0978271289338399684e2;

    let x = a.abs();

    if x < 1.0 {
        return 1.0 - cephes_erf(a);
    }

    let z = -a * a;
    if z < -MAXLOG {
        // Underflow
        return if a < 0.0 { 2.0 } else { 0.0 };
    }

    let exp_z = cephes_expx2(a, -1);

    let (p, q) = if x < 8.0 {
        (polevl(x, &P), p1evl(x, &Q))
    } else {
        (polevl(x, &R), p1evl(x, &S))
    };

    let mut y = (exp_z * p) / q;
    if a < 0.0 {
        y = 2.0 - y;
    }
    y
}

/// Exponentially scaled erfc function: exp(x²) * erfc(x).
///
/// Intended for use in the normal tail where cancellation makes `erfc` sensitive.
/// In this module it's called with non-negative `x` (typically |z|/sqrt(2)).
#[allow(clippy::excessive_precision)]
pub fn cephes_erfce(x: f64) -> f64 {
    // For |x| <= ~26, exp(x^2) will not overflow in f64, and using the
    // defining identity often improves agreement with reference values.
    let ax = x.abs();
    if ax <= 26.0 {
        return (x * x).exp() * cephes_erfc(x);
    }
    // For large x, fall back to the stable Cephes-style approximation
    const P: [f64; 9] = [
        2.46196981473530512524e-10,
        5.64189564831068821977e-1,
        7.46321056442269912687e0,
        4.86371970985681366614e1,
        1.96520832956077098242e2,
        5.26445194995477358631e2,
        9.34528527171957607540e2,
        1.02755188689515710272e3,
        5.57535335369399327526e2,
    ];

    const Q: [f64; 8] = [
        1.32281951154744992508e1,
        8.67072140885989742329e1,
        3.54937778887819891062e2,
        9.75708501743205489753e2,
        1.82390916687909736289e3,
        2.24633760818710981792e3,
        1.65666309194161350182e3,
        5.57535340817727675546e2,
    ];

    const R: [f64; 6] = [
        5.64189583547755073984e-1,
        1.27536670759978104416e0,
        5.01905042251180477414e0,
        6.16021097993053585195e0,
        7.40974269950448939160e0,
        2.97886665372100240670e0,
    ];

    const S: [f64; 6] = [
        2.26052863220117276590e0,
        9.39603524938001434673e0,
        1.20489539808096656605e1,
        1.70814450747565897222e1,
        9.60896809063285878198e0,
        3.36907645100081516050e0,
    ];

    if x < 8.0 {
        polevl(x, &P) / p1evl(x, &Q)
    } else {
        polevl(x, &R) / p1evl(x, &S)
    }
}

/// Computes exp(x*x) with suppressed error amplification.
///
/// The naive implementation of exp(x²) amplifies rounding errors in x².
/// This function represents x as an exact multiple of M plus a residual,
/// where M is chosen so exp(m*m) doesn't overflow/underflow and |x-m| is small.
///
/// # Arguments
///
/// * `x` - Input value
/// * `sign` - If negative, computes exp(-x²); if non-negative, computes exp(x²)
#[allow(clippy::excessive_precision)]
fn cephes_expx2(x: f64, sign: i32) -> f64 {
    const MAXLOG: f64 = 7.0978271289338399684e2;
    #[cfg(target_arch = "x86")]
    const M: f64 = 32.0;
    #[cfg(target_arch = "x86")]
    const MINV: f64 = 0.03125;
    #[cfg(not(target_arch = "x86"))]
    const M: f64 = 128.0;
    #[cfg(not(target_arch = "x86"))]
    const MINV: f64 = 0.0078125;

    let mut x = x.abs();

    if sign < 0 {
        x = -x;
    }

    // Represent x as exact multiple of M plus residual
    let m = MINV * (M * x + 0.5).floor();
    let f = x - m;

    // x² = m² + 2mf + f²
    let u = m * m;
    let u1 = 2.0 * m * f + f * f;

    let (u, u1) = if sign < 0 { (-u, -u1) } else { (u, u1) };

    if u + u1 > MAXLOG {
        return f64::INFINITY;
    }

    u.exp() * u1.exp()
}

/// Evaluates a polynomial with coefficients in descending-power order (Cephes).
///
/// If coeffs = [c0, c1, ..., cn], computes:
///   c0*x^n + c1*x^(n-1) + ... + cn
fn polevl(x: f64, coeffs: &[f64]) -> f64 {
    // Cephes expects coeffs in descending-power order:
    // coeffs[0] is the leading coefficient.
    let mut result = 0.0;
    for &c in coeffs.iter() {
        result = result * x + c;
    }
    result
}

/// Evaluates a polynomial with an implicit leading coefficient 1.0 (Cephes p1evl).
///
/// If coeffs = [c0, c1, ..., c_{n-1}], computes:
///   x^n + c0*x^(n-1) + ... + c_{n-1}
///
/// Cephes does this as:
///   ans = x + c0
///   ans = ans*x + c1
///   ...
fn p1evl(x: f64, coeffs: &[f64]) -> f64 {
    // Cephes p1evl: evaluate a polynomial of degree coeffs.len()
    // with *implicit leading coefficient 1.0* in descending-power order.
    //
    // C reference:
    //   ans = x + coef[0];
    //   for i=1..N-1: ans = ans*x + coef[i];
    //
    // Here, coeffs are in descending-power order, matching Cephes tables.
    let mut result = x + coeffs[0];
    for &c in coeffs.iter().skip(1) {
        result = result * x + c;
    }
    result
}

/// Computes the inverse of the standard normal CDF (probit function).
///
/// Uses Acklam's algorithm with highly accurate rational approximation.
/// Maximum error is approximately 1.15×10⁻⁹.
///
/// # Arguments
///
/// * `p` - Probability (0 < p < 1)
///
/// # Returns
///
/// The z-score such that P(Z ≤ z) = p for a standard normal distribution.
///
/// # Example
///
/// ```
/// # use linreg_core::distributions::normal_inverse_cdf;
/// let z = normal_inverse_cdf(0.975);
/// assert!((z - 1.96).abs() < 0.01); // approximately 1.96 for p=0.975
/// ```
#[allow(clippy::excessive_precision)]
#[allow(clippy::manual_clamp)]
pub fn normal_inverse_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    // ------------------------------------------------------------------------
    // Initial rational approximation (Acklam)
    // ------------------------------------------------------------------------
    let mut z = if p > P_LOW && p < P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num / den
    } else {
        // Tail regions
        let q = if p < P_LOW {
            (-2.0 * p.ln()).sqrt()
        } else {
            (-2.0 * (-p).ln_1p()).sqrt() // ln(1 - p) computed stably
        };

        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        let x = num / den;

        if p < P_LOW {
            x
        } else {
            -x
        }
    };

    // ------------------------------------------------------------------------
    // One Newton refinement step:
    //   If p <= 0.5: refine using Phi(z) - p
    //   If p >  0.5: refine using Q(z) - (1-p) to avoid cancellation near 1
    // ------------------------------------------------------------------------
    if z.is_finite() && z.abs() <= 10.0 {
        // phi(z) = exp(-0.5 z^2) / sqrt(2π)
        let pdf = (-0.5 * z * z).exp() / 2.5066282746310005;

        if pdf > 0.0 {
            if p <= 0.5 {
                let pz = normal_cdf_cephes(z);
                if pz.is_finite() {
                    z -= (pz - p) / pdf;
                }
            } else {
                use std::f64::consts::FRAC_1_SQRT_2;
                let q_target = 1.0 - p;

                // For Newton refinement in the upper tail, compute Q(z) via erfc directly:
                // Q(z) = 0.5 * erfc(z / sqrt(2))
                let qz = 0.5 * cephes_erfc(z * FRAC_1_SQRT_2);

                if qz.is_finite() {
                    z += (qz - q_target) / pdf;
                }
            }
        }
    }

    z
}

/// Computes the inverse Student's t-distribution function (quantile function).
///
/// Finds the value t such that P(T ≤ t) = p using the Newton-Raphson method
/// with a normal approximation as the initial guess.
///
/// # Arguments
///
/// * `p` - Probability (0 < p < 1)
/// * `df` - Degrees of freedom
///
/// # Returns
///
/// The t-statistic such that the CDF at that value equals p.
///
/// # Example
///
/// ```
/// # use linreg_core::distributions::student_t_inverse_cdf;
/// let t = student_t_inverse_cdf(0.975, 20.0);
/// assert!(t > 2.0); // approximately 2.086 for df=20, p=0.975
/// ```
#[allow(clippy::manual_clamp)]
pub fn student_t_inverse_cdf(p: f64, df: f64) -> f64 {
    if !df.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    // Initial guess using Normal approximation, clamped to reasonable range
    let mut x = normal_inverse_cdf(p).clamp(-10.0, 10.0);

    // Newton-Raphson with step size limiting for stability
    for _ in 0..50 {
        let cdf = student_t_cdf(x, df);
        // PDF of t-distribution
        let pdf = (ln_gamma((df + 1.0) / 2.0) - ln_gamma(df / 2.0)).exp()
            / ((df * PI).sqrt() * (1.0 + x * x / df).powf((df + 1.0) / 2.0));

        let diff = cdf - p;
        if diff.abs() < 1e-12 {
            break;
        }
        if pdf < 1e-15 {
            break;
        } // Avoid division by near-zero

        // Limit step size to prevent divergence
        let step = (diff / pdf).clamp(-2.0, 2.0);
        x -= step;

        // Hard clamp to prevent runaway
        if x < -20.0 {
            x = -20.0;
        }
        if x > 20.0 {
            x = 20.0;
        }
    }

    x
}
