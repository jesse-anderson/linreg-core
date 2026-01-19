//! Custom implementation of statistical distributions to replace statrs dependency.
//! Includes Log Gamma, Incomplete Beta, Incomplete Gamma, Student's T, F-distribution, Chi-Squared, and Normal distribution functions.

use std::f64::consts::PI;

// ============================================================================
// Core Math Functions
// ============================================================================
//
// Fundamental operations for gamma and beta functions, which form the
// building blocks for statistical distributions.

/// Computes the natural logarithm of the gamma function.
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
pub fn ln_gamma(z: f64) -> f64 {
    if z <= 0.0 {
        // Reflection formula: Gamma(1-z) * Gamma(z) = pi / sin(pi*z)
        let val = (PI * z).sin();
        if val == 0.0 { return f64::INFINITY; } // Singularity
        return (PI / val).ln() - ln_gamma(1.0 - z);
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
pub fn inc_beta(x: f64, a: f64, b: f64) -> f64 {
    if x < 0.0 || x > 1.0 {
        return f64::NAN;
    }
    if x == 0.0 { return 0.0; }
    if x == 1.0 { return 1.0; }

    // Symmetry relation to optimize convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - inc_beta(1.0 - x, b, a);
    }

    let lbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - lbeta).exp();

    // Lentz's method for continued fraction
    // Target: 1 / (1 + d1/(1 + d2/...))
    // We evaluate 1 + d1/(1 + d2/...) and invert result
    
    let max_iter = 200;
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
        if (delta - 1.0).abs() < epsilon { break; }

        // Even step: d_{2m+2} (bb)
        // d2, d4, ...
        let bb = (m_f + 1.0) * (b - m_f - 1.0) * x / ((a + 2.0 * m_f + 1.0) * (a + 2.0 * m_f + 2.0));
        delta = evaluate_fraction_step(bb, &mut d, &mut c, tiny);
        f *= delta;
        if (delta - 1.0).abs() < epsilon { break; }
    }

    // Result is front * (1/f) / a
    front / (f * a)
}

fn evaluate_fraction_step(val: f64, d: &mut f64, c: &mut f64, tiny: f64) -> f64 {
    *d = 1.0 + val * *d;
    if d.abs() < tiny { *d = tiny; }
    *d = 1.0 / *d;
    
    *c = 1.0 + val / *c;
    if c.abs() < tiny { *c = tiny; }
    
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
    if x <= 0.0 { return 0.0; }
    
    let gln = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    let max_iter = 100;
    let epsilon = 3.0e-7;

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
    
    let max_iter = 100;
    let epsilon = 3.0e-7;
    
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    
    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny { d = tiny; }
        c = b + an / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < epsilon { break; }
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
pub fn inc_gamma_upper(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 { return 1.0; }
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
pub fn student_t_cdf(t: f64, df: f64) -> f64 {
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
pub fn fisher_snedecor_cdf(f: f64, d1: f64, d2: f64) -> f64 {
    if f <= 0.0 { return 0.0; }
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
pub fn chi_squared_survival(x: f64, k: f64) -> f64 {
    if x <= 0.0 { return 1.0; }
    inc_gamma_upper(k / 2.0, x / 2.0)
}

/// Computes the standard normal cumulative distribution function.
///
/// Returns P(Z ≤ z) for a standard normal distribution using the error
/// function approximation (Abramowitz and Stegun 7.1.26).
///
/// The relationship is: Φ(z) = 0.5 * (1 + erf(z/√2))
///
/// # Arguments
///
/// * `z` - z-score value
///
/// # Returns
///
/// The probability that a standard normal random variable is less than or equal to z.
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
    let x_abs = z.abs() / 2.0_f64.sqrt();  // Divide by sqrt(2) for normal CDF

    let t = 1.0 / (1.0 + P * x_abs);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x_abs * x_abs).exp();

    // Φ(z) = 0.5 * (1 + erf(z/√2)) = 0.5 + 0.5 * sign * erf_approx
    0.5 + 0.5 * sign * y
}

/// Computes the standard normal cumulative distribution function using Cephes algorithm.
///
/// Returns P(Z ≤ z) for a standard normal distribution using the
/// Cephes algorithm (high-precision rational approximation matching R's pnorm).
///
/// This implementation ports the ndtr function from the Cephes math library,
/// which is the same algorithm used by R for pnorm. The relationship is:
/// Φ(z) = 0.5 * (1 + erf(z/√2)) = 0.5 * erfc(-z/√2)
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
pub fn normal_cdf_cephes(z: f64) -> f64 {
    use std::f64::consts::FRAC_1_SQRT_2;
    // Cephes constants - SQRTH = sqrt(0.5) = 1/sqrt(2)
    const SQRTH: f64 = FRAC_1_SQRT_2;

    let x = z * SQRTH;
    let z_abs = x.abs();

    if z_abs < 1.0 {
        // For |x| < 1, use erf approximation directly
        0.5 + 0.5 * cephes_erf(x)
    } else {
        // For |x| >= 1, use erfc with exp(-x²/2) via expx2 to avoid error amplification
        let mut y = 0.5 * cephes_erfce(z_abs);
        // Multiply by exp(-a²/2) where a = z (original input)
        let exp_factor = cephes_expx2(z, -1);
        y *= exp_factor.sqrt();
        if x > 0.0 {
            y = 1.0 - y;
        }
        y
    }
}

/// Computes the error function erf(x) using Cephes rational approximation.
///
/// For |x| < 1: erf(x) = x * P4(x²)/Q5(x²)
/// For |x| >= 1: erf(x) = 1 - erfc(x)
///
/// Accuracy: Relative error < 3.7e-16 (IEEE double precision)
fn cephes_erf(x: f64) -> f64 {
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
fn cephes_erfc(a: f64) -> f64 {
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

/// Exponentially scaled erfc function: exp(x²) * erfc(x)
///
/// Valid for x > 1. Used by normal_cdf to avoid error amplification.
fn cephes_erfce(x: f64) -> f64 {
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

/// Evaluates a polynomial with given coefficients.
///
/// Computes: P[0] + P[1]*x + P[2]*x² + ... + P[n]*x^n
fn polevl(x: f64, coeffs: &[f64]) -> f64 {
    let mut result = 0.0;
    for &c in coeffs.iter().rev() {
        result = result * x + c;
    }
    result
}

/// Evaluates a polynomial with given coefficients (no constant term).
///
/// Cephes p1evl: computes x + P[0] + P[1]*x + P[2]*x² + ...
/// which is equivalent to x * (1 + P[1] + P[2]*x + ...) + P[0]
///
/// This is used for rational approximations where the denominator is
/// implicitly 1 + P[1]*x + P[2]*x² + ... but we want the full polynomial.
fn p1evl(x: f64, coeffs: &[f64]) -> f64 {
    // C implementation: ans = x + *p++; while(--N) ans = ans*x + *p++;
    let mut result = x;
    for &c in coeffs.iter() {
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
pub fn normal_inverse_cdf(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if p == 0.5 { return 0.0; }

    // Coefficients for rational approximation
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

    // Rational approximation for central region
    if p > P_LOW && p < P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        return num / den;
    }

    // Rational approximation for tails
    let (sign, q) = if p < P_LOW {
        (-1.0, (-2.0 * p.ln()).sqrt())
    } else {
        (1.0, (-2.0 * (1.0 - p).ln()).sqrt())
    };

    let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
    let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;

    sign * (q - num / den)
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
pub fn student_t_inverse_cdf(p: f64, df: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if p == 0.5 { return 0.0; }

    // Initial guess using Normal approximation, clamped to reasonable range
    let mut x = normal_inverse_cdf(p).clamp(-10.0, 10.0);

    // Newton-Raphson with step size limiting for stability
    for _ in 0..50 {
        let cdf = student_t_cdf(x, df);
        // PDF of t-distribution
        let pdf = (ln_gamma((df + 1.0) / 2.0) - ln_gamma(df / 2.0)) .exp()
            / ((df * PI).sqrt() * (1.0 + x * x / df).powf((df + 1.0) / 2.0));

        let diff = cdf - p;
        if diff.abs() < 1e-12 { break; }
        if pdf < 1e-15 { break; } // Avoid division by near-zero

        // Limit step size to prevent divergence
        let step = (diff / pdf).clamp(-2.0, 2.0);
        x -= step;

        // Hard clamp to prevent runaway
        if x < -20.0 { x = -20.0; }
        if x > 20.0 { x = 20.0; }
    }

    x
}

// ============================================================================
// Unit Tests
// ============================================================================
//
// All functions validated against R reference values to ensure accuracy.

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    const TOL: f64 = 1e-8;      // General tolerance (reserved for future use)
    const TOL_LOOSE: f64 = 1e-6; // For iterative methods

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ========================================================================
    // Log Gamma Tests (R: lgamma)
    // ========================================================================
    #[test]
    fn test_ln_gamma() {
        let cases = [
            (0.5,  0.572364942924700),
            (1.0,  0.000000000000000),
            (1.5, -0.120782237635245),
            (2.0,  0.000000000000000),
            (5.0,  3.178053830347946),
            (10.0, 12.801827480081469),
            (11.5, 16.292000476567242),
            (100.0, 359.134205369575398),
        ];

        for (z, expected) in cases {
            let result = ln_gamma(z);
            assert!(
                approx_eq(result, expected, TOL_LOOSE),
                "ln_gamma({}) = {}, expected {}", z, result, expected
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
                "inc_beta({}, {}, {}) = {}, expected {}", x, a, b, result, expected
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
                "student_t_cdf({}, {}) = {}, expected {}", t, df, result, expected
            );
        }
    }

    // ========================================================================
    // Two-tailed p-value Tests (R: 2 * pt(-|t|, df))
    // ========================================================================
    #[test]
    fn test_two_tailed_p_value() {
        use crate::core::two_tailed_p_value;

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
                "two_tailed_p_value({}, {}) = {}, expected {}", t, df, result, expected
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
                "fisher_snedecor_cdf({}, {}, {}) = {}, expected {}", f, df1, df2, result, expected
            );
        }
    }

    // ========================================================================
    // F p-value Tests (R: pf(f, df1, df2, lower.tail=FALSE))
    // ========================================================================
    #[test]
    fn test_f_p_value() {
        use crate::core::f_p_value;

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
                "f_p_value({}, {}, {}) = {}, expected {}", f, df1, df2, result, expected
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
                "chi_squared_survival({}, {}) = {}, expected {}", x, k, result, expected
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
                "student_t_inverse_cdf({}, {}) = {}, expected {}", p, df, result, expected
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
                "normal_inverse_cdf({}) = {}, expected {}", p, result, expected
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
        use crate::core::two_tailed_p_value;

        // P-values must be in [0, 1]
        for t in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            for df in [1.0, 5.0, 10.0, 21.0, 100.0] {
                let p = two_tailed_p_value(t, df);
                assert!(
                    p >= 0.0 && p <= 1.0,
                    "two_tailed_p_value({}, {}) = {} is out of bounds [0,1]", t, df, p
                );
            }
        }
    }
}
