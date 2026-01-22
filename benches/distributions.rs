//! Statistical distribution benchmarks.
//!
//! Benchmarks the custom statistical distribution functions including:
//! - Log gamma (Lanczos approximation)
//! - Incomplete beta function
//! - Student's t CDF
//! - F-distribution CDF
//! - Chi-squared survival
//! - Normal CDF and inverse CDF

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linreg_core::distributions::{
    chi_squared_survival, fisher_snedecor_cdf, inc_beta, ln_gamma, normal_cdf_cephes,
    normal_inverse_cdf, student_t_cdf, student_t_inverse_cdf,
};

/// Benchmarks ln_gamma across different input ranges.
fn bench_ln_gamma(c: &mut Criterion) {
    let values: Vec<f64> = vec![
        0.5,   // Special case (sqrt(pi))
        1.0,   // Gamma(1) = 1
        2.0,   // Gamma(2) = 1
        5.0,   // Small integer
        10.0,  // Medium integer
        50.0,  // Large integer
        100.0, // Very large integer
        2.5,   // Fractional
        17.5,  // Larger fractional
        0.1,   // Small positive
        0.01,  // Very small positive
        -0.5,  // Negative (uses reflection)
        -2.5,  // Negative fractional
    ];

    let mut group = c.benchmark_group("ln_gamma");

    for &val in &values {
        group.bench_with_input(
            BenchmarkId::new("value", format!("{:.2}", val)),
            &val,
            |b, &val| b.iter(|| ln_gamma(black_box(val))),
        );
    }

    group.finish();
}

/// Benchmarks inc_beta (incomplete beta function).
fn bench_inc_beta(c: &mut Criterion) {
    // Format: (x, a, b)
    let cases = vec![
        (0.5, 1.0, 1.0),   // Simple case
        (0.3, 2.0, 3.0),   // Small parameters
        (0.5, 5.0, 10.0),  // Medium parameters
        (0.7, 10.0, 20.0), // Larger parameters
        (0.2, 0.5, 0.5),   // Fractional shape params
        (0.5, 50.0, 50.0), // Large parameters
        (0.9, 5.0, 2.0),   // Near upper bound
        (0.1, 10.0, 5.0),  // Near lower bound
    ];

    let mut group = c.benchmark_group("inc_beta");

    for &(x, a, b) in &cases {
        group.bench_with_input(
            BenchmarkId::new("params", format!("{:.2}_{}_{}", x, a, b)),
            &(x, a, b),
            |bencher, &(x, a, b)| {
                bencher.iter(|| inc_beta(black_box(x), black_box(a), black_box(b)))
            },
        );
    }

    group.finish();
}

/// Benchmarks Student's t CDF.
fn bench_student_t_cdf(c: &mut Criterion) {
    // Format: (t, df)
    let cases = vec![
        (0.0, 1.0),    // Center, df=1 (Cauchy)
        (1.0, 1.0),    // df=1
        (2.0, 5.0),    // Typical t-stat
        (1.96, 20.0),  // Critical value, df=20
        (1.96, 100.0), // Critical value, high df (approx normal)
        (3.0, 10.0),   // Extreme value
        (-2.5, 15.0),  // Negative
        (5.0, 2.0),    // Extreme, low df
    ];

    let mut group = c.benchmark_group("student_t_cdf");

    for &(t, df) in &cases {
        group.bench_with_input(
            BenchmarkId::new("params", format!("{:.2}_{}", t, df)),
            &(t, df),
            |b, &(t, df)| b.iter(|| student_t_cdf(black_box(t), black_box(df))),
        );
    }

    group.finish();
}

/// Benchmarks Student's t inverse CDF (quantile function).
fn bench_student_t_inverse_cdf(c: &mut Criterion) {
    // Probabilities and degrees of freedom
    let p_values = vec![0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
    let df_values = vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0];

    let mut group = c.benchmark_group("student_t_inverse_cdf");

    for &df in &df_values {
        for &p in &p_values {
            group.bench_with_input(
                BenchmarkId::new("params", format!("{:.3}_{}", p, df)),
                &(p, df),
                |b, &(p, df)| b.iter(|| student_t_inverse_cdf(black_box(p), black_box(df))),
            );
        }
    }

    group.finish();
}

/// Benchmarks F-distribution CDF.
fn bench_f_cdf(c: &mut Criterion) {
    // Format: (f, d1, d2)
    let cases = vec![
        (1.0, 1.0, 1.0),    // Base case
        (4.0, 5.0, 10.0),   // Typical
        (3.0, 10.0, 20.0),  // Larger dfs
        (2.0, 50.0, 100.0), // Very large dfs
        (10.0, 2.0, 10.0),  // Large F
        (0.5, 5.0, 5.0),    // Small F
        (5.0, 1.0, 100.0),  // d1=1 (extreme skew)
    ];

    let mut group = c.benchmark_group("fisher_snedecor_cdf");

    for &(f, d1, d2) in &cases {
        group.bench_with_input(
            BenchmarkId::new("params", format!("{:.2}_{}_{}", f, d1, d2)),
            &(f, d1, d2),
            |b, &(f, d1, d2)| {
                b.iter(|| fisher_snedecor_cdf(black_box(f), black_box(d1), black_box(d2)))
            },
        );
    }

    group.finish();
}

/// Benchmarks chi-squared survival function (p-value).
fn bench_chi_squared_survival(c: &mut Criterion) {
    // Format: (x, k)
    let cases = vec![
        (3.84, 1.0),   // Critical value, df=1
        (5.99, 2.0),   // Critical value, df=2
        (10.0, 5.0),   // Typical
        (20.0, 10.0),  // Medium
        (50.0, 25.0),  // Larger
        (100.0, 50.0), // Large
        (1.0, 1.0),    // Small chi-squared
    ];

    let mut group = c.benchmark_group("chi_squared_survival");

    for &(x, k) in &cases {
        group.bench_with_input(
            BenchmarkId::new("params", format!("{:.2}_{}", x, k)),
            &(x, k),
            |b, &(x, k)| b.iter(|| chi_squared_survival(black_box(x), black_box(k))),
        );
    }

    group.finish();
}

/// Benchmarks normal CDF (Cephes implementation).
fn bench_normal_cdf(c: &mut Criterion) {
    let z_values = vec![
        -5.0, -4.0, -3.0, -2.5, -2.0, -1.96, -1.0, -0.5, 0.0, 0.5, 1.0, 1.96, 2.0, 2.5, 3.0, 4.0,
        5.0,
    ];

    let mut group = c.benchmark_group("normal_cdf_cephes");

    for &z in &z_values {
        group.bench_with_input(BenchmarkId::new("z", format!("{:.2}", z)), &z, |b, &z| {
            b.iter(|| normal_cdf_cephes(black_box(z)))
        });
    }

    group.finish();
}

/// Benchmarks normal inverse CDF (probit function).
fn bench_normal_inverse_cdf(c: &mut Criterion) {
    let p_values = vec![
        0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999,
    ];

    let mut group = c.benchmark_group("normal_inverse_cdf");

    for &p in &p_values {
        group.bench_with_input(BenchmarkId::new("p", format!("{:.4}", p)), &p, |b, &p| {
            b.iter(|| normal_inverse_cdf(black_box(p)))
        });
    }

    group.finish();
}

criterion_group!(
    distributions,
    bench_ln_gamma,
    bench_inc_beta,
    bench_student_t_cdf,
    bench_student_t_inverse_cdf,
    bench_f_cdf,
    bench_chi_squared_survival,
    bench_normal_cdf,
    bench_normal_inverse_cdf
);
criterion_main!(distributions);
