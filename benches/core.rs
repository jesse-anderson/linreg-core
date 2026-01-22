//! Core OLS regression benchmarks.
//!
//! Benchmarks the main `ols_regression` function across varying dataset sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use linreg_core::core::ols_regression;

/// Generates a synthetic dataset with the given dimensions.
///
/// Uses a more complex pattern to avoid singular matrices.
///
/// # Arguments
///
/// * `n` - Number of observations
/// * `k` - Number of predictors
///
/// # Returns
///
/// A tuple of (y, x_vars, names) suitable for `ols_regression`
fn generate_data(n: usize, k: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<String>) {
    let mut y = Vec::with_capacity(n);
    let mut x_vars: Vec<Vec<f64>> = (0..k).map(|_| Vec::with_capacity(n)).collect();

    // Use different frequencies to avoid perfect collinearity
    let frequencies: Vec<f64> = (0..k)
        .map(|j| 1.0 + (j as f64) * 0.3 + ((j as f64) * 0.7).sqrt())
        .collect();

    // Generate synthetic data: y = 1.0 + sum(coef_j * x_j) + noise
    for i in 0..n {
        let t = (i as f64) / 100.0; // Time-like variable

        // True coefficients: intercept=1.0, others varying
        let mut y_val = 1.0;
        for j in 0..k {
            // Mix of sin, cos, and linear terms for diversity
            let freq = frequencies[j];
            let x_val = (t * freq).sin()
                + 0.5 * (t * freq * 0.7).cos()
                + 0.1 * (i as f64 * (j + 1) as f64 * 0.01).sin();
            x_vars[j].push(x_val);
            y_val += (j + 1) as f64 * 0.5 * x_val;
        }

        // Add noise to avoid perfect fit
        y_val += (i as f64 * 0.13).sin() * 0.3;
        y.push(y_val);
    }

    // Variable names
    let mut names = vec!["Intercept".to_string()];
    for j in 0..k {
        names.push(format!("X{}", j + 1));
    }

    (y, x_vars, names)
}

/// Benchmarks OLS regression across different dataset sizes.
fn bench_ols_sizes(c: &mut Criterion) {
    let sizes = vec![
        (10, 2),    // Tiny: 10 obs, 2 predictors
        (50, 3),    // Small: 50 obs, 3 predictors
        (100, 5),   // Medium-small: 100 obs, 5 predictors
        (500, 10),  // Medium: 500 obs, 10 predictors
        (1000, 20), // Large: 1000 obs, 20 predictors
        (5000, 50), // XLarge: 5000 obs, 50 predictors
    ];

    for &(n, k) in &sizes {
        let (y, x_vars, names) = generate_data(n, k);

        let mut group = c.benchmark_group("ols_regression");
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(y, x_vars, names),
            |b, (y, x_vars, names)| {
                b.iter(|| {
                    ols_regression(black_box(y), black_box(x_vars), black_box(names)).unwrap()
                })
            },
        );
        group.finish();
    }
}

/// Benchmarks OLS regression with fixed observations, varying predictors.
fn bench_ols_predictors(c: &mut Criterion) {
    let n = 1000; // Fixed observations
    let k_values = vec![2, 5, 10, 20, 50, 100];

    let mut group = c.benchmark_group("ols_predictors");

    for k in k_values {
        let (y, x_vars, names) = generate_data(n, k);

        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter(|| ols_regression(black_box(&y), black_box(&x_vars), black_box(&names)).unwrap())
        });
    }

    group.finish();
}

/// Benchmarks OLS regression with fixed predictors, varying observations.
fn bench_ols_observations(c: &mut Criterion) {
    let k = 10; // Fixed predictors
    let n_values = vec![100, 500, 1000, 5000, 10000];

    let mut group = c.benchmark_group("ols_observations");

    for n in n_values {
        let (y, x_vars, names) = generate_data(n, k);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| ols_regression(black_box(&y), black_box(&x_vars), black_box(&names)).unwrap())
        });
    }

    group.finish();
}

criterion_group!(
    regression,
    bench_ols_sizes,
    bench_ols_predictors,
    bench_ols_observations
);
criterion_main!(regression);
