//! Pressure and stress test benchmarks.
//!
//! These benchmarks test performance at extreme scales.
//!
//! # Memory Requirements
//!
//! Some benchmarks require significant memory:
//! - 10,000 obs × 100 predictors ≈ 8 MB
//! - 10,000 obs × 1,000 predictors ≈ 80 MB
//! - 10,000 obs × 10,000 predictors ≈ 800 MB
//! - 10,000 obs × 100,000 predictors ≈ 8 GB
//!
//! The extreme pressure test (100,000 predictors) is marked as
//! "sample_size = 1" to avoid long test times.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use linreg_core::core::ols_regression;
use linreg_core::linalg::Matrix;
use std::time::Duration;

/// Generates synthetic data for pressure testing.
/// Uses diverse patterns to avoid singular matrices.
fn generate_pressure_data(n: usize, k: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<String>) {
    let mut y = Vec::with_capacity(n);
    let mut x_vars: Vec<Vec<f64>> = (0..k).map(|_| Vec::with_capacity(n)).collect();

    // Use different frequencies for each predictor to avoid collinearity
    let frequencies: Vec<f64> = (0..k)
        .map(|j| 1.0 + (j as f64) * 0.3 + ((j as f64) * 0.7).sqrt())
        .collect();

    for i in 0..n {
        let t = (i as f64) / 100.0;
        let mut y_val = 1.0;

        for j in 0..k {
            let freq = frequencies[j];
            let x_val = (t * freq).sin()
                + 0.5 * (t * freq * 0.7).cos()
                + 0.1 * (i as f64 * (j + 1) as f64 * 0.01).sin();
            x_vars[j].push(x_val);
            y_val += ((j + 1) as f64) * 0.1 * x_val;
        }

        // Add noise
        y_val += (i as f64 * 0.13).sin() * 0.3;
        y.push(y_val);
    }

    let mut names = vec!["Intercept".to_string()];
    for j in 0..k {
        names.push(format!("X{}", j + 1));
    }

    (y, x_vars, names)
}

/// Pressure test: Large number of observations.
fn bench_pressure_observations(c: &mut Criterion) {
    let k = 50; // Fixed predictors
    let n_values = vec![10_000, 50_000, 100_000];

    let mut group = c.benchmark_group("pressure_observations");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(30));

    for &n in &n_values {
        let (y, x_vars, names) = generate_pressure_data(n, k);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| ols_regression(black_box(&y), black_box(&x_vars), black_box(&names)).unwrap())
        });
    }

    group.finish();
}

/// Pressure test: Many predictors.
fn bench_pressure_predictors(c: &mut Criterion) {
    let n = 10_000; // Fixed observations
    let k_values = vec![100, 500, 1_000, 5_000];

    let mut group = c.benchmark_group("pressure_predictors");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(30));

    for &k in &k_values {
        let (y, x_vars, names) = generate_pressure_data(n, k);

        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter(|| ols_regression(black_box(&y), black_box(&x_vars), black_box(&names)).unwrap())
        });
    }

    group.finish();
}

/// Pressure test: Large matrix operations.
fn bench_pressure_matrix_ops(c: &mut Criterion) {
    // QR decomposition is the bottleneck for large matrices
    let sizes = vec![(10_000, 100), (10_000, 500), (10_000, 1_000)];

    let mut group = c.benchmark_group("pressure_qr");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(30));

    for &(rows, cols) in &sizes {
        let data: Vec<f64> = (0..(rows * cols))
            .map(|i| {
                let x = (i as f64) * 0.0001;
                x.sin() + x.cos()
            })
            .collect();
        let m = Matrix::new(rows, cols, data);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", rows, cols)),
            &(rows, cols),
            |b, _| b.iter(|| black_box(&m).qr()),
        );
    }

    group.finish();
}

/// EXTREME pressure test: Very high dimensional data.
///
/// This test benchmarks the limit of what the library can handle.
/// Requires significant memory for the largest tests.
fn bench_pressure_extreme(c: &mut Criterion) {
    let mut group = c.benchmark_group("pressure_extreme");
    group.warm_up_time(Duration::from_secs(10));
    // Extreme tests use sample_size = 1 to avoid long test times
    group.sample_size(1);

    // 10,000 observations × 10,000 predictors (≈800 MB of data)
    let (y, x_vars, names) = generate_pressure_data(10_000, 10_000);
    group.bench_function("10k_obs_10k_pred", |b| {
        b.iter(|| {
            // This is a stress test - may fail on systems with <4GB RAM
            let result = ols_regression(black_box(&y), black_box(&x_vars), black_box(&names));
            if let Ok(r) = result {
                black_box(r);
            }
        })
    });

    // 5,000 observations × 20,000 predictors (≈800 MB of data)
    let (y, x_vars, names) = generate_pressure_data(5_000, 20_000);
    group.bench_function("5k_obs_20k_pred", |b| {
        b.iter(|| {
            let result = ols_regression(black_box(&y), black_box(&x_vars), black_box(&names));
            if let Ok(r) = result {
                black_box(r);
            }
        })
    });

    // The ultimate stress test: 10,000 obs × 100,000 predictors (≈8 GB)
    // Enabled for systems with 16GB+ RAM
    let (y, x_vars, names) = generate_pressure_data(10_000, 100_000);
    group.bench_function("10k_obs_100k_pred", |b| {
        b.iter(|| {
            let result = ols_regression(black_box(&y), black_box(&x_vars), black_box(&names));
            if let Ok(r) = result {
                black_box(r);
            }
        })
    });

    group.finish();
}

/// Memory allocation benchmark.
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    group.sample_size(10);

    let sizes = vec![(1_000, 100), (5_000, 500), (10_000, 1_000)];

    for &(n, k) in &sizes {
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, &(n, k)| {
                b.iter(|| {
                    // Measure the cost of allocating data
                    let y: Vec<f64> = vec![0.0; n];
                    let x_vars: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0; n]).collect();
                    black_box((y, x_vars));
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    pressure,
    bench_pressure_observations,
    bench_pressure_predictors,
    bench_pressure_matrix_ops,
    bench_pressure_extreme,
    bench_memory_allocation
);
criterion_main!(pressure);
