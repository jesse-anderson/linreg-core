//! Core regression benchmarks.
//!
//! Benchmarks regression fitting functions across varying dataset sizes:
//! - OLS regression
//! - Ridge regression (L2)
//! - Lasso regression (L1)
//! - Elastic Net regression (L1 + L2)
//! - WLS regression (weighted least squares)
//! - LOESS (locally estimated scatterplot smoothing)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use linreg_core::core::ols_regression;
use linreg_core::linalg::Matrix;
use linreg_core::loess::{loess_fit, LoessOptions};
use linreg_core::regularized::{
    elastic_net_fit, elastic_net_path, lasso_fit, make_lambda_path, ridge_fit, ElasticNetOptions,
    LambdaPathOptions, LassoFitOptions, RidgeFitOptions,
};
use linreg_core::weighted_regression::wls_regression;

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

/// Builds a design matrix (n × (k+1)) with intercept column from x_vars.
fn build_design_matrix(x_vars: &[Vec<f64>], n: usize, k: usize) -> Matrix {
    let mut data = Vec::with_capacity(n * (k + 1));
    for i in 0..n {
        data.push(1.0); // Intercept
        for j in 0..k {
            data.push(x_vars[j][i]);
        }
    }
    Matrix::new(n, k + 1, data)
}

/// Benchmarks Ridge regression across dataset sizes.
fn bench_ridge_sizes(c: &mut Criterion) {
    let sizes = vec![
        (50, 3),
        (100, 5),
        (500, 10),
        (1000, 20),
        (5000, 50),
    ];

    let mut group = c.benchmark_group("ridge_regression");

    for &(n, k) in &sizes {
        let (y, x_vars, _) = generate_data(n, k);
        let x = build_design_matrix(&x_vars, n, k);
        let options = RidgeFitOptions {
            lambda: 1.0,
            standardize: true,
            intercept: true,
            ..Default::default()
        };

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| ridge_fit(black_box(&x), black_box(&y), black_box(&options)).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmarks Lasso regression across dataset sizes.
fn bench_lasso_sizes(c: &mut Criterion) {
    let sizes = vec![
        (50, 3),
        (100, 5),
        (500, 10),
        (1000, 20),
        (5000, 50),
    ];

    let mut group = c.benchmark_group("lasso_regression");

    for &(n, k) in &sizes {
        let (y, x_vars, _) = generate_data(n, k);
        let x = build_design_matrix(&x_vars, n, k);
        let options = LassoFitOptions {
            lambda: 0.1,
            standardize: true,
            intercept: true,
            ..Default::default()
        };

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| lasso_fit(black_box(&x), black_box(&y), black_box(&options)).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmarks Elastic Net regression across dataset sizes.
fn bench_elastic_net_sizes(c: &mut Criterion) {
    let sizes = vec![
        (50, 3),
        (100, 5),
        (500, 10),
        (1000, 20),
        (5000, 50),
    ];

    let mut group = c.benchmark_group("elastic_net_regression");

    for &(n, k) in &sizes {
        let (y, x_vars, _) = generate_data(n, k);
        let x = build_design_matrix(&x_vars, n, k);
        let options = ElasticNetOptions {
            lambda: 0.1,
            alpha: 0.5,
            standardize: true,
            intercept: true,
            ..Default::default()
        };

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    elastic_net_fit(black_box(&x), black_box(&y), black_box(&options)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks WLS regression across dataset sizes.
fn bench_wls_sizes(c: &mut Criterion) {
    let sizes = vec![
        (50, 3),
        (100, 5),
        (500, 10),
        (1000, 20),
        (5000, 50),
    ];

    let mut group = c.benchmark_group("wls_regression");

    for &(n, k) in &sizes {
        let (y, x_vars, _) = generate_data(n, k);
        // Generate varying weights
        let weights: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64 * 0.07).sin().abs()).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    wls_regression(black_box(&y), black_box(&x_vars), black_box(&weights)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks LOESS fitting across dataset sizes.
fn bench_loess_sizes(c: &mut Criterion) {
    // LOESS is O(n²) so keep sizes modest
    let sizes = vec![
        (50, 1),
        (100, 1),
        (500, 1),
        (1000, 1),
        (100, 2),
        (500, 2),
    ];

    let mut group = c.benchmark_group("loess_fit");

    for &(n, k) in &sizes {
        let (y, x_vars, _) = generate_data(n, k);
        let options = LoessOptions {
            span: 0.75,
            degree: 1,
            robust_iterations: 0,
            n_predictors: k,
            ..Default::default()
        };

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    loess_fit(black_box(&y), black_box(&x_vars), black_box(&options)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks elastic net regularization path across dataset sizes.
fn bench_elastic_net_path_sizes(c: &mut Criterion) {
    let sizes = vec![
        (100, 5),
        (500, 10),
        (1000, 20),
    ];

    let mut group = c.benchmark_group("elastic_net_path");

    for &(n, k) in &sizes {
        let (y, x_vars, _) = generate_data(n, k);
        let x = build_design_matrix(&x_vars, n, k);
        let path_options = LambdaPathOptions {
            nlambda: 50,
            ..Default::default()
        };
        let fit_options = ElasticNetOptions {
            lambda: 0.1, // Will be overridden by path
            alpha: 0.5,
            standardize: true,
            intercept: true,
            ..Default::default()
        };

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    elastic_net_path(
                        black_box(&x),
                        black_box(&y),
                        black_box(&path_options),
                        black_box(&fit_options),
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks lambda path generation across dataset sizes.
fn bench_make_lambda_path(c: &mut Criterion) {
    let sizes = vec![
        (100, 5),
        (500, 10),
        (1000, 20),
        (5000, 50),
    ];

    let mut group = c.benchmark_group("make_lambda_path");

    for &(n, k) in &sizes {
        let (y, x_vars, _) = generate_data(n, k);
        let x = build_design_matrix(&x_vars, n, k);
        let options = LambdaPathOptions::default();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    make_lambda_path(
                        black_box(&x),
                        black_box(&y),
                        black_box(&options),
                        black_box(None),
                        black_box(Some(0)),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    regression,
    bench_ols_sizes,
    bench_ols_predictors,
    bench_ols_observations,
    bench_ridge_sizes,
    bench_lasso_sizes,
    bench_elastic_net_sizes,
    bench_elastic_net_path_sizes,
    bench_make_lambda_path,
    bench_wls_sizes,
    bench_loess_sizes
);
criterion_main!(regression);
