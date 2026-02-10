//! Diagnostic test benchmarks.
//!
//! Benchmarks all statistical diagnostic tests:
//! - Rainbow test (linearity)
//! - Harvey-Collier test (linearity)
//! - RESET test (specification error)
//! - Breusch-Pagan test (heteroscedasticity)
//! - White test (heteroscedasticity)
//! - Jarque-Bera test (normality)
//! - Durbin-Watson test (autocorrelation)
//! - Breusch-Godfrey test (higher-order autocorrelation)
//! - Shapiro-Wilk test (normality)
//! - Anderson-Darling test (normality)
//! - Cook's Distance (influence)
//! - DFBETAS (coefficient influence)
//! - DFFITS (fitted value influence)
//! - VIF (variance inflation factor)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linreg_core::core::ols_regression;
use linreg_core::diagnostics::{
    anderson_darling_test, breusch_godfrey_test, breusch_pagan_test, cooks_distance_test,
    dfbetas_test, dffits_test, durbin_watson_test, harvey_collier_test, jarque_bera_test,
    rainbow_test, reset_test, shapiro_wilk_test, vif_test, white_test, BGTestType,
    HarveyCollierMethod, RainbowMethod, ResetType, WhiteMethod,
};

/// Generates a synthetic dataset for diagnostic benchmarks.
/// Uses diverse patterns to avoid singular matrices.
fn generate_diagnostic_data(n: usize, k: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut y = Vec::with_capacity(n);
    let mut x_vars: Vec<Vec<f64>> = (0..k).map(|_| Vec::with_capacity(n)).collect();

    // Use different frequencies for each predictor
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
            y_val += (j + 1) as f64 * 0.5 * x_val;
        }

        // Add noise
        y_val += (i as f64 * 0.13).sin() * 0.3;
        y.push(y_val);
    }

    (y, x_vars)
}

/// Benchmarks Rainbow test across dataset sizes.
fn bench_rainbow_test(c: &mut Criterion) {
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20), (5000, 50)];

    let mut group = c.benchmark_group("rainbow_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    rainbow_test(
                        black_box(&y),
                        black_box(&x_vars),
                        black_box(0.5),
                        black_box(RainbowMethod::R),
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks Harvey-Collier test across dataset sizes.
///
/// Harvey-Collier uses recursive residuals which require well-conditioned matrices.
/// Uses a single predictor with simple integer sequence to match the test's own
/// unit tests pattern, avoiding multi-predictor collinearity issues.
fn bench_harvey_collier_test(c: &mut Criterion) {
    let sizes = vec![50, 100, 500, 1000];

    let mut group = c.benchmark_group("harvey_collier_test");

    for &n in &sizes {
        // Simple linear data with small noise (matches unit test pattern)
        let y: Vec<f64> = (1..=n)
            .map(|i| 1.0 + 2.0 * i as f64 + 0.01 * ((i % 7) as f64 - 3.0))
            .collect();
        let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_1", n)),
            &n,
            |b, _| {
                b.iter(|| {
                    harvey_collier_test(
                        black_box(&y),
                        black_box(&[x.clone()]),
                        black_box(HarveyCollierMethod::R),
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks Breusch-Pagan test across dataset sizes.
fn bench_breusch_pagan_test(c: &mut Criterion) {
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20), (5000, 50)];

    let mut group = c.benchmark_group("breusch_pagan_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| b.iter(|| breusch_pagan_test(black_box(&y), black_box(&x_vars)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmarks White test across dataset sizes.
fn bench_white_test(c: &mut Criterion) {
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20)];

    let mut group = c.benchmark_group("white_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    white_test(black_box(&y), black_box(&x_vars), black_box(WhiteMethod::R))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks Jarque-Bera test across dataset sizes.
fn bench_jarque_bera_test(c: &mut Criterion) {
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20), (5000, 50)];

    let mut group = c.benchmark_group("jarque_bera_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| b.iter(|| jarque_bera_test(black_box(&y), black_box(&x_vars)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmarks Durbin-Watson test across dataset sizes.
fn bench_durbin_watson_test(c: &mut Criterion) {
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20), (5000, 50)];

    let mut group = c.benchmark_group("durbin_watson_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| b.iter(|| durbin_watson_test(black_box(&y), black_box(&x_vars)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmarks Shapiro-Wilk test across dataset sizes.
fn bench_shapiro_wilk_test(c: &mut Criterion) {
    // Shapiro-Wilk is limited to n <= 5000
    let sizes = vec![
        (10, 2),
        (50, 3),
        (100, 5),
        (500, 10),
        (1000, 20),
        (5000, 50),
    ];

    let mut group = c.benchmark_group("shapiro_wilk_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| b.iter(|| shapiro_wilk_test(black_box(&y), black_box(&x_vars)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmarks Anderson-Darling test across dataset sizes.
fn bench_anderson_darling_test(c: &mut Criterion) {
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20), (5000, 50)];

    let mut group = c.benchmark_group("anderson_darling_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| b.iter(|| anderson_darling_test(black_box(&y), black_box(&x_vars)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmarks Cook's Distance across dataset sizes.
fn bench_cooks_distance_test(c: &mut Criterion) {
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20), (5000, 50)];

    let mut group = c.benchmark_group("cooks_distance_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| b.iter(|| cooks_distance_test(black_box(&y), black_box(&x_vars)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmarks RESET test across dataset sizes.
fn bench_reset_test(c: &mut Criterion) {
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20), (5000, 50)];

    let mut group = c.benchmark_group("reset_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    reset_test(
                        black_box(&y),
                        black_box(&x_vars),
                        black_box(&[2, 3]),
                        black_box(ResetType::Fitted),
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks Breusch-Godfrey test across dataset sizes.
fn bench_breusch_godfrey_test(c: &mut Criterion) {
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20), (5000, 50)];

    let mut group = c.benchmark_group("breusch_godfrey_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    breusch_godfrey_test(
                        black_box(&y),
                        black_box(&x_vars),
                        black_box(2),
                        black_box(BGTestType::Chisq),
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks DFBETAS across dataset sizes.
fn bench_dfbetas_test(c: &mut Criterion) {
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20)];

    let mut group = c.benchmark_group("dfbetas_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| b.iter(|| dfbetas_test(black_box(&y), black_box(&x_vars)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmarks DFFITS across dataset sizes.
fn bench_dffits_test(c: &mut Criterion) {
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20)];

    let mut group = c.benchmark_group("dffits_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| b.iter(|| dffits_test(black_box(&y), black_box(&x_vars)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmarks VIF across dataset sizes.
fn bench_vif_test(c: &mut Criterion) {
    // VIF requires at least 2 predictors
    let sizes = vec![(50, 3), (100, 5), (500, 10), (1000, 20), (5000, 50)];

    let mut group = c.benchmark_group("vif_test");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| b.iter(|| vif_test(black_box(&y), black_box(&x_vars)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmarks running all diagnostics on a dataset (full pipeline).
fn bench_full_diagnostics(c: &mut Criterion) {
    let sizes = vec![(100, 5), (500, 10), (1000, 20)];

    let mut group = c.benchmark_group("full_diagnostics");

    for &(n, k) in &sizes {
        let (y, x_vars) = generate_diagnostic_data(n, k);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    // Run all diagnostic tests (except Harvey-Collier which fails on synthetic data)
                    let _ = black_box(
                        &ols_regression(
                            black_box(&y),
                            black_box(&x_vars),
                            &vec!["Intercept".into(); k + 1],
                        )
                        .unwrap(),
                    );

                    let _ = black_box(&rainbow_test(&y, &x_vars, 0.5, RainbowMethod::R).unwrap());
                    // Skip harvey_collier_test - sensitive to synthetic data patterns
                    let _ = black_box(&reset_test(&y, &x_vars, &[2, 3], ResetType::Fitted).unwrap());
                    let _ = black_box(&breusch_pagan_test(&y, &x_vars).unwrap());
                    let _ = black_box(&jarque_bera_test(&y, &x_vars).unwrap());
                    let _ = black_box(&durbin_watson_test(&y, &x_vars).unwrap());
                    let _ = black_box(&breusch_godfrey_test(&y, &x_vars, 2, BGTestType::Chisq).unwrap());
                    let _ = black_box(&shapiro_wilk_test(&y, &x_vars).unwrap());
                    let _ = black_box(&anderson_darling_test(&y, &x_vars).unwrap());
                    let _ = black_box(&cooks_distance_test(&y, &x_vars).unwrap());
                    let _ = black_box(&dfbetas_test(&y, &x_vars).unwrap());
                    let _ = black_box(&dffits_test(&y, &x_vars).unwrap());
                    let _ = black_box(&vif_test(&y, &x_vars).unwrap());
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    diagnostics,
    bench_rainbow_test,
    bench_harvey_collier_test,
    bench_breusch_pagan_test,
    bench_white_test,
    bench_jarque_bera_test,
    bench_durbin_watson_test,
    bench_breusch_godfrey_test,
    bench_shapiro_wilk_test,
    bench_anderson_darling_test,
    bench_cooks_distance_test,
    bench_reset_test,
    bench_dfbetas_test,
    bench_dffits_test,
    bench_vif_test,
    bench_full_diagnostics
);
criterion_main!(diagnostics);
