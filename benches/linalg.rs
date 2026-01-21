//! Linear algebra benchmarks.
//!
//! Benchmarks matrix operations, QR decomposition, and linear solvers.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use linreg_core::linalg::{Matrix, vec_mean, vec_dot, vec_l2_norm};

/// Creates a random matrix of given dimensions.
fn make_matrix(rows: usize, cols: usize) -> Matrix {
    let data: Vec<f64> = (0..(rows * cols))
        .map(|i| {
            let x = (i as f64) * 0.01;
            x.sin() + x.cos()
        })
        .collect();
    Matrix::new(rows, cols, data)
}

/// Benchmarks matrix transpose.
fn bench_transpose(c: &mut Criterion) {
    let sizes = vec![
        (10, 10),
        (50, 50),
        (100, 100),
        (500, 500),
        (1000, 1000),
        (100, 500),
        (500, 100),
    ];

    let mut group = c.benchmark_group("matrix_transpose");

    for &(rows, cols) in &sizes {
        let m = make_matrix(rows, cols);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", rows, cols)),
            &(rows, cols),
            |b, _| b.iter(|| black_box(&m).transpose()),
        );
    }

    group.finish();
}

/// Benchmarks matrix multiplication.
fn bench_matmul(c: &mut Criterion) {
    // Format: (A_rows, A_cols, B_cols) - B_rows must equal A_cols
    let sizes = vec![
        (10, 10, 10),
        (50, 50, 50),
        (100, 100, 100),
        (200, 200, 200),
        (100, 50, 100),
        (500, 100, 50),
        (1000, 100, 100),
    ];

    let mut group = c.benchmark_group("matrix_matmul");

    for &(a_rows, a_cols, b_cols) in &sizes {
        let a = make_matrix(a_rows, a_cols);
        let b = make_matrix(a_cols, b_cols);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}_{}", a_rows, a_cols, b_cols)),
            &(a_rows, a_cols, b_cols),
            |bencher, _| bencher.iter(|| black_box(&a).matmul(black_box(&b))),
        );
    }

    group.finish();
}

/// Benchmarks matrix-vector multiplication.
fn bench_mul_vec(c: &mut Criterion) {
    let sizes = vec![
        (10, 10),
        (50, 50),
        (100, 100),
        (500, 100),
        (1000, 100),
        (5000, 100),
        (10000, 100),
    ];

    let mut group = c.benchmark_group("matrix_mul_vec");

    for &(rows, cols) in &sizes {
        let m = make_matrix(rows, cols);
        let v: Vec<f64> = (0..cols).map(|i| (i as f64 * 0.1).sin()).collect();

        group.throughput(Throughput::Elements(rows as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", rows, cols)),
            &(rows, cols),
            |b, _| b.iter(|| black_box(&m).mul_vec(black_box(&v))),
        );
    }

    group.finish();
}

/// Benchmarks QR decomposition using Householder reflections.
fn bench_qr_decomposition(c: &mut Criterion) {
    // QR requires rows >= cols (tall matrices)
    let sizes = vec![
        (10, 5),
        (50, 10),
        (100, 20),
        (500, 50),
        (1000, 100),
        (5000, 100),
        (10000, 100),
        (1000, 500),
    ];

    let mut group = c.benchmark_group("qr_decomposition");

    for &(rows, cols) in &sizes {
        let m = make_matrix(rows, cols);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", rows, cols)),
            &(rows, cols),
            |b, _| b.iter(|| black_box(&m).qr()),
        );
    }

    group.finish();
}

/// Benchmarks upper triangular matrix inversion.
fn bench_invert_upper_triangular(c: &mut Criterion) {
    let sizes = vec![5, 10, 20, 50, 100, 200, 500];

    let mut group = c.benchmark_group("invert_upper_triangular");

    for &n in &sizes {
        // Create an upper triangular matrix
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                data[i * n + j] = (i + j + 1) as f64 * 0.1;
            }
            // Ensure non-zero diagonal
            data[i * n + i] = (i + 1) as f64;
        }
        let m = Matrix::new(n, n, data);

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &n,
            |b, _| b.iter(|| black_box(&m).invert_upper_triangular()),
        );
    }

    group.finish();
}

/// Benchmarks full matrix inversion via QR decomposition.
fn bench_matrix_invert(c: &mut Criterion) {
    let sizes = vec![5, 10, 20, 50, 100, 200];

    let mut group = c.benchmark_group("matrix_invert");

    for &n in &sizes {
        let m = make_matrix(n, n);

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &n,
            |b, _| b.iter(|| black_box(&m).invert()),
        );
    }

    group.finish();
}

/// Benchmarks vector operations.
fn bench_vector_ops(c: &mut Criterion) {
    let sizes = vec![10, 100, 1000, 10000, 100000];

    // vec_mean
    let mut group = c.benchmark_group("vec_mean");
    for &n in &sizes {
        let v: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| vec_mean(black_box(&v)))
        });
    }
    group.finish();

    // vec_dot
    let mut group = c.benchmark_group("vec_dot");
    for &n in &sizes {
        let v1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let v2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.02).cos()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| vec_dot(black_box(&v1), black_box(&v2)))
        });
    }
    group.finish();

    // vec_l2_norm
    let mut group = c.benchmark_group("vec_l2_norm");
    for &n in &sizes {
        let v: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| vec_l2_norm(black_box(&v)))
        });
    }
    group.finish();
}

criterion_group!(
    linalg,
    bench_transpose,
    bench_matmul,
    bench_mul_vec,
    bench_qr_decomposition,
    bench_invert_upper_triangular,
    bench_matrix_invert,
    bench_vector_ops
);
criterion_main!(linalg);
