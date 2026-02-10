//! Linear algebra benchmarks.
//!
//! Benchmarks matrix operations, decompositions, solvers, and vector operations.
//! Covers all hot-path operations used by OLS, regularized regression, diagnostics,
//! WLS, and LOESS fitting.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use linreg_core::linalg::{
    fit_and_predict_linpack, fit_ols_linpack, vec_dot, vec_l2_norm, vec_mean, vec_sub, Matrix,
};

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

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(&m).invert_upper_triangular())
        });
    }

    group.finish();
}

/// Benchmarks full matrix inversion via QR decomposition.
fn bench_matrix_invert(c: &mut Criterion) {
    let sizes = vec![5, 10, 20, 50, 100, 200];

    let mut group = c.benchmark_group("matrix_invert");

    for &n in &sizes {
        let m = make_matrix(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(&m).invert())
        });
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

/// Benchmarks SVD decomposition.
///
/// SVD is the fallback solver for ill-conditioned matrices in WLS/LOESS.
/// O(mn²) for an m×n matrix, significantly more expensive than QR.
fn bench_svd(c: &mut Criterion) {
    // SVD is expensive — keep sizes modest
    let sizes = vec![
        (10, 5),
        (50, 10),
        (100, 20),
        (500, 50),
        (1000, 100),
    ];

    let mut group = c.benchmark_group("svd");

    for &(rows, cols) in &sizes {
        let m = make_matrix(rows, cols);

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", rows, cols)),
            &(rows, cols),
            |b, _| b.iter(|| black_box(&m).svd()),
        );
    }

    group.finish();
}

/// Benchmarks SVD solve (pseudoinverse solution).
///
/// Used as fallback when QR fails on singular/near-singular matrices.
fn bench_svd_solve(c: &mut Criterion) {
    let sizes = vec![
        (10, 5),
        (50, 10),
        (100, 20),
        (500, 50),
    ];

    let mut group = c.benchmark_group("svd_solve");

    for &(rows, cols) in &sizes {
        let m = make_matrix(rows, cols);
        let svd_result = m.svd();
        let rhs: Vec<f64> = (0..rows).map(|i| (i as f64 * 0.03).sin()).collect();

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", rows, cols)),
            &(rows, cols),
            |b, _| b.iter(|| black_box(&m).svd_solve(black_box(&svd_result), black_box(&rhs))),
        );
    }

    group.finish();
}

/// Benchmarks chol2inv_from_qr (Cholesky-like inverse via QR).
///
/// Used by Harvey-Collier test for recursive residual computation.
/// Operates on square matrices (p × p design matrix subsets).
fn bench_chol2inv_from_qr(c: &mut Criterion) {
    let sizes = vec![5, 10, 20, 50, 100, 200];

    let mut group = c.benchmark_group("chol2inv_from_qr");

    for &n in &sizes {
        // Create a well-conditioned square matrix (rows of a design matrix)
        let m = make_matrix(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(&m).chol2inv_from_qr())
        });
    }

    group.finish();
}

/// Benchmarks vec_sub (element-wise vector subtraction).
///
/// Used in residual computation throughout OLS and diagnostics (5+ calls).
fn bench_vec_sub(c: &mut Criterion) {
    let sizes = vec![10, 100, 1000, 10000, 100000];

    let mut group = c.benchmark_group("vec_sub");

    for &n in &sizes {
        let v1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let v2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.02).cos()).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| vec_sub(black_box(&v1), black_box(&v2)))
        });
    }

    group.finish();
}

/// Benchmarks LINPACK QR OLS solver (fit_ols_linpack).
///
/// Used by White test for auxiliary regressions. Different code path from .qr().
fn bench_fit_ols_linpack(c: &mut Criterion) {
    let sizes = vec![
        (50, 10),
        (100, 20),
        (500, 50),
        (1000, 100),
    ];

    let mut group = c.benchmark_group("fit_ols_linpack");

    for &(rows, cols) in &sizes {
        let m = make_matrix(rows, cols);
        let y: Vec<f64> = (0..rows).map(|i| (i as f64 * 0.03).sin() + 1.0).collect();

        group.throughput(Throughput::Elements(rows as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", rows, cols)),
            &(rows, cols),
            |b, _| b.iter(|| fit_ols_linpack(black_box(&y), black_box(&m))),
        );
    }

    group.finish();
}

/// Benchmarks LINPACK QR OLS solver with predictions (fit_and_predict_linpack).
///
/// Used by White test. Returns fitted values directly.
fn bench_fit_and_predict_linpack(c: &mut Criterion) {
    let sizes = vec![
        (50, 10),
        (100, 20),
        (500, 50),
        (1000, 100),
    ];

    let mut group = c.benchmark_group("fit_and_predict_linpack");

    for &(rows, cols) in &sizes {
        let m = make_matrix(rows, cols);
        let y: Vec<f64> = (0..rows).map(|i| (i as f64 * 0.03).sin() + 1.0).collect();

        group.throughput(Throughput::Elements(rows as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", rows, cols)),
            &(rows, cols),
            |b, _| b.iter(|| fit_and_predict_linpack(black_box(&y), black_box(&m))),
        );
    }

    group.finish();
}

/// Benchmarks column dot product via scalar .get() access.
///
/// This is the #1 BLAS/SIMD optimization target. The elastic net coordinate
/// descent inner loop (elastic_net.rs:826-828) computes:
///   sum += x.get(i, j) * residuals[i]   for i in 0..n
/// This is a column-dot-product pattern that would map to BLAS ddot or SIMD.
fn bench_column_dot_scalar(c: &mut Criterion) {
    let sizes = vec![
        (100, 20),
        (500, 50),
        (1000, 100),
        (5000, 100),
        (10000, 100),
    ];

    let mut group = c.benchmark_group("column_dot_scalar_get");

    for &(n, p) in &sizes {
        let x = make_matrix(n, p);
        let residuals: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).sin()).collect();
        let col = p / 2; // Pick a middle column

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, p)),
            &(n, p),
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for i in 0..n {
                        sum += x.get(i, col) * residuals[i];
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks column axpy (residual update) via scalar .get() access.
///
/// This is the #2 BLAS/SIMD optimization target. The elastic net coordinate
/// descent inner loop (elastic_net.rs:858-860) computes:
///   residuals[i] -= x.get(i, j) * delta   for i in 0..n
/// This is a column-axpy pattern that would map to BLAS daxpy or SIMD.
fn bench_column_axpy_scalar(c: &mut Criterion) {
    let sizes = vec![
        (100, 20),
        (500, 50),
        (1000, 100),
        (5000, 100),
        (10000, 100),
    ];

    let mut group = c.benchmark_group("column_axpy_scalar_get");

    for &(n, p) in &sizes {
        let x = make_matrix(n, p);
        let col = p / 2;
        let delta = 0.42;

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, p)),
            &(n, p),
            |b, _| {
                b.iter(|| {
                    let mut residuals: Vec<f64> =
                        (0..n).map(|i| (i as f64 * 0.07).sin()).collect();
                    for i in 0..n {
                        residuals[i] -= x.get(i, col) * delta;
                    }
                    black_box(residuals)
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks a full coordinate descent iteration (dot + threshold + axpy).
///
/// Simulates one feature update from the elastic net inner loop to measure
/// the combined cost of the column-dot and column-axpy pattern together,
/// which is the actual bottleneck profile.
fn bench_coordinate_descent_step(c: &mut Criterion) {
    let sizes = vec![
        (100, 20),
        (500, 50),
        (1000, 100),
        (5000, 100),
        (10000, 100),
    ];

    let mut group = c.benchmark_group("coord_descent_step");

    for &(n, p) in &sizes {
        let x = make_matrix(n, p);
        let col = p / 2;
        let lambda = 0.1;
        let alpha = 0.5;
        let col_norm_sq = 1.0; // Normalized column

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("size", format!("{}_{}", n, p)),
            &(n, p),
            |b, _| {
                b.iter(|| {
                    let mut residuals: Vec<f64> =
                        (0..n).map(|i| (i as f64 * 0.07).sin()).collect();
                    let beta_old = 0.5;

                    // Column dot product: rho = X_j' * residuals + norm_sq * beta
                    let mut partial = 0.0;
                    for i in 0..n {
                        partial += x.get(i, col) * residuals[i];
                    }
                    let rho = partial + col_norm_sq * beta_old;

                    // Soft threshold
                    let threshold = lambda * alpha;
                    let numerator = if rho > threshold {
                        rho - threshold
                    } else if rho < -threshold {
                        rho + threshold
                    } else {
                        0.0
                    };
                    let beta_new = numerator / (col_norm_sq + lambda * (1.0 - alpha));

                    // Column axpy: residuals -= X_j * (beta_new - beta_old)
                    let delta = beta_new - beta_old;
                    if delta != 0.0 {
                        for i in 0..n {
                            residuals[i] -= x.get(i, col) * delta;
                        }
                    }

                    black_box((beta_new, residuals))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    linalg,
    bench_transpose,
    bench_matmul,
    bench_mul_vec,
    bench_qr_decomposition,
    bench_svd,
    bench_svd_solve,
    bench_chol2inv_from_qr,
    bench_invert_upper_triangular,
    bench_matrix_invert,
    bench_fit_ols_linpack,
    bench_fit_and_predict_linpack,
    bench_vector_ops,
    bench_vec_sub,
    bench_column_dot_scalar,
    bench_column_axpy_scalar,
    bench_coordinate_descent_step
);
criterion_main!(linalg);
