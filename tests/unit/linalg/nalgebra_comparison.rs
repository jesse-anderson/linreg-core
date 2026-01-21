// ============================================================================
// Nalgebra Comparison Tests
// ============================================================================
//
// Compare our custom implementation against nalgebra to debug differences.
// These tests are ONLY for development/debugging - they verify our custom
// implementation produces the same results as the established nalgebra library.
//
// DISABLED: nalgebra not in dependencies
// To enable, add nalgebra to dev-dependencies and uncomment the module below.

#[allow(dead_code)]
use super::common::assert_close;

#[allow(dead_code)]
use linreg_core::linalg::Matrix;

// Nalgebra comparison tests - enabled when nalgebra is available (dev-dependencies)
/*
#[cfg(test)]
mod nalgebra_comparison {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    const TOLERANCE: f64 = 1e-9;

    /// Helper to convert our Matrix to nalgebra DMatrix
    fn to_nalgebra_matrix(m: &Matrix) -> DMatrix<f64> {
        DMatrix::from_row_slice(m.rows, m.cols, &m.data)
    }

    /// Helper to convert nalgebra DMatrix to our Matrix
    fn from_nalgebra_matrix(nm: &DMatrix<f64>) -> Matrix {
        Matrix::new(nm.nrows(), nm.ncols(), nm.iter().cloned().collect())
    }

    #[test]
    fn compare_qr_decomposition() {
        // Test matrix: same as our QR tests
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,  // Changed from 9.0 for full rank
            10.0, 11.0, 12.0,
        ];
        let our_m = Matrix::new(4, 3, data.clone());
        let na_m = to_nalgebra_matrix(&our_m);

        // Our QR
        let (our_q, our_r) = our_m.qr();

        // Nalgebra QR
        let na_qr = na_m.qr();
        let (na_q, na_r) = na_qr.unpack();

        println!("\n=== QR Decomposition Comparison ===");
        println!("Our Q (first 3 rows):");
        for i in 0..3.min(our_q.rows) {
            for j in 0..our_q.cols.min(3) {
                print!("{:12.6} ", our_q.get(i, j));
            }
            println!();
        }
        println!("\nNalgebra Q (first 3 rows):");
        for i in 0..3.min(na_q.nrows()) {
            for j in 0..na_q.ncols().min(3) {
                print!("{:12.6} ", na_q[(i, j)]);
            }
            println!();
        }

        println!("\nOur R:");
        for i in 0..our_r.rows.min(4) {
            for j in 0..our_r.cols.min(3) {
                print!("{:12.6} ", our_r.get(i, j));
            }
            println!();
        }
        println!("\nNalgebra R:");
        for i in 0..na_r.nrows().min(4) {
            for j in 0..na_r.ncols().min(3) {
                print!("{:12.6} ", na_r[(i, j)]);
            }
            println!();
        }

        // Note: Q and R can differ by sign conventions, but A = Q*R should be the same
        // Verify reconstruction matches
        let our_reconstructed = our_q.matmul(&our_r);
        let na_reconstructed = &na_q * &na_r;

        println!("\nReconstruction comparison (our A vs Q*R):");
        for i in 0..our_m.rows {
            for j in 0..our_m.cols {
                let our_val = our_reconstructed.get(i, j);
                let orig_val = our_m.get(i, j);
                println!("  [{},{}]: original={:.6}, qr_recon={:.6}, diff={:.2e}",
                    i, j, orig_val, our_val, (orig_val - our_val).abs());
            }
        }

        // Both should reconstruct original
        for i in 0..our_m.rows {
            for j in 0..our_m.cols {
                assert_close(
                    our_reconstructed.get(i, j),
                    our_m.get(i, j),
                    TOLERANCE,
                    &format!("our reconstruction [{},{}]", i, j)
                );
                assert_close(
                    na_reconstructed[(i, j)],
                    our_m.get(i, j),
                    TOLERANCE,
                    &format!("nalgebra reconstruction [{},{}]", i, j)
                );
            }
        }
    }

    #[test]
    fn compare_matrix_multiplication() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];  // 2x2
        let b_data = vec![5.0, 6.0, 7.0, 8.0];  // 2x2

        let our_a = Matrix::new(2, 2, a_data.clone());
        let our_b = Matrix::new(2, 2, b_data.clone());

        let na_a = to_nalgebra_matrix(&our_a);
        let na_b = to_nalgebra_matrix(&our_b);

        let our_result = our_a.matmul(&our_b);
        let na_result = &na_a * &na_b;

        println!("\n=== Matrix Multiplication Comparison ===");
        println!("A:\n{:?}", &a_data);
        println!("B:\n{:?}", &b_data);
        println!("Our result: {:?}", our_result.data);
        println!("Nalgebra result: {:?}", na_result.iter().collect::<Vec<_>>());

        for i in 0..2 {
            for j in 0..2 {
                assert_close(
                    our_result.get(i, j),
                    na_result[(i, j)],
                    TOLERANCE,
                    &format!("matmul [{},{}]", i, j)
                );
            }
        }
    }

    #[test]
    fn compare_matrix_inverse() {
        // Use a well-conditioned matrix
        let data = vec![
            4.0, 7.0, 2.0,
            3.0, 6.0, 1.0,
            2.0, 5.0, 3.0,
        ];

        let our_m = Matrix::new(3, 3, data.clone());
        let na_m = to_nalgebra_matrix(&our_m);

        let our_inv = our_m.invert().expect("should invert");
        let na_inv = na_m.clone().try_inverse().expect("nalgebra should invert");

        println!("\n=== Matrix Inverse Comparison ===");
        println!("Original matrix:");
        for i in 0..3 {
            println!("  [{},{},{}]", data[i*3], data[i*3+1], data[i*3+2]);
        }

        println!("\nOur inverse:");
        for i in 0..3 {
            println!("  [{},{},{}]", our_inv.get(i,0), our_inv.get(i,1), our_inv.get(i,2));
        }

        println!("\nNalgebra inverse:");
        for i in 0..3 {
            println!("  [{},{},{}]", na_inv[(i,0)], na_inv[(i,1)], na_inv[(i,2)]);
        }

        // Compare values
        for i in 0..3 {
            for j in 0..3 {
                assert_close(
                    our_inv.get(i, j),
                    na_inv[(i, j)],
                    1e-8,  // Looser tolerance for inversion
                    &format!("inverse [{},{}]", i, j)
                );
            }
        }

        // Verify A * A^-1 = I for both
        let our_product = our_m.matmul(&our_inv);
        let na_product = &na_m * &na_inv;

        println!("\nOur A * A^-1:");
        for i in 0..3 {
            println!("  [{},{},{}]", our_product.get(i,0), our_product.get(i,1), our_product.get(i,2));
        }

        println!("\nNalgebra A * A^-1:");
        for i in 0..3 {
            println!("  [{},{},{}]", na_product[(i,0)], na_product[(i,1)], na_product[(i,2)]);
        }
    }

    #[test]
    fn compare_chol2inv() {
        // X is a 4x3 matrix (tall, as used in OLS)
        let x_data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,  // Changed from 9.0 for full rank
            2.0, 3.0, 4.0,
        ];

        let our_x = Matrix::new(4, 3, x_data.clone());
        let na_x = to_nalgebra_matrix(&our_x);

        // Our chol2inv
        let our_result = our_x.chol2inv_from_qr().expect("chol2inv should work");

        // Nalgebra: compute X'X then invert
        let na_xt = na_x.transpose();
        let na_xtx = &na_xt * &na_x;
        let na_result = na_xtx.try_inverse().expect("X'X should be invertible");

        println!("\n=== chol2inv Comparison ===");
        println!("X (4x3):");
        for i in 0..4 {
            println!("  [{},{},{}]", x_data[i*3], x_data[i*3+1], x_data[i*3+2]);
        }

        println!("\nOur (X'X)^(-1):");
        for i in 0..3 {
            println!("  [{},{},{}]", our_result.get(i,0), our_result.get(i,1), our_result.get(i,2));
        }

        println!("\nNalgebra (X'X)^(-1):");
        for i in 0..3 {
            println!("  [{},{},{}]", na_result[(i,0)], na_result[(i,1)], na_result[(i,2)]);
        }

        println!("\nDifferences:");
        for i in 0..3 {
            for j in 0..3 {
                let our_val = our_result.get(i, j);
                let na_val = na_result[(i, j)];
                let diff = (our_val - na_val).abs();
                println!("  [{},{}]: our={:.10e}, na={:.10e}, diff={:.2e}",
                    i, j, our_val, na_val, diff);
            }
        }

        // Compare
        for i in 0..3 {
            for j in 0..3 {
                assert_close(
                    our_result.get(i, j),
                    na_result[(i, j)],
                    1e-9,
                    &format!("chol2inv [{},{}]", i, j)
                );
            }
        }
    }

    #[test]
    fn compare_ols_solution() {
        // Test the complete OLS solution: beta = (X'X)^(-1) X' y
        // This is what we actually use in regression

        let x_data = vec![
            1.0, 1.0,  // intercept, x1
            1.0, 2.0,
            1.0, 3.0,
            1.0, 4.0,
            1.0, 5.0,
        ];
        let y_data = vec![2.1, 4.9, 7.1, 9.8, 12.2];  // Approximately y = 0.5 + 2.4*x

        let our_x = Matrix::new(5, 2, x_data.clone());

        let na_x = to_nalgebra_matrix(&our_x);
        let na_y = DMatrix::from_column_slice(5, 1, &y_data);

        // Our solution: beta = (X'X)^(-1) X' y
        let our_xtx_inv = our_x.chol2inv_from_qr().expect("chol2inv");
        let our_xt = our_x.transpose();
        let our_xty = our_xt.mul_vec(&y_data);
        let our_beta = our_xtx_inv.mul_vec(&our_xty);

        // Nalgebra solution
        let na_xt = na_x.transpose();
        let na_xtx = &na_xt * &na_x;
        let na_xtx_inv = na_xtx.try_inverse().expect("X'X invertible");
        let na_xty = &na_xt * &na_y;
        let na_beta = &na_xtx_inv * &na_xty;

        println!("\n=== OLS Solution Comparison ===");
        println!("X (design matrix with intercept):");
        for i in 0..5 {
            println!("  [{},{}]", x_data[i*2], x_data[i*2+1]);
        }
        println!("y: {:?}", y_data);

        println!("\nOur beta: [{:.10}, {:.10}]", our_beta[0], our_beta[1]);
        println!("Nalgebra beta: [{:.10}, {:.10}]", na_beta[0], na_beta[1]);

        println!("\nActual expected values: intercept≈-0.31, slope≈2.51");

        // Both should give same results
        assert_close(our_beta[0], na_beta[0], 1e-9, "beta[0] (intercept)");
        assert_close(our_beta[1], na_beta[1], 1e-9, "beta[1] (slope)");
    }

    #[test]
    fn compare_qr_orthogonality() {
        // Q should be orthogonal: Q^T * Q = I
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ];

        let our_m = Matrix::new(3, 3, data);
        let na_m = to_nalgebra_matrix(&our_m);

        let (our_q, _) = our_m.qr();
        let na_qr = na_m.qr();
        let (na_q, _) = na_qr.unpack();

        // Our Q^T * Q
        let our_qt = our_q.transpose();
        let our_qt_q = our_qt.matmul(&our_q);

        // Nalgebra Q^T * Q
        let na_qt = na_q.transpose();
        let na_qt_q = &na_qt * &na_q;

        println!("\n=== Q Orthogonality Comparison ===");
        println!("Our Q^T * Q (should be identity):");
        for i in 0..3 {
            println!("  [{},{},{}]", our_qt_q.get(i,0), our_qt_q.get(i,1), our_qt_q.get(i,2));
        }

        println!("\nNalgebra Q^T * Q:");
        for i in 0..3 {
            println!("  [{},{},{}]", na_qt_q[(i,0)], na_qt_q[(i,1)], na_qt_q[(i,2)]);
        }

        // Both should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_close(our_qt_q.get(i, j), expected, 1e-9, &format!("our Q^T*Q [{},{}]", i, j));
                assert_close(na_qt_q[(i, j)], expected, 1e-9, &format!("na Q^T*Q [{},{}]", i, j));
            }
        }
    }

    #[test]
    fn compare_transpose() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];  // 3x4 matrix

        let our_m = Matrix::new(3, 4, data);
        let na_m = to_nalgebra_matrix(&our_m);

        let our_t = our_m.transpose();
        let na_t = na_m.transpose();

        println!("\n=== Transpose Comparison ===");
        println!("Original: 3x4, Transposed: 4x3");
        println!("Our transpose:");
        for i in 0..4 {
            println!("  [{},{},{}]", our_t.get(i,0), our_t.get(i,1), our_t.get(i,2));
        }

        println!("\nNalgebra transpose:");
        for i in 0..4 {
            println!("  [{},{},{}]", na_t[(i,0)], na_t[(i,1)], na_t[(i,2)]);
        }

        // Compare dimensions
        assert_eq!(our_t.rows, 4, "transposed rows should be 4");
        assert_eq!(our_t.cols, 3, "transposed cols should be 3");

        // Compare values
        for i in 0..4 {
            for j in 0..3 {
                assert_close(
                    our_t.get(i, j),
                    na_t[(i, j)],
                    TOLERANCE,
                    &format!("transpose [{},{}]", i, j)
                );
            }
        }
    }

    #[test]
    fn compare_mul_vec() {
        // Matrix x vector: used in OLS for Q^T * y
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ];  // 3x3 matrix
        let vec = vec![2.0, 3.0, 4.0];

        let our_m = Matrix::new(3, 3, data);
        let na_m = to_nalgebra_matrix(&our_m);
        let na_vec = DVector::from_vec(vec.clone());

        let our_result = our_m.mul_vec(&vec);
        let na_result = &na_m * &na_vec;

        println!("\n=== Matrix x Vector Comparison ===");
        println!("Vector: {:?}", vec);
        println!("Our result: {:?}", our_result);
        println!("Nalgebra result: {:?}", na_result.iter().copied().collect::<Vec<_>>());

        for i in 0..3 {
            assert_close(
                our_result[i],
                na_result[i],
                TOLERANCE,
                &format!("mul_vec [{}]", i)
            );
        }
    }

    #[test]
    fn compare_invert_upper_triangular() {
        // Create an upper triangular matrix
        let data = vec![
            2.0, 3.0, 1.0,
            0.0, 4.0, 2.0,
            0.0, 0.0, 3.0,
        ];  // Upper triangular

        let our_m = Matrix::new(3, 3, data.clone());
        let na_m = to_nalgebra_matrix(&our_m);

        let our_inv = our_m.invert_upper_triangular().expect("should invert upper triangular");

        // Nalgebra doesn't have a specialized upper triangular inverse,
        // so we use the general inverse for comparison
        let na_inv = na_m.try_inverse().expect("nalgebra should invert");

        println!("\n=== Upper Triangular Inverse Comparison ===");
        println!("Original (upper triangular):");
        for i in 0..3 {
            println!("  [{},{},{}]", data[i*3], data[i*3+1], data[i*3+2]);
        }

        println!("\nOur inverse:");
        for i in 0..3 {
            println!("  [{},{},{}]", our_inv.get(i,0), our_inv.get(i,1), our_inv.get(i,2));
        }

        println!("\nNalgebra inverse:");
        for i in 0..3 {
            println!("  [{},{},{}]", na_inv[(i,0)], na_inv[(i,1)], na_inv[(i,2)]);
        }

        // Compare values
        for i in 0..3 {
            for j in 0..3 {
                assert_close(
                    our_inv.get(i, j),
                    na_inv[(i, j)],
                    1e-9,
                    &format!("upper triangular inverse [{},{}]", i, j)
                );
            }
        }

        // Verify A * A^-1 = I
        let our_product = our_m.matmul(&our_inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_close(
                    our_product.get(i, j),
                    expected,
                    1e-9,
                    &format!("A * A^-1 = I [{},{}]", i, j)
                );
            }
        }
    }

    #[test]
    fn compare_qr_solve() {
        // Test QR-based linear system solve: Ax = b
        // Solution: x = R^(-1) * Q^T * b
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,  // Full rank 3x3
        ];
        let b = vec![6.0, 15.0, 25.0];  // RHS vector

        let our_a = Matrix::new(3, 3, data);
        let na_a = to_nalgebra_matrix(&our_a);
        let na_b = DVector::from_vec(b.clone());

        // Our QR solve: x = R^(-1) * Q^T * b
        let (our_q, our_r) = our_a.qr();
        let our_qt = our_q.transpose();
        let our_qtb = our_qt.mul_vec(&b);
        let our_r_inv = our_r.invert_upper_triangular().expect("R should be invertible");
        let our_x_mat = our_r_inv.matmul(&Matrix::new(3, 1, our_qtb));
        let our_x: Vec<f64> = (0..3).map(|i| our_x_mat.get(i, 0)).collect();

        // Nalgebra QR solve
        let na_qr = na_a.qr();
        let na_x = na_qr.solve(&na_b).expect("nalgebra should solve");

        println!("\n=== QR Solve Comparison ===");
        println!("b vector: {:?}", b);
        println!("Our solution: {:?}", our_x);
        println!("Nalgebra solution: {:?}", na_x.iter().copied().collect::<Vec<f64>>());

        // Compare solutions
        for i in 0..3 {
            assert_close(
                our_x[i],
                na_x[i],
                1e-9,
                &format!("QR solve x[{}]", i)
            );
        }

        // Verify Ax = b for our solution
        let verification = our_a.mul_vec(&our_x);
        println!("\nVerification (A * x):");
        for i in 0..3 {
            println!("  [{}] = {:.6} (expected {:.6}), diff = {:.2e}",
                i, verification[i], b[i], (verification[i] - b[i]).abs());
            assert_close(
                verification[i],
                b[i],
                1e-9,
                &format!("Ax = b verification [{}]", i)
            );
        }
    }
}
*/
