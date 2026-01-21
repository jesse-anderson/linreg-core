// ============================================================================
// Property-Based Tests
// ============================================================================
//
// Property-based tests using the proptest crate to verify mathematical
// properties hold for random inputs.

use linreg_core::linalg::{Matrix, vec_dot, vec_l2_norm, vec_scale};
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_double_transpose(rows in 1..10usize, cols in 1..10usize) {
        // (A^T)^T = A for any matrix
        let size = rows * cols;
        let values: Vec<f64> = (0..size).map(|i| i as f64 * 1.0).collect();
        let a = Matrix::new(rows, cols, values);
        let at = a.transpose();
        let att = at.transpose();

        prop_assert_eq!(att.rows, rows);
        prop_assert_eq!(att.cols, cols);
        for i in 0..rows {
            for j in 0..cols {
                prop_assert!((att.get(i, j) - a.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn prop_transpose_product(
        m1 in 2..5usize,
        n in 2..5usize,
        m2 in 2..5usize,
    ) {
        // (A * B)^T = B^T * A^T
        let a_vals: Vec<f64> = (0..(m1 * n)).map(|i| i as f64 * 1.0).collect();
        let b_vals: Vec<f64> = (0..(n * m2)).map(|i| i as f64 * 1.0).collect();

        let a = Matrix::new(m1, n, a_vals);
        let b = Matrix::new(n, m2, b_vals);

        let ab = a.matmul(&b);
        let ab_t = ab.transpose();

        let a_t = a.transpose();
        let b_t = b.transpose();
        let bt_at = b_t.matmul(&a_t);

        prop_assert_eq!(ab_t.rows, bt_at.rows);
        prop_assert_eq!(ab_t.cols, bt_at.cols);

        for i in 0..ab_t.rows {
            for j in 0..ab_t.cols {
                let diff = (ab_t.get(i, j) - bt_at.get(i, j)).abs();
                prop_assert!(diff < 1e-8, "Difference at [{},{}]: {} vs {}", i, j, ab_t.get(i, j), bt_at.get(i, j));
            }
        }
    }

    #[test]
    fn prop_q_orthogonal(
        m in 2..6usize,
        n_max in 2usize..6usize,
    ) {
        // Q from QR decomposition is always orthogonal: Q^T * Q ≈ I
        let n = n_max.min(m);
        let a_vals: Vec<f64> = (0..(m * n)).map(|i| (i as f64) * 2.0 - 10.0).collect();
        let a = Matrix::new(m, n, a_vals);

        let (q, _r) = a.qr();
        let q_t = q.transpose();
        let qt_q = q_t.matmul(&q);

        // Check diagonal is close to 1
        for i in 0..q.cols {
            let diff = (qt_q.get(i, i) - 1.0).abs();
            prop_assert!(diff < 1e-6, "Diagonal element {} not close to 1: {}", i, qt_q.get(i, i));
        }

        // Check off-diagonal is close to 0
        for i in 0..q.cols {
            for j in 0..q.cols {
                if i != j {
                    let val = qt_q.get(i, j).abs();
                    prop_assert!(val < 1e-6, "Off-diagonal [{},{}] not close to 0: {}", i, j, val);
                }
            }
        }
    }

    // ========================================================================
    // Matrix Multiplication Properties
    // ========================================================================

    #[test]
    fn prop_matmul_associative(
        m in 2..4usize,
        n in 2..4usize,
        p in 2..4usize,
    ) {
        // (A * B) * C = A * (B * C) when dimensions are compatible
        let a_vals: Vec<f64> = (0..(m * n)).map(|i| (i as f64) * 0.1).collect();
        let b_vals: Vec<f64> = (0..(n * p)).map(|i| (i as f64) * 0.1).collect();
        let c_vals: Vec<f64> = (0..(p * p)).map(|i| (i as f64) * 0.1).collect();

        let a = Matrix::new(m, n, a_vals);
        let b = Matrix::new(n, p, b_vals);
        let c = Matrix::new(p, p, c_vals);

        let ab_c = a.matmul(&b).matmul(&c);
        let a_bc = a.matmul(&b.matmul(&c));

        prop_assert_eq!(ab_c.rows, a_bc.rows);
        prop_assert_eq!(ab_c.cols, a_bc.cols);

        for i in 0..ab_c.rows {
            for j in 0..ab_c.cols {
                let diff = (ab_c.get(i, j) - a_bc.get(i, j)).abs();
                prop_assert!(diff < 1e-8, "Associativity failed at [{},{}]: diff = {}", i, j, diff);
            }
        }
    }

    #[test]
    fn prop_matmul_distributive(
        m in 2..4usize,
        n in 2..4usize,
        p in 2..4usize,
    ) {
        // A * (B + C) = A*B + A*C when dimensions are compatible
        let a_vals: Vec<f64> = (0..(m * n)).map(|i| (i as f64) * 0.1).collect();
        let b_vals: Vec<f64> = (0..(n * p)).map(|i| (i as f64) * 0.1).collect();
        let c_vals: Vec<f64> = (0..(n * p)).map(|i| ((i + 100) as f64) * 0.1).collect();

        let a = Matrix::new(m, n, a_vals);
        let b = Matrix::new(n, p, b_vals);
        let c_orig = Matrix::new(n, p, c_vals.clone());
        let mut c = Matrix::new(n, p, c_vals);

        // Compute B + C manually
        for i in 0..n {
            for j in 0..p {
                let sum = b.get(i, j) + c.get(i, j);
                c.set(i, j, sum);
            }
        }

        // Left side: A * (B + C)
        let left = a.matmul(&c);

        // Right side: A*B + A*C
        let ab = a.matmul(&b);
        let ac = a.matmul(&c_orig);
        let mut right = Matrix::zeros(m, p);
        for i in 0..m {
            for j in 0..p {
                let sum = ab.get(i, j) + ac.get(i, j);
                right.set(i, j, sum);
            }
        }

        for i in 0..m {
            for j in 0..p {
                let diff = (left.get(i, j) - right.get(i, j)).abs();
                prop_assert!(diff < 1e-8, "Distributivity failed at [{},{}]: diff = {}", i, j, diff);
            }
        }
    }

    #[test]
    fn prop_matmul_identity(
        n in 1..10usize,
    ) {
        // A * I = I * A = A for any square matrix A
        let size = n * n;
        let a_vals: Vec<f64> = (0..size).map(|i| (i as f64) * 0.5).collect();
        let a = Matrix::new(n, n, a_vals);
        let identity = Matrix::identity(n);

        let ai = a.matmul(&identity);
        let ia = identity.matmul(&a);

        for i in 0..n {
            for j in 0..n {
                prop_assert!((ai.get(i, j) - a.get(i, j)).abs() < 1e-10);
                prop_assert!((ia.get(i, j) - a.get(i, j)).abs() < 1e-10);
            }
        }
    }

    // ========================================================================
    // QR Decomposition Properties
    // ========================================================================

    #[test]
    fn prop_qr_reconstruction(
        m in 2..6usize,
        n_max in 2usize..6usize,
    ) {
        // A = Q * R (QR decomposition reconstructs original matrix)
        let n = n_max.min(m);
        let a_vals: Vec<f64> = (0..(m * n)).map(|i| (i as f64) * 0.3).collect();
        let a = Matrix::new(m, n, a_vals);

        let (q, r) = a.qr();
        let reconstructed = q.matmul(&r);

        for i in 0..m {
            for j in 0..n {
                let diff = (reconstructed.get(i, j) - a.get(i, j)).abs();
                prop_assert!(diff < 1e-6, "Reconstruction failed at [{},{}]: diff = {}", i, j, diff);
            }
        }
    }

    #[test]
    fn prop_r_upper_triangular(
        m in 2..6usize,
        n_max in 2usize..6usize,
    ) {
        // R from QR decomposition is upper triangular
        let n = n_max.min(m);
        let a_vals: Vec<f64> = (0..(m * n)).map(|i| (i as f64) * 0.7).collect();
        let a = Matrix::new(m, n, a_vals);

        let (_q, r) = a.qr();

        // Check lower triangle is zero
        for i in 1..r.rows {
            for j in 0..i.min(r.cols) {
                prop_assert!(r.get(i, j).abs() < 1e-6, "R not upper triangular at [{},{}]: {}", i, j, r.get(i, j));
            }
        }
    }

    // ========================================================================
    // Inversion Properties
    // ========================================================================

    #[test]
    fn prop_invert_is_diagonal(
        n in 2..5usize,
    ) {
        // For a diagonal matrix, the inverse is just 1/diagonal elements
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = (i + 2) as f64; // Avoid division by zero
        }

        let d = Matrix::new(n, n, data);
        let inv = d.invert();

        prop_assert!(inv.is_some());

        let inv = inv.unwrap();
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 / ((i + 2) as f64) } else { 0.0 };
                let diff = (inv.get(i, j) - expected).abs();
                prop_assert!(diff < 1e-10, "Diagonal inverse failed at [{},{}]", i, j);
            }
        }
    }

    #[test]
    fn prop_identity_inverse(
        n in 1..10usize,
    ) {
        // I^(-1) = I (identity matrix is its own inverse)
        let identity = Matrix::identity(n);
        let inv = identity.invert();

        prop_assert!(inv.is_some());

        let inv = inv.unwrap();
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                prop_assert!((inv.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    // ========================================================================
    // Vector Function Properties
    // ========================================================================

    #[test]
    fn prop_vec_dot_symmetric(seed in 0u64..10000u64) {
        // a · b = b · a (dot product is commutative)
        let mut rng = seed;
        let a: Vec<f64> = (0..10).map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            (rng as f64) / 10000.0
        }).collect();
        let b: Vec<f64> = (0..10).map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            (rng as f64) / 10000.0
        }).collect();

        let ab = vec_dot(&a, &b);
        let ba = vec_dot(&b, &a);

        prop_assert!((ab - ba).abs() < 1e-10);
    }

    #[test]
    fn prop_vec_dot_bilinear(seed in 0u64..10000u64) {
        // (α·a) · b = a · (α·b) = α · (a · b) (dot product is bilinear)
        // Note: Floating-point arithmetic may cause small differences in the order of operations
        let mut rng = seed;
        let a: Vec<f64> = (0..10).map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            (rng as f64) / 10000.0
        }).collect();
        let b: Vec<f64> = (0..10).map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            (rng as f64) / 10000.0
        }).collect();
        let alpha = 2.5;

        let alpha_a = vec_scale(&a, alpha);
        let alpha_a_dot_b = vec_dot(&alpha_a, &b);

        let alpha_b = vec_scale(&b, alpha);
        let a_dot_alpha_b = vec_dot(&a, &alpha_b);

        // Use relative tolerance since values can vary widely
        let max_val = alpha_a_dot_b.abs().max(a_dot_alpha_b.abs());
        let tolerance = 1e-10 * max_val.max(1.0);

        prop_assert!((alpha_a_dot_b - a_dot_alpha_b).abs() < tolerance,
            "Bilinearity failed: {} vs {}", alpha_a_dot_b, a_dot_alpha_b);
    }

    #[test]
    fn prop_vec_l2_norm_triangle_inequality(seed in 0u64..10000u64) {
        // ||a + b|| ≤ ||a|| + ||b|| (triangle inequality)
        let mut rng = seed;
        let a: Vec<f64> = (0..10).map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng as f64) % 10.0) - 5.0
        }).collect();
        let b: Vec<f64> = (0..10).map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng as f64) % 10.0) - 5.0
        }).collect();

        let a_plus_b: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let norm_a_plus_b = vec_l2_norm(&a_plus_b);
        let norm_a = vec_l2_norm(&a);
        let norm_b = vec_l2_norm(&b);

        prop_assert!(norm_a_plus_b <= norm_a + norm_b + 1e-10,
            "Triangle inequality: {} <= {} + {}",
            norm_a_plus_b, norm_a, norm_b
        );
    }

    #[test]
    fn prop_vec_l2_norm_homogeneous(seed in 0u64..10000u64) {
        // ||α·a|| = |α| · ||a|| (norm is homogeneous)
        let mut rng = seed;
        let a: Vec<f64> = (0..10).map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng as f64) % 10.0) + 1.0
        }).collect();
        let alpha = 2.5;

        let alpha_a = vec_scale(&a, alpha);
        let norm_alpha_a = vec_l2_norm(&alpha_a);
        let norm_a = vec_l2_norm(&a);

        let expected = alpha.abs() * norm_a;

        prop_assert!((norm_alpha_a - expected).abs() < 1e-10,
            "Homogeneity: {} == {} * {}",
            norm_alpha_a, alpha, norm_a
        );
    }

    // ========================================================================
    // Matrix-Vector Multiplication Properties
    // ========================================================================

    #[test]
    fn prop_mul_vec_linear(
        m in 2..5usize,
        n in 2..5usize,
    ) {
        // M · (α·v + β·w) = α·(M·v) + β·(M·w) (linearity)
        let m_vals: Vec<f64> = (0..(m * n)).map(|i| (i as f64) * 0.1).collect();
        let matrix = Matrix::new(m, n, m_vals);

        let v: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();
        let w: Vec<f64> = (0..n).map(|i| ((i + n) as f64) * 0.3).collect();

        let alpha = 2.0;
        let beta = 3.0;

        // Left side: M · (α·v + β·w)
        let alpha_v = vec_scale(&v, alpha);
        let beta_w = vec_scale(&w, beta);
        let alpha_v_plus_beta_w: Vec<f64> = alpha_v.iter().zip(beta_w.iter()).map(|(x, y)| x + y).collect();
        let left = matrix.mul_vec(&alpha_v_plus_beta_w);

        // Right side: α·(M·v) + β·(M·w)
        let mv = matrix.mul_vec(&v);
        let mw = matrix.mul_vec(&w);
        let mut right = vec![0.0; m];
        for i in 0..m {
            right[i] = alpha * mv[i] + beta * mw[i];
        }

        for i in 0..m {
            prop_assert!((left[i] - right[i]).abs() < 1e-8, "Linearity failed at [{}]", i);
        }
    }
}
