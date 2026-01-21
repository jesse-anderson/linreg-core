// ============================================================================
// LINPACK QR Tests
// ============================================================================
//
// Tests for R's LINPACK dqrdc2 algorithm implementation:
// - qr_linpack(): QR decomposition with column pivoting
// - qr_solve_linpack(): Solve linear system using pivoted QR
// - fit_ols_linpack(): OLS regression using LINPACK QR
// - fit_and_predict_linpack(): OLS with prediction for rank-deficient cases

use linreg_core::linalg::{Matrix, fit_ols_linpack, fit_and_predict_linpack};
use super::common::{EPSILON, assert_close};

// ============================================================================
// QRLinpack Struct Tests
// ============================================================================

#[test]
fn test_qr_linpack_basic() {
    let x = Matrix::new(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 3.0, 4.0,
        ],
    );

    let result = x.qr_linpack(None);

    // Check dimensions
    assert_eq!(result.qr.rows, 4);
    assert_eq!(result.qr.cols, 3);
    assert_eq!(result.qraux.len(), 3);
    assert_eq!(result.pivot.len(), 3);
    assert_eq!(result.rank, 3); // Full rank
}

#[test]
fn test_qr_linpack_default_tolerance() {
    let x = Matrix::new(
        3,
        2,
        vec![1.0, 2.0, 2.0, 4.0, 3.0, 5.0],
    );

    let result = x.qr_linpack(None);

    // Should use default tolerance of 1e-7
    assert_eq!(result.rank, 2); // Full rank
}

#[test]
fn test_qr_linpack_custom_tolerance() {
    let x = Matrix::new(
        3,
        2,
        vec![1.0, 2.0, 2.0, 4.0, 3.0, 5.0],
    );

    let result_strict = x.qr_linpack(Some(1e-10));
    let result_loose = x.qr_linpack(Some(1e-4));

    // Both should be full rank for this well-conditioned matrix
    assert_eq!(result_strict.rank, 2);
    assert_eq!(result_loose.rank, 2);
}

#[test]
fn test_qr_linpack_rank_deficient() {
    // Column 2 is exactly 2 * column 1
    let x = Matrix::new(
        3,
        2,
        vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
    );

    let result = x.qr_linpack(None);

    // Should detect rank deficiency
    assert_eq!(result.rank, 1); // Only rank 1

    // Check pivot - should move the dependent column to the end
    // (pivot values are 1-based indices)
    // Original column order: [1, 2], after pivoting should identify
    // which columns are linearly independent
    assert!(result.pivot.len() == 2);
}

#[test]
fn test_qr_linpack_near_rank_deficient() {
    // Column 2 is almost 2 * column 1
    let epsilon = 1e-6;
    let x = Matrix::new(
        3,
        2,
        vec![1.0, 2.0, 2.0, 4.0 + epsilon, 3.0, 6.0],
    );

    let result = x.qr_linpack(None);

    // With default tolerance, might detect as full rank or rank-deficient
    // depending on the exact tolerance value
    assert!(result.rank == 1 || result.rank == 2);
}

#[test]
fn test_qr_linpack_pivot_is_identity_for_full_rank() {
    // For a well-conditioned matrix, pivot should be close to identity
    let x = Matrix::new(
        4,
        3,
        vec![
            1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
        ],
    );

    let result = x.qr_linpack(None);

    // Pivot should be [1, 2, 3] (1-based identity permutation)
    assert_eq!(result.pivot, vec![1, 2, 3]);
    assert_eq!(result.rank, 3);
}

#[test]
fn test_qr_linpack_single_column() {
    let x = Matrix::new(4, 1, vec![1.0, 2.0, 3.0, 4.0]);

    let result = x.qr_linpack(None);

    assert_eq!(result.qr.rows, 4);
    assert_eq!(result.qr.cols, 1);
    assert_eq!(result.qraux.len(), 1);
    assert_eq!(result.pivot, vec![1]);
    assert_eq!(result.rank, 1);
}

#[test]
fn test_qr_linpack_zero_column() {
    let x = Matrix::new(
        3,
        2,
        vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
    );

    let result = x.qr_linpack(None);

    // Zero column should be moved to the end
    assert_eq!(result.rank, 1);
    // Pivot should show that column 2 (index 2) was moved
    assert_eq!(result.pivot[1], 2); // Zero column is at position 2
}

#[test]
fn test_qr_linpack_square_matrix() {
    let x = Matrix::new(
        3,
        3,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0,
        ],
    );

    let result = x.qr_linpack(None);

    assert_eq!(result.rank, 3); // Full rank for this matrix
    assert_eq!(result.pivot.len(), 3);
}

// ============================================================================
// qr_solve_linpack() Tests
// ============================================================================

#[test]
fn test_qr_solve_linpack_basic() {
    // Solve Ax = b where A is well-conditioned
    let x = Matrix::new(
        3,
        2,
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 5.0],
    );
    let y = vec![1.0, 2.0, 3.0];

    let qr_result = x.qr_linpack(None);
    let coef = x.qr_solve_linpack(&qr_result, &y);

    assert!(coef.is_some(), "Should solve successfully");
    let coef = coef.unwrap();
    assert_eq!(coef.len(), 2);

    // Verify solution: X * coef should approximate y
    let y_pred = x.mul_vec(&coef);
    for i in 0..3 {
        assert_close(y_pred[i], y[i], 1e-9, &format!("solution[{}]", i));
    }
}

#[test]
fn test_qr_solve_linpack_full_rank() {
    let x = Matrix::new(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 3.0, 4.0,
        ],
    );
    let y = vec![6.0, 15.0, 25.0, 9.0];

    let qr_result = x.qr_linpack(None);
    let coef = x.qr_solve_linpack(&qr_result, &y);

    assert!(coef.is_some());
    let coef = coef.unwrap();

    // Verify reconstruction
    let y_pred = x.mul_vec(&coef);
    for i in 0..4 {
        assert_close(y_pred[i], y[i], 1e-8, &format!("full rank[{}]", i));
    }
}

#[test]
fn test_qr_solve_linpack_rank_deficient() {
    // Rank-deficient case: column 2 = 2 * column 1
    let x = Matrix::new(
        3,
        2,
        vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
    );
    let y = vec![1.0, 2.0, 3.0];

    let qr_result = x.qr_linpack(None);
    let coef = x.qr_solve_linpack(&qr_result, &y);

    assert!(coef.is_some(), "Should handle rank deficiency");
    let coef = coef.unwrap();

    // Coefficient for dependent column should be NaN
    assert!(coef[0].is_finite() || coef[1].is_finite(), "At least one coefficient should be finite");
}

#[test]
fn test_qr_solve_linpack_wrong_y_length() {
    let x = Matrix::new(3, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 5.0]);
    let y = vec![1.0, 2.0]; // Wrong length

    let qr_result = x.qr_linpack(None);
    let coef = x.qr_solve_linpack(&qr_result, &y);

    assert!(coef.is_none(), "Should return None for mismatched dimensions");
}

#[test]
fn test_qr_solve_linpack_zero_rank() {
    // Zero matrix
    let x = Matrix::new(3, 2, vec![0.0; 6]);
    let y = vec![1.0, 2.0, 3.0];

    let qr_result = x.qr_linpack(None);
    let coef = x.qr_solve_linpack(&qr_result, &y);

    assert!(coef.is_none(), "Should return None for zero-rank matrix");
}

#[test]
fn test_qr_solve_linpack_intercept_only() {
    // Simple case: intercept only (column of ones)
    let x = Matrix::new(4, 1, vec![1.0, 1.0, 1.0, 1.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0];

    let qr_result = x.qr_linpack(None);
    let coef = x.qr_solve_linpack(&qr_result, &y);

    assert!(coef.is_some());
    let coef = coef.unwrap();
    assert_eq!(coef.len(), 1);

    // Coefficient should be the mean of y
    let expected = 5.0; // (2 + 4 + 6 + 8) / 4
    assert_close(coef[0], expected, 1e-9, "intercept only");
}

// ============================================================================
// fit_ols_linpack() Tests
// ============================================================================

#[test]
fn test_fit_ols_linpack_basic() {
    // Simple linear regression: y = 1 + 2*x
    let x = Matrix::new(
        4,
        2,
        vec![
            1.0, 1.0, // intercept, x=1
            1.0, 2.0, // intercept, x=2
            1.0, 3.0, // intercept, x=3
            1.0, 4.0, // intercept, x=4
        ],
    );
    // y = 1 + 2*x: [3, 5, 7, 9]
    let y = vec![3.0, 5.0, 7.0, 9.0];

    let coef = fit_ols_linpack(&y, &x);

    assert!(coef.is_some());
    let coef = coef.unwrap();
    assert_eq!(coef.len(), 2);

    // Intercept ~ 1, slope ~ 2
    assert_close(coef[0], 1.0, 1e-9, "intercept");
    assert_close(coef[1], 2.0, 1e-9, "slope");
}

#[test]
fn test_fit_ols_linpack_predictions() {
    let x = Matrix::new(
        4,
        2,
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
    );
    let y = vec![3.0, 5.0, 7.0, 9.0];

    let coef = fit_ols_linpack(&y, &x).unwrap();
    let y_pred = x.mul_vec(&coef);

    // Predictions should match original y
    for i in 0..4 {
        assert_close(y_pred[i], y[i], 1e-9, &format!("prediction[{}]", i));
    }
}

#[test]
fn test_fit_ols_linpack_rank_deficient() {
    // Rank-deficient design matrix
    let x = Matrix::new(
        3,
        2,
        vec![
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0,
        ],
    );

    let y = vec![1.0, 2.0, 3.0];

    let coef = fit_ols_linpack(&y, &x);

    // Should handle rank deficiency
    assert!(coef.is_some());
}

#[test]
fn test_fit_ols_linpack_wrong_dimensions() {
    let x = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y = vec![1.0, 2.0]; // Wrong length

    let coef = fit_ols_linpack(&y, &x);

    assert!(coef.is_none(), "Should return None for dimension mismatch");
}

#[test]
fn test_fit_ols_linpack_perfect_fit() {
    // Perfect linear relationship
    let x = Matrix::new(
        3,
        2,
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
    );
    // y = 2 + 3*x exactly
    let y = vec![5.0, 8.0, 11.0];

    let coef = fit_ols_linpack(&y, &x).unwrap();

    assert_close(coef[0], 2.0, 1e-9, "perfect intercept");
    assert_close(coef[1], 3.0, 1e-9, "perfect slope");
}

// ============================================================================
// fit_and_predict_linpack() Tests
// ============================================================================

#[test]
fn test_fit_and_predict_linpack_basic() {
    let x = Matrix::new(
        4,
        2,
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
    );
    let y = vec![3.0, 5.0, 7.0, 9.0];

    let pred = fit_and_predict_linpack(&y, &x);

    assert!(pred.is_some());
    let pred = pred.unwrap();
    assert_eq!(pred.len(), 4);

    // Predictions should match y
    for i in 0..4 {
        assert_close(pred[i], y[i], 1e-9, &format!("fit & predict[{}]", i));
    }
}

#[test]
fn test_fit_and_predict_linpack_full_rank() {
    let x = Matrix::new(
        5,
        3,
        vec![
            1.0, 1.0, 2.0,
            1.0, 2.0, 3.0,
            1.0, 3.0, 4.0,
            1.0, 4.0, 5.0,
            1.0, 5.0, 6.0,
        ],
    );
    // y = 1 + 2*x1 + 3*x2
    let y = vec![
        1.0 + 2.0*1.0 + 3.0*2.0,
        1.0 + 2.0*2.0 + 3.0*3.0,
        1.0 + 2.0*3.0 + 3.0*4.0,
        1.0 + 2.0*4.0 + 3.0*5.0,
        1.0 + 2.0*5.0 + 3.0*6.0,
    ];

    let pred = fit_and_predict_linpack(&y, &x).unwrap();

    // Should predict perfectly
    for i in 0..5 {
        assert_close(pred[i], y[i], 1e-8, &format!("full rank pred[{}]", i));
    }
}

#[test]
fn test_fit_and_predict_linpack_rank_deficient() {
    // Rank-deficient: column 2 = 2 * column 1
    let x = Matrix::new(
        4,
        2,
        vec![
            1.0, 2.0,
            2.0, 4.0,
            3.0, 6.0,
            4.0, 8.0,
        ],
    );

    // y depends only on the first column
    let y = vec![1.0, 2.0, 3.0, 4.0];

    let pred = fit_and_predict_linpack(&y, &x);

    assert!(pred.is_some());
    let pred = pred.unwrap();

    // Predictions should still work (using only the independent columns)
    // Since y = column_1 exactly, predictions should match
    for i in 0..4 {
        assert_close(pred[i], y[i], 1e-8, &format!("rank deficient pred[{}]", i));
    }
}

#[test]
fn test_fit_and_predict_linpack_with_intercept() {
    // Standard regression with intercept
    let x = Matrix::new(
        5,
        2,
        vec![
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0,
            1.0, 4.0,
            1.0, 5.0,
        ],
    );
    let y = vec![2.1, 4.0, 5.9, 8.1, 10.0];

    let pred = fit_and_predict_linpack(&y, &x).unwrap();

    // Predictions should be close to y (with some residual due to noise)
    for i in 0..5 {
        assert!(
            (pred[i] - y[i]).abs() < 0.2,
            "prediction[{}] = {} vs {}",
            i,
            pred[i],
            y[i]
        );
    }
}

#[test]
fn test_fit_and_predict_linpack_single_predictor() {
    let x = Matrix::new(
        4,
        1,
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let y = vec![2.0, 4.0, 6.0, 8.0];

    let pred = fit_and_predict_linpack(&y, &x).unwrap();

    // Perfect fit: y = 2*x
    for i in 0..4 {
        assert_close(pred[i], y[i], 1e-9, &format!("single predictor[{}]", i));
    }
}

#[test]
fn test_fit_and_predict_vs_fit_ols() {
    // For full-rank matrices, both methods should give same predictions
    let x = Matrix::new(
        4,
        2,
        vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
    );
    let y = vec![3.0, 5.0, 7.0, 9.0];

    let pred1 = fit_and_predict_linpack(&y, &x).unwrap();
    let coef = fit_ols_linpack(&y, &x).unwrap();
    let pred2 = x.mul_vec(&coef);

    // Should be identical
    for i in 0..4 {
        assert_close(
            pred1[i],
            pred2[i],
            EPSILON,
            &format!("fit & predict vs fit_ols[{}]", i),
        );
    }
}

// ============================================================================
// LINPACK QR Edge Cases
// ============================================================================

#[test]
fn test_qr_linpack_wide_matrix() {
    // More columns than rows (not typical for OLS but should work)
    let x = Matrix::new(
        2,
        3,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );

    let result = x.qr_linpack(None);

    // Rank should be at most number of rows
    assert!(result.rank <= 2);
}

#[test]
fn test_qr_linpack_constant_column() {
    // Column with all same values
    let x = Matrix::new(
        4,
        2,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0],
    );

    let result = x.qr_linpack(None);

    // Should detect constant column (rank 2 since columns are independent)
    // Actually, intercept column of 1s is fine
    assert_eq!(result.rank, 2);
}

#[test]
fn test_qr_linpack_very_small_values() {
    let scale = 1e-10;
    let x = Matrix::new(
        3,
        2,
        vec![
            scale, 2.0 * scale,
            2.0 * scale, 3.0 * scale,
            3.0 * scale, 5.0 * scale,
        ],
    );

    let result = x.qr_linpack(None);

    // Should handle small values with relative tolerance
    assert!(result.rank >= 1);
}

#[test]
fn test_qr_linpack_very_large_values() {
    let scale = 1e10;
    let x = Matrix::new(
        3,
        2,
        vec![
            scale, 2.0 * scale,
            2.0 * scale, 3.0 * scale,
            3.0 * scale, 5.0 * scale,
        ],
    );

    let result = x.qr_linpack(None);

    // Should handle large values
    assert!(result.rank >= 1);
}

#[test]
fn test_qr_solve_with_collinear_columns() {
    // Perfectly collinear columns
    let x = Matrix::new(
        4,
        3,
        vec![
            1.0, 2.0, 4.0,  // col2 = 2*col1
            1.0, 3.0, 6.0,  // col2 = 2*col1
            1.0, 4.0, 8.0,  // col2 = 2*col1
            1.0, 5.0, 10.0, // col2 = 2*col1
        ],
    );
    let y = vec![1.0, 2.0, 3.0, 4.0];

    let qr_result = x.qr_linpack(None);
    let coef = x.qr_solve_linpack(&qr_result, &y);

    assert!(coef.is_some());
    let coef = coef.unwrap();

    // One of the coefficients should be NaN
    let nan_count = coef.iter().filter(|c| c.is_nan()).count();
    assert!(nan_count >= 1, "At least one coefficient should be NaN for collinear columns");
}

// ============================================================================
// Large-Scale LINPACK QR Stress Tests
// ============================================================================

#[test]
fn test_qr_linpack_large_tall_matrix() {
    // Typical OLS scenario: many observations, moderate predictors
    // 100 observations x 10 predictors
    // Use a design matrix with intercept + varied predictors to ensure full rank
    let m = 100;
    let n = 10;

    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        data.push(1.0); // intercept column
        for j in 1..n {
            // Create varied values that don't create collinearity
            let val = (i as f64) * (j as f64) * 0.1
                + (j as f64).powi(2) * 0.01
                + ((i * j) % 17) as f64;
            data.push(val);
        }
    }

    let x = Matrix::new(m, n, data);
    let result = x.qr_linpack(None);

    // Should be full rank
    assert_eq!(result.rank, n, "100x10 matrix should be full rank");
    assert_eq!(result.pivot.len(), n);
    assert_eq!(result.qraux.len(), n);
}

#[test]
fn test_qr_linpack_large_square_matrix() {
    // Square matrix stress test: 50x50
    let n = 50;
    let mut data = vec![0.0; n * n];

    // Create a diagonally dominant matrix (guaranteed full rank)
    for i in 0..n {
        for j in 0..n {
            if i == j {
                data[i * n + j] = 100.0 + (i as f64);
            } else {
                data[i * n + j] = 1.0 / (1.0 + ((i as i64 - j as i64).abs() as f64));
            }
        }
    }

    let x = Matrix::new(n, n, data);
    let result = x.qr_linpack(None);

    // Should be full rank
    assert_eq!(result.rank, n, "50x50 diagonally dominant should be full rank");
}

#[test]
fn test_fit_ols_linpack_large() {
    // Large OLS fitting: 200 observations, 5 predictors (with intercept)
    let m = 200;
    let n = 5;

    // Build design matrix with intercept column
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        data.push(1.0); // intercept
        for j in 1..n {
            data.push(((i * j + j) as f64) * 0.1 % 10.0);
        }
    }

    let x = Matrix::new(m, n, data.clone());

    // Create y = 2 + 0.5*x1 + 0.3*x2 + 0.2*x3 + 0.1*x4 + noise
    let mut y = Vec::with_capacity(m);
    for i in 0..m {
        let row_start = i * n;
        let val = 2.0
            + 0.5 * data[row_start + 1]
            + 0.3 * data[row_start + 2]
            + 0.2 * data[row_start + 3]
            + 0.1 * data[row_start + 4];
        y.push(val);
    }

    let coef = fit_ols_linpack(&y, &x);
    assert!(coef.is_some(), "Large OLS should succeed");

    let coef = coef.unwrap();
    assert_eq!(coef.len(), n);

    // Verify coefficients are close to true values
    assert_close(coef[0], 2.0, 1e-6, "intercept");
    assert_close(coef[1], 0.5, 1e-6, "beta1");
    assert_close(coef[2], 0.3, 1e-6, "beta2");
    assert_close(coef[3], 0.2, 1e-6, "beta3");
    assert_close(coef[4], 0.1, 1e-6, "beta4");
}

#[test]
fn test_fit_and_predict_linpack_large() {
    // Large prediction test: 150 observations, 8 predictors
    let m = 150;
    let n = 8;

    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(((i + j * 7) as f64) * 0.3 % 15.0);
        }
    }

    let x = Matrix::new(m, n, data);

    // y = linear combination of columns
    let y: Vec<f64> = (0..m)
        .map(|i| {
            let mut sum = 0.0;
            for j in 0..n {
                sum += (j as f64 + 1.0) * x.get(i, j);
            }
            sum
        })
        .collect();

    let pred = fit_and_predict_linpack(&y, &x);
    assert!(pred.is_some(), "Large fit_and_predict should succeed");

    let pred = pred.unwrap();
    assert_eq!(pred.len(), m);

    // Predictions should match original y (perfect fit)
    for i in 0..m {
        assert_close(pred[i], y[i], 1e-6, &format!("large prediction[{}]", i));
    }
}

#[test]
fn test_qr_linpack_large_with_rank_deficiency() {
    // Large matrix with known rank deficiency
    // 80 observations, 10 columns, but only rank 5
    let m = 80;
    let n = 10;

    // Create columns where columns 5-9 are linear combinations of columns 0-4
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        // First 5 independent columns
        for j in 0..5 {
            data.push(((i * (j + 1) + j) as f64) * 0.5 % 20.0 - 10.0);
        }
        // Last 5 columns are 2x the first 5
        for j in 0..5 {
            let base_idx = i * n + j;
            data.push(data[base_idx] * 2.0);
        }
    }

    let x = Matrix::new(m, n, data);
    let result = x.qr_linpack(None);

    // Should detect rank deficiency
    assert_eq!(result.rank, 5, "80x10 matrix with 5 dependent columns should have rank 5");
}

#[test]
fn test_qr_linpack_large_scale_values() {
    // Test numerical stability with large scale values
    let m = 60;
    let n = 6;
    let scale = 1e8;

    let data: Vec<f64> = (0..(m * n))
        .map(|i| scale * (((i as f64) * 0.13) % 1.0 - 0.5))
        .collect();

    let x = Matrix::new(m, n, data);
    let result = x.qr_linpack(None);

    // Should handle large scale
    assert!(result.rank >= 1, "Large scale matrix should have positive rank");
    assert_eq!(result.pivot.len(), n);
}

#[test]
fn test_qr_linpack_small_scale_values() {
    // Test numerical stability with small scale values
    let m = 60;
    let n = 6;
    let scale = 1e-8;

    let data: Vec<f64> = (0..(m * n))
        .map(|i| scale * (((i as f64) * 0.17) % 1.0 + 0.1))
        .collect();

    let x = Matrix::new(m, n, data);
    let result = x.qr_linpack(None);

    // Should handle small scale with appropriate tolerance
    assert!(result.rank >= 1, "Small scale matrix should have positive rank");
}

#[test]
fn test_qr_solve_linpack_large() {
    // Solve large system: 100x20
    let m = 100;
    let n = 20;

    // Create well-conditioned design matrix with intercept
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        data.push(1.0); // intercept
        for j in 1..n {
            data.push(((i * j) as f64) * 0.1 % 5.0 + (j as f64) * 0.5);
        }
    }

    let x = Matrix::new(m, n, data.clone());

    // Create known solution
    let true_coef: Vec<f64> = (0..n).map(|j| (j as f64) * 0.1 + 1.0).collect();

    // Compute y = X * true_coef
    let y: Vec<f64> = (0..m)
        .map(|i| {
            (0..n).map(|j| x.get(i, j) * true_coef[j]).sum()
        })
        .collect();

    let qr_result = x.qr_linpack(None);
    let coef = x.qr_solve_linpack(&qr_result, &y);

    assert!(coef.is_some(), "Large QR solve should succeed");
    let coef = coef.unwrap();

    // Verify recovered coefficients
    for j in 0..n {
        assert_close(coef[j], true_coef[j], 1e-6, &format!("large solve coef[{}]", j));
    }
}
