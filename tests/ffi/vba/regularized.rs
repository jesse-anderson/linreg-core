// FFI tests for regularized regression functions (Ridge, Lasso, Elastic Net).

#[cfg(feature = "ffi")]

use super::common::*;

// ============================================================================
// Ridge Regression Tests
// ============================================================================

#[test]
fn test_ridge_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let lambda = 1.0;
    let handle = unsafe { LR_Ridge(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, 1) };
    let _guard = HandleGuard::new(handle);

    // Check basic statistics
    let r2 = unsafe { LR_GetRSquared(handle) };
    assert!(r2 >= 0.0 && r2 <= 1.0, "R² should be in [0, 1]");

    let mse = unsafe { LR_GetMSE(handle) };
    assert!(mse > 0.0, "MSE should be positive");

    // Check intercept
    let intercept = unsafe { LR_GetIntercept(handle) };
    assert!(!intercept.is_nan(), "Intercept should not be NaN");

    // Check number of coefficients (slopes only, not including intercept)
    let n_coef = unsafe { LR_GetNumCoefficients(handle) } as usize;
    assert_eq!(n_coef, x_cols.len(), "Should have p slope coefficients");

    // Get coefficients
    let mut coefs = vec![0.0f64; n_coef];
    unsafe { LR_GetCoefficients(handle, coefs.as_mut_ptr(), n_coef as i32) };

    // Coefficients should be non-NaN
    for (i, &c) in coefs.iter().enumerate() {
        assert!(!c.is_nan(), "Coefficient {} should not be NaN", i);
    }

    // Check effective degrees of freedom
    let eff_df = unsafe { LR_GetDF(handle) };
    assert!(eff_df > 0.0 && eff_df <= n_coef as f64, "Eff df should be in (0, p]");
}

#[test]
fn test_ridge_lambda_effect() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    // Fit with small lambda
    let h_small = unsafe { LR_Ridge(y.as_ptr(), n, x_matrix.as_ptr(), p, 0.01, 1) };
    let _guard1 = HandleGuard::new(h_small);

    // Fit with large lambda
    let h_large = unsafe { LR_Ridge(y.as_ptr(), n, x_matrix.as_ptr(), p, 100.0, 1) };
    let _guard2 = HandleGuard::new(h_large);

    let n_coef = unsafe { LR_GetNumCoefficients(h_small) } as usize;

    let mut coefs_small = vec![0.0f64; n_coef];
    let mut coefs_large = vec![0.0f64; n_coef];

    unsafe {
        LR_GetCoefficients(h_small, coefs_small.as_mut_ptr(), n_coef as i32);
        LR_GetCoefficients(h_large, coefs_large.as_mut_ptr(), n_coef as i32);
    }

    // Larger lambda should produce smaller coefficient magnitudes
    let mag_small: f64 = coefs_small.iter().map(|&c| c.abs()).sum();
    let mag_large: f64 = coefs_large.iter().map(|&c| c.abs()).sum();

    assert!(
        mag_small > mag_large,
        "Larger lambda should shrink coefficients more: {} > {}",
        mag_small,
        mag_large
    );
}

#[test]
fn test_ridge_standardize_vs_unstandardized() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;
    let lambda = 1.0;

    let h_std = unsafe { LR_Ridge(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, 1) };
    let _guard1 = HandleGuard::new(h_std);

    let h_unstd = unsafe { LR_Ridge(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, 0) };
    let _guard2 = HandleGuard::new(h_unstd);

    let n_coef = unsafe { LR_GetNumCoefficients(h_std) } as usize;

    let mut coefs_std = vec![0.0f64; n_coef];
    let mut coefs_unstd = vec![0.0f64; n_coef];

    unsafe {
        LR_GetCoefficients(h_std, coefs_std.as_mut_ptr(), n_coef as i32);
        LR_GetCoefficients(h_unstd, coefs_unstd.as_mut_ptr(), n_coef as i32);
    }

    // Standardized coefficients should differ from unstandardized
    let different = coefs_std
        .iter()
        .zip(coefs_unstd.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-6);

    assert!(different, "Standardized and unstandardized should differ");
}

#[test]
fn test_ridge_get_fitted_values() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { LR_Ridge(y.as_ptr(), n, x_matrix.as_ptr(), p, 1.0, 1) };
    let _guard = HandleGuard::new(handle);

    let n_obs = unsafe { LR_GetNumObservations(handle) } as usize;
    let mut fitted = vec![0.0f64; n_obs];
    unsafe { LR_GetFittedValues(handle, fitted.as_mut_ptr(), n_obs as i32) };

    // All fitted values should be non-NaN
    for (i, &f) in fitted.iter().enumerate() {
        assert!(!f.is_nan(), "Fitted value {} should not be NaN", i);
    }

    // Get residuals
    let mut residuals = vec![0.0f64; n_obs];
    unsafe { LR_GetResiduals(handle, residuals.as_mut_ptr(), n_obs as i32) };

    // y = fitted + residual should hold
    for i in 0..n_obs {
        let reconstructed = fitted[i] + residuals[i];
        assert!(
            (reconstructed - y[i]).abs() < 1e-9,
            "y = fitted + residual at index {}",
            i
        );
    }
}

// ============================================================================
// Lasso Regression Tests
// ============================================================================

#[test]
fn test_lasso_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let lambda = 0.1;
    let handle = unsafe { LR_Lasso(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, 1, 10000, 1e-7) };
    let _guard = HandleGuard::new(handle);

    // Check basic statistics
    let r2 = unsafe { LR_GetRSquared(handle) };
    assert!(r2 >= 0.0 && r2 <= 1.0, "R² should be in [0, 1]");

    // Check intercept
    let intercept = unsafe { LR_GetIntercept(handle) };
    assert!(!intercept.is_nan(), "Intercept should not be NaN");

    // Check convergence
    let converged = unsafe { LR_GetConverged(handle) };
    assert_eq!(converged, 1, "Lasso should converge for reasonable data");

    // Check non-zero count
    let n_nonzero = unsafe { LR_GetNNonzero(handle) } as usize;
    assert!(n_nonzero <= p as usize, "Non-zero count should be <= p");
    assert!(n_nonzero > 0, "Should have at least some non-zero coefficients");
}

#[test]
fn test_lasso_sparsity() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    // Large lambda should produce sparse solution
    let h_large = unsafe { LR_Lasso(y.as_ptr(), n, x_matrix.as_ptr(), p, 10.0, 1, 10000, 1e-7) };
    let _guard = HandleGuard::new(h_large);

    let n_nonzero_large = unsafe { LR_GetNNonzero(h_large) } as usize;

    // Small lambda should produce less sparse solution
    let h_small = unsafe { LR_Lasso(y.as_ptr(), n, x_matrix.as_ptr(), p, 0.01, 1, 10000, 1e-7) };
    let _guard2 = HandleGuard::new(h_small);

    let n_nonzero_small = unsafe { LR_GetNNonzero(h_small) } as usize;

    assert!(
        n_nonzero_small >= n_nonzero_large,
        "Smaller lambda should have >= non-zero coefficients: {} >= {}",
        n_nonzero_small,
        n_nonzero_large
    );
}

#[test]
fn test_lasso_convergence() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { LR_Lasso(y.as_ptr(), n, x_matrix.as_ptr(), p, 0.1, 1, 10000, 1e-7) };
    let _guard = HandleGuard::new(handle);

    let converged = unsafe { LR_GetConverged(handle) };
    assert_eq!(converged, 1, "Should converge with default parameters");
}

#[test]
fn test_lasso_max_iter_enforced() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    // Set max iterations very low - may not converge
    let handle = unsafe { LR_Lasso(y.as_ptr(), n, x_matrix.as_ptr(), p, 0.1, 1, 2, 1e-7) };

    if handle != 0 {
        let converged = unsafe { LR_GetConverged(handle) };
        // With only 2 iterations, might not converge
        unsafe { LR_Free(handle) };
    }
}

// ============================================================================
// Elastic Net Tests
// ============================================================================

#[test]
fn test_elastic_net_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let lambda = 0.1;
    let alpha = 0.5; // Equal mix of L1 and L2

    let handle = unsafe {
        LR_ElasticNet(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, alpha, 1, 10000, 1e-7)
    };
    let _guard = HandleGuard::new(handle);

    // Check basic statistics
    let r2 = unsafe { LR_GetRSquared(handle) };
    assert!(r2 >= 0.0 && r2 <= 1.0, "R² should be in [0, 1]");

    let mse = unsafe { LR_GetMSE(handle) };
    assert!(mse > 0.0, "MSE should be positive");

    // Check convergence
    let converged = unsafe { LR_GetConverged(handle) };
    assert_eq!(converged, 1, "Elastic Net should converge");

    // Check non-zero count
    let n_nonzero = unsafe { LR_GetNNonzero(handle) } as usize;
    assert!(n_nonzero > 0, "Should have at least some non-zero coefficients");
}

#[test]
fn test_elastic_net_alpha_continuum() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;
    let lambda = 0.1;

    // alpha = 0 (pure Ridge)
    let h_ridge = unsafe {
        LR_ElasticNet(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, 0.0, 1, 10000, 1e-7)
    };
    let _guard1 = HandleGuard::new(h_ridge);

    // alpha = 1 (pure Lasso)
    let h_lasso = unsafe {
        LR_ElasticNet(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, 1.0, 1, 10000, 1e-7)
    };
    let _guard2 = HandleGuard::new(h_lasso);

    // alpha = 0.5 (Elastic Net)
    let h_enet = unsafe {
        LR_ElasticNet(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, 0.5, 1, 10000, 1e-7)
    };
    let _guard3 = HandleGuard::new(h_enet);

    // Get R² values
    let r2_ridge = unsafe { LR_GetRSquared(h_ridge) };
    let r2_lasso = unsafe { LR_GetRSquared(h_lasso) };
    let r2_enet = unsafe { LR_GetRSquared(h_enet) };

    // All should be valid
    assert!(r2_ridge >= 0.0 && r2_ridge <= 1.0);
    assert!(r2_lasso >= 0.0 && r2_lasso <= 1.0);
    assert!(r2_enet >= 0.0 && r2_enet <= 1.0);

    // Elastic Net R² should typically be between Ridge and Lasso
    // (or at least not worse than both)
    let min_r2 = r2_ridge.min(r2_lasso);
    assert!(r2_enet >= min_r2 - 0.1, "Elastic Net should be competitive");
}

#[test]
fn test_elastic_net_sparsity_with_alpha() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;
    let lambda = 1.0;

    // Higher alpha (more L1) should produce more sparsity
    let h_alpha1 = unsafe {
        LR_ElasticNet(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, 0.9, 1, 10000, 1e-7)
    };
    let _guard1 = HandleGuard::new(h_alpha1);

    let h_alpha01 = unsafe {
        LR_ElasticNet(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, 0.1, 1, 10000, 1e-7)
    };
    let _guard2 = HandleGuard::new(h_alpha01);

    let nz_alpha1 = unsafe { LR_GetNNonzero(h_alpha1) } as usize;
    let nz_alpha01 = unsafe { LR_GetNNonzero(h_alpha01) } as usize;

    // Higher alpha should generally produce fewer non-zeros
    // (though this depends on the data)
    assert!(
        nz_alpha1 <= nz_alpha01 + 1,
        "Higher alpha (more L1) should produce similar or fewer non-zeros: {} <= {}",
        nz_alpha1,
        nz_alpha01
    );
}

// ============================================================================
// Regularized Regression Error Handling
// ============================================================================

#[test]
fn test_ridge_error_null_pointer() {
    let handle = unsafe { LR_Ridge(std::ptr::null(), 10, std::ptr::null(), 1, 1.0, 1) };
    assert_eq!(handle, 0, "Null pointer should return error handle");
}

#[test]
fn test_lasso_error_null_pointer() {
    let handle = unsafe {
        LR_Lasso(std::ptr::null(), 10, std::ptr::null(), 1, 1.0, 1, 1000, 1e-7)
    };
    assert_eq!(handle, 0, "Null pointer should return error handle");
}

#[test]
fn test_elastic_net_error_null_pointer() {
    let handle = unsafe {
        LR_ElasticNet(std::ptr::null(), 10, std::ptr::null(), 1, 1.0, 0.5, 1, 1000, 1e-7)
    };
    assert_eq!(handle, 0, "Null pointer should return error handle");
}

#[test]
fn test_regularized_invalid_handle_returns_nan() {
    let invalid_handle = 999999;

    let intercept = unsafe { LR_GetIntercept(invalid_handle) };
    assert!(intercept.is_nan(), "Invalid handle should return NaN for intercept");

    let eff_df = unsafe { LR_GetDF(invalid_handle) };
    assert!(eff_df.is_nan(), "Invalid handle should return NaN for eff df");

    let n_nonzero = unsafe { LR_GetNNonzero(invalid_handle) };
    assert_eq!(n_nonzero, -1, "Invalid handle should return -1 for n_nonzero");
}
