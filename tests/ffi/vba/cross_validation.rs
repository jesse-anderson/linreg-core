// FFI tests for cross-validation functions.

#[cfg(feature = "ffi")]

use super::common::*;

// ============================================================================
// K-Fold OLS Cross-Validation Tests
// ============================================================================

#[test]
fn test_kfold_ols_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;
    let k = 5; // 5-fold CV

    let handle = unsafe { LR_KFoldOLS(y.as_ptr(), n, x_matrix.as_ptr(), p, k) };
    let _guard = HandleGuard::new(handle);

    // Check that we got a valid result
    let n_folds = unsafe { LR_GetCVNFolds(handle) } as usize;
    assert_eq!(n_folds, k as usize, "Number of folds should match requested k");

    // Get statistics
    let mean_mse = unsafe { LR_GetCVMeanMSE(handle) };
    let std_mse = unsafe { LR_GetCVStdMSE(handle) };
    let mean_rmse = unsafe { LR_GetCVMeanRMSE(handle) };
    let std_rmse = unsafe { LR_GetCVStdRMSE(handle) };
    let mean_r2 = unsafe { LR_GetCVMeanR2(handle) };

    // All values should be finite and non-negative where appropriate
    assert!(mean_mse.is_finite() && mean_mse >= 0.0, "Mean MSE should be non-negative");
    assert!(std_mse.is_finite() && std_mse >= 0.0, "Std MSE should be non-negative");
    assert!(mean_rmse.is_finite() && mean_rmse >= 0.0, "Mean RMSE should be non-negative");
    assert!(std_rmse.is_finite() && std_rmse >= 0.0, "Std RMSE should be non-negative");
    assert!(mean_r2.is_finite(), "Mean R² should be finite");

    // Verify Jensen's inequality: mean_rmse < sqrt(mean_mse) for concave sqrt function
    // Within each fold: rmse = sqrt(mse) by definition
    // Across folds: mean(sqrt(mse_i)) < sqrt(mean(mse_i)) when MSE values differ
    let sqrt_mean_mse = mean_mse.sqrt();
    assert!(
        mean_rmse <= sqrt_mean_mse + 1e-10,
        "mean_rmse ({}) should be <= sqrt(mean_mse) ({}) due to Jensen's inequality",
        mean_rmse, sqrt_mean_mse
    );
}

// ============================================================================
// K-Fold Ridge Cross-Validation Tests
// ============================================================================

#[test]
fn test_kfold_ridge_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;
    let k = 5;
    let lambda = 1.0;

    let handle = unsafe {
        LR_KFoldRidge(
            y.as_ptr(),
            n,
            x_matrix.as_ptr(),
            p,
            lambda,
            1, // standardize
            k,
        )
    };
    let _guard = HandleGuard::new(handle);

    let n_folds = unsafe { LR_GetCVNFolds(handle) } as usize;
    assert_eq!(n_folds, k as usize);

    let mean_mse = unsafe { LR_GetCVMeanMSE(handle) };
    let mean_r2 = unsafe { LR_GetCVMeanR2(handle) };

    assert!(mean_mse.is_finite() && mean_mse > 0.0);
    assert!(mean_r2.is_finite());
}

#[test]
fn test_kfold_ridge_lambda_effect() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;
    let k = 5;

    // Small lambda
    let h_small = unsafe { LR_KFoldRidge(y.as_ptr(), n, x_matrix.as_ptr(), p, 0.01, 1, k) };
    let _guard1 = HandleGuard::new(h_small);

    // Large lambda
    let h_large = unsafe { LR_KFoldRidge(y.as_ptr(), n, x_matrix.as_ptr(), p, 100.0, 1, k) };
    let _guard2 = HandleGuard::new(h_large);

    let mse_small = unsafe { LR_GetCVMeanMSE(h_small) };
    let mse_large = unsafe { LR_GetCVMeanMSE(h_large) };

    assert!(mse_small.is_finite());
    assert!(mse_large.is_finite());

    // Large lambda may have different MSE (could be higher or lower depending on data)
    assert!(mse_small > 0.0 && mse_large > 0.0);
}

// ============================================================================
// K-Fold Lasso Cross-Validation Tests
// ============================================================================

#[test]
fn test_kfold_lasso_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;
    let k = 5;
    let lambda = 0.1;

    let handle = unsafe {
        LR_KFoldLasso(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, 1, k)
    };
    let _guard = HandleGuard::new(handle);

    let n_folds = unsafe { LR_GetCVNFolds(handle) } as usize;
    assert_eq!(n_folds, k as usize);

    let mean_mse = unsafe { LR_GetCVMeanMSE(handle) };
    let mean_r2 = unsafe { LR_GetCVMeanR2(handle) };

    assert!(mean_mse.is_finite() && mean_mse > 0.0);
    assert!(mean_r2.is_finite());
}

// ============================================================================
// K-Fold Elastic Net Cross-Validation Tests
// ============================================================================

#[test]
fn test_kfold_elastic_net_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;
    let k = 5;
    let lambda = 0.1;
    let alpha = 0.5;

    let handle = unsafe {
        LR_KFoldElasticNet(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, alpha, 1, k)
    };
    let _guard = HandleGuard::new(handle);

    let n_folds = unsafe { LR_GetCVNFolds(handle) } as usize;
    assert_eq!(n_folds, k as usize);

    let mean_mse = unsafe { LR_GetCVMeanMSE(handle) };
    let mean_rmse = unsafe { LR_GetCVMeanRMSE(handle) };
    let mean_r2 = unsafe { LR_GetCVMeanR2(handle) };

    assert!(mean_mse.is_finite() && mean_mse > 0.0);
    assert!(mean_rmse.is_finite() && mean_rmse > 0.0);
    assert!(mean_r2.is_finite());

    // Verify Jensen's inequality: mean_rmse < sqrt(mean_mse)
    let sqrt_mean_mse = mean_mse.sqrt();
    assert!(
        mean_rmse <= sqrt_mean_mse + 1e-10,
        "mean_rmse ({}) should be <= sqrt(mean_mse) ({})",
        mean_rmse, sqrt_mean_mse
    );
}

#[test]
fn test_kfold_elastic_net_alpha_variations() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;
    let k = 5;
    let lambda = 0.1;

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let handle = unsafe {
            LR_KFoldElasticNet(y.as_ptr(), n, x_matrix.as_ptr(), p, lambda, alpha, 1, k)
        };

        if handle != 0 {
            let mean_mse = unsafe { LR_GetCVMeanMSE(handle) };
            assert!(mean_mse.is_finite(), "Mean MSE should be finite for alpha={}", alpha);
            unsafe { LR_Free(handle) };
        }
    }
}

// ============================================================================
// K-Fold Error Handling Tests
// ============================================================================

#[test]
fn test_kfold_null_pointers() {
    let cv_fns: &[unsafe extern "system" fn(*const f64, i32, *const f64, i32, i32) -> usize] =
        &[LR_KFoldOLS];

    for &cv_fn in cv_fns {
        let handle = unsafe { cv_fn(std::ptr::null(), 10, std::ptr::null(), 1, 5) };
        assert_eq!(handle, 0, "K-fold CV should return 0 on null pointer");
    }
}

#[test]
fn test_kfold_invalid_k() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    // k > n should still work (implementation uses min(k, n))
    let handle = unsafe { LR_KFoldOLS(y.as_ptr(), n, x_matrix.as_ptr(), p, 1000) };
    if handle != 0 {
        unsafe { LR_Free(handle) };
    }
}

#[test]
fn test_kfold_invalid_handle() {
    let invalid_handle = 999999;

    let n_folds = unsafe { LR_GetCVNFolds(invalid_handle) };
    assert_eq!(n_folds, -1, "Invalid handle should return -1 for n_folds");

    let mean_mse = unsafe { LR_GetCVMeanMSE(invalid_handle) };
    assert!(mean_mse.is_nan(), "Invalid handle should return NaN for mean MSE");

    let mean_r2 = unsafe { LR_GetCVMeanR2(invalid_handle) };
    assert!(mean_r2.is_nan(), "Invalid handle should return NaN for mean R²");
}

// ============================================================================
// K-Fold Edge Cases
// ============================================================================

#[test]
fn test_kfold_small_dataset() {
    // Very small dataset
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let x_matrix = columns_to_row_major(&[x]);

    let n = 4;
    let p = 1;
    let k = 2; // Only 2 folds for 4 observations

    let handle = unsafe { LR_KFoldOLS(y.as_ptr(), n, x_matrix.as_ptr(), p, k) };

    if handle != 0 {
        let _guard = HandleGuard::new(handle);

        let mean_mse = unsafe { LR_GetCVMeanMSE(handle) };
        assert!(mean_mse.is_finite());

        let n_folds = unsafe { LR_GetCVNFolds(handle) };
        assert!(n_folds > 0);
    }
}

#[test]
fn test_kfold_consistency() {
    // Same data should produce same results
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;
    let k = 5;

    let h1 = unsafe { LR_KFoldOLS(y.as_ptr(), n, x_matrix.as_ptr(), p, k) };
    let h2 = unsafe { LR_KFoldOLS(y.as_ptr(), n, x_matrix.as_ptr(), p, k) };

    let mse1 = unsafe { LR_GetCVMeanMSE(h1) };
    let mse2 = unsafe { LR_GetCVMeanMSE(h2) };

    assert_eq!(mse1, mse2, "Same data should produce same CV MSE");

    unsafe {
        LR_Free(h1);
        LR_Free(h2);
    }
}

#[test]
fn test_kfold_rmse_relationship() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { LR_KFoldOLS(y.as_ptr(), n, x_matrix.as_ptr(), p, 5) };
    let _guard = HandleGuard::new(handle);

    let mean_mse = unsafe { LR_GetCVMeanMSE(handle) };
    let mean_rmse = unsafe { LR_GetCVMeanRMSE(handle) };

    // Verify the mathematical relationship
    // Within each fold: rmse = sqrt(mse)
    // Across folds: mean_rmse <= sqrt(mean_mse) due to Jensen's inequality
    let sqrt_mean_mse = mean_mse.sqrt();
    assert!(
        mean_rmse <= sqrt_mean_mse + 1e-10,
        "mean_rmse ({}) should be <= sqrt(mean_mse) ({})",
        mean_rmse, sqrt_mean_mse
    );
    assert!(mean_rmse.is_finite() && mean_rmse > 0.0, "RMSE should be positive");
}
