// FFI tests for OLS regression functions.
//
// Tests the LR_OLS function and all associated getter functions.

#[cfg(feature = "ffi")]

use super::common::*;

// ============================================================================
// LR_OLS Function Tests
// ============================================================================

#[test]
fn test_ols_simple_linear_regression() {
    let (y, x) = simple_linear_data();
    let x_matrix = columns_to_row_major(&[x]);

    let handle = unsafe { LR_OLS(y.as_ptr(), y.len() as i32, x_matrix.as_ptr(), 1) };
    let _guard = HandleGuard::new(handle);

    // Verify R² is high (should be > 0.99 for nearly perfect linear data)
    let r2 = unsafe { LR_GetRSquared(handle) };
    assert!(r2 > 0.99, "R² should be > 0.99 for linear data, got {}", r2);

    // Verify adjusted R² is close to R²
    let adj_r2 = unsafe { LR_GetAdjRSquared(handle) };
    assert!((adj_r2 - r2).abs() < 0.01, "Adj R² should be close to R²");

    // Get coefficients
    let n_coef = unsafe { LR_GetNumCoefficients(handle) } as usize;
    assert_eq!(n_coef, 2, "Should have 2 coefficients (intercept + slope)");

    let mut coefs = vec![0.0f64; n_coef];
    let written = unsafe { LR_GetCoefficients(handle, coefs.as_mut_ptr(), n_coef as i32) };
    assert_eq!(written, n_coef as i32);

    // Intercept should be ~2.0, slope should be ~3.0
    assert!((coefs[0] - 2.0).abs() < 0.1, "Intercept should be ~2.0, got {}", coefs[0]);
    assert!((coefs[1] - 3.0).abs() < 0.1, "Slope should be ~3.0, got {}", coefs[1]);

    // Verify standard errors are positive
    let mut ses = vec![0.0f64; n_coef];
    unsafe { LR_GetStdErrors(handle, ses.as_mut_ptr(), n_coef as i32) };
    assert!(ses.iter().all(|&se| se > 0.0), "All standard errors should be positive");

    // Verify t-stats are reasonable
    let mut tstats = vec![0.0f64; n_coef];
    unsafe { LR_GetTStats(handle, tstats.as_mut_ptr(), n_coef as i32) };
    assert!(tstats.iter().all(|&t| t.abs() > 1.0), "All t-stats should be significant");

    // Verify p-values are small for significant coefficients
    let mut pvals = vec![0.0f64; n_coef];
    unsafe { LR_GetPValues(handle, pvals.as_mut_ptr(), n_coef as i32) };
    assert!(pvals.iter().all(|&p| p < 0.05), "All p-values should be < 0.05");

    // Check MSE
    let mse = unsafe { LR_GetMSE(handle) };
    assert!(mse > 0.0, "MSE should be positive");

    // Check F-statistic
    let f_stat = unsafe { LR_GetFStatistic(handle) };
    assert!(f_stat > 0.0, "F-statistic should be positive");

    let f_pval = unsafe { LR_GetFPValue(handle) };
    assert!(f_pval < 0.05, "F-test p-value should be significant");

    // Check number of observations
    let n_obs = unsafe { LR_GetNumObservations(handle) } as usize;
    assert_eq!(n_obs, y.len(), "Observation count should match input");
}

#[test]
fn test_ols_multiple_regression() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { LR_OLS(y.as_ptr(), n, x_matrix.as_ptr(), p) };
    let _guard = HandleGuard::new(handle);

    // Basic sanity checks
    let r2 = unsafe { LR_GetRSquared(handle) };
    assert!(r2 > 0.5 && r2 <= 1.0, "R² should be between 0.5 and 1.0");

    let n_coef = unsafe { LR_GetNumCoefficients(handle) } as usize;
    assert_eq!(n_coef, x_cols.len() + 1, "Should have p+1 coefficients");

    // Get residuals
    let n_obs = unsafe { LR_GetNumObservations(handle) } as usize;
    let mut residuals = vec![0.0f64; n_obs];
    unsafe {
        LR_GetResiduals(handle, residuals.as_mut_ptr(), n_obs as i32);
    }

    // Residuals should sum to approximately zero
    let sum_residuals: f64 = residuals.iter().sum();
    assert!(
        sum_residuals.abs() < 1e-10,
        "Residuals should sum to ~0, got {}",
        sum_residuals
    );

    // Get fitted values
    let mut fitted = vec![0.0f64; n_obs];
    unsafe {
        LR_GetFittedValues(handle, fitted.as_mut_ptr(), n_obs as i32);
    }

    // y = fitted + residuals
    for i in 0..n_obs {
        let reconstructed = fitted[i] + residuals[i];
        assert!(
            (reconstructed - y[i]).abs() < 1e-9,
            "y = fitted + residual at index {}",
            i
        );
    }
}

#[test]
fn test_ols_error_null_pointer() {
    // Null y_ptr should return handle 0
    let handle = unsafe { LR_OLS(std::ptr::null(), 10, std::ptr::null(), 1) };
    assert_eq!(handle, 0, "Null pointer should return error handle");

    let msg = get_last_error_message();
    assert!(
        msg.contains("null pointer") || msg.contains("Null"),
        "Error message should mention null pointer"
    );
}

#[test]
fn test_ols_error_invalid_dimensions() {
    let y = vec![1.0, 2.0];
    let x = vec![1.0];

    // n <= 0 should error
    let handle = unsafe { LR_OLS(y.as_ptr(), 0, x.as_ptr(), 1) };
    assert_eq!(handle, 0, "Non-positive n should return error handle");

    let handle = unsafe { LR_OLS(y.as_ptr(), -1, x.as_ptr(), 1) };
    assert_eq!(handle, 0, "Negative n should return error handle");

    // p <= 0 should error
    let handle = unsafe { LR_OLS(y.as_ptr(), 2, x.as_ptr(), 0) };
    assert_eq!(handle, 0, "Non-positive p should return error handle");
}

#[test]
fn test_ols_buffer_too_small() {
    let (y, x) = simple_linear_data();
    let x_matrix = columns_to_row_major(&[x]);

    let handle = unsafe {
        LR_OLS(
            y.as_ptr(),
            y.len() as i32,
            x_matrix.as_ptr(),
            1,
        )
    };
    let _guard = HandleGuard::new(handle);

    let n_coef = unsafe { LR_GetNumCoefficients(handle) } as usize;

    // Request fewer elements than available - should only write what fits
    let mut coefs = vec![0.0f64; 1];
    let written = unsafe { LR_GetCoefficients(handle, coefs.as_mut_ptr(), 1) };
    assert_eq!(written, 1, "Should write only 1 element when buffer is size 1");
}

#[test]
fn test_ols_coefficient_consistency() {
    // Verify that coef / se = t_stat
    let (y, x) = simple_linear_data();
    let x_matrix = columns_to_row_major(&[x]);

    let handle = unsafe {
        LR_OLS(
            y.as_ptr(),
            y.len() as i32,
            x_matrix.as_ptr(),
            1,
        )
    };
    let _guard = HandleGuard::new(handle);

    let n_coef = unsafe { LR_GetNumCoefficients(handle) } as usize;

    let mut coefs = vec![0.0f64; n_coef];
    let mut ses = vec![0.0f64; n_coef];
    let mut tstats = vec![0.0f64; n_coef];

    unsafe {
        LR_GetCoefficients(handle, coefs.as_mut_ptr(), n_coef as i32);
        LR_GetStdErrors(handle, ses.as_mut_ptr(), n_coef as i32);
        LR_GetTStats(handle, tstats.as_mut_ptr(), n_coef as i32);
    }

    for i in 0..n_coef {
        let calculated_t = coefs[i] / ses[i];
        assert!(
            (calculated_t - tstats[i]).abs() < 1e-9,
            "t-stat should equal coef/se for coefficient {}",
            i
        );
    }
}

#[test]
fn test_ols_model_fit_statistics() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);

    let handle = unsafe {
        LR_OLS(
            y.as_ptr(),
            y.len() as i32,
            x_matrix.as_ptr(),
            x_cols.len() as i32,
        )
    };
    let _guard = HandleGuard::new(handle);

    let r2 = unsafe { LR_GetRSquared(handle) };
    let adj_r2 = unsafe { LR_GetAdjRSquared(handle) };
    let mse = unsafe { LR_GetMSE(handle) };
    let n_obs = unsafe { LR_GetNumObservations(handle) } as usize;
    let n_coef = unsafe { LR_GetNumCoefficients(handle) } as usize;

    // Adj R² should be <= R² (always true for OLS)
    assert!(
        adj_r2 <= r2 + 1e-9,
        "Adj R² should be <= R², got {} vs {}",
        adj_r2,
        r2
    );

    // R² should be between 0 and 1
    assert!(r2 >= 0.0 && r2 <= 1.0, "R² should be in [0, 1]");

    // MSE should be positive
    assert!(mse > 0.0, "MSE should be positive");

    // Residual degrees of freedom = n - k
    let df_residual = n_obs - n_coef;
    assert!(
        df_residual > 0,
        "Should have positive residual degrees of freedom"
    );
}

#[test]
fn test_ols_get_last_error() {
    let mut buffer = vec![0u8; 256];

    // Before any error, message should be empty or "(unknown error)"
    let len = unsafe { LR_GetLastError(buffer.as_mut_ptr(), 256) };
    // Either 0 (no error set) or some message

    // Trigger an error
    let handle = unsafe { LR_OLS(std::ptr::null(), 10, std::ptr::null(), 1) };
    assert_eq!(handle, 0);

    // Now we should have an error message
    let len = unsafe { LR_GetLastError(buffer.as_mut_ptr(), 256) };
    assert!(len > 0, "Should have error message after failed call");

    let msg = String::from_utf8_lossy(&buffer[..len as usize]);
    assert!(!msg.is_empty(), "Error message should not be empty");
}

#[test]
fn test_ols_handle_reuse() {
    let (y, x) = simple_linear_data();
    let x_matrix = columns_to_row_major(&[x]);

    // Create first handle
    let h1 = unsafe { LR_OLS(y.as_ptr(), y.len() as i32, x_matrix.as_ptr(), 1) };
    assert_ne!(h1, 0);

    // Create second handle (should be different)
    let h2 = unsafe { LR_OLS(y.as_ptr(), y.len() as i32, x_matrix.as_ptr(), 1) };
    assert_ne!(h2, 0);
    assert_ne!(h1, h2, "Each call should return a unique handle");

    // Both should have valid R²
    let r2_1 = unsafe { LR_GetRSquared(h1) };
    let r2_2 = unsafe { LR_GetRSquared(h2) };
    assert_eq!(r2_1, r2_2, "Same data should produce same R²");

    unsafe {
        LR_Free(h1);
        LR_Free(h2);
    }
}

#[test]
fn test_ols_double_free_is_safe() {
    let (y, x) = simple_linear_data();
    let x_matrix = columns_to_row_major(&[x]);

    let handle = unsafe { LR_OLS(y.as_ptr(), y.len() as i32, x_matrix.as_ptr(), 1) };
    assert_ne!(handle, 0);

    unsafe {
        LR_Free(handle);
        LR_Free(handle); // Should not crash
    }
}

#[test]
fn test_ols_invalid_handle_returns_nan() {
    // Use an invalid handle that was never returned
    let invalid_handle = 999999;

    let r2 = unsafe { LR_GetRSquared(invalid_handle) };
    assert!(r2.is_nan(), "Invalid handle should return NaN");

    let mse = unsafe { LR_GetMSE(invalid_handle) };
    assert!(mse.is_nan(), "Invalid handle should return NaN");

    let n_coef = unsafe { LR_GetNumCoefficients(invalid_handle) };
    assert_eq!(n_coef, -1, "Invalid handle should return -1 for count");
}

#[test]
fn test_ols_single_predictor() {
    // Test with exactly one predictor
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let handle = unsafe { LR_OLS(y.as_ptr(), 5, x.as_ptr(), 1) };
    let _guard = HandleGuard::new(handle);

    let r2 = unsafe { LR_GetRSquared(handle) };
    assert_eq!(r2, 1.0, "Perfect linear relationship should have R² = 1");

    let n_coef = unsafe { LR_GetNumCoefficients(handle) } as usize;
    assert_eq!(n_coef, 2, "Should have intercept + 1 slope");

    let mut coefs = vec![0.0f64; 2];
    unsafe { LR_GetCoefficients(handle, coefs.as_mut_ptr(), 2) };

    // y = 0.5 * x, so intercept = 0, slope = 0.5
    assert!((coefs[0]).abs() < 1e-9, "Intercept should be ~0");
    assert!((coefs[1] - 0.5).abs() < 1e-9, "Slope should be 0.5");
}

#[test]
fn test_ols_with_constant_x() {
    // When all x values are the same, we should still get a result
    // (though statistical validity is questionable)
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    let handle = unsafe { LR_OLS(y.as_ptr(), 5, x.as_ptr(), 1) };

    // This might fail due to perfect collinearity (intercept = constant x)
    // or it might succeed; the important thing is it doesn't crash
    if handle != 0 {
        unsafe { LR_Free(handle) };
    }
}
