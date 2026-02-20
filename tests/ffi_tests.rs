//! FFI Integration Tests
//!
//! This test validates the FFI layer by directly calling the FFI functions
//! that are exported when the `ffi` feature is enabled.
//!
//! Run with: cargo test --features ffi --test ffi_tests

#[cfg(feature = "ffi")]
mod ffi_tests {
    // Import the FFI module - this gives us access to all the FFI functions
    use linreg_core::ffi;

    // ========================================================================
    // Common Test Utilities
    // ========================================================================

    /// Helper to retrieve the last error message from the FFI layer
    fn get_last_error_message() -> String {
        let mut buffer = vec![0u8; 512];
        unsafe {
            let written = ffi::ols::LR_GetLastError(buffer.as_mut_ptr(), 512);
            if written > 0 {
                String::from_utf8_lossy(&buffer[..written as usize]).to_string()
            } else {
                "(unknown error)".to_string()
            }
        }
    }

    /// Converts column vectors to row-major matrix (FFI format)
    fn columns_to_row_major(columns: &[Vec<f64>]) -> Vec<f64> {
        let n = columns.first().map(|c| c.len()).unwrap_or(0);
        let p = columns.len();
        let mut result = Vec::with_capacity(n * p);

        for row in 0..n {
            for col in 0..p {
                result.push(columns[col][row]);
            }
        }
        result
    }

    /// RAII guard for automatic handle cleanup
    struct HandleGuard(usize);

    impl HandleGuard {
        fn new(handle: usize) -> Option<Self> {
            if handle != 0 {
                Some(Self(handle))
            } else {
                None
            }
        }

        fn handle(&self) -> usize {
            self.0
        }
    }

    impl Drop for HandleGuard {
        fn drop(&mut self) {
            unsafe {
                ffi::ols::LR_Free(self.0);
            }
        }
    }

    /// Test data fixtures
    fn simple_linear_data() -> (Vec<f64>, Vec<f64>) {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
        (y, x)
    }

    fn mtcars_subset() -> (Vec<f64>, Vec<Vec<f64>>) {
        let y = vec![
            21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2,
            10.4, 10.4, 14.7, 32.4, 30.4, 33.9,
        ];

        let cyl = vec![6.0, 6.0, 4.0, 6.0, 8.0, 6.0, 8.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0];
        let disp = vec![160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472.0, 460.0, 440.0, 78.7, 75.7, 71.1];
        let hp = vec![110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0, 123.0, 180.0, 180.0, 180.0, 205.0, 215.0, 230.0, 66.0, 52.0, 65.0];
        let wt = vec![2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440, 3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835];

        (y, vec![cyl, disp, hp, wt])
    }

    // ========================================================================
    // Version and Init Tests
    // ========================================================================

    #[test]
    fn test_version() {
        let mut buffer = vec![0u8; 64];
        unsafe {
            let written = ffi::utils::LR_Version(buffer.as_mut_ptr(), 64);
            assert!(written > 0, "Version should write bytes");

            let version = String::from_utf8_lossy(&buffer[..written as usize]);
            assert!(version.contains('.'), "Version should contain dots: {}", version);
        }
    }

    #[test]
    fn test_init() {
        unsafe {
            let result = ffi::utils::LR_Init();
            assert!(result >= 0, "Init should return non-negative");
        }
    }

    // ========================================================================
    // OLS Regression Tests
    // ========================================================================

    #[test]
    fn test_ols_simple_linear() {
        let (y, x) = simple_linear_data();
        let x_matrix = columns_to_row_major(&[x.clone()]);

        unsafe {
            let handle = ffi::ols::LR_OLS(y.as_ptr(), y.len() as i32, x_matrix.as_ptr(), 1);

            let guard = HandleGuard::new(handle).expect("OLS should succeed");

            let r2 = ffi::ols::LR_GetRSquared(guard.handle());
            assert!(r2 > 0.99, "R² should be > 0.99 for linear data, got {}", r2);

            let n_coef = ffi::ols::LR_GetNumCoefficients(guard.handle()) as usize;
            assert_eq!(n_coef, 2, "Should have 2 coefficients");

            let mut coefs = vec![0.0f64; n_coef];
            ffi::ols::LR_GetCoefficients(guard.handle(), coefs.as_mut_ptr(), n_coef as i32);

            assert!((coefs[0] - 2.0).abs() < 0.1, "Intercept should be ~2.0");
            assert!((coefs[1] - 3.0).abs() < 0.1, "Slope should be ~3.0");
        }
    }

    #[test]
    fn test_ols_multiple_regression() {
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;

        unsafe {
            let handle = ffi::ols::LR_OLS(y.as_ptr(), n, x_matrix.as_ptr(), p);

            let guard = HandleGuard::new(handle).expect("OLS should succeed");

            let r2 = ffi::ols::LR_GetRSquared(guard.handle());
            assert!(r2 >= 0.5 && r2 <= 1.0, "R² should be in valid range");

            let adj_r2 = ffi::ols::LR_GetAdjRSquared(guard.handle());
            assert!(adj_r2 <= r2 + 0.01, "Adj R² should be <= R²");

            let mse = ffi::ols::LR_GetMSE(guard.handle());
            assert!(mse > 0.0, "MSE should be positive");
        }
    }

    #[test]
    fn test_ols_error_null_pointer() {
        unsafe {
            let handle = ffi::ols::LR_OLS(std::ptr::null(), 10, std::ptr::null(), 1);
            assert_eq!(handle, 0, "Null pointer should return error handle");

            let msg = get_last_error_message();
            assert!(msg.contains("null") || msg.contains("Null") || msg.contains("0"),
                "Error should mention null pointer or zero: {}", msg);
        }
    }

    #[test]
    fn test_ols_vector_getters() {
        let (y, x) = simple_linear_data();
        let x_matrix = columns_to_row_major(&[x]);

        unsafe {
            let handle = ffi::ols::LR_OLS(y.as_ptr(), y.len() as i32, x_matrix.as_ptr(), 1);

            let guard = HandleGuard::new(handle).expect("OLS should succeed");

            let n_coef = ffi::ols::LR_GetNumCoefficients(guard.handle()) as usize;

            // Test all vector getters
            let mut coefs = vec![0.0f64; n_coef];
            let mut ses = vec![0.0f64; n_coef];
            let mut tstats = vec![0.0f64; n_coef];
            let mut pvals = vec![0.0f64; n_coef];

            ffi::ols::LR_GetCoefficients(guard.handle(), coefs.as_mut_ptr(), n_coef as i32);
            ffi::ols::LR_GetStdErrors(guard.handle(), ses.as_mut_ptr(), n_coef as i32);
            ffi::ols::LR_GetTStats(guard.handle(), tstats.as_mut_ptr(), n_coef as i32);
            ffi::ols::LR_GetPValues(guard.handle(), pvals.as_mut_ptr(), n_coef as i32);

            // Verify all are valid
            for i in 0..n_coef {
                assert!(!coefs[i].is_nan(), "Coefficient {} should be valid", i);
                assert!(ses[i] > 0.0, "SE {} should be positive", i);
                assert!(!tstats[i].is_nan(), "t-stat {} should be valid", i);
                assert!(pvals[i] >= 0.0 && pvals[i] <= 1.0, "p-value {} should be in [0,1]", i);
            }
        }
    }

    #[test]
    fn test_ols_residuals_and_fitted() {
        let (y, x) = simple_linear_data();
        let x_matrix = columns_to_row_major(&[x]);

        unsafe {
            let handle = ffi::ols::LR_OLS(y.as_ptr(), y.len() as i32, x_matrix.as_ptr(), 1);

            let guard = HandleGuard::new(handle).expect("OLS should succeed");

            let n_obs = ffi::ols::LR_GetNumObservations(guard.handle()) as usize;

            let mut residuals = vec![0.0f64; n_obs];
            let mut fitted = vec![0.0f64; n_obs];

            ffi::ols::LR_GetResiduals(guard.handle(), residuals.as_mut_ptr(), n_obs as i32);
            ffi::ols::LR_GetFittedValues(guard.handle(), fitted.as_mut_ptr(), n_obs as i32);

            // Verify y = fitted + residuals
            for i in 0..n_obs {
                let reconstructed = fitted[i] + residuals[i];
                assert!((reconstructed - y[i]).abs() < 1e-9,
                    "y = fitted + residual at index {}: {} = {} + {}", i, y[i], fitted[i], residuals[i]);
            }
        }
    }

    #[test]
    fn test_ols_handle_cleanup() {
        let (y, x) = simple_linear_data();
        let x_matrix = columns_to_row_major(&[x]);

        unsafe {
            let handle = ffi::ols::LR_OLS(y.as_ptr(), y.len() as i32, x_matrix.as_ptr(), 1);
            assert_ne!(handle, 0);

            // Double free should be safe
            ffi::ols::LR_Free(handle);
            ffi::ols::LR_Free(handle);
        }
    }

    #[test]
    fn test_ols_invalid_handle() {
        unsafe {
            let invalid = 999999;

            let r2 = ffi::ols::LR_GetRSquared(invalid);
            assert!(r2.is_nan(), "Invalid handle should return NaN");

            let mse = ffi::ols::LR_GetMSE(invalid);
            assert!(mse.is_nan(), "Invalid handle should return NaN");

            let n_coef = ffi::ols::LR_GetNumCoefficients(invalid);
            assert_eq!(n_coef, -1, "Invalid handle should return -1");
        }
    }

    // ========================================================================
    // Ridge Regression Tests
    // ========================================================================

    #[test]
    fn test_ridge_basic() {
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;

        unsafe {
            let handle = ffi::regularized::LR_Ridge(y.as_ptr(), n, x_matrix.as_ptr(), p, 1.0, 1);

            let guard = HandleGuard::new(handle).expect("Ridge should succeed");

            let r2 = ffi::ols::LR_GetRSquared(guard.handle());
            assert!(r2 >= 0.0 && r2 <= 1.0);

            let intercept = ffi::regularized::LR_GetIntercept(guard.handle());
            assert!(!intercept.is_nan());

            let n_coef = ffi::ols::LR_GetNumCoefficients(guard.handle()) as usize;
            assert_eq!(n_coef, x_cols.len());
        }
    }

    // ========================================================================
    // Lasso Regression Tests
    // ========================================================================

    #[test]
    fn test_lasso_basic() {
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;

        unsafe {
            let handle = ffi::regularized::LR_Lasso(y.as_ptr(), n, x_matrix.as_ptr(), p, 0.1, 1, 10000, 1e-7);

            let guard = HandleGuard::new(handle).expect("Lasso should succeed");

            let converged = ffi::regularized::LR_GetConverged(guard.handle());
            assert_eq!(converged, 1, "Lasso should converge");

            let n_nonzero = ffi::regularized::LR_GetNNonzero(guard.handle()) as usize;
            assert!(n_nonzero > 0 && n_nonzero <= x_cols.len());
        }
    }

    // ========================================================================
    // Elastic Net Tests
    // ========================================================================

    #[test]
    fn test_elastic_net_basic() {
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;

        unsafe {
            let handle = ffi::regularized::LR_ElasticNet(y.as_ptr(), n, x_matrix.as_ptr(), p, 0.1, 0.5, 1, 10000, 1e-7);

            let guard = HandleGuard::new(handle).expect("Elastic Net should succeed");

            let r2 = ffi::ols::LR_GetRSquared(guard.handle());
            assert!(r2 >= 0.0 && r2 <= 1.0);

            let converged = ffi::regularized::LR_GetConverged(guard.handle());
            assert_eq!(converged, 1);
        }
    }

    // ========================================================================
    // Diagnostic Tests
    // ========================================================================

    #[test]
    fn test_breusch_pagan() {
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;

        unsafe {
            let handle = ffi::diagnostics::LR_BreuschPagan(y.as_ptr(), n, x_matrix.as_ptr(), p);

            let guard = HandleGuard::new(handle).expect("Breusch-Pagan should succeed");

            let stat = ffi::diagnostics::LR_GetStatistic(guard.handle());
            let p_val = ffi::diagnostics::LR_GetPValue(guard.handle());
            let df = ffi::diagnostics::LR_GetTestDF(guard.handle());

            assert!(!stat.is_nan());
            assert!(!p_val.is_nan());
            assert!(p_val >= 0.0 && p_val <= 1.0);
            // df is often 0.0 for diagnostic tests that don't report it
            assert!(df.is_nan() || df >= 0.0);
        }
    }

    #[test]
    fn test_durbin_watson() {
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;

        unsafe {
            let handle = ffi::diagnostics::LR_DurbinWatson(y.as_ptr(), n, x_matrix.as_ptr(), p);

            let guard = HandleGuard::new(handle).expect("Durbin-Watson should succeed");

            let stat = ffi::diagnostics::LR_GetStatistic(guard.handle());
            let autocorr = ffi::diagnostics::LR_GetAutocorrelation(guard.handle());

            assert!(!stat.is_nan());
            assert!(stat >= 0.0 && stat <= 4.0, "DW should be in [0, 4]");

            // Autocorrelation = 1 - DW/2
            let expected_autocorr = 1.0 - stat / 2.0;
            assert!((autocorr - expected_autocorr).abs() < 1e-9);
        }
    }

    #[test]
    fn test_diagnostics_null_pointer() {
        unsafe {
            let handle = ffi::diagnostics::LR_JarqueBera(std::ptr::null(), 10, std::ptr::null(), 1);
            assert_eq!(handle, 0, "Null pointer should error");
        }
    }

    // ========================================================================
    // Prediction Intervals Tests
    // ========================================================================

    #[test]
    fn test_prediction_intervals_basic() {
        let y_train: Vec<f64> = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_train: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_train_matrix = columns_to_row_major(&[x_train.clone()]);

        let x_new = vec![2.0, 3.0];
        let n_new = x_new.len() as i32;

        unsafe {
            let handle = ffi::prediction_intervals::LR_PredictionIntervals(
                y_train.as_ptr(),
                y_train.len() as i32,
                x_train_matrix.as_ptr(),
                1,
                x_new.as_ptr(),
                n_new,
                0.05,
            );

            let guard = HandleGuard::new(handle).expect("Prediction intervals should succeed");

            let mut predicted = vec![0.0f64; n_new as usize];
            let mut lower = vec![0.0f64; n_new as usize];
            let mut upper = vec![0.0f64; n_new as usize];
            let mut se_pred = vec![0.0f64; n_new as usize];

            ffi::prediction_intervals::LR_GetPredicted(guard.handle(), predicted.as_mut_ptr(), n_new);
            ffi::prediction_intervals::LR_GetLowerBound(guard.handle(), lower.as_mut_ptr(), n_new);
            ffi::prediction_intervals::LR_GetUpperBound(guard.handle(), upper.as_mut_ptr(), n_new);
            ffi::prediction_intervals::LR_GetSEPred(guard.handle(), se_pred.as_mut_ptr(), n_new);

            for i in 0..n_new as usize {
                assert!(predicted[i].is_finite());
                assert!(lower[i] <= predicted[i]);
                assert!(predicted[i] <= upper[i]);
                assert!(se_pred[i] > 0.0);
            }
        }
    }

    // ========================================================================
    // Cross-Validation Tests
    // ========================================================================

    #[test]
    fn test_kfold_ols() {
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;
        let k = 5;

        unsafe {
            let handle = ffi::cross_validation::LR_KFoldOLS(y.as_ptr(), n, x_matrix.as_ptr(), p, k);

            let guard = HandleGuard::new(handle).expect("K-Fold OLS should succeed");

            let mean_mse = ffi::cross_validation::LR_GetCVMeanMSE(guard.handle());
            let mean_rmse = ffi::cross_validation::LR_GetCVMeanRMSE(guard.handle());
            let mean_r2 = ffi::cross_validation::LR_GetCVMeanR2(guard.handle());

            assert!(mean_mse.is_finite() && mean_mse > 0.0);
            assert!(mean_rmse.is_finite() && mean_rmse > 0.0);
            assert!(mean_r2.is_finite());
        }
    }

    #[test]
    fn test_kfold_ridge() {
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;

        unsafe {
            let handle = ffi::cross_validation::LR_KFoldRidge(y.as_ptr(), n, x_matrix.as_ptr(), p, 1.0, 1, 5);

            let guard = HandleGuard::new(handle).expect("K-Fold Ridge should succeed");

            let mean_mse = ffi::cross_validation::LR_GetCVMeanMSE(guard.handle());
            assert!(mean_mse.is_finite() && mean_mse > 0.0);
        }
    }

    #[test]
    fn test_kfold_rmse_mse_relationship() {
        // Test the mathematical relationship between MSE and RMSE in cross-validation
        //
        // Within each fold: rmse = sqrt(mse) by definition
        // Across folds: mean_rmse < sqrt(mean_mse) due to Jensen's inequality
        // (sqrt is a concave function, so average of sqrt < sqrt of average)
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;
        let k = 5;

        unsafe {
            let handle = ffi::cross_validation::LR_KFoldOLS(y.as_ptr(), n, x_matrix.as_ptr(), p, k);

            let guard = HandleGuard::new(handle).expect("K-Fold OLS should succeed");

            let mean_mse = ffi::cross_validation::LR_GetCVMeanMSE(guard.handle());
            let mean_rmse = ffi::cross_validation::LR_GetCVMeanRMSE(guard.handle());

            assert!(mean_mse.is_finite() && mean_mse > 0.0, "MSE should be positive");
            assert!(mean_rmse.is_finite() && mean_rmse > 0.0, "RMSE should be positive");

            // Due to Jensen's inequality for the concave sqrt function:
            // mean(sqrt(x_i)) < sqrt(mean(x_i)) when the x_i values differ
            // In other words: mean_rmse < sqrt(mean_mse)
            let sqrt_mean_mse = mean_mse.sqrt();

            // The relationship should hold: mean_rmse <= sqrt(mean_mse)
            // (equal only if all folds have exactly the same MSE)
            assert!(
                mean_rmse <= sqrt_mean_mse + 1e-10,
                "mean_rmse ({}) should be <= sqrt(mean_mse) ({}) due to Jensen's inequality",
                mean_rmse,
                sqrt_mean_mse
            );

            // Also verify the values are in a reasonable relationship
            // If they were equal, all folds would have the same MSE (unlikely)
            // We expect some difference due to varying fold performance
            let ratio = mean_rmse / sqrt_mean_mse;
            assert!(ratio > 0.9 && ratio < 1.0,
                "RMSE ratio should be close to but less than 1, got {}", ratio);
        }
    }

    #[test]
    fn test_kfold_all_variance_metrics() {
        // Test that all CV variance metrics are finite and non-negative
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;

        unsafe {
            let handle = ffi::cross_validation::LR_KFoldOLS(y.as_ptr(), n, x_matrix.as_ptr(), p, 5);

            let guard = HandleGuard::new(handle).expect("K-Fold OLS should succeed");

            // Mean metrics
            let mean_mse = ffi::cross_validation::LR_GetCVMeanMSE(guard.handle());
            let mean_rmse = ffi::cross_validation::LR_GetCVMeanRMSE(guard.handle());
            let mean_r2 = ffi::cross_validation::LR_GetCVMeanR2(guard.handle());

            // Std metrics
            let std_mse = ffi::cross_validation::LR_GetCVStdMSE(guard.handle());
            let std_rmse = ffi::cross_validation::LR_GetCVStdRMSE(guard.handle());

            // All means should be finite
            assert!(mean_mse.is_finite());
            assert!(mean_rmse.is_finite());
            assert!(mean_r2.is_finite());

            // MSE and RMSE should be positive
            assert!(mean_mse > 0.0);
            assert!(mean_rmse > 0.0);

            // Standard deviations should be non-negative
            assert!(std_mse >= 0.0);
            assert!(std_rmse >= 0.0);
        }
    }

    // ========================================================================
    // Influence Diagnostics Tests
    // ========================================================================

    #[test]
    fn test_cooks_distance() {
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;

        unsafe {
            let handle = ffi::diagnostics::LR_CooksDistance(y.as_ptr(), n, x_matrix.as_ptr(), p);

            let guard = HandleGuard::new(handle).expect("Cook's Distance should succeed");

            let vec_len = ffi::ols::LR_GetVectorLength(guard.handle()) as usize;
            assert_eq!(vec_len, y.len());

            let mut distances = vec![0.0f64; vec_len];
            ffi::ols::LR_GetVector(guard.handle(), distances.as_mut_ptr(), vec_len as i32);

            for (i, &d) in distances.iter().enumerate() {
                assert!(d >= 0.0, "Cook's distance at {} should be non-negative", i);
            }
        }
    }

    #[test]
    fn test_vif() {
        let (y, x_cols) = mtcars_subset();
        let x_matrix = columns_to_row_major(&x_cols);
        let n = y.len() as i32;
        let p = x_cols.len() as i32;

        unsafe {
            let handle = ffi::diagnostics::LR_VIF(y.as_ptr(), n, x_matrix.as_ptr(), p);

            let guard = HandleGuard::new(handle).expect("VIF should succeed");

            let vec_len = ffi::ols::LR_GetVectorLength(guard.handle()) as usize;
            assert_eq!(vec_len, x_cols.len());

            let mut vifs = vec![0.0f64; vec_len];
            ffi::ols::LR_GetVector(guard.handle(), vifs.as_mut_ptr(), vec_len as i32);

            for (i, &vif) in vifs.iter().enumerate() {
                assert!(vif >= 1.0, "VIF at {} should be >= 1.0", i);
            }
        }
    }
}
