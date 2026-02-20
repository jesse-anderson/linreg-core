//! Common utilities and FFI function access for VBA/C#/C++ FFI tests.
//!
//! This module provides test fixtures and helper functions for testing
//! the FFI layer from within Rust. Instead of re-declaring FFI functions
//! with `extern "system"`, we import them directly from `linreg_core::ffi`,
//! which allows the tests to run as proper Rust integration tests.
//!
//! # Test Data Fixtures
//!
//! - `simple_linear_data()` - Perfect linear relationship (y = 2 + 3x)
//! - `mtcars_subset()` - Real regression data (20 observations, 4 predictors)
//! - `heteroscedastic_data()` - For heteroscedasticity tests
//!
//! # Helper Functions
//!
//! - `columns_to_row_major()` - Converts column vectors to FFI's row-major format
//! - `HandleGuard` - RAII wrapper for automatic handle cleanup
//! - `get_last_error_message()` - Retrieves FFI error messages
//! - `read_vector_result()` - Reads Vector results via LR_GetVector
//! - `read_matrix_result()` - Reads Matrix results via LR_GetMatrix
//! - `read_prediction_intervals()` - Reads prediction interval arrays
//! - `assert_valid_handle()` - Asserts non-zero handle with error message

use linreg_core::ffi;

// Re-export FFI functions for convenient use in tests
pub use ffi::ols::{
    LR_Free, LR_GetLastError, LR_GetRSquared, LR_GetAdjRSquared, LR_GetFStatistic,
    LR_GetFPValue, LR_GetMSE, LR_GetNumCoefficients, LR_GetNumObservations, LR_GetCoefficients,
    LR_GetStdErrors, LR_GetTStats, LR_GetPValues, LR_GetResiduals, LR_GetFittedValues,
    LR_GetVectorLength, LR_GetVector, LR_GetMatrixRows, LR_GetMatrixCols, LR_GetMatrix,
    LR_OLS,
};
pub use ffi::regularized::{
    LR_Ridge, LR_Lasso, LR_ElasticNet, LR_GetIntercept, LR_GetDF, LR_GetNNonzero, LR_GetConverged,
};
pub use ffi::diagnostics::{
    LR_GetStatistic, LR_GetPValue, LR_GetTestDF, LR_GetAutocorrelation,
    LR_BreuschPagan, LR_JarqueBera, LR_ShapiroWilk, LR_AndersonDarling, LR_HarveyCollier,
    LR_White, LR_Rainbow, LR_Reset, LR_DurbinWatson, LR_BreuschGodfrey,
    LR_CooksDistance, LR_DFFITS, LR_VIF, LR_DFBETAS,
};
pub use ffi::prediction_intervals::{
    LR_PredictionIntervals, LR_GetPredicted, LR_GetLowerBound, LR_GetUpperBound, LR_GetSEPred,
};
pub use ffi::cross_validation::{
    LR_KFoldOLS, LR_KFoldRidge, LR_KFoldLasso, LR_KFoldElasticNet,
    LR_GetCVNFolds, LR_GetCVMeanMSE, LR_GetCVStdMSE, LR_GetCVMeanRMSE, LR_GetCVStdRMSE, LR_GetCVMeanR2,
};
pub use ffi::utils::{LR_Version, LR_Init};

// ============================================================================
// Test Data Fixtures
// ============================================================================

/// Simple linear regression test data: y = 2 + 3x
pub fn simple_linear_data() -> (Vec<f64>, Vec<f64>) {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
    (y, x)
}

/// Multiple regression test data (mtcars subset - 20 observations, 4 predictors)
pub fn mtcars_subset() -> (Vec<f64>, Vec<Vec<f64>>) {
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

/// Heteroscedastic data (variance increases with x)
pub fn heteroscedastic_data() -> (Vec<f64>, Vec<f64>) {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| 2.0 + 3.0 * xi + (i as f64) * 0.5)
        .collect();
    (y, x)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Converts column vectors to row-major matrix format expected by FFI
///
/// # Arguments
/// * `columns` - Column vectors (each vec is one column)
///
/// # Returns
/// A flat f64 slice in row-major order (row 0 col 0, row 0 col 1, ...)
pub fn columns_to_row_major(columns: &[Vec<f64>]) -> Vec<f64> {
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
///
/// Wraps an FFI handle and automatically calls LR_Free when dropped.
/// Returns None if the handle is 0 (error), Some otherwise.
pub struct HandleGuard {
    pub handle: usize,
}

impl HandleGuard {
    /// Creates a new guard. Returns None if handle is 0 (error).
    pub fn new(handle: usize) -> Option<Self> {
        if handle != 0 {
            Some(Self { handle })
        } else {
            None
        }
    }

    /// Returns the underlying handle value
    pub fn handle(&self) -> usize {
        self.handle
    }
}

impl Drop for HandleGuard {
    fn drop(&mut self) {
        unsafe {
            ffi::ols::LR_Free(self.handle);
        }
    }
}

/// Helper to retrieve the last error message from the FFI layer
pub fn get_last_error_message() -> String {
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

/// Reads a vector result from a handle using LR_GetVectorLength and LR_GetVector
pub fn read_vector_result(handle: usize) -> Vec<f64> {
    let len = unsafe { ffi::ols::LR_GetVectorLength(handle) } as usize;
    assert!(len > 0, "Vector result should have positive length");

    let mut buffer = vec![0.0f64; len];
    let written = unsafe { ffi::ols::LR_GetVector(handle, buffer.as_mut_ptr(), len as i32) };
    assert_eq!(written, len as i32, "Vector read should return expected length");

    buffer
}

/// Reads a matrix result from a handle
///
/// # Returns
/// (rows, cols, data) where data is a flat row-major vector
pub fn read_matrix_result(handle: usize) -> (usize, usize, Vec<f64>) {
    let rows = unsafe { ffi::ols::LR_GetMatrixRows(handle) } as usize;
    let cols = unsafe { ffi::ols::LR_GetMatrixCols(handle) } as usize;

    assert!(rows > 0 && cols > 0, "Matrix should have positive dimensions");

    let total = rows * cols;
    let mut data = vec![0.0f64; total];
    let written = unsafe { ffi::ols::LR_GetMatrix(handle, data.as_mut_ptr(), total as i32) };
    assert_eq!(written, total as i32, "Matrix read should return all elements");

    (rows, cols, data)
}

/// Reads prediction interval results from a handle
///
/// # Returns
/// (predicted, lower_bound, upper_bound, se_pred)
pub fn read_prediction_intervals(handle: usize, n_new: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut predicted = vec![0.0f64; n_new];
    let mut lower = vec![0.0f64; n_new];
    let mut upper = vec![0.0f64; n_new];
    let mut se_pred = vec![0.0f64; n_new];

    unsafe {
        let n = n_new as i32;
        assert_eq!(ffi::prediction_intervals::LR_GetPredicted(handle, predicted.as_mut_ptr(), n), n);
        assert_eq!(ffi::prediction_intervals::LR_GetLowerBound(handle, lower.as_mut_ptr(), n), n);
        assert_eq!(ffi::prediction_intervals::LR_GetUpperBound(handle, upper.as_mut_ptr(), n), n);
        assert_eq!(ffi::prediction_intervals::LR_GetSEPred(handle, se_pred.as_mut_ptr(), n), n);
    }

    (predicted, lower, upper, se_pred)
}

/// Asserts that a handle is non-zero (valid).
///
/// Panics with the FFI error message if the handle is 0.
pub fn assert_valid_handle(handle: usize, context: &str) {
    assert_ne!(
        handle, 0,
        "{}: FFI call returned error handle (0): {}",
        context,
        get_last_error_message()
    );
}
