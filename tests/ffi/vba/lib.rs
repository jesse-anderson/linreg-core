// FFI Integration Tests for VBA/C#/C++ consumers
//
// This test module validates the FFI layer by calling exported functions
// directly through raw pointers, simulating how external consumers (VBA,
// C#, C++, etc.) interact with the library.
//
// # Running the Tests
//
// ```bash
// cargo test --package linreg-core --features 'ffi' --test ffi_tests
// ```
//
// Or from the root:
//
// ```bash
// cargo test --features ffi --tests ffi
// ```
//
// # Test Organization
//
// - `common` - Shared utilities and FFI declarations
// - `ols` - OLS regression FFI tests
// - `regularized` - Ridge/Lasso/ElasticNet FFI tests
// - `diagnostics` - Diagnostic test FFI tests
// - `prediction_intervals` - Prediction interval FFI tests
// - `cross_validation` - Cross-validation FFI tests
// - `utilities` - Utility function tests (version, init, error handling)

// Only compile when FFI feature is enabled
#![cfg(feature = "ffi")]

mod vba;

// Re-export for convenience
pub use vba::common;
pub use vba::cross_validation;
pub use vba::diagnostics;
pub use vba::ols;
pub use vba::prediction_intervals;
pub use vba::regularized;
pub use vba::utilities;

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod integration {
    use super::*;

    #[test]
    fn test_ffi_version_available() {
        let mut buffer = vec![0u8; 64];
        let written = unsafe { common::LR_Version(buffer.as_mut_ptr(), 64) };
        assert!(written > 0, "FFI version function should work");
    }

    #[test]
    fn test_ffi_init_success() {
        let result = unsafe { common::LR_Init() };
        assert!(result >= 0, "FFI init should succeed");
    }

    #[test]
    fn test_ffi_full_ols_workflow() {
        use common::*;

        let (y, x) = common::simple_linear_data();
        let x_matrix = common::columns_to_row_major(&[x]);

        // Fit model
        let handle = unsafe {
            LR_OLS(
                y.as_ptr(),
                y.len() as i32,
                x_matrix.as_ptr(),
                1,
            )
        };
        assert_ne!(handle, 0, "OLS should succeed");

        // Get results
        let r2 = unsafe { LR_GetRSquared(handle) };
        assert!(r2 > 0.9, "R² should be high for linear data");

        let n_coef = unsafe { LR_GetNumCoefficients(handle) };
        assert_eq!(n_coef, 2, "Should have 2 coefficients");

        // Clean up
        unsafe { LR_Free(handle) };
    }
}
