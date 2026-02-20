//! FFI Integration Tests for VBA/C#/C++ consumers
//!
//! This test module validates the FFI layer by calling exported functions
//! directly through the library's public API, simulating how external
//! consumers (VBA, C#, C++, etc.) interact with the DLL.
//!
//! # Running the Tests
//!
//! ```bash
//! cargo test --features ffi --test ffi_vba_tests
//! ```

// Only compile when FFI feature is enabled
#[cfg(feature = "ffi")]

mod vba {
    include!("ffi/vba/mod.rs");
}

// Re-export common items individually to avoid module name conflict
#[cfg(feature = "ffi")]
pub use vba::common::{
    columns_to_row_major, simple_linear_data, mtcars_subset,
    HandleGuard,
    // OLS functions
    LR_Free, LR_OLS, LR_GetRSquared, LR_GetNumCoefficients,
    // Utilities
    LR_Version, LR_Init,
};

// Re-export other modules
#[cfg(feature = "ffi")]
pub use vba::{cross_validation, diagnostics, ols, prediction_intervals, regularized, utilities};

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(all(test, feature = "ffi"))]
mod integration {
    use super::*;

    #[test]
    fn test_ffi_version_available() {
        let mut buffer = vec![0u8; 64];
        let written = unsafe { LR_Version(buffer.as_mut_ptr(), 64) };
        assert!(written > 0, "FFI version function should work");
    }

    #[test]
    fn test_ffi_init_success() {
        let result = unsafe { LR_Init() };
        assert!(result >= 0, "FFI init should succeed");
    }

    #[test]
    fn test_ffi_full_ols_workflow() {
        let (y, x) = simple_linear_data();
        let x_matrix = columns_to_row_major(&[x]);

        // Fit model
        let handle = unsafe { LR_OLS(y.as_ptr(), y.len() as i32, x_matrix.as_ptr(), 1) };
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
