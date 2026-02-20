// FFI Tests for VBA/C#/C++ consumers
//
// This module tests the FFI layer to ensure compatibility with external consumers.
// Tests call the FFI functions directly through the library's public API,
// simulating how VBA, C#, or C++ would interact with the DLL.
//
// # Structure
//
// - `mod.rs` - Module exports
// - `common.rs` - Shared utilities and test fixtures
// - `ols.rs` - OLS regression FFI tests
// - `regularized.rs` - Ridge/Lasso/ElasticNet FFI tests
// - `diagnostics.rs` - Diagnostic test FFI tests
// - `prediction_intervals.rs` - Prediction interval FFI tests
// - `cross_validation.rs` - Cross-validation FFI tests
// - `utilities.rs` - Utility function tests (version, init, error handling)

// Only compile these tests when the FFI feature is enabled
#[cfg(feature = "ffi")]

pub mod common;
pub mod cross_validation;
pub mod diagnostics;
pub mod ols;
pub mod prediction_intervals;
pub mod regularized;
pub mod utilities;
