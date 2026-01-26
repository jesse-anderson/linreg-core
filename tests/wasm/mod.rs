// ============================================================================
// WASM Tests Module
// ============================================================================
//
// Browser-based WASM tests using wasm-bindgen-test framework.
// Tests cover OLS regression, diagnostic tests, error handling, CSV parsing,
// regularized regression, and domain checking.
//
// Run with: wasm-pack test --chrome --headless

#![cfg(target_arch = "wasm32")]

pub mod fixtures;
pub mod integration_tests;
pub mod diagnostic_tests;
pub mod ols_tests;
pub mod regularized_tests;
pub mod utility_tests;

// Re-export all tests at the module level
pub use fixtures::*;
pub use integration_tests::*;
pub use diagnostic_tests::*;
pub use ols_tests::*;
pub use regularized_tests::*;
pub use utility_tests::*;
