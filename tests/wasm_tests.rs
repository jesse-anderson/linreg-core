// ============================================================================
// WASM Integration Tests
// ============================================================================
//
// Browser-based WASM tests using wasm-bindgen-test framework.
// Tests cover OLS regression, diagnostic tests, error handling, CSV parsing,
// regularized regression, and domain checking.
//
// Run with: wasm-pack test --chrome --headless

#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;
use wasm_bindgen_test::*;

// Import all test modules
mod wasm;

// Re-export all tests at the crate level so wasm-bindgen-test can find them
use wasm::fixtures;
use wasm::integration_tests;
use wasm::diagnostic_tests;
use wasm::ols_tests;
use wasm::regularized_tests;
use wasm::utility_tests;
