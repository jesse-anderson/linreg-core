// ============================================================================
// Unit Tests for linreg-core
// ============================================================================
//
// This directory contains unit tests for internal library functionality.
// Tests are organized by module.

#![cfg(not(target_arch = "wasm32"))]

mod core_tests;
mod diagnostics_tests;
mod distributions_tests;
mod error_tests;
mod glmnet_algorithm_tests;  // GLMNET algorithm validation tests
mod input_validation_tests;
mod linalg;
mod regularized_tests;
