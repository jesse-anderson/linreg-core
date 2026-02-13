// ============================================================================
// Unit Tests for linreg-core
// ============================================================================
//
// This directory contains unit tests for internal library functionality.
// Tests are organized by module.

#![cfg(not(target_arch = "wasm32"))]

mod core_tests;
mod cross_validation_tests;  // K-Fold Cross Validation tests
mod diagnostics;        // New modular test structure for diagnostics
mod diagnostics_tests;
mod distributions; // New modular test structure for distributions
                  // Old: distributions_tests (kept for now for backward compatibility)
mod distributions_tests; // Legacy file - will be removed after migration
mod error_tests;
mod glmnet_algorithm_tests;  // GLMNET algorithm validation tests
mod input_validation_tests;
mod linalg;
mod loess_tests;  // LOESS non-parametric regression tests
mod regularized_tests;
mod serialization_tests;  // Model save/load tests
