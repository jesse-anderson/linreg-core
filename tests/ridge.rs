//! Ridge regression validation tests
//!
//! This module contains comprehensive validation tests for ridge regression.

mod ridge_modules;

#[test]
fn run_all_ridge_tests() {
    // This module is organized into submodules:
    // - baseline: Basic smoke tests and baseline values
    // - verification: Manual calculation verification
    // - glmnet_audit: Comparison with R's glmnet
    println!("Ridge regression tests are in ridge_modules/ subdirectory");
    println!("Run with: cargo test --test ridge");
}
