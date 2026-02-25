// XLL Integration Tests
//
// Tests the XLL add-in layer by calling UDF functions directly with
// constructed XLOPER12 inputs, simulating how Excel calls them.
//
// # Structure
//
// - `common.rs` - XLOPER12 builders, result inspectors, test fixtures
// - `version.rs` - LINREG.VERSION() UDF tests
// - `ols.rs` - LINREG.OLS() UDF tests
// - `errors.rs` - Error case tests (null, missing, empty cells, propagation)
// - `diagnostics.rs` - Diagnostic test UDF tests (14 diagnostics)
// - `cross_validation.rs` - K-Fold CV UDF tests (OLS, Ridge, Lasso, Elastic Net)
// - `prediction_intervals.rs` - Prediction interval UDF tests
//
// # Running
//
// ```bash
// cargo test --features xll --test xll_tests
// ```

pub mod common;
pub mod version;
pub mod ols;
pub mod wls;
pub mod regularized;
pub mod errors;
pub mod diagnostics;
pub mod cross_validation;
pub mod prediction_intervals;
pub mod polynomial;
