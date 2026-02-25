// ============================================================================
// Result Classes for Python Bindings
// ============================================================================
//! Result classes for Python bindings.
//!
//! This module defines Python-wrapper classes for all result types
//! returned by linreg-core functions. These classes provide Pythonic
//! access to regression outputs, diagnostic test results, and other
//! computed values.
//!
//! ## Categories
//!
//! - **Regression Results**: `OLSResult`, `RidgeResult`, `LassoResult`, `ElasticNetResult`, `WLSResult`, `LoessResult`, `PolynomialResult`
//! - **Diagnostic Results**: `DiagnosticResult`, `RainbowTestResult`, `WhiteTestResult`, etc.
//! - **Utility Results**: `LambdaPathResult`, `CSVResult`, `FoldResult`, `CVResult`
//! - **Prediction Intervals**: `PredictionIntervalResult`
//! - **Feature Importance**: `StandardizedCoefficientsResult`, `ShapResult`, `PermutationImportanceResult`, `VifRankingResult`
//!
// Split into sub-modules by category:
//   - regression.rs  : OLS, Ridge, Lasso, ElasticNet, WLS, LOESS, Polynomial
//   - diagnostics.rs : All diagnostic test result types
//   - utils.rs       : LambdaPath, CSV, cross-validation, prediction intervals
//   - feature_importance.rs : Feature importance result types

pub mod regression;
pub mod diagnostics;
pub mod utils;
pub mod feature_importance;

#[cfg(feature = "python")]
pub use regression::*;
#[cfg(feature = "python")]
pub use diagnostics::*;
#[cfg(feature = "python")]
pub use utils::*;
#[cfg(feature = "python")]
pub use feature_importance::*;
