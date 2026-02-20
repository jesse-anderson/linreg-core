// ============================================================================
// Result Classes for Python Bindings
// ============================================================================
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
