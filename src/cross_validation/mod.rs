// ============================================================================
// Cross Validation Module
// ============================================================================

//! K-Fold Cross Validation for linear regression models.
//!
//! This module provides cross-validation functionality for estimating out-of-sample
//! prediction error and selecting optimal hyperparameters (e.g., lambda for
//! regularized regression).
//!
//! # Supported Models
//!
//! - **OLS** — Ordinary Least Squares regression
//! - **Ridge** — L2-regularized regression
//! - **Lasso** — L1-regularized regression
//! - **Elastic Net** — Combined L1/L2 regularization
//!
//! # Basic Usage
//!
//! ```rust
//! use linreg_core::cross_validation::{kfold_cv_ols, KFoldOptions};
//!
//! let y = vec![2.5, 3.7, 4.2, 5.1, 6.3, 7.0, 7.5, 8.1];
//! let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0, 4.5, 5.5, 6.0];
//! let names = vec!["Intercept".into(), "X1".into(), "X2".into()];
//!
//! let options = KFoldOptions {
//!     n_folds: 5,
//!     shuffle: true,
//!     seed: Some(42),
//! };
//!
//! let result = kfold_cv_ols(&y, &[x1, x2], &names, &options)?;
//! println!("CV RMSE: {:.4} (+/- {:.4})", result.mean_rmse, result.std_rmse);
//! # Ok::<(), linreg_core::Error>(())
//! ```

// Submodules
pub mod kfold;
pub mod metrics;
pub mod splits;
pub mod types;

// Re-export public API
pub use types::{CVResult, FoldResult, KFoldOptions};

// Re-export CV functions
pub use kfold::{kfold_cv_elastic_net, kfold_cv_lasso, kfold_cv_ols, kfold_cv_ridge};
