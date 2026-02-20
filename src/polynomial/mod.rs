//! # Polynomial Regression
//!
//! This module provides polynomial regression by transforming a single predictor
//! into polynomial features and fitting linear regression via the existing OLS solver.
//!
//! ## Mathematical Foundation
//!
//! Polynomial regression models the relationship between `y` and `x` as:
//!
//! ```text
//! y = β₀ + β₁x + β₂x² + β₃x³ + … + β_d·x^d + ε
//! ```
//!
//! While non-linear in `x`, this is **linear in the parameters** `β`, so all
//! OLS machinery (QR decomposition, statistics, diagnostics) applies unchanged.
//!
//! ## Multicollinearity
//!
//! Polynomial features are inherently correlated (x² correlates with x, etc.).
//! This does **not** affect prediction accuracy but inflates coefficient variance.
//!
//! **Mitigation strategies:**
//! 1. **Centering** (`center: true`): Subtract the mean of x before raising to
//!    powers. Substantially reduces correlation between terms. Recommended for
//!    degree ≥ 3.
//! 2. **Regularization**: Use `polynomial_ridge`, `polynomial_lasso`, or
//!    `polynomial_elastic_net` from the [`regularized`] sub-module.
//!
//! ## Example
//!
//! ```rust
//! use linreg_core::polynomial::{polynomial_regression, predict, PolynomialOptions};
//!
//! // Training data with a quadratic relationship
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();
//!
//! let options = PolynomialOptions {
//!     degree: 2,
//!     center: true,   // reduce multicollinearity
//!     ..Default::default()
//! };
//! let fit = polynomial_regression(&y, &x, &options).unwrap();
//! println!("R² = {}", fit.ols_output.r_squared);
//!
//! // Predict at new points — centering is applied automatically
//! let preds = predict(&fit, &[6.0, 7.0]).unwrap();
//! ```

pub mod features;
pub mod fit;
pub mod predict;
pub mod regularized;
pub mod types;

// Public API re-exports
pub use fit::polynomial_regression;
pub use predict::predict;
pub use regularized::{polynomial_elastic_net, polynomial_lasso, polynomial_ridge};
pub use types::{PolynomialFit, PolynomialOptions};
