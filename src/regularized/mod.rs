//! Ridge and Lasso regression (glmnet-compatible implementations).
//!
//! This module provides regularized regression methods that are compatible with
//! R's glmnet package. The implementations follow the same objective functions,
//! standardization conventions, and scaling approaches as glmnet.
//!
//! # Objective Function
//!
//! The elastic net objective (which includes ridge and lasso as special cases) is:
//!
//! ```text
//! minimize over (β₀, β):
//!
//!     (1/(2n)) * Σᵢ (yᵢ - β₀ - xᵢᵀβ)²
//!     + λ * [(1 - α) * ||β||₂² / 2 + α * ||β||₁]
//! ```
//!
//! Where:
//! - `α = 0`: Ridge regression (L2 penalty)
//! - `α = 1`: Lasso regression (L1 penalty)
//! - `β₀` (intercept) is **never penalized**
//! - `λ` controls the overall penalty strength
//!
//! # Standardization
//!
//! By default, predictors are standardized before fitting (matching glmnet's
//! `standardize=TRUE` default):
//!
//! - Each column of X is centered to mean 0 (if intercept is used)
//! - Each column is scaled to unit variance
//! - Coefficients are returned on the **original scale**
//!
//! # Compatibility with glmnet
//!
//! These implementations match R's glmnet behavior:
//!
//! - Same objective function form
//! - Same standardization defaults
//! - Intercept is never penalized
//! - Coefficients are returned on original data scale
//!
//! # Modules
//!
//! - [`preprocess`] - Data standardization utilities
//! - [`elastic_net`] - Core elastic net implementation
//! - [`ridge`] - Ridge regression (L2 penalty)
//! - [`lasso`] - Lasso regression (L1 penalty)
//! - [`path`] - Lambda path generation for regularization paths

pub mod elastic_net;
pub mod lasso;
pub mod path;
pub mod preprocess;
pub mod ridge;

// Re-exports for convenience
pub use elastic_net::{elastic_net_fit, elastic_net_path, ElasticNetFit, ElasticNetOptions};
pub use lasso::{lasso_fit, LassoFit, LassoFitOptions};
pub use path::{make_lambda_path, LambdaPathOptions};
pub use preprocess::{standardize_xy, unstandardize_coefficients, StandardizationInfo};
pub use ridge::{ridge_fit, RidgeFit, RidgeFitOptions};