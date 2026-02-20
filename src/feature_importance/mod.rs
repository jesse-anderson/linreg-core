//! Feature importance metrics for regression models.
//!
//! This module provides various methods for assessing the relative importance
//! of predictor variables in regression models. These metrics help with:
//!
//! - **Model interpretation** - Understanding which variables drive predictions
//! - **Model selection** - Identifying which features actually matter
//! - **Diagnostics** - Detecting problematic predictor relationships
//! - **Communication** - Explaining model behavior to stakeholders
//!
//! # Available Metrics
//!
//! | Metric | Description | Model Support | Complexity |
//! |--------|-------------|---------------|------------|
//! | Standardized Coefficients | Coefficients scaled by SD of X and Y | OLS, Ridge, Lasso, ENet, Polynomial | * |
//! | Linear SHAP | Exact SHAP values for linear models | OLS (exact), Regularized (with caveat) | * |
//! | Permutation Importance | Performance drop when feature is shuffled | All models | *** |
//! | VIF Ranking | Variance Inflation Factor based ranking | OLS, Ridge, Lasso, ENet | * |
//!
//! # Example
//!
//! ```rust
//! use linreg_core::core::ols_regression;
//! use linreg_core::{
//!     standardized_coefficients, vif_ranking, shap_values_linear,
//!     permutation_importance_ols, PermutationImportanceOptions,
//! };
//!
//! let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
//! let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];
//! let names = vec!["Intercept".into(), "X1".into(), "X2".into()];
//!
//! let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names)?;
//!
//! // Compute standardized coefficients
//! let std_coefs = standardized_coefficients(&fit.coefficients, &[x1.clone(), x2.clone()])?;
//!
//! // Compute SHAP values (local explanations)
//! let shap = shap_values_linear(&[x1.clone(), x2.clone()], &fit.coefficients)?;
//!
//! // Compute permutation importance
//! let perm_imp = permutation_importance_ols(
//!     &y,
//!     &[x1, x2],
//!     &fit,
//!     &PermutationImportanceOptions::default()
//! )?;
//!
//! // Compute VIF ranking
//! let vif_rank_result = vif_ranking(&fit.vif);
//!
//! println!("Standardized coefficients: {:?}", std_coefs);
//! println!("VIF ranking: {:?}", vif_rank_result);
//! println!("Permutation importance: {:?}", perm_imp.importance);
//! # Ok::<(), linreg_core::Error>(())
//! ```

pub mod permutation;
pub mod shap;
pub mod standardized;
pub mod types;
pub mod vif;

// Re-exports for convenience
pub use permutation::{
    permutation_importance_elastic_net, permutation_importance_lasso,
    permutation_importance_loess, permutation_importance_ols, permutation_importance_ols_named,
    permutation_importance_ridge,
};
pub use shap::{
    shap_values_elastic_net, shap_values_lasso, shap_values_linear,
    shap_values_linear_named, shap_values_polynomial, shap_values_ridge,
};
pub use standardized::{standardized_coefficients, standardized_coefficients_named};
pub use vif::vif_ranking;

// Re-export types
pub use types::{
    PermutationImportanceOptions, PermutationImportanceOutput, ShapOutput,
    StandardizedCoefficientsOutput, VifRankingOutput,
};
