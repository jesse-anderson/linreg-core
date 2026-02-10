//! WASM-specific bindings for linreg-core
//!
//! This module contains all WebAssembly bindings using wasm-bindgen.
//! These functions are only compiled when the "wasm" feature is enabled.
//!
//! All WASM functions accept and return JSON strings for JavaScript interoperability.
//!
//! # Module Structure
//!
//! - [`csv`] - CSV parsing for WASM
//! - [`regression`] - OLS and weighted least squares regression
//! - [`diagnostics`] - All diagnostic test wrappers
//! - [`regularized`] - Ridge, Lasso, Elastic Net regression
//! - [`loess`] - LOESS regression fitting and prediction
//! - [`stats`] - Statistical utility functions
//! - [`domain`] - Domain checking for security
//! - [`tests`] - Test and validation functions

#![cfg(feature = "wasm")]

pub mod csv;
pub mod diagnostics;
pub mod domain;
pub mod loess;
pub mod regression;
pub mod regularized;
pub mod stats;
pub mod tests;

// Re-export all #[wasm_bindgen] functions for external use
pub use csv::parse_csv;
pub use domain::check_domain;
pub use loess::{loess_fit, loess_predict};
pub use regression::{ols_regression, wls_regression};
pub use regularized::{elastic_net_regression, lasso_regression, make_lambda_path, ridge_regression};
pub use stats::{
    get_normal_inverse, get_t_cdf, get_t_critical, stats_correlation, stats_mean, stats_median,
    stats_quantile, stats_stddev, stats_variance,
};

// Re-export all diagnostic test functions
pub use diagnostics::{
    anderson_darling_test, breusch_godfrey_test, breusch_pagan_test, cooks_distance_test,
    dfbetas_test, dffits_test, harvey_collier_test, jarque_bera_test, durbin_watson_test,
    python_white_test, r_white_test, rainbow_test, reset_test, shapiro_wilk_test, vif_test,
    white_test,
};

// Re-export test functions
pub use tests::{get_version, test, test_ci, test_housing_regression, test_r_accuracy, test_t_critical};
