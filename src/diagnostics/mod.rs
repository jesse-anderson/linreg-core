// ============================================================================
// Diagnostics Module
// ============================================================================

//! Statistical diagnostic tests for linear regression assumptions.
//!
//! This module provides a comprehensive suite of diagnostic tests to validate
//! the assumptions of ordinary least squares (OLS) regression. Each test is
//! implemented in its own file for easier maintenance.
//!
//! # Available Tests
//!
//! ## Linearity Tests
//!
//! - **Rainbow Test** (`rainbow.rs`) - Tests whether the relationship between
//!   predictors and response is linear
//! - **Harvey-Collier Test** (`harvey_collier.rs`) - Tests for functional form
//!   misspecification using recursive residuals
//!
//! ## Heteroscedasticity Tests
//!
//! - **Breusch-Pagan Test** (`breusch_pagan.rs`) - Tests for constant variance
//!   of residuals (studentized/Koenker variant)
//! - **White Test** (`white.rs`) - More general test for heteroscedasticity
//!   that does not assume a specific form
//!
//! ## Normality Tests
//!
//! - **Jarque-Bera Test** (`jarque_bera.rs`) - Tests normality using skewness
//!   and kurtosis
//! - **Shapiro-Wilk Test** (`shapiro_wilk.rs`) - Powerful normality test for
//!   small to moderate samples (n ≤ 5000)
//! - **Anderson-Darling Test** (`anderson_darling.rs`) - Tail-sensitive test for
//!   normality
//!
//! ## Autocorrelation Tests
//!
//! - **Durbin-Watson Test** (`durbin_watson.rs`) - Tests for first-order
//!   autocorrelation in residuals
//!
//! ## Influence Measures
//!
//! - **Cook's Distance** (`cooks_distance.rs`) - Identifies influential
//!   observations that may affect regression results

// Submodules
mod anderson_darling;
mod breusch_pagan;
mod cooks_distance;
mod durbin_watson;
mod harvey_collier;
mod helpers;
mod jarque_bera;
mod rainbow;
mod shapiro_wilk;
mod types;
mod white;

// Re-export types
pub use types::{
    CooksDistanceResult, DiagnosticTestResult, RainbowMethod, RainbowSingleResult,
    RainbowTestOutput, WhiteMethod, WhiteSingleResult, WhiteTestOutput,
};

// Re-export test functions
pub use anderson_darling::{anderson_darling_test, anderson_darling_test_raw};
pub use breusch_pagan::breusch_pagan_test;
pub use cooks_distance::cooks_distance_test;
pub use durbin_watson::{durbin_watson_test, DurbinWatsonResult};
pub use harvey_collier::harvey_collier_test;
pub use jarque_bera::jarque_bera_test;
pub use rainbow::rainbow_test;
pub use shapiro_wilk::{shapiro_wilk_test, shapiro_wilk_test_raw};
pub use white::{python_white_method, r_white_method, white_test};

// Re-export helper functions that are used elsewhere
pub use helpers::{f_p_value, two_tailed_p_value, validate_regression_data};
