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
//! - **RESET Test** (`reset.rs`) - Ramsey's Regression Specification Error Test
//!   for detecting omitted variables or incorrect functional form
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
//! - **Breusch-Godfrey Test** (`breusch_godfrey.rs`) - Tests for higher-order
//!   serial correlation (LM test)
//!
//! ## Influence Measures
//!
//! - **Cook's Distance** (`cooks_distance.rs`) - Identifies influential
//!   observations that may affect regression results
//! - **DFBETAS** (`dfbetas.rs`) - Measures influence of each observation on each coefficient
//! - **DFFITS** (`dffits.rs`) - Measures influence of each observation on its fitted value
//!
//! ## Multicollinearity Tests
//!
//! - **VIF** (`vif.rs`) - Variance Inflation Factor for detecting multicollinearity

// Submodules
mod anderson_darling;
mod breusch_godfrey;
mod breusch_pagan;
mod cooks_distance;
mod dfbetas;
mod dffits;
mod durbin_watson;
mod harvey_collier;
mod helpers;
mod jarque_bera;
mod rainbow;
mod reset;
mod shapiro_wilk;
mod types;
mod white;
mod vif;

// Re-export types
pub use types::{
    CooksDistanceResult, DfbetasResult, DffitsResult, DiagnosticTestResult, RainbowMethod,
    RainbowSingleResult, RainbowTestOutput, VifDetail, VifDiagnosticResult, WhiteMethod,
    WhiteSingleResult, WhiteTestOutput, HarveyCollierMethod,
};

// Re-export test functions
pub use anderson_darling::{anderson_darling_test, anderson_darling_test_raw};
pub use breusch_godfrey::{breusch_godfrey_test, BGTestType, BreuschGodfreyResult};
pub use breusch_pagan::breusch_pagan_test;
pub use cooks_distance::cooks_distance_test;
pub use dfbetas::dfbetas_test;
pub use dffits::dffits_test;
pub use durbin_watson::{durbin_watson_test, DurbinWatsonResult};
pub use harvey_collier::harvey_collier_test;
pub use jarque_bera::jarque_bera_test;
pub use rainbow::rainbow_test;
pub use reset::{reset_test, ResetType};
pub use shapiro_wilk::{shapiro_wilk_test, shapiro_wilk_test_raw};
pub use white::{python_white_method, r_white_method, white_test};
pub use vif::vif_test;

// Re-export helper functions that are used elsewhere
pub use helpers::{f_p_value, two_tailed_p_value, validate_regression_data};
