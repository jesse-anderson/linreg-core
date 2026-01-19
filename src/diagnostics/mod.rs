// ============================================================================
// Diagnostics Module
// ============================================================================
//
// Statistical diagnostic tests for linear regression assumptions.
// Each test is in its own file for easier maintenance.
//
// ## Available Tests
//
// - **Rainbow Test** (`rainbow.rs`): Tests for linearity
// - **Harvey-Collier Test** (`harvey_collier.rs`): Tests for functional form
// - **Breusch-Pagan Test** (`breusch_pagan.rs`): Tests for heteroscedasticity
// - **White Test** (`white.rs`): Tests for heteroscedasticity (more general)
// - **Jarque-Bera Test** (`jarque_bera.rs`): Tests for normality of residuals
// - **Durbin-Watson Test** (`durbin_watson.rs`): Tests for autocorrelation
// - **Shapiro-Wilk Test** (`shapiro_wilk.rs`): Tests for normality
// - **Anderson-Darling Test** (`anderson_darling.rs`): Tests for normality
// - **Cook's Distance** (`cooks_distance.rs`): Identifies influential observations

// Submodules
mod types;
mod helpers;
mod rainbow;
mod harvey_collier;
mod breusch_pagan;
mod white;
mod jarque_bera;
mod durbin_watson;
mod shapiro_wilk;
mod anderson_darling;
mod cooks_distance;

// Re-export types
pub use types::{
    DiagnosticTestResult,
    RainbowSingleResult,
    RainbowTestOutput,
    RainbowMethod,
    WhiteSingleResult,
    WhiteTestOutput,
    WhiteMethod,
    CooksDistanceResult,
};

// Re-export test functions
pub use rainbow::rainbow_test;
pub use harvey_collier::harvey_collier_test;
pub use breusch_pagan::breusch_pagan_test;
pub use white::{white_test, r_white_method, python_white_method};
pub use jarque_bera::jarque_bera_test;
pub use durbin_watson::{durbin_watson_test, DurbinWatsonResult};
pub use shapiro_wilk::{shapiro_wilk_test, shapiro_wilk_test_raw};
pub use anderson_darling::{anderson_darling_test, anderson_darling_test_raw};
pub use cooks_distance::cooks_distance_test;

// Re-export helper functions that are used elsewhere
pub use helpers::{two_tailed_p_value, f_p_value};
