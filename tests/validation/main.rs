// ============================================================================
// Validation Tests for linreg-core
// ============================================================================
//
// These tests validate the Rust implementation against reference values from
// R and Python statistical libraries.

#![cfg(not(target_arch = "wasm32"))]
//
// Module Organization:
// - common.rs:          Shared utilities, data structures, loaders
// - core_regression.rs: Housing data R/Python validation
// - anderson_darling.rs: Anderson-Darling normality test validation
// - breusch_pagan.rs:   Breusch-Pagan heteroscedasticity test validation
// - breusch_godfrey.rs: Breusch-Godfrey autocorrelation test validation
// - cooks_distance.rs:  Cook's Distance validation
// - durbin_watson.rs:   Durbin-Watson autocorrelation test validation
// - harvey_collier.rs:  Harvey-Collier linearity test validation
// - jarque_bera.rs:     Jarque-Bera normality test validation
// - rainbow.rs:         Rainbow linearity test validation
// - regularized.rs:     Ridge & Lasso glmnet validation
// - reset.rs:           RESET test for functional form validation
// - shapiro_wilk.rs:    Shapiro-Wilk normality test validation
// - white.rs:           White heteroscedasticity test validation
// - dfbetas.rs:         DFBETAS influence measure validation
// - dffits.rs:          DFFITS influence measure validation
// - vif.rs:             VIF multicollinearity measure validation
// - loess.rs:           LOESS non-parametric regression validation
//
// To regenerate the reference values, run the scripts in verification/:
//   R:      cd verification/scripts/runners && Rscript run_all_diagnostics_r.R
//   Python: cd verification/scripts/runners && python run_all_diagnostics_python.py

pub mod common;

mod anderson_darling;
mod breusch_godfrey;
mod breusch_pagan;
mod cooks_distance;
mod core_regression;
mod cross_validation;  // K-Fold Cross Validation tests
mod dfbetas;
mod dffits;
mod durbin_watson;
mod vif;
mod wls;
mod elastic_net;
mod harvey_collier;
mod jarque_bera;
mod loess;
mod ols_by_dataset;
mod rainbow;
mod regularized;
mod reset;
mod shapiro_wilk;
mod white;


