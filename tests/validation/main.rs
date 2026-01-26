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
// - breusch_pagan.rs:   Breusch-Pagan heteroscedasticity test validation
// - white.rs:           White heteroscedasticity test validation
// - rainbow.rs:         Rainbow linearity test validation
// - reset.rs:           RESET test for functional form validation
// - cooks_distance.rs:  Cook's Distance validation
// - shapiro_wilk.rs:    Shapiro-Wilk normality test validation
// - anderson_darling.rs: Anderson-Darling normality test validation
// - regularized.rs:     Ridge & Lasso glmnet validation
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
mod elastic_net;
mod ols_by_dataset;
mod rainbow;
mod regularized;
mod reset;
mod shapiro_wilk;
mod white;


