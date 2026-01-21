// ============================================================================
// Validation Tests for linreg-core
// ============================================================================
//
// These tests validate the Rust implementation against reference values from
// R and Python statistical libraries.
//
// Module Organization:
// - common.rs:          Shared utilities, data structures, loaders
// - core_regression.rs: Housing data R/Python validation
// - breusch_pagan.rs:   Breusch-Pagan heteroscedasticity test validation
// - white.rs:           White heteroscedasticity test validation
// - rainbow.rs:         Rainbow linearity test validation
// - cooks_distance.rs:  Cook's Distance validation
// - shapiro_wilk.rs:    Shapiro-Wilk normality test validation
// - anderson_darling.rs: Anderson-Darling normality test validation
// - regularized.rs:     Ridge & Lasso glmnet validation
//
// To regenerate the reference values, run the scripts in verification/:
//   R:      cd verification/scripts/runners && Rscript run_all_diagnostics_r.R
//   Python: cd verification/scripts/runners && python run_all_diagnostics_python.py

pub mod common;

mod core_regression;
mod breusch_pagan;
mod white;
mod rainbow;
mod cooks_distance;
mod shapiro_wilk;
mod anderson_darling;
mod regularized;
mod ols_by_dataset;
