//! # linreg-core
//!
//! A lightweight, self-contained linear regression library in pure Rust.
//!
//! **No external math dependencies.** All linear algebra (matrices, QR decomposition)
//! and statistical functions (distributions, hypothesis tests) are implemented from
//! scratch. Compiles to WebAssembly for browser use or runs as a native Rust crate.
//!
//! ## What This Does
//!
//! - **OLS Regression** — Ordinary Least Squares with numerically stable QR decomposition
//! - **Regularized Regression** — Ridge, Lasso, and Elastic Net via coordinate descent
//! - **Diagnostic Tests** — 8+ statistical tests for validating regression assumptions
//! - **WASM Support** — Same API works in browsers via WebAssembly
//!
//! ## Quick Start
//!
//! ### Native Rust
//!
//! Add to `Cargo.toml` (no WASM overhead):
//!
//! ```toml
//! [dependencies]
//! linreg-core = { version = "0.5", default-features = false }
//! ```
//!
//! ```rust
//! use linreg_core::core::ols_regression;
//!
//! let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
//! let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];
//! let names = vec!["Intercept".into(), "Temp".into(), "Pressure".into()];
//!
//! let result = ols_regression(&y, &[x1, x2], &names)?;
//! println!("R²: {}", result.r_squared);
//! println!("F-statistic: {}", result.f_statistic);
//! # Ok::<(), linreg_core::Error>(())
//! ```
//!
//! ### WebAssembly (JavaScript)
//!
//! ```toml
//! [dependencies]
//! linreg-core = "0.5"
//! ```
//!
//! Build with `wasm-pack build --target web`, then use in JavaScript:
//!
//! ```text
//! import init, { ols_regression } from './linreg_core.js';
//! await init();
//!
//! const result = JSON.parse(ols_regression(
//!     JSON.stringify([2.5, 3.7, 4.2, 5.1, 6.3]),
//!     JSON.stringify([[1,2,3,4,5], [2,4,5,4,3]]),
//!     JSON.stringify(["Intercept", "X1", "X2"])
//! ));
//! console.log("R²:", result.r_squared);
//! ```
//!
//! ## Regularized Regression
//!
//! ```no_run
//! use linreg_core::regularized::{ridge, lasso};
//! use linreg_core::linalg::Matrix;
//!
//! let x = Matrix::new(100, 3, vec![0.0; 300]);
//! let y = vec![0.0; 100];
//!
//! // Ridge regression (L2 penalty - shrinks coefficients, handles multicollinearity)
//! let ridge_result = ridge::ridge_fit(&x, &y, &ridge::RidgeFitOptions {
//!     lambda: 1.0,
//!     intercept: true,
//!     standardize: true,
//!     ..Default::default()
//! })?;
//!
//! // Lasso regression (L1 penalty - does variable selection by zeroing coefficients)
//! let lasso_result = lasso::lasso_fit(&x, &y, &lasso::LassoFitOptions {
//!     lambda: 0.1,
//!     intercept: true,
//!     standardize: true,
//!     ..Default::default()
//! })?;
//! # Ok::<(), linreg_core::Error>(())
//! ```
//!
//! ## Diagnostic Tests
//!
//! After fitting a model, validate its assumptions:
//!
//! | Test | Tests For | Use When |
//! |------|-----------|----------|
//! | [`diagnostics::rainbow_test`] | Linearity | Checking if relationships are linear |
//! | [`diagnostics::harvey_collier_test`] | Functional form | Suspecting model misspecification |
//! | [`diagnostics::breusch_pagan_test`] | Heteroscedasticity | Variance changes with predictors |
//! | [`diagnostics::white_test`] | Heteroscedasticity | More general than Breusch-Pagan |
//! | [`diagnostics::shapiro_wilk_test`] | Normality | Small to moderate samples (n ≤ 5000) |
//! | [`diagnostics::jarque_bera_test`] | Normality | Large samples, skewness/kurtosis |
//! | [`diagnostics::anderson_darling_test`] | Normality | Tail-sensitive, any sample size |
//! | [`diagnostics::durbin_watson_test`] | Autocorrelation | Time series or ordered data |
//! | [`diagnostics::cooks_distance_test`] | Influential points | Identifying high-impact observations |
//!
//! ```rust
//! use linreg_core::diagnostics::{rainbow_test, breusch_pagan_test, RainbowMethod};
//!
//! # let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
//! # let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];
//! // Rainbow test for linearity
//! let rainbow = rainbow_test(&y, &[x1.clone(), x2.clone()], 0.5, RainbowMethod::R)?;
//! if rainbow.r_result.as_ref().map_or(false, |r| r.p_value < 0.05) {
//!     println!("Warning: relationship may be non-linear");
//! }
//!
//! // Breusch-Pagan test for heteroscedasticity
//! let bp = breusch_pagan_test(&y, &[x1, x2])?;
//! if bp.p_value < 0.05 {
//!     println!("Warning: residuals have non-constant variance");
//! }
//! # Ok::<(), linreg_core::Error>(())
//! ```
//!
//! ## Feature Flags
//!
//! | Flag | Default | Description |
//! |------|---------|-------------|
//! | `wasm` | Yes | Enables WASM bindings and browser support |
//! | `validation` | No | Includes test data for validation tests |
//!
//! For native-only builds (smaller binary, no WASM deps):
//!
//! ```toml
//! linreg-core = { version = "0.5", default-features = false }
//! ```
//!
//! ## Why This Library?
//!
//! - **Zero dependencies** — No `nalgebra`, no `statrs`, no `ndarray`. Pure Rust.
//! - **Validated** — Outputs match R's `lm()` and Python's `statsmodels`
//! - **WASM-ready** — Same code runs natively and in browsers
//! - **Small** — Core is ~2000 lines, compiles quickly
//! - **Permissive license** — MIT OR Apache-2.0
//!
//! ## Module Structure
//!
//! - [`core`] — OLS regression, coefficients, residuals, VIF
//! - [`regularized`] — Ridge, Lasso, Elastic Net, regularization paths
//! - [`diagnostics`] — All diagnostic tests (linearity, heteroscedasticity, normality, autocorrelation)
//! - [`distributions`] — Statistical distributions (t, F, χ², normal, beta, gamma)
//! - [`linalg`] — Matrix operations, QR decomposition, linear system solver
//! - [`error`] — Error types and Result alias
//!
//! ## Links
//!
//! - [Repository](https://github.com/jesse-anderson/linreg-core)
//! - [Documentation](https://docs.rs/linreg-core)
//! - [Examples](https://github.com/jesse-anderson/linreg-core/tree/main/examples)
//!
//! ## Disclaimer
//!
//! This library is under active development and has not reached 1.0 stability.
//! While outputs are validated against R and Python implementations, **do not
//! use this library for critical applications** (medical, financial, safety-critical
//! systems) without independent verification. See the LICENSE for full terms.
//! The software is provided "as is" without warranty of any kind.

// Import core modules (always available)
pub mod core;
pub mod diagnostics;
pub mod distributions;
pub mod error;
pub mod linalg;
pub mod loess;
pub mod regularized;
pub mod stats;

// Python bindings (only compiled when "python" feature is enabled)
// Module structure: src/python/ with mod.rs, error.rs, types.rs, results.rs
#[cfg(feature = "python")]
pub mod python;

// WASM bindings (only compiled when "wasm" feature is enabled)
// Module structure: src/wasm.rs - contains all wasm-bindgen exports
#[cfg(feature = "wasm")]
pub mod wasm;

// Unit tests are now in tests/unit/ directory
// - error_tests.rs -> tests/unit/error_tests.rs
// - core_tests.rs -> tests/unit/core_tests.rs
// - linalg_tests.rs -> tests/unit/linalg_tests.rs
// - validation_tests.rs -> tests/validation/main.rs
// - diagnostics_tests.rs: disabled (references unimplemented functions)

// Re-export public API (always available)
pub use core::{aic, aic_python, bic, bic_python, log_likelihood, RegressionOutput, VifResult};
pub use diagnostics::{
    BGTestType, BreuschGodfreyResult, CooksDistanceResult, DiagnosticTestResult,
    RainbowMethod, RainbowSingleResult, RainbowTestOutput, ResetType,
    WhiteMethod, WhiteSingleResult, WhiteTestOutput,
};
pub use loess::{loess_fit, LoessFit, LoessOptions};

// Re-export core test functions with different names to avoid WASM conflicts
pub use diagnostics::rainbow_test as rainbow_test_core;
pub use diagnostics::white_test as white_test_core;

pub use error::{error_json, error_to_json, Error, Result};