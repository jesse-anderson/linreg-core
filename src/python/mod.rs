// ============================================================================
// Python Bindings Module for linreg-core
// ============================================================================
// This module provides Python bindings using PyO3 with native Python types.
//
// Module structure:
// - error.rs: Custom exception types
// - types.rs: Type conversion utilities for native types
// - results.rs: Result class definitions
// - mod.rs: Main module exports and function definitions

#[cfg(feature = "python")]
use pyo3::prelude::*;

// Public sub-modules
pub mod error;
pub mod results;
pub mod types;

// Re-export commonly used items
#[cfg(feature = "python")]
pub use error::{error_to_pyerr, validation_error, LinregError, DataValidationError};
#[cfg(feature = "python")]
pub use results::*;
#[cfg(feature = "python")]
pub use types::*;

// ============================================================================
// Include the actual Python binding functions
// ============================================================================
// The functions are split into separate submodules for organization:
// - ols.rs: OLS regression
// - regularized.rs: Ridge, Lasso, Elastic Net
// - diagnostics.rs: Diagnostic tests
// - stats.rs: Statistical utilities
// - csv.rs: CSV parsing

// Include the function implementations
include!("ols_impl.rs");
include!("regularized_impl.rs");
include!("loess_impl.rs");
include!("diagnostics_impl.rs");
include!("stats_impl.rs");
include!("csv_impl.rs");
include!("wls_impl.rs");

// ============================================================================
// Python Module Definition
// ============================================================================

#[cfg(feature = "python")]
#[pymodule]
fn linreg_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register custom exceptions
    m.add_class::<LinregError>()?;
    m.add_class::<DataValidationError>()?;

    // Register result classes (for native API - Phase 2+)
    m.add_class::<PyOLSResult>()?;
    m.add_class::<PyRidgeResult>()?;
    m.add_class::<PyLassoResult>()?;
    m.add_class::<PyElasticNetResult>()?;
    m.add_class::<PyLoessResult>()?;
    m.add_class::<PyLambdaPathResult>()?;
    m.add_class::<PyDiagnosticResult>()?;
    m.add_class::<PyDurbinWatsonResult>()?;
    m.add_class::<PyCooksDistanceResult>()?;
    m.add_class::<PyDfbetasResult>()?;
    m.add_class::<PyDffitsResult>()?;
    m.add_class::<PyBreuschGodfreyResult>()?;
    m.add_class::<PyVifDetail>()?;
    m.add_class::<PyVifTestResult>()?;
    m.add_class::<PyRainbowTestResult>()?;
    m.add_class::<PyWhiteTestResult>()?;
    m.add_class::<PyCSVResult>()?;
    m.add_class::<PyWlsResult>()?;

    // OLS Regression
    m.add_function(wrap_pyfunction!(ols_regression, m)?)?;

    // Regularized Regression
    m.add_function(wrap_pyfunction!(ridge_regression, m)?)?;
    m.add_function(wrap_pyfunction!(lasso_regression, m)?)?;
    m.add_function(wrap_pyfunction!(elastic_net_regression, m)?)?;
    m.add_function(wrap_pyfunction!(make_lambda_path, m)?)?;

    // LOESS Regression
    m.add_function(wrap_pyfunction!(loess_fit, m)?)?;
    m.add_function(wrap_pyfunction!(loess_predict, m)?)?;

    // WLS Regression
    m.add_function(wrap_pyfunction!(wls_regression, m)?)?;

    // Statistical Utilities
    m.add_function(wrap_pyfunction!(get_t_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(get_t_critical, m)?)?;
    m.add_function(wrap_pyfunction!(get_normal_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;

    // Diagnostic Tests
    m.add_function(wrap_pyfunction!(rainbow_test, m)?)?;
    m.add_function(wrap_pyfunction!(harvey_collier_test, m)?)?;
    m.add_function(wrap_pyfunction!(breusch_pagan_test, m)?)?;
    m.add_function(wrap_pyfunction!(white_test, m)?)?;
    m.add_function(wrap_pyfunction!(r_white_test, m)?)?;
    m.add_function(wrap_pyfunction!(python_white_test, m)?)?;
    m.add_function(wrap_pyfunction!(jarque_bera_test, m)?)?;
    m.add_function(wrap_pyfunction!(durbin_watson_test, m)?)?;
    m.add_function(wrap_pyfunction!(shapiro_wilk_test, m)?)?;
    m.add_function(wrap_pyfunction!(anderson_darling_test, m)?)?;
    m.add_function(wrap_pyfunction!(cooks_distance_test, m)?)?;
    m.add_function(wrap_pyfunction!(dfbetas_test, m)?)?;
    m.add_function(wrap_pyfunction!(dffits_test, m)?)?;
    m.add_function(wrap_pyfunction!(reset_test, m)?)?;
    m.add_function(wrap_pyfunction!(breusch_godfrey_test, m)?)?;
    m.add_function(wrap_pyfunction!(vif_test, m)?)?;

    // Descriptive Statistics
    m.add_function(wrap_pyfunction!(stats_mean, m)?)?;
    m.add_function(wrap_pyfunction!(stats_variance, m)?)?;
    m.add_function(wrap_pyfunction!(stats_stddev, m)?)?;
    m.add_function(wrap_pyfunction!(stats_median, m)?)?;
    m.add_function(wrap_pyfunction!(stats_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(stats_correlation, m)?)?;

    // CSV Parsing
    m.add_function(wrap_pyfunction!(parse_csv, m)?)?;

    Ok(())
}
