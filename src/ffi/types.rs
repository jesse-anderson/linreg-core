//! FFI result types for the handle store.
//!
//! Defines the `FitResult` enum that wraps every kind of result the DLL
//! can produce, plus lightweight helper structs for diagnostics and
//! prediction intervals.

use crate::core::RegressionOutput;
use crate::cross_validation::CVResult;
use crate::prediction_intervals::PredictionIntervalOutput;
use crate::regularized::elastic_net::ElasticNetFit;
use crate::regularized::lasso::LassoFit;
use crate::regularized::ridge::RidgeFit;
use crate::weighted_regression::WlsFit;

/// Lightweight result for a diagnostic test that produces a statistic + p-value.
///
/// Durbin-Watson has no p-value; in that case `p_value` is `f64::NAN` and
/// `autocorrelation` holds `1 - DW/2`.
pub struct DiagnosticResult {
    /// Primary test statistic.
    pub statistic: f64,
    /// Two-tailed p-value, or `f64::NAN` if the test does not produce one.
    pub p_value: f64,
    /// Degrees of freedom (used by Breusch-Godfrey; 0.0 if not applicable).
    pub df: f64,
    /// Estimated autocorrelation ρ ≈ 1 − DW/2 (DW only; 0.0 otherwise).
    pub autocorrelation: f64,
}

/// All storable result variants.
pub enum FitResult {
    /// OLS regression result.
    Ols(RegressionOutput),
    /// Ridge regression result.
    Ridge(RidgeFit),
    /// Lasso regression result.
    Lasso(LassoFit),
    /// Elastic Net regression result.
    ElasticNet(ElasticNetFit),
    /// WLS regression result.
    Wls(WlsFit),
    /// Diagnostic test result (statistic + optional p-value).
    Diagnostic(DiagnosticResult),
    /// Prediction interval result.
    PredictionInterval(PredictionIntervalOutput),
    /// A flat vector of f64 values.
    ///
    /// Used for: Cook's distances, DFFITS values, VIF values, lambda path.
    Vector(Vec<f64>),
    /// A flat row-major matrix of f64 values with explicit dimensions.
    ///
    /// Used for: DFBETAS (rows = observations, cols = parameters).
    Matrix {
        data: Vec<f64>,
        rows: usize,
        cols: usize,
    },
    /// K-Fold cross-validation result.
    CV(CVResult),
}
