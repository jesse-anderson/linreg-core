use serde::{Deserialize, Serialize};

/// Options for polynomial regression fitting.
///
/// # Fields
///
/// * `degree` - Polynomial degree (≥ 1). degree=1 is simple linear regression.
/// * `center` - Whether to center x before creating polynomial features.
///   Centering (subtracting the mean) reduces multicollinearity between
///   x, x², x³, etc. and improves numerical stability. Recommended for degree ≥ 3.
/// * `standardize` - Whether to standardize polynomial features (z-score).
///   Useful for regularization but not required for plain OLS.
/// * `intercept` - Whether to include an intercept term. Typically `true`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolynomialOptions {
    pub degree: usize,
    pub center: bool,
    pub standardize: bool,
    pub intercept: bool,
}

impl Default for PolynomialOptions {
    fn default() -> Self {
        PolynomialOptions {
            degree: 2,
            center: false,
            standardize: false,
            intercept: true,
        }
    }
}

/// Result of a polynomial regression fit.
///
/// Wraps the standard OLS [`RegressionOutput`] with additional polynomial-specific
/// information required for prediction and interpretation.
///
/// [`RegressionOutput`]: crate::core::RegressionOutput
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolynomialFit {
    /// Underlying OLS regression output (coefficients, R², diagnostics, etc.)
    pub ols_output: crate::core::RegressionOutput,

    /// Polynomial degree used.
    pub degree: usize,

    /// Whether x was centered before fitting.
    pub centered: bool,

    /// Mean of x used for centering (0.0 if centering was not applied).
    pub x_mean: f64,

    /// Standard deviation of the linear x term after centering (1.0 if not standardized).
    /// Kept for backward compatibility; the per-feature version is in `feature_stds`.
    pub x_std: f64,

    /// Whether features were standardized.
    pub standardized: bool,

    /// Number of polynomial features (excluding intercept); equals `degree`.
    pub n_features: usize,

    /// Names of all model terms, starting with "Intercept".
    /// Length = degree + 1.
    pub feature_names: Vec<String>,

    /// Per-feature means used for standardization (one entry per polynomial term,
    /// index 0 = linear x, index 1 = x², …). Empty when `standardized = false`.
    pub feature_means: Vec<f64>,

    /// Per-feature standard deviations used for standardization (one entry per
    /// polynomial term, index 0 = linear x, index 1 = x², …).
    /// Empty when `standardized = false`.
    pub feature_stds: Vec<f64>,
}
