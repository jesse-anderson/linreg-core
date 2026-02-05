//! LOESS type definitions
//!
//! Contains the main data structures for LOESS fitting.

/// Surface computation method for LOESS
///
/// Controls whether fitted values are computed directly at each point
/// or interpolated from a fitted surface.
///
/// # Variants
///
/// * **Direct** - Computes exact local fits at each data point (slower, more accurate)
/// * **Interpolate** - Uses Hermite interpolation from vertex fits (faster, R's default)
///
/// # Comparison with R
///
/// R's `loess()` function uses `surface = "interpolate"` by default, which computes
/// fits at a subset of "vertex" points and uses Hermite interpolation for other points.
/// This is faster but produces approximate values at most data points.
///
/// The `Direct` method computes exact local fits at every point, which is more accurate
/// but slower. Use `Direct` when:
/// - Validating against other implementations
/// - Need exact fitted values
/// - Dataset is small enough that speed doesn't matter
///
/// # Example
///
/// ```
/// use linreg_core::loess::LoessSurface;
///
/// // Direct surface (exact, slower)
/// let surface = LoessSurface::Direct;
/// assert_eq!(surface.as_str(), "direct");
///
/// // Interpolate surface (approximate, faster - R's default)
/// let surface = LoessSurface::Interpolate;
/// assert_eq!(surface.as_str(), "interpolate");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum LoessSurface {
    /// Exact local fits at each data point (more accurate, slower)
    #[default]
    Direct,
    /// Hermite interpolation from vertex fits (R's default, faster)
    Interpolate,
}

impl LoessSurface {
    /// Returns the string representation matching R's `surface` parameter
    pub fn as_str(&self) -> &'static str {
        match self {
            LoessSurface::Direct => "direct",
            LoessSurface::Interpolate => "interpolate",
        }
    }
}

/// LOESS fit parameters
///
/// Configuration options for LOESS fitting.
///
/// # Example
///
/// ```
/// use linreg_core::loess::{LoessOptions, LoessSurface};
///
/// // Default options: span=0.75, degree=1, no robustness
/// let options = LoessOptions::default();
/// assert_eq!(options.span, 0.75);
/// assert_eq!(options.degree, 1);
/// assert_eq!(options.robust_iterations, 0);
///
/// // Custom options for wiggly curve with robust fitting
/// let options = LoessOptions {
///     span: 0.5,
///     degree: 2,
///     robust_iterations: 2,
///     n_predictors: 1,
///     surface: LoessSurface::Direct,
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoessOptions {
    /// Span (fraction of data used in each local fit)
    ///
    /// Range: 0.0 to 1.0
    /// - Smaller values (e.g., 0.25) produce wiggly curves that follow data closely
    /// - Larger values (e.g., 0.75) produce smoother curves
    pub span: f64,

    /// Degree of local polynomial: 0 (constant), 1 (linear), or 2 (quadratic)
    pub degree: usize,

    /// Number of robustness iterations (0 for non-robust fit)
    ///
    /// Typical values are 0 (no robustness) or 2 (standard robust fitting).
    /// Robust fitting uses a biweight function to downweight outliers.
    pub robust_iterations: usize,

    /// Number of predictor variables
    pub n_predictors: usize,

    /// Surface computation method
    ///
    /// - `Direct`: Exact local fits at each point (default, more accurate)
    /// - `Interpolate`: Hermite interpolation from vertex fits (R's default, faster)
    pub surface: LoessSurface,
}

impl Default for LoessOptions {
    fn default() -> Self {
        Self {
            span: 0.75,
            degree: 1,
            robust_iterations: 0,
            n_predictors: 1,
            surface: LoessSurface::Direct,
        }
    }
}

/// LOESS fit result
///
/// Contains the fitted values and information about the fit.
///
/// # Example
///
/// ```
/// use linreg_core::loess::{loess_fit, LoessOptions};
///
/// let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
///
/// let options = LoessOptions::default();
/// let result = loess_fit(&y, &[x], &options).unwrap();
///
/// // Fitted values should have same length as input
/// assert_eq!(result.fitted.len(), y.len());
/// // Each fitted value should be finite
/// assert!(result.fitted.iter().all(|v| v.is_finite()));
/// // Check fit parameters
/// assert_eq!(result.span, 0.75);  // default span
/// assert_eq!(result.degree, 1);   // default degree
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoessFit {
    /// Fitted values (smoothed y values) at each observation point
    pub fitted: Vec<f64>,

    /// Predicted values at query points (if any)
    pub predictions: Option<Vec<f64>>,

    /// Span parameter used for fitting
    pub span: f64,

    /// Degree of polynomial used for fitting
    pub degree: usize,

    /// Number of robustness iterations performed
    pub robust_iterations: usize,

    /// Surface computation method used
    pub surface: LoessSurface,
}
