//! Elastic Net regression (L1 + L2 regularized linear regression).
//!
//! This module provides a generalized elastic net implementation using cyclical
//! coordinate descent with soft-thresholding and active set convergence strategies.
//! It serves as the core engine for both Lasso (`alpha=1.0`) and Ridge (`alpha=0.0`).
//!
//! # Objective Function
//!
//! Minimizes over `(β₀, β)`:
//!
//! ```text
//! (1/(2n)) * ||y - β₀ - Xβ||² + λ * [ (1-α)||β||₂²/2 + α||β||₁ ]
//! ```
//!
//! Note on scaling: The internal implementation works with standardized data (unit norm columns).
//! The lambda parameter is adjusted internally to match the scale expected by the formulation above.

use crate::core::{aic, bic, log_likelihood};
use crate::error::{Error, Result};
use crate::linalg::Matrix;
use crate::regularized::preprocess::{
    predict, standardize_xy, unstandardize_coefficients, StandardizeOptions,
};

#[cfg(feature = "wasm")]
use serde::Serialize;

/// Soft-thresholding operator: S(z, γ) = sign(z) * max(|z| - γ, 0)
///
/// This is the key operation in Lasso and Elastic Net regression that applies
/// the L1 penalty, producing sparse solutions by shrinking small values to zero.
///
/// # Arguments
///
/// * `z` - Input value to be thresholded
/// * `gamma` - Threshold value (must be non-negative)
///
/// # Returns
///
/// - `z - gamma` if `z > gamma`
/// - `z + gamma` if `z < -gamma`
/// - `0` otherwise (when `|z| <= gamma`)
///
/// # Panics
///
/// Panics if `gamma` is negative.
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::elastic_net::soft_threshold;
/// // Values above threshold are reduced
/// assert_eq!(soft_threshold(5.0, 2.0), 3.0);
///
/// // Values below threshold are set to zero
/// assert_eq!(soft_threshold(1.0, 2.0), 0.0);
///
/// // Negative values work symmetrically
/// assert_eq!(soft_threshold(-5.0, 2.0), -3.0);
/// assert_eq!(soft_threshold(-1.0, 2.0), 0.0);
/// ```
#[inline]
pub fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if gamma < 0.0 {
        panic!("Soft threshold gamma must be non-negative");
    }
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

/// Options for elastic net fitting.
///
/// Configuration options for elastic net regression, which combines L1 and L2 penalties.
///
/// # Fields
///
/// - `lambda` - Regularization strength (≥ 0, higher = more regularization)
/// - `alpha` - Mixing parameter (0 = Ridge, 1 = Lasso, 0.5 = equal mix)
/// - `intercept` - Whether to include an intercept term
/// - `standardize` - Whether to standardize predictors to unit variance
/// - `max_iter` - Maximum coordinate descent iterations
/// - `tol` - Convergence tolerance on coefficient changes
/// - `penalty_factor` - Optional per-feature penalty multipliers
/// - `warm_start` - Optional initial coefficient values for warm starts
/// - `weights` - Optional observation weights
/// - `coefficient_bounds` - Optional (lower, upper) bounds for each coefficient
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::elastic_net::ElasticNetOptions;
/// let options = ElasticNetOptions {
///     lambda: 0.1,
///     alpha: 0.5,  // Equal mix of L1 and L2
///     intercept: true,
///     standardize: true,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug)]
pub struct ElasticNetOptions {
    /// Regularization strength (lambda >= 0)
    pub lambda: f64,
    /// Elastic net mixing parameter (0 <= alpha <= 1).
    /// alpha=1 is Lasso, alpha=0 is Ridge.
    pub alpha: f64,
    /// Whether to include an intercept term
    pub intercept: bool,
    /// Whether to standardize predictors
    pub standardize: bool,
    /// Maximum coordinate descent iterations
    pub max_iter: usize,
    /// Convergence tolerance on coefficient changes
    pub tol: f64,
    /// Per-feature penalty factors (optional).
    /// If None, all features have penalty factor 1.0.
    pub penalty_factor: Option<Vec<f64>>,
    /// Initial coefficients for warm start (optional).
    /// If provided, optimization starts from these values instead of zero.
    /// Used for efficient pathwise coordinate descent.
    pub warm_start: Option<Vec<f64>>,
    /// Observation weights (optional).
    /// If provided, must have length equal to the number of observations.
    /// Weights are normalized to sum to 1 internally.
    pub weights: Option<Vec<f64>>,
    /// Coefficient bounds: (lower, upper) for each predictor.
    /// If None, uses (-inf, +inf) for all coefficients (no bounds).
    ///
    /// The bounds vector length must equal the number of predictors (excluding intercept).
    /// For each predictor, the coefficient will be clamped to [lower, upper] after
    /// each coordinate descent update.
    ///
    /// # Examples
    /// * Non-negative least squares: `Some(vec![(0.0, f64::INFINITY); p])`
    /// * Upper bound only: `Some(vec![(-f64::INFINITY, 10.0); p])`
    /// * Both bounds: `Some(vec![(-5.0, 5.0); p])`
    ///
    /// # Notes
    /// * Bounds are applied to coefficients on the ORIGINAL scale, not standardized scale
    /// * The intercept is never bounded
    /// * Each pair must satisfy `lower <= upper`
    pub coefficient_bounds: Option<Vec<(f64, f64)>>,
}

impl Default for ElasticNetOptions {
    fn default() -> Self {
        ElasticNetOptions {
            lambda: 1.0,
            alpha: 1.0, // Lasso default
            intercept: true,
            standardize: true,
            max_iter: 100000,
            tol: 1e-7,
            penalty_factor: None,
            warm_start: None,
            weights: None,
            coefficient_bounds: None,
        }
    }
}

/// Result of an elastic net fit.
///
/// Contains the fitted model coefficients, convergence information, and diagnostic metrics.
///
/// # Fields
///
/// - `lambda` - The regularization strength used
/// - `alpha` - The elastic net mixing parameter (0 = Ridge, 1 = Lasso)
/// - `intercept` - Intercept coefficient (never penalized)
/// - `coefficients` - Slope coefficients (may be sparse for high alpha)
/// - `fitted_values` - Predicted values on training data
/// - `residuals` - Residuals (y - fitted_values)
/// - `n_nonzero` - Number of non-zero coefficients (excluding intercept)
/// - `iterations` - Number of coordinate descent iterations performed
/// - `converged` - Whether the algorithm converged
/// - `r_squared` - Coefficient of determination
/// - `adj_r_squared` - Adjusted R²
/// - `mse` - Mean squared error
/// - `rmse` - Root mean squared error
/// - `mae` - Mean absolute error
/// - `log_likelihood` - Log-likelihood of the model (for model comparison)
/// - `aic` - Akaike Information Criterion (lower = better)
/// - `bic` - Bayesian Information Criterion (lower = better)
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::elastic_net::{elastic_net_fit, ElasticNetOptions};
/// # use linreg_core::linalg::Matrix;
/// # let y = vec![2.0, 4.0, 6.0, 8.0];
/// # let x = Matrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
/// # let options = ElasticNetOptions { lambda: 0.1, alpha: 0.5, intercept: true, standardize: true, ..Default::default() };
/// let fit = elastic_net_fit(&x, &y, &options).unwrap();
///
/// // Access fit results
/// println!("Lambda: {}, Alpha: {}", fit.lambda, fit.alpha);
/// println!("Non-zero coefficients: {}", fit.n_nonzero);
/// println!("Converged: {}", fit.converged);
/// println!("R²: {}", fit.r_squared);
/// println!("AIC: {}", fit.aic);
/// # Ok::<(), linreg_core::Error>(())
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(Serialize))]
pub struct ElasticNetFit {
    pub lambda: f64,
    pub alpha: f64,
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
    pub n_nonzero: usize,
    pub iterations: usize,
    pub converged: bool,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
}

use crate::regularized::path::{make_lambda_path, LambdaPathOptions};

/// Fits an elastic net regularization path.
///
/// This is the most efficient way to fit models for multiple lambda values.
/// It performs data standardization once and uses warm starts to speed up
/// convergence along the path.
///
/// # Arguments
///
/// * `x` - Design matrix
/// * `y` - Response vector
/// * `path_options` - Options for generating the lambda path
/// * `fit_options` - Options for the elastic net fit (alpha, tol, etc.)
///
/// # Returns
///
/// A vector of `ElasticNetFit` structs, one for each lambda in the path.
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::elastic_net::{elastic_net_path, ElasticNetOptions};
/// # use linreg_core::regularized::path::LambdaPathOptions;
/// # use linreg_core::linalg::Matrix;
/// let y = vec![2.0, 4.0, 6.0, 8.0];
/// let x = Matrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
///
/// let path_options = LambdaPathOptions {
///     nlambda: 10,
///     ..Default::default()
/// };
/// let fit_options = ElasticNetOptions {
///     alpha: 0.5,
///     ..Default::default()
/// };
///
/// let path = elastic_net_path(&x, &y, &path_options, &fit_options).unwrap();
/// assert_eq!(path.len(), 10); // One fit per lambda
///
/// // First model has strongest regularization (fewest non-zero coefficients)
/// println!("Non-zero at lambda_max: {}", path[0].n_nonzero);
/// // Last model has weakest regularization (most non-zero coefficients)
/// println!("Non-zero at lambda_min: {}", path.last().unwrap().n_nonzero);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn elastic_net_path(
    x: &Matrix,
    y: &[f64],
    path_options: &LambdaPathOptions,
    fit_options: &ElasticNetOptions,
) -> Result<Vec<ElasticNetFit>> {
    let n = x.rows;
    let p = x.cols;

    if y.len() != n {
        return Err(Error::DimensionMismatch(format!(
            "Length of y ({}) must match number of rows in X ({})",
            y.len(), n
        )));
    }

    // 1. Standardize X and y ONCE
    let standardization_options = StandardizeOptions {
        intercept: fit_options.intercept,
        standardize_x: fit_options.standardize,
        standardize_y: fit_options.intercept,
        weights: fit_options.weights.clone(),
    };

    let (x_standardized, y_standardized, standardization_info) = standardize_xy(x, y, &standardization_options);

    // 2. Generate lambda path
    // If lambdas are not provided in options (which they aren't in LambdaPathOptions, 
    // it just controls generation), we generate them.
    // NOTE: If the user wants specific lambdas, they should probably use a different API
    // or we could add `lambdas: Option<&[f64]>` to this function.
    // For now, we strictly generate them.
    
    // We need to account for penalty factors in lambda generation if provided
    let intercept_col = if fit_options.intercept { Some(0) } else { None };
    let lambdas = make_lambda_path(
        &x_standardized,
        &y_standardized, // y_standardized is centered if intercept=true
        path_options, 
        fit_options.penalty_factor.as_deref(), 
        intercept_col
    );

    // 3. Loop over lambdas with warm starts
    let mut fits = Vec::with_capacity(lambdas.len());
    let mut coefficients_standardized = vec![0.0; p]; // Initialize at 0

    // Determine unpenalized columns
    let first_penalized_column_index = if fit_options.intercept { 1 } else { 0 };

    // Calculate scale factor for converting Internal lambdas to Public (user-facing) lambdas
    // make_lambda_path returns Internal lambdas (for standardized data)
    // We use these directly in the solver, but scale them for user reporting
    let y_scale_factor = standardization_info.y_scale.unwrap_or(1.0);
    // Public lambda = Internal lambda * y_scale_factor
    // This converts from standardized scale to original data scale
    let lambda_conversion_factor = if y_scale_factor > 1e-12 {
        y_scale_factor
    } else {
        1.0
    };

    for &lambda_standardized_value in &lambdas {
        // The path generation returns lambdas on the internal scale (for standardized data),
        // which are used directly in coordinate descent without additional scaling.
        let lambda_standardized = lambda_standardized_value;

        // Transform coefficient bounds to standardized scale
        // Bounds on original scale need to be converted: coefficients_standardized = beta_orig * x_scale / y_scale
        let bounds_standardized: Option<Vec<(f64, f64)>> = fit_options.coefficient_bounds.as_ref().map(|bounds| {
            let y_scale = standardization_info.y_scale.unwrap_or(1.0);
            bounds.iter().enumerate().map(|(j, &(lower, upper))| {
                // For each predictor j in original scale, the corresponding column
                // in the standardized matrix is at index j+1 (col 0 is intercept)
                let std_idx = j + 1;
                let x_scale_predictor_j = if std_idx < standardization_info.x_scale.len() {
                    standardization_info.x_scale[std_idx]
                } else {
                    1.0
                };
                let scale_factor = x_scale_predictor_j / y_scale;
                (lower * scale_factor, upper * scale_factor)
            }).collect()
        });

        let (iterations, converged) = coordinate_descent(
            &x_standardized,
            &y_standardized,
            &mut coefficients_standardized,
            lambda_standardized,
            fit_options.alpha,
            first_penalized_column_index,
            fit_options.max_iter,
            fit_options.tol,
            fit_options.penalty_factor.as_deref(),
            bounds_standardized.as_deref(),
            &standardization_info.column_squared_norms,
        )?;

        // Unstandardize coefficients for output
        let (intercept, beta_orig) = unstandardize_coefficients(&coefficients_standardized, &standardization_info);

        // Count non-zeros
        let n_nonzero = beta_orig.iter().filter(|&&b| b.abs() > 0.0).count();

        // Fitted values & residuals
        let fitted = predict(x, intercept, &beta_orig);
        let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(yi, yh)| yi - yh).collect();

        // Statistics
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
        let mae: f64 = residuals.iter().map(|r| r.abs()).sum::<f64>() / n as f64;

        let r_squared = if ss_tot > 1e-10 { 1.0 - ss_res / ss_tot } else { 1.0 };
        let eff_df = 1.0 + n_nonzero as f64;
        let adj_r_squared = if ss_tot > 1e-10 && n > eff_df as usize {
            1.0 - (1.0 - r_squared) * ((n - 1) as f64 / (n as f64 - eff_df))
        } else {
            r_squared
        };
        let mse = ss_res / (n as f64 - eff_df).max(1.0);

        // Model selection criteria
        let ll = log_likelihood(n, mse, ss_res);
        let n_coef = beta_orig.len() + 1; // coefficients + intercept
        let aic_val = aic(ll, n_coef);
        let bic_val = bic(ll, n_coef, n);

        // Convert Internal lambda to Public (user-facing) lambda for reporting
        // Public = Internal * y_scale_var * n (to match R's glmnet reporting)
        let lambda_original_scale = lambda_standardized_value * lambda_conversion_factor;

        fits.push(ElasticNetFit {
            lambda: lambda_original_scale,
            alpha: fit_options.alpha,
            intercept,
            coefficients: beta_orig,
            fitted_values: fitted,
            residuals,
            n_nonzero,
            iterations,
            converged,
            r_squared,
            adj_r_squared,
            mse,
            rmse: mse.sqrt(),
            mae,
            log_likelihood: ll,
            aic: aic_val,
            bic: bic_val,
        });
    }

    Ok(fits)
}

/// Fits elastic net regression for a single (lambda, alpha) pair.
///
/// Elastic net combines L1 (Lasso) and L2 (Ridge) penalties:
/// - `alpha = 1.0` is pure Lasso (L1 only)
/// - `alpha = 0.0` is pure Ridge (L2 only)
/// - `alpha = 0.5` is an equal mix
///
/// # Arguments
///
/// * `x` - Design matrix (n rows × p columns including intercept)
/// * `y` - Response variable (n observations)
/// * `options` - Configuration options for elastic net regression
///
/// # Returns
///
/// An `ElasticNetFit` containing coefficients, convergence info, and metrics.
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::elastic_net::{elastic_net_fit, ElasticNetOptions};
/// # use linreg_core::linalg::Matrix;
/// let y = vec![2.0, 4.0, 6.0, 8.0];
/// let x = Matrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
///
/// // Elastic net with 50% L1, 50% L2
/// let options = ElasticNetOptions {
///     lambda: 0.1,
///     alpha: 0.5,
///     intercept: true,
///     standardize: true,
///     ..Default::default()
/// };
///
/// let fit = elastic_net_fit(&x, &y, &options).unwrap();
/// assert!(fit.converged);
/// println!("R²: {}", fit.r_squared);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn elastic_net_fit(x: &Matrix, y: &[f64], options: &ElasticNetOptions) -> Result<ElasticNetFit> {
    if options.lambda < 0.0 {
        return Err(Error::InvalidInput("Lambda must be non-negative".into()));
    }
    if options.alpha < 0.0 || options.alpha > 1.0 {
        return Err(Error::InvalidInput("Alpha must be between 0 and 1".into()));
    }

    let n = x.rows;
    let p = x.cols;

    if y.len() != n {
        return Err(Error::DimensionMismatch(format!(
            "Length of y ({}) must match number of rows in X ({})",
            y.len(),
            n
        )));
    }

    // Validate coefficient bounds
    let n_predictors = if options.intercept { p - 1 } else { p };
    if let Some(ref bounds) = options.coefficient_bounds {
        if bounds.len() != n_predictors {
            return Err(Error::InvalidInput(format!(
                "Coefficient bounds length ({}) must match number of predictors ({})",
                bounds.len(), n_predictors
            )));
        }
        for (i, &(lower, upper)) in bounds.iter().enumerate() {
            if lower > upper {
                return Err(Error::InvalidInput(format!(
                    "Coefficient bounds for predictor {}: lower ({}) must be <= upper ({})",
                    i, lower, upper
                )));
            }
            // Note: We allow (-inf, +inf) as it represents "no bounds" for that predictor
            // This is useful for having mixed bounded/unbounded predictors
        }
    }

    // Standardize X and y
    // glmnet convention: y is always centered/scaled if intercept is present
    let standardization_options = StandardizeOptions {
        intercept: options.intercept,
        standardize_x: options.standardize,
        standardize_y: options.intercept,
        weights: options.weights.clone(),
    };

    let (x_standardized, y_standardized, standardization_info) = standardize_xy(x, y, &standardization_options);

    // Adjust lambda for scaling
    // The path generation returns internal lambdas (for standardized data),
    // which are used directly in coordinate descent.
    //
    // For single-lambda fits, the user provides "public" lambda values
    // (like R reports), which need to be converted to "internal" scale:
    //   lambda_standardized_value = lambda_original_scale / y_scale
    let y_scale_factor = standardization_info.y_scale.unwrap_or(1.0);
    let lambda_standardized = if y_scale_factor > 1e-12 {
        options.lambda / y_scale_factor
    } else {
        options.lambda
    };

    // DEBUG: Print scaling info
    // #[cfg(debug_assertions)]
    // {
    //     eprintln!("DEBUG elastic_net_fit: user_lambda = {}, y_scale = {}, lambda_standardized = {}",
    //              options.lambda, y_scale_factor, lambda_standardized);
    // }

    // Initial coefficients (all zeros)
    let mut coefficients_standardized = vec![0.0; p];

    // Determine unpenalized columns (e.g. intercept column 0 if manually added,
    // but standardize_xy handles the intercept externally usually.
    // If intercept=true, standardize_xy centers data and we don't penalize an implicit intercept.
    // Here we assume x contains PREDICTORS only if intercept is handled by standardization centering.
    // However, the `Matrix` struct might include a column of 1s if the user passed it.
    // `standardize_xy` treats all columns in X as predictors to be standardized.
    // If options.intercept is true, we compute the intercept from the means later.
    // We assume X passed here does NOT contain a manual intercept column of 1s unless
    // the user explicitly wants to penalize it (which is weird) or turned off intercept in options.
    // For now, we penalize all columns in X according to penalty_factors.

    // Check if we assume X has an intercept column at 0 that we should skip?
    // The previous ridge/lasso implementations had a `first_penalized_column_index` logic:
    // `let first_penalized_column_index = if options.intercept { 1 } else { 0 };`
    // This implies `x` might have a column of 1s.
    // GLMNET convention usually takes x matrix of predictors only.
    // `standardize_xy` calculates means for ALL columns.
    // If column 0 is all 1s, std dev is 0, standardization might fail or set to 0.
    // Let's stick to the previous `lasso.rs` logic: if intercept is requested, we ignore column 0?
    // `lasso.rs`: "Determine which columns are penalized. first_penalized_column_index = if options.intercept { 1 } else { 0 }"
    // This strongly suggests the input Matrix `x` is expected to have a column of 1s at index 0 if intercept=true.
    // We will preserve this behavior for compatibility with existing tests.
    // i.e. this is going to be hell to refactor and I'm idly typing my thoughts away...
    // This is a naive implementation anyways and only one head of the hydra that is glmnet.
    let first_penalized_column_index = if options.intercept { 1 } else { 0 };

    // Warm start initialization
    if let Some(warm) = &options.warm_start {
        // warm contains slope coefficients on ORIGINAL scale
        // We need to transform them to STANDARDIZED scale
        // coefficients_standardized = beta_orig * x_scale / y_scale
        let y_scale = standardization_info.y_scale.unwrap_or(1.0);

        if first_penalized_column_index == 1 {
            // Case 1: Intercept at col 0
            // warm start vector should correspond to cols 1..p (slopes)
            // coefficients_standardized[0] stays 0.0 (intercept of centered data is 0)
            if warm.len() == p - 1 {
                for j in 1..p {
                    coefficients_standardized[j] = warm[j - 1] * standardization_info.x_scale[j] / y_scale;
                }
            } else {
                // If dimensions don't match, ignore warm start or warn?
                // For safety in this "todo" fix, we'll just ignore mismatched warm starts to avoid panics,
                // but usually this indicates a caller error.
                // Given I can't print warnings easily here, I'll ignore or maybe assume warm includes intercept?
                // If warm has length p, maybe it includes intercept? But ElasticNetFit.coefficients excludes it.
                // Let's stick to: warm start matches slopes.
            }
        } else {
            // Case 2: No intercept column
            if warm.len() == p {
                for j in 0..p {
                    coefficients_standardized[j] = warm[j] * standardization_info.x_scale[j] / y_scale;
                }
            }
        }
    }

    // Transform coefficient bounds to standardized scale
    // Bounds on original scale need to be converted: coefficients_standardized = beta_orig * x_scale / y_scale
    let bounds_standardized: Option<Vec<(f64, f64)>> = options.coefficient_bounds.as_ref().map(|bounds| {
        let y_scale = standardization_info.y_scale.unwrap_or(1.0);
        bounds.iter().enumerate().map(|(j, &(lower, upper))| {
            // For each predictor j in original scale, the corresponding column
            // in the standardized matrix is at index j+1 (col 0 is intercept)
            let std_idx = j + 1;
            let x_scale_predictor_j = if std_idx < standardization_info.x_scale.len() {
                standardization_info.x_scale[std_idx]
            } else {
                1.0
            };
            let scale_factor = x_scale_predictor_j / y_scale;
            (lower * scale_factor, upper * scale_factor)
        }).collect()
    });

    let (iterations, converged) = coordinate_descent(
        &x_standardized,
        &y_standardized,
        &mut coefficients_standardized,
        lambda_standardized,
        options.alpha,
        first_penalized_column_index,
        options.max_iter,
        options.tol,
        options.penalty_factor.as_deref(),
        bounds_standardized.as_deref(),
        &standardization_info.column_squared_norms,
    )?;

    // Unstandardize
    let (intercept, beta_orig) = unstandardize_coefficients(&coefficients_standardized, &standardization_info);

    // Count nonzero (excluding intercept)
    // beta_orig contains slopes. If first_penalized_column_index=1, coefficients_standardized[0] was 0.
    // The coefficients returned should correspond to the columns of X (excluding the manual intercept if present?).
    // `unstandardize_coefficients` handles the mapping.
    let n_nonzero = beta_orig.iter().filter(|&&b| b.abs() > 0.0).count();

    // Fitted values
    let fitted = predict(x, intercept, &beta_orig);
    let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(yi, yh)| yi - yh).collect();

    // Statistics
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
    let mae: f64 = residuals.iter().map(|r| r.abs()).sum::<f64>() / n as f64;

    let r_squared = if ss_tot > 1e-10 { 1.0 - ss_res / ss_tot } else { 1.0 };

    // Effective DF approximation for Elastic Net
    // df ≈ n_nonzero for Lasso
    // df ≈ trace(S) for Ridge
    // We use a naive approximation here: n_nonzero
    let eff_df = 1.0 + n_nonzero as f64;
    let adj_r_squared = if ss_tot > 1e-10 && n > eff_df as usize {
        1.0 - (1.0 - r_squared) * ((n - 1) as f64 / (n as f64 - eff_df))
    } else {
        r_squared
    };

    let mse = ss_res / (n as f64 - eff_df).max(1.0);

    // Model selection criteria
    let ss_res: f64 = residuals.iter().map(|&r| r * r).sum();
    let ll = log_likelihood(n, mse, ss_res);
    let n_coef = beta_orig.len() + 1; // coefficients + intercept
    let aic_val = aic(ll, n_coef);
    let bic_val = bic(ll, n_coef, n);

    Ok(ElasticNetFit {
        lambda: options.lambda,
        alpha: options.alpha,
        intercept,
        coefficients: beta_orig,
        fitted_values: fitted,
        residuals,
        n_nonzero,
        iterations,
        converged,
        r_squared,
        adj_r_squared,
        mse,
        rmse: mse.sqrt(),
        mae,
        log_likelihood: ll,
        aic: aic_val,
        bic: bic_val,
    })
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn coordinate_descent(
    x: &Matrix,
    y: &[f64],
    beta: &mut [f64],
    lambda: f64,
    alpha: f64,
    first_penalized_column_index: usize,
    max_iter: usize,
    tol: f64,
    penalty_factor: Option<&[f64]>,
    bounds: Option<&[(f64, f64)]>,
    column_squared_norms: &[f64],  // Column squared norms (for coordinate descent update)
) -> Result<(usize, bool)> {
    let n = x.rows;
    let p = x.cols;

    // Residuals r = y - Xβ
    // Initialize with all betas zero -> residuals = y
    // If y contains infinity/NaN, residuals will too
    let mut residuals = y.to_vec();

    // Check for non-finite residuals initially - if present, we can't optimize
    if residuals.iter().any(|r| !r.is_finite()) {
        return Ok((0, false));
    }

    // Handle non-zero initial betas (warm starts)
    for j in 0..p {
        if beta[j] != 0.0 {
            for i in 0..n {
                residuals[i] -= x.get(i, j) * beta[j];
            }
        }
    }

    // Active set: indices of non-zero coefficients
    let mut active_set = vec![false; p];

    let mut converged = false;
    let mut iter = 0;

    while iter < max_iter {
        let mut maximum_coefficient_change = 0.0;

        // --- Full Pass ---
        for j in first_penalized_column_index..p {
            if update_feature(j, x, &mut residuals, beta, lambda, alpha, penalty_factor, bounds, column_squared_norms, &mut maximum_coefficient_change) {
                active_set[j] = true;
            }
        }
        iter += 1;

        if maximum_coefficient_change < tol {
            converged = true;
            break;
        }

        // --- Active Set Loop ---
        loop {
            if iter >= max_iter { break; }

            let mut active_set_coefficient_change = 0.0;
            let mut active_count = 0;

            for j in first_penalized_column_index..p {
                if active_set[j] {
                    update_feature(j, x, &mut residuals, beta, lambda, alpha, penalty_factor, bounds, column_squared_norms, &mut active_set_coefficient_change);
                    active_count += 1;

                    if beta[j] == 0.0 {
                       active_set[j] = false;
                    }
                }
            }

            iter += 1;

            if active_set_coefficient_change < tol {
                break;
            }

            if active_count == 0 {
                break;
            }
        }
    }

    Ok((iter, converged))
}

#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn update_feature(
    j: usize,
    x: &Matrix,
    residuals: &mut [f64],
    beta: &mut [f64],
    lambda: f64,
    alpha: f64,
    penalty_factor: Option<&[f64]>,
    bounds: Option<&[(f64, f64)]>,
    column_squared_norms: &[f64],  // Column squared norms (for coordinate descent update)
    maximum_coefficient_change: &mut f64
) -> bool {
    // Penalty factor
    let penalty_factor_value = penalty_factor.and_then(|v| v.get(j)).copied().unwrap_or(1.0);
    if penalty_factor_value == f64::INFINITY {
        beta[j] = 0.0;
        return false;
    }

    let n = x.rows;
    let coefficient_previous = beta[j];

    // Calculate partial residual correlation (rho)
    // residuals currently = y - Sum(Xk * beta_k)
    // We want r_partial = y - Sum_{k!=j}(Xk * beta_k) = residuals + Xj * beta_j
    // rho = Xj^T * r_partial = Xj^T * residuals + (Xj^T * Xj) * beta_j
    // where Xj^T * Xj = column_squared_norms[j] (the squared norm of column j after standardization)

    let mut partial_correlation_unscaled = 0.0;
    for i in 0..n {
        partial_correlation_unscaled += x.get(i, j) * residuals[i];
    }
    // Use column_squared_norms[j] instead of assuming 1.0
    let rho = partial_correlation_unscaled + column_squared_norms[j] * coefficient_previous;

    // Soft thresholding
    // Numerator: S(rho, lambda * alpha * penalty_factor_value)
    let threshold = lambda * alpha * penalty_factor_value;
    let soft_threshold_result = soft_threshold(rho, threshold);

    // Denominator
    // Elastic net denominator: column_squared_norms[j] + lambda * (1 - alpha) * penalty_factor_value
    // This matches glmnet's formula
    let denominator_with_ridge_penalty = column_squared_norms[j] + lambda * (1.0 - alpha) * penalty_factor_value;

    let mut coefficient_updated = soft_threshold_result / denominator_with_ridge_penalty;

    // Apply coefficient bounds (clamping) if provided
    // Bounds clamp the calculated value to [lower, upper]
    if let Some(bounds) = bounds {
        // bounds[j-1] because bounds is indexed by predictor (excluding intercept)
        // and j starts at first_penalized_column_index (usually 1 for intercept models)
        let bounds_idx = j.saturating_sub(1);
        if let Some((lower, upper)) = bounds.get(bounds_idx) {
            coefficient_updated = coefficient_updated.max(*lower).min(*upper);
        }
    }

    // Update residuals if beta changed
    if coefficient_updated != coefficient_previous {
        let coefficient_change = coefficient_updated - coefficient_previous;
        for i in 0..n {
            // residuals_new = residuals_old - x_j * coefficient_change
            residuals[i] -= x.get(i, j) * coefficient_change;
        }
        beta[j] = coefficient_updated;
        *maximum_coefficient_change = maximum_coefficient_change.max(coefficient_change.abs());
        true // changed
    } else {
        false // no change
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_threshold_basic_cases() {
        // Test soft_threshold function edge cases
        assert_eq!(soft_threshold(5.0, 2.0), 3.0); // z > gamma
        assert_eq!(soft_threshold(-5.0, 2.0), -3.0); // z < -gamma
        assert_eq!(soft_threshold(1.0, 2.0), 0.0); // |z| <= gamma
        assert_eq!(soft_threshold(2.0, 2.0), 0.0); // z == gamma
        assert_eq!(soft_threshold(-2.0, 2.0), 0.0); // z == -gamma
    }

    #[test]
    fn test_soft_threshold_zero() {
        assert_eq!(soft_threshold(0.0, 0.0), 0.0);
        assert_eq!(soft_threshold(5.0, 0.0), 5.0);
        assert_eq!(soft_threshold(-5.0, 0.0), -5.0);
    }

    #[test]
    #[should_panic(expected = "Soft threshold gamma must be non-negative")]
    fn test_soft_threshold_negative_gamma_panics() {
        soft_threshold(1.0, -1.0);
    }

    #[test]
    fn test_elastic_net_options_default() {
        let options = ElasticNetOptions::default();
        assert_eq!(options.lambda, 1.0);
        assert_eq!(options.alpha, 1.0);  // Default is 1.0 (Lasso)
        assert!(options.intercept);
        assert!(options.standardize);
        assert_eq!(options.max_iter, 100000);
        assert_eq!(options.tol, 1e-7);
        assert!(options.penalty_factor.is_none());
        assert!(options.warm_start.is_none());
        assert!(options.coefficient_bounds.is_none());
    }

    #[test]
    fn test_elastic_net_fit_simple() {
        // Simple linear relationship: y = 2*x + 1
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let x1: Vec<f64> = (1..=5).map(|i| i as f64).collect();

        // Build matrix with intercept column
        let n = 5;
        let p = 1;
        let mut x_data = vec![1.0; n * (p + 1)];  // Start with all 1s for intercept
        for i in 0..n {
            x_data[i * (p + 1) + 1] = x1[i];  // Fill in predictor column
        }
        let x = Matrix::new(n, p + 1, x_data);

        let options = ElasticNetOptions {
            lambda: 0.01,  // Small lambda for minimal regularization
            alpha: 0.5,
            intercept: true,
            standardize: true,
            ..Default::default()
        };

        let result = elastic_net_fit(&x, &y, &options);
        assert!(result.is_ok());

        let fit = result.unwrap();
        assert!(fit.converged);
        // Coefficients should be close to [1, 2] (intercept, slope)
        assert!((fit.intercept - 1.0).abs() < 0.5);
        assert!((fit.coefficients[0] - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_elastic_net_fit_with_penalty_factor() {
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let x1: Vec<f64> = (1..=5).map(|i| i as f64).collect();

        let n = 5;
        let p = 1;
        let mut x_data = vec![1.0; n * (p + 1)];
        for i in 0..n {
            x_data[i * (p + 1) + 1] = x1[i];
        }
        let x = Matrix::new(n, p + 1, x_data);

        let options = ElasticNetOptions {
            lambda: 0.1,
            alpha: 0.5,
            penalty_factor: Some(vec![1.0]),
            intercept: true,
            standardize: true,
            ..Default::default()
        };

        let result = elastic_net_fit(&x, &y, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_elastic_net_fit_with_coefficient_bounds() {
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let x1: Vec<f64> = (1..=5).map(|i| i as f64).collect();

        let n = 5;
        let p = 1;
        let mut x_data = vec![1.0; n * (p + 1)];
        for i in 0..n {
            x_data[i * (p + 1) + 1] = x1[i];
        }
        let x = Matrix::new(n, p + 1, x_data);

        let options = ElasticNetOptions {
            lambda: 0.01,
            alpha: 0.5,
            coefficient_bounds: Some(vec![(0.0, 3.0)]), // Bound slope to [0, 3]
            intercept: true,
            standardize: true,
            ..Default::default()
        };

        let result = elastic_net_fit(&x, &y, &options);
        assert!(result.is_ok());

        let fit = result.unwrap();
        // Coefficient should be within bounds
        assert!(fit.coefficients[0] >= 0.0);
        assert!(fit.coefficients[0] <= 3.0);
    }

    #[test]
    fn test_elastic_net_pure_lasso() {
        // alpha = 1.0 means pure Lasso
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let x1: Vec<f64> = (1..=5).map(|i| i as f64).collect();

        let n = 5;
        let p = 1;
        let mut x_data = vec![1.0; n * (p + 1)];
        for i in 0..n {
            x_data[i * (p + 1) + 1] = x1[i];
        }
        let x = Matrix::new(n, p + 1, x_data);

        let options = ElasticNetOptions {
            lambda: 1.0,
            alpha: 1.0,  // Pure Lasso
            intercept: true,
            standardize: true,
            ..Default::default()
        };

        let result = elastic_net_fit(&x, &y, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_elastic_net_pure_ridge() {
        // alpha = 0.0 means pure Ridge
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let x1: Vec<f64> = (1..=5).map(|i| i as f64).collect();

        let n = 5;
        let p = 1;
        let mut x_data = vec![1.0; n * (p + 1)];
        for i in 0..n {
            x_data[i * (p + 1) + 1] = x1[i];
        }
        let x = Matrix::new(n, p + 1, x_data);

        let options = ElasticNetOptions {
            lambda: 0.1,
            alpha: 0.0,  // Pure Ridge
            intercept: true,
            standardize: true,
            ..Default::default()
        };

        let result = elastic_net_fit(&x, &y, &options);
        assert!(result.is_ok());

        let fit = result.unwrap();
        // Ridge shouldn't zero out coefficients
        assert!(fit.n_nonzero >= 1);
    }

    #[test]
    fn test_elastic_fit_no_intercept() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1: Vec<f64> = (1..=5).map(|i| i as f64).collect();

        let n = 5;
        let p = 1;
        let x = Matrix::new(n, p, x1);  // No intercept column

        let options = ElasticNetOptions {
            lambda: 0.01,
            alpha: 0.5,
            intercept: false,  // No intercept
            standardize: true,
            ..Default::default()
        };

        let result = elastic_net_fit(&x, &y, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_elastic_net_with_warm_start() {
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let x1: Vec<f64> = (1..=5).map(|i| i as f64).collect();

        let n = 5;
        let p = 1;
        let mut x_data = vec![1.0; n * (p + 1)];
        for i in 0..n {
            x_data[i * (p + 1) + 1] = x1[i];
        }
        let x = Matrix::new(n, p + 1, x_data);

        let warm = vec![1.5];

        let options = ElasticNetOptions {
            lambda: 0.1,
            alpha: 0.5,
            intercept: true,
            standardize: true,
            warm_start: Some(warm),
            ..Default::default()
        };

        let result = elastic_net_fit(&x, &y, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_elastic_net_multivariate() {
        // Multiple predictors
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let x1: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];

        let n = 5;
        let p = 2;
        let mut x_data = vec![1.0; n * (p + 1)];  // Intercept column
        for i in 0..n {
            x_data[i * (p + 1) + 1] = x1[i];
            x_data[i * (p + 1) + 2] = x2[i];
        }
        let x = Matrix::new(n, p + 1, x_data);

        let options = ElasticNetOptions {
            lambda: 0.1,
            alpha: 0.5,
            intercept: true,
            standardize: true,
            ..Default::default()
        };

        let result = elastic_net_fit(&x, &y, &options);
        assert!(result.is_ok());

        let fit = result.unwrap();
        assert_eq!(fit.coefficients.len(), 2); // Two predictors
    }
}
