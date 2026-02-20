//! Permutation importance for feature importance.
//!
//! Permutation importance measures the decrease in model performance (typically R²)
//! when a single predictor's values are randomly shuffled. This breaks the
//! relationship between that feature and the target while preserving the
//! distribution of all other predictors.
//!
//! # Algorithm
//!
//! 1. Compute baseline model performance (e.g., R²) on original data
//! 2. For each predictor j:
//!    a. Randomly permute (shuffle) column j
//!    b. Refit the OLS model on the permuted data
//!    c. Compute performance metric (R²) on permuted data
//!    d. Importance = baseline_R² - permuted_R²
//! 3. Average across multiple permutations for stability
//!
//! # Interpretation
//!
//! - Higher values = more important (shuffling causes larger performance drop)
//! - Values close to 0 = feature has no effect on predictions
//! - Negative values possible = feature happened to help by chance in this sample
//!
//! # Example
//!
//! ```
//! # use linreg_core::feature_importance::{permutation_importance_ols, PermutationImportanceOptions};
//! # use linreg_core::core::ols_regression;
//! let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
//! let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];
//! let names = vec!["Intercept".into(), "X1".into(), "X2".into()];
//!
//! let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names)?;
//!
//! let options = PermutationImportanceOptions {
//!     n_permutations: 50,
//!     seed: Some(42),
//!     compute_intervals: false,
//!     interval_confidence: 0.95,
//! };
//!
//! let importance = permutation_importance_ols(&y, &[x1, x2], &fit, &options)?;
//!
//! println!("Importance: {:?}", importance.importance);
//! # Ok::<(), linreg_core::Error>(())
//! ```

use crate::core::{ols_regression, RegressionOutput};
use crate::distributions::normal_inverse_cdf;
use crate::error::{Error, Result};
use crate::feature_importance::types::{PermutationImportanceOptions, PermutationImportanceOutput};
use crate::linalg::Matrix;

/// Simple deterministic seeded random number generator for permutation.
///
/// Uses a simple linear congruential generator for reproducibility without
/// external dependencies. Not cryptographically secure but sufficient for
/// permutation importance.
#[derive(Clone, Debug)]
struct SeededRng {
    state: u64,
}

impl SeededRng {
    fn new(seed: u64) -> Self {
        SeededRng { state: seed.wrapping_add(1) }
    }

    /// Generates a random u64 in [0, 2^64)
    #[inline]
    fn next(&mut self) -> u64 {
        // Linear congruential generator (LCG)
        // Using constants from Numerical Recipes
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }

    /// Generates a random f64 in [0, 1)
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Fisher-Yates shuffle using the seeded RNG.
#[inline]
fn shuffle(vec: &mut Vec<f64>, rng: &mut SeededRng) {
    let n = vec.len();
    if n <= 1 {
        return;
    }

    for i in (1..n).rev() {
        // Pick random index from 0..=i
        let j = (rng.next_f64() * (i as f64 + 1.0)).floor() as usize;
        vec.swap(i, j);
    }
}

/// Computes R² (coefficient of determination).
fn r_squared(y: &[f64], predictions: &[f64]) -> f64 {
    if y.len() != predictions.len() || y.is_empty() {
        return f64::NAN;
    }

    let y_mean = y.iter().sum::<f64>() / y.len() as f64;

    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = y
        .iter()
        .zip(predictions.iter())
        .map(|(&yi, &yp)| (yi - yp).powi(2))
        .sum();

    if ss_tot == 0.0 {
        return if ss_res == 0.0 { 1.0 } else { f64::NAN };
    }

    1.0 - ss_res / ss_tot
}

/// Makes predictions using OLS coefficients.
fn predict_ols(x: &Matrix, coefficients: &[f64]) -> Vec<f64> {
    let n = x.rows;
    let p = x.cols;

    let mut predictions = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..p {
            sum += x.get(i, j) * coefficients[j];
        }
        predictions[i] = sum;
    }
    predictions
}

/// Computes permutation importance for OLS regression.
///
/// Permutation importance measures the decrease in R² when each predictor
/// is randomly shuffled. Higher values indicate more important features.
///
/// # Arguments
///
/// * `y` - Response variable
/// * `x_vars` - Predictor variables (each `Vec<f64>` is a column)
/// * `fit` - Fitted OLS model result
/// * `options` - Configuration options for permutation
///
/// # Returns
///
/// A [`PermutationImportanceOutput`] containing:
/// - Variable names
/// - Importance scores (baseline_R² - permuted_R² for each feature)
/// - Baseline R² score
/// - Number of permutations performed
///
/// # Errors
///
/// * [`Error::InvalidInput`] - if dimensions don't match
/// * [`Error::InsufficientData`] - if insufficient data for permutations
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::{permutation_importance_ols, PermutationImportanceOptions};
/// # use linreg_core::core::ols_regression;
/// let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];
/// let names = vec!["Intercept".into(), "X1".into(), "X2".into()];
///
/// let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names)?;
///
/// let importance = permutation_importance_ols(
///     &y,
///     &[x1, x2],
///     &fit,
///     &PermutationImportanceOptions::default()
/// )?;
///
/// // Higher importance = more important feature
/// let ranking = importance.ranking();
/// println!("Most important: {}", ranking[0].0);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn permutation_importance_ols(
    y: &[f64],
    x_vars: &[Vec<f64>],
    fit: &RegressionOutput,
    options: &PermutationImportanceOptions,
) -> Result<PermutationImportanceOutput> {
    let n = y.len();
    let k = x_vars.len();

    // Validate inputs
    if n <= 1 {
        return Err(Error::InsufficientData {
            required: 2,
            available: n,
        });
    }

    for (i, var) in x_vars.iter().enumerate() {
        if var.len() != n {
            return Err(Error::InvalidInput(format!(
                "x_vars[{}] has {} elements, expected {}",
                i, var.len(), n
            )));
        }
    }

    // Baseline R²
    let baseline_r2 = fit.r_squared;

    // Initialize RNG
    let mut rng = options.seed.map(SeededRng::new).unwrap_or_else(|| {
        // Use a fixed default seed for reproducibility when no seed provided
        SeededRng::new(42)
    });

    // Create design matrix for predictions
    let p = k + 1; // predictors + intercept
    let mut x_data = vec![0.0; n * p];
    for (row, _yi) in y.iter().enumerate() {
        x_data[row * p] = 1.0; // intercept
        for (col, x_var) in x_vars.iter().enumerate() {
            x_data[row * p + col + 1] = x_var[row];
        }
    }

    // Compute importance for each feature
    let mut importance = vec![0.0; k];
    let mut variable_names = Vec::with_capacity(k);

    // Prepare variable names for refitting
    let perm_names: Vec<String> = (0..=k).map(|i| {
        if i == 0 { "Intercept".to_string() } else { format!("X{}", i) }
    }).collect();

    for j in 0..k {
        variable_names.push(format!("X{}", j + 1));

        let mut perm_importance_sum = 0.0;

        // Extract the column to shuffle (copy for each iteration)
        let original_column: Vec<f64> = (0..n)
            .map(|row| x_data[row * p + j + 1])
            .collect();

        for _iter in 0..options.n_permutations {
            // Clone the column for this iteration
            let mut column = original_column.clone();

            // Shuffle the column
            shuffle(&mut column, &mut rng);

            // Create permuted predictor variables
            let x_permuted_vars: Vec<Vec<f64>> = (0..k).map(|col_idx| {
                (0..n).map(|row| {
                    if col_idx == j {
                        // Use shuffled column
                        column[row]
                    } else {
                        // Use original data
                        x_data[row * p + col_idx + 1]
                    }
                }).collect()
            }).collect();

            // Refit OLS model on permuted data
            let perm_fit = match ols_regression(y, &x_permuted_vars, &perm_names) {
                Ok(f) => f,
                Err(_) => continue, // Skip if fitting fails
            };

            // Importance = baseline - permuted R²
            if perm_fit.r_squared.is_finite() {
                perm_importance_sum += baseline_r2 - perm_fit.r_squared;
            }
        }

        importance[j] = perm_importance_sum / options.n_permutations as f64;
    }

    Ok(PermutationImportanceOutput {
        variable_names,
        importance,
        baseline_score: baseline_r2,
        n_permutations: options.n_permutations,
        seed: options.seed,
        importance_std_err: None,
        interval_lower: None,
        interval_upper: None,
        interval_confidence: None,
    })
}

/// Computes permutation importance with custom variable names.
///
/// This version allows specifying variable names for clearer output.
///
/// # Arguments
///
/// * `y` - Response variable
/// * `x_vars` - Predictor variables
/// * `fit` - Fitted OLS model
/// * `options` - Configuration options
/// * `variable_names` - Names for each predictor variable
///
/// # Returns
///
/// A [`PermutationImportanceOutput`] with the specified variable names.
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::{permutation_importance_ols_named, PermutationImportanceOptions};
/// # use linreg_core::core::ols_regression;
/// let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];
/// let names = vec!["Intercept".into(), "Temperature".into(), "Pressure".into()];
/// let pred_names = vec!["Temperature".into(), "Pressure".into()];
///
/// let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names)?;
///
/// let importance = permutation_importance_ols_named(
///     &y,
///     &[x1, x2],
///     &fit,
///     &PermutationImportanceOptions::default(),
///     &pred_names
/// )?;
///
/// assert_eq!(importance.variable_names, pred_names);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn permutation_importance_ols_named(
    y: &[f64],
    x_vars: &[Vec<f64>],
    fit: &RegressionOutput,
    options: &PermutationImportanceOptions,
    variable_names: &[String],
) -> Result<PermutationImportanceOutput> {
    let k = x_vars.len();

    if variable_names.len() != k {
        return Err(Error::InvalidInput(format!(
            "variable_names length ({}) must equal x_vars length ({})",
            variable_names.len(),
            k
        )));
    }

    let mut result = permutation_importance_ols(y, x_vars, fit, options)?;
    result.variable_names = variable_names.to_vec();
    Ok(result)
}

/// Computes permutation importance for Ridge regression.
///
/// This function handles the standardized nature of Ridge regression by:
/// 1. Permuting the original (unscaled) data
/// 2. Applying the same standardization that would be used during training
/// 3. Computing predictions using the fitted coefficients
///
/// # Arguments
///
/// * `y` - Response variable
/// * `x_original` - Original predictor variables (each `Vec<f64>` is a column)
/// * `fit` - Fitted Ridge regression model
/// * `options` - Configuration options for permutation
///
/// # Returns
///
/// A [`PermutationImportanceOutput`] containing importance scores
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::{permutation_importance_ridge, PermutationImportanceOptions};
/// # use linreg_core::regularized::{ridge_fit, RidgeFitOptions};
/// # use linreg_core::linalg::Matrix;
/// let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
/// let x = Matrix::new(5, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0]);
///
/// let options = RidgeFitOptions {
///     lambda: 1.0,
///     standardize: true,
///     ..Default::default()
/// };
/// let fit = ridge_fit(&x, &y, &options).unwrap();
///
/// let perm_options = PermutationImportanceOptions::default();
/// let importance = permutation_importance_ridge(&y, &[vec![1.0,2.0,3.0,4.0,5.0]], &fit, &perm_options)?;
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn permutation_importance_ridge(
    y: &[f64],
    x_original: &[Vec<f64>],
    fit: &crate::regularized::RidgeFit,
    options: &PermutationImportanceOptions,
) -> Result<PermutationImportanceOutput> {
    permutation_importance_regularized_helper(
        y,
        x_original,
        &fit.coefficients,
        fit.intercept,
        fit.r_squared,
        options,
        "Ridge",
    )
}

/// Computes permutation importance for Lasso regression.
///
/// This function handles the standardized nature of Lasso regression by:
/// 1. Permuting the original (unscaled) data
/// 2. Applying the same standardization that would be used during training
/// 3. Computing predictions using the fitted coefficients
///
/// # Arguments
///
/// * `y` - Response variable
/// * `x_original` - Original predictor variables (each `Vec<f64>` is a column)
/// * `fit` - Fitted Lasso regression model
/// * `options` - Configuration options for permutation
///
/// # Returns
///
/// A [`PermutationImportanceOutput`] containing importance scores
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::{permutation_importance_lasso, PermutationImportanceOptions};
/// # use linreg_core::regularized::{lasso_fit, LassoFitOptions};
/// # use linreg_core::linalg::Matrix;
/// let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
/// let x = Matrix::new(5, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0]);
///
/// let options = LassoFitOptions {
///     lambda: 0.1,
///     standardize: true,
///     ..Default::default()
/// };
/// let fit = lasso_fit(&x, &y, &options).unwrap();
///
/// let perm_options = PermutationImportanceOptions::default();
/// let importance = permutation_importance_lasso(&y, &[vec![1.0,2.0,3.0,4.0,5.0]], &fit, &perm_options)?;
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn permutation_importance_lasso(
    y: &[f64],
    x_original: &[Vec<f64>],
    fit: &crate::regularized::LassoFit,
    options: &PermutationImportanceOptions,
) -> Result<PermutationImportanceOutput> {
    permutation_importance_regularized_helper(
        y,
        x_original,
        &fit.coefficients,
        fit.intercept,
        fit.r_squared,
        options,
        "Lasso",
    )
}

/// Computes permutation importance for Elastic Net regression.
///
/// This function handles the standardized nature of Elastic Net regression by:
/// 1. Permuting the original (unscaled) data
/// 2. Applying the same standardization that would be used during training
/// 3. Computing predictions using the fitted coefficients
///
/// # Arguments
///
/// * `y` - Response variable
/// * `x_original` - Original predictor variables (each `Vec<f64>` is a column)
/// * `fit` - Fitted Elastic Net regression model
/// * `options` - Configuration options for permutation
///
/// # Returns
///
/// A [`PermutationImportanceOutput`] containing importance scores
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::{permutation_importance_elastic_net, PermutationImportanceOptions};
/// # use linreg_core::regularized::{elastic_net_fit, ElasticNetOptions};
/// # use linreg_core::linalg::Matrix;
/// let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
/// let x = Matrix::new(5, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0]);
///
/// let options = ElasticNetOptions {
///     lambda: 0.1,
///     alpha: 0.5,
///     standardize: true,
///     ..Default::default()
/// };
/// let fit = elastic_net_fit(&x, &y, &options).unwrap();
///
/// let perm_options = PermutationImportanceOptions::default();
/// let importance = permutation_importance_elastic_net(&y, &[vec![1.0,2.0,3.0,4.0,5.0]], &fit, &perm_options)?;
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn permutation_importance_elastic_net(
    y: &[f64],
    x_original: &[Vec<f64>],
    fit: &crate::regularized::ElasticNetFit,
    options: &PermutationImportanceOptions,
) -> Result<PermutationImportanceOutput> {
    permutation_importance_regularized_helper(
        y,
        x_original,
        &fit.coefficients,
        fit.intercept,
        fit.r_squared,
        options,
        "ElasticNet",
    )
}

/// Helper function for permutation importance with standardized models.
///
/// For regularized regression (Ridge/Lasso/ElasticNet), the models are typically
/// fit on standardized data. This function:
/// 1. Computes standardization parameters from original data
/// 2. Permutes each feature individually
/// 3. Re-standardizes the permuted data
/// 4. Computes predictions using the fitted coefficients
fn permutation_importance_regularized_helper(
    y: &[f64],
    x_original: &[Vec<f64>],
    coefficients: &[f64],
    intercept: f64,
    baseline_r2: f64,
    options: &PermutationImportanceOptions,
    _model_name: &str,
) -> Result<PermutationImportanceOutput> {
    let n = y.len();
    let k = x_original.len();

    // Validate inputs
    if n <= 1 {
        return Err(Error::InsufficientData {
            required: 2,
            available: n,
        });
    }

    for (i, var) in x_original.iter().enumerate() {
        if var.len() != n {
            return Err(Error::InvalidInput(format!(
                "x_original[{}] has {} elements, expected {}",
                i, var.len(), n
            )));
        }
    }

    // Initialize RNG
    let mut rng = options.seed.map(SeededRng::new).unwrap_or_else(|| {
        SeededRng::new(42)
    });

    // Compute standardization parameters from original data
    let mut x_means = vec![0.0; k];
    let mut x_stds = vec![0.0; k];

    for j in 0..k {
        x_means[j] = crate::stats::mean(&x_original[j]);
        let variance = crate::stats::variance(&x_original[j]);
        x_stds[j] = variance.sqrt();
        if x_stds[j] == 0.0 {
            x_stds[j] = 1.0; // Avoid division by zero
        }
    }

    // Compute y mean
    let y_mean = crate::stats::mean(y);

    // Compute importance for each feature
    let mut importance = vec![0.0; k];
    let mut variable_names = Vec::with_capacity(k);

    // For computing confidence intervals
    let mut importance_samples: Vec<Vec<f64>> = if options.compute_intervals {
        vec![vec![0.0; options.n_permutations]; k]
    } else {
        vec![]
    };

    for j in 0..k {
        variable_names.push(format!("X{}", j + 1));

        let mut perm_importance_sum = 0.0;

        for iter in 0..options.n_permutations {
            // Clone original data
            let mut x_permuted: Vec<Vec<f64>> = x_original.to_vec();

            // Shuffle column j
            shuffle(&mut x_permuted[j], &mut rng);

            // Standardize and make predictions
            let predictions = predict_standardized(
                &x_permuted,
                coefficients,
                intercept,
                &x_means,
                &x_stds,
                y_mean,
            );

            // Compute permuted R²
            let perm_r2 = r_squared(y, &predictions);

            // Importance = baseline - permuted
            let imp = baseline_r2 - perm_r2;
            if imp.is_finite() {
                perm_importance_sum += imp;
                if options.compute_intervals {
                    importance_samples[j][iter] = imp;
                }
            }
        }

        importance[j] = perm_importance_sum / options.n_permutations as f64;
    }

    // Compute confidence intervals if requested
    let (importance_std_err, interval_lower, interval_upper, interval_confidence) =
        if options.compute_intervals {
            let alpha = 1.0 - options.interval_confidence;
            // Use t-distribution with n_permutations - 1 degrees of freedom
            // For simplicity, we use normal approximation which is valid for n_permutations >= 30
            let z_score = normal_inverse_cdf(1.0 - alpha / 2.0);

            let mut std_err = vec![0.0; k];
            let mut lower = vec![0.0; k];
            let mut upper = vec![0.0; k];

            for j in 0..k {
                let mean = importance[j];
                let sample_mean: f64 = importance_samples[j].iter().sum::<f64>() / options.n_permutations as f64;
                let sample_variance: f64 = importance_samples[j]
                    .iter()
                    .map(|&x| (x - sample_mean).powi(2))
                    .sum::<f64>() / (options.n_permutations - 1) as f64;
                std_err[j] = sample_variance.sqrt();

                let margin = z_score * std_err[j];
                lower[j] = mean - margin;
                upper[j] = mean + margin;
            }

            (
                Some(std_err),
                Some(lower),
                Some(upper),
                Some(options.interval_confidence),
            )
        } else {
            (None, None, None, None)
        };

    Ok(PermutationImportanceOutput {
        variable_names,
        importance,
        baseline_score: baseline_r2,
        n_permutations: options.n_permutations,
        seed: options.seed,
        importance_std_err,
        interval_lower,
        interval_upper,
        interval_confidence,
    })
}

/// Makes predictions using standardized features.
///
/// For regularized models, predictions are:
/// y_pred = intercept + Σ(coefᵢ × (xᵢ - meanᵢ) / stdᵢ)
fn predict_standardized(
    x_vars: &[Vec<f64>],
    coefficients: &[f64],
    intercept: f64,
    x_means: &[f64],
    x_stds: &[f64],
    y_mean: f64,
) -> Vec<f64> {
    let n = x_vars[0].len();
    let k = x_vars.len();

    let mut predictions = vec![0.0; n];

    for i in 0..n {
        let mut sum = intercept;

        for j in 0..k {
            let x_std = (x_vars[j][i] - x_means[j]) / x_stds[j];
            sum += coefficients[j] * x_std;
        }

        predictions[i] = sum + y_mean;
    }

    predictions
}

/// Computes permutation importance for LOESS regression.
///
/// For LOESS, since the fit doesn't store the original data, we need to
/// re-fit the LOESS model on permuted data each time. This is computationally
/// expensive but provides model-agnostic feature importance.
///
/// # Arguments
///
/// * `y` - Response variable
/// * `x_vars` - Predictor variables (each `Vec<f64>` is a column)
/// * `span` - Span parameter used in original fit
/// * `degree` - Degree of polynomial used in original fit
/// * `options` - Configuration options for permutation
///
/// # Returns
///
/// A [`PermutationImportanceOutput`] containing importance scores
///
/// # Note
///
/// This function is computationally expensive as it re-fits the LOESS model
/// for each permutation of each feature. Use smaller `n_permutations` for
/// large datasets.
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::{permutation_importance_loess, PermutationImportanceOptions};
/// let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// let options = PermutationImportanceOptions {
///     n_permutations: 10,
///     seed: Some(42),
///     compute_intervals: false,
///     interval_confidence: 0.95,
/// };
///
/// let importance = permutation_importance_loess(&y, &[x], 0.75, 1, &options)?;
/// println!("Importance: {:?}", importance.importance);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn permutation_importance_loess(
    y: &[f64],
    x_vars: &[Vec<f64>],
    span: f64,
    degree: usize,
    options: &PermutationImportanceOptions,
) -> Result<PermutationImportanceOutput> {
    use crate::loess::{loess_fit, LoessOptions, LoessSurface};

    let n = y.len();
    let k = x_vars.len();

    // Validate inputs
    if n <= 1 {
        return Err(Error::InsufficientData {
            required: 2,
            available: n,
        });
    }

    for (i, var) in x_vars.iter().enumerate() {
        if var.len() != n {
            return Err(Error::InvalidInput(format!(
                "x_vars[{}] has {} elements, expected {}",
                i, var.len(), n
            )));
        }
    }

    // Compute baseline R² from LOESS fit
    let baseline_options = LoessOptions {
        span,
        degree,
        robust_iterations: 0,
        n_predictors: k,
        surface: LoessSurface::Direct,
    };

    let baseline_fit = loess_fit(y, x_vars, &baseline_options)?;
    let baseline_r2 = r_squared(y, &baseline_fit.fitted);

    // Initialize RNG
    let mut rng = options.seed.map(SeededRng::new).unwrap_or_else(|| {
        SeededRng::new(42)
    });

    // Compute importance for each feature
    let mut importance = vec![0.0; k];
    let mut variable_names = Vec::with_capacity(k);

    for j in 0..k {
        variable_names.push(format!("X{}", j + 1));

        let mut perm_importance_sum = 0.0;

        for _iter in 0..options.n_permutations {
            // Clone and permute data
            let mut x_permuted: Vec<Vec<f64>> = x_vars.to_vec();
            shuffle(&mut x_permuted[j], &mut rng);

            // Fit LOESS on permuted data
            match loess_fit(y, &x_permuted, &baseline_options) {
                Ok(perm_fit) => {
                    let perm_r2 = r_squared(y, &perm_fit.fitted);
                    if perm_r2.is_finite() {
                        perm_importance_sum += baseline_r2 - perm_r2;
                    }
                }
                Err(_) => {
                    // If LOESS fit fails, count as maximum importance
                    // (model completely broke when feature was shuffled)
                    perm_importance_sum += baseline_r2;
                }
            }
        }

        importance[j] = perm_importance_sum / options.n_permutations as f64;
    }

    Ok(PermutationImportanceOutput {
        variable_names,
        importance,
        baseline_score: baseline_r2,
        n_permutations: options.n_permutations,
        seed: options.seed,
        importance_std_err: None,
        interval_lower: None,
        interval_upper: None,
        interval_confidence: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ols_regression;

    #[test]
    fn test_permutation_importance_basic() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Strongly correlated with y
        let x2 = vec![2.0, 4.0, 1.0, 3.0, 2.0, 4.0]; // Some noise, weaker correlation
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        let options = PermutationImportanceOptions {
            n_permutations: 10,
            seed: Some(42),
            compute_intervals: false,
            interval_confidence: 0.95,
        };

        let importance = permutation_importance_ols(
            &y,
            &[x1, x2],
            &fit,
            &options
        ).unwrap();

        // X1 should have higher importance than X2 (more correlated with y)
        assert!(importance.importance[0] > importance.importance[1]);
        assert_eq!(importance.n_permutations, 10);
        assert!(importance.baseline_score > 0.9); // Strong correlation
    }

    #[test]
    fn test_permutation_importance_ranking() {
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Strong correlation
        let x2 = vec![0.1, 0.2, 0.3, 0.4, 0.5]; // Also correlated but weaker effect
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        let options = PermutationImportanceOptions {
            n_permutations: 20,
            seed: Some(42),
            compute_intervals: false,
            interval_confidence: 0.95,
        };

        let importance = permutation_importance_ols(
            &y,
            &[x1, x2],
            &fit,
            &options
        ).unwrap();

        let ranking = importance.ranking();

        // Most important should be first
        assert!(ranking[0].1 >= ranking[1].1);
    }

    #[test]
    fn test_permutation_importance_named() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let names = vec!["Intercept".into(), "Temp".into()];
        let pred_names = vec!["Temperature".into()];

        let fit = ols_regression(&y, &[x1.clone()], &names).unwrap();

        let options = PermutationImportanceOptions::default();
        let importance = permutation_importance_ols_named(
            &y,
            &[x1],
            &fit,
            &options,
            &pred_names
        ).unwrap();

        assert_eq!(importance.variable_names, pred_names);
    }

    #[test]
    fn test_seeded_rng_reproducibility() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut rng1 = SeededRng::new(42);
        let mut rng2 = SeededRng::new(42);

        let mut data1 = data.clone();
        let mut data2 = data.clone();

        shuffle(&mut data1, &mut rng1);
        shuffle(&mut data2, &mut rng2);

        assert_eq!(data1, data2); // Same seed = same shuffle
    }

    #[test]
    fn test_seeded_rng_different_seeds() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut rng1 = SeededRng::new(42);
        let mut rng2 = SeededRng::new(12345); // Very different seed

        let mut data1 = data.clone();
        let mut data2 = data.clone();

        shuffle(&mut data1, &mut rng1);
        shuffle(&mut data2, &mut rng2);

        assert_ne!(data1, data2); // Different seeds = different shuffles
    }

    #[test]
    fn test_r_squared_perfect() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let r2 = r_squared(&y, &predictions);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_r_squared_no_fit() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_mean = 3.0;
        let predictions = vec![y_mean; 5];

        let r2 = r_squared(&y, &predictions);
        assert!((r2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict_ols() {
        let x_data = vec![
            1.0, 1.0,  // obs 0: intercept, x1
            1.0, 2.0,  // obs 1
            1.0, 3.0,  // obs 2
        ];
        let x = Matrix::new(3, 2, x_data);
        let coefficients = vec![1.0, 2.0]; // intercept=1, coef=2

        let preds = predict_ols(&x, &coefficients);

        assert_eq!(preds[0], 1.0 + 2.0 * 1.0); // 3.0
        assert_eq!(preds[1], 1.0 + 2.0 * 2.0); // 5.0
        assert_eq!(preds[2], 1.0 + 2.0 * 3.0); // 7.0
    }

    #[test]
    fn test_permutation_importance_insufficient_data() {
        let y = vec![1.0];
        let x1 = vec![1.0];
        let names = vec!["Intercept".into(), "X1".into()];

        // ols_regression needs n > k + 1, so this should fail
        let fit_result = ols_regression(&y, &[x1.clone()], &names);
        assert!(fit_result.is_err());

        // Test with 3 observations (minimum valid for 1 predictor)
        let y2 = vec![1.0, 2.0, 3.0];
        let x2 = vec![1.0, 2.0, 3.0];
        let names2 = vec!["Intercept".into(), "X1".into()];

        let fit = ols_regression(&y2, &[x2.clone()], &names2).unwrap();
        let options = PermutationImportanceOptions::default();

        // Permutation importance should work with 3 obs
        let result = permutation_importance_ols(&y2, &[x2], &fit, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_permutation_importance_mismatched_dimensions() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let x2 = vec![1.0, 2.0]; // Wrong length (2 vs 4)
        let names = vec!["Intercept".into(), "X1".into(), "X2".into()];

        let fit = ols_regression(&y, &[x1.clone(), x1.clone()], &names).unwrap();
        let options = PermutationImportanceOptions::default();

        let result = permutation_importance_ols(&y, &[x1.clone(), x2.clone()], &fit, &options);
        assert!(result.is_err());
    }
}
