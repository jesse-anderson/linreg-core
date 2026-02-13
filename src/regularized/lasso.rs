//! Lasso regression (L1-regularized linear regression).
//!
//! This module provides a wrapper around the elastic net implementation with `alpha=1.0`.

use crate::error::Result;
use crate::linalg::Matrix;
use crate::regularized::elastic_net::{elastic_net_fit, ElasticNetOptions};
use crate::regularized::preprocess::predict;
use crate::serialization::types::ModelType;
use crate::impl_serialization;
use serde::{Deserialize, Serialize};

pub use crate::regularized::elastic_net::soft_threshold;

/// Options for lasso regression fitting.
///
/// Configuration options for lasso regression (L1-regularized linear regression).
///
/// # Fields
///
/// - `lambda` - Regularization strength (≥ 0, higher = more sparsity)
/// - `intercept` - Whether to include an intercept term
/// - `standardize` - Whether to standardize predictors to unit variance
/// - `max_iter` - Maximum coordinate descent iterations
/// - `tol` - Convergence tolerance on coefficient changes
/// - `penalty_factor` - Optional per-feature penalty multipliers
/// - `warm_start` - Optional initial coefficient values for warm starts
/// - `weights` - Optional observation weights
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::lasso::LassoFitOptions;
/// let options = LassoFitOptions {
///     lambda: 0.1,
///     intercept: true,
///     standardize: true,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug)]
pub struct LassoFitOptions {
    pub lambda: f64,
    pub intercept: bool,
    pub standardize: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub penalty_factor: Option<Vec<f64>>,
    pub warm_start: Option<Vec<f64>>,
    pub weights: Option<Vec<f64>>, // Observation weights
}

impl Default for LassoFitOptions {
    fn default() -> Self {
        LassoFitOptions {
            lambda: 1.0,
            intercept: true,
            standardize: true,
            max_iter: 100000,
            tol: 1e-7, // Match ElasticNetOptions default
            penalty_factor: None,
            warm_start: None,
            weights: None,
        }
    }
}

/// Result of a lasso regression fit.
///
/// Contains the fitted model coefficients, convergence information, and diagnostic metrics.
///
/// # Fields
///
/// - `lambda` - The regularization strength used
/// - `intercept` - Intercept coefficient (never penalized)
/// - `coefficients` - Slope coefficients (some may be exactly zero due to L1 penalty)
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
/// # use linreg_core::regularized::lasso::{lasso_fit, LassoFitOptions};
/// # use linreg_core::linalg::Matrix;
/// # let y = vec![2.0, 4.0, 6.0, 8.0];
/// # let x = Matrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
/// # let options = LassoFitOptions { lambda: 0.01, intercept: true, standardize: true, ..Default::default() };
/// let fit = lasso_fit(&x, &y, &options).unwrap();
///
/// // Check convergence and sparsity
/// println!("Converged: {}", fit.converged);
/// println!("Non-zero coefficients: {}", fit.n_nonzero);
/// println!("Iterations: {}", fit.iterations);
///
/// // Access model coefficients
/// println!("Intercept: {}", fit.intercept);
/// println!("Slopes: {:?}", fit.coefficients);
/// println!("AIC: {}", fit.aic);
/// # Ok::<(), linreg_core::Error>(())
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LassoFit {
    pub lambda: f64,
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

/// Fits lasso regression for a single lambda value.
///
/// Lasso regression adds an L1 penalty to the coefficients, which performs
/// automatic variable selection by shrinking some coefficients to exactly zero.
/// The intercept is never penalized.
///
/// # Arguments
///
/// * `x` - Design matrix (n rows × p columns including intercept)
/// * `y` - Response variable (n observations)
/// * `options` - Configuration options for lasso regression
///
/// # Returns
///
/// A `LassoFit` containing coefficients, convergence info, and metrics.
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::lasso::{lasso_fit, LassoFitOptions};
/// # use linreg_core::linalg::Matrix;
/// let y = vec![2.0, 4.0, 6.0, 8.0];
/// let x = Matrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
///
/// let options = LassoFitOptions {
///     lambda: 0.01,
///     intercept: true,
///     standardize: true,
///     ..Default::default()
/// };
///
/// let fit = lasso_fit(&x, &y, &options).unwrap();
/// assert!(fit.converged);
/// assert!(fit.n_nonzero <= 1); // At most 1 non-zero coefficient
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn lasso_fit(x: &Matrix, y: &[f64], options: &LassoFitOptions) -> Result<LassoFit> {
    let en_options = ElasticNetOptions {
        lambda: options.lambda,
        alpha: 1.0, // Lasso
        intercept: options.intercept,
        standardize: options.standardize,
        max_iter: options.max_iter,
        tol: options.tol,
        penalty_factor: options.penalty_factor.clone(),
        warm_start: options.warm_start.clone(),
        weights: options.weights.clone(),
        coefficient_bounds: None,
    };

    let fit = elastic_net_fit(x, y, &en_options)?;

    Ok(LassoFit {
        lambda: fit.lambda,
        intercept: fit.intercept,
        coefficients: fit.coefficients,
        fitted_values: fit.fitted_values,
        residuals: fit.residuals,
        n_nonzero: fit.n_nonzero,
        iterations: fit.iterations,
        converged: fit.converged,
        r_squared: fit.r_squared,
        adj_r_squared: fit.adj_r_squared,
        mse: fit.mse,
        rmse: fit.rmse,
        mae: fit.mae,
        log_likelihood: fit.log_likelihood,
        aic: fit.aic,
        bic: fit.bic,
    })
}

/// Makes predictions using a lasso regression fit.
///
/// Computes predictions for new observations using the fitted lasso regression model.
///
/// # Arguments
///
/// * `fit` - Fitted lasso regression model
/// * `x_new` - New design matrix (same number of columns as training data)
///
/// # Returns
///
/// Vector of predicted values.
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::lasso::{lasso_fit, predict_lasso, LassoFitOptions};
/// # use linreg_core::linalg::Matrix;
/// // Training data
/// let y = vec![2.0, 4.0, 6.0, 8.0];
/// let x = Matrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
///
/// let options = LassoFitOptions {
///     lambda: 0.01,
///     intercept: true,
///     standardize: true,
///     ..Default::default()
/// };
/// let fit = lasso_fit(&x, &y, &options).unwrap();
///
/// // Predict on new data
/// let x_new = Matrix::new(2, 2, vec![1.0, 5.0, 1.0, 6.0]);
/// let predictions = predict_lasso(&fit, &x_new);
///
/// assert_eq!(predictions.len(), 2);
/// // Predictions should be close to [10.0, 12.0] for the linear relationship y = 2*x
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn predict_lasso(fit: &LassoFit, x_new: &Matrix) -> Vec<f64> {
    predict(x_new, fit.intercept, &fit.coefficients)
}

// ============================================================================
// Model Serialization Traits
// ============================================================================

// Generate ModelSave and ModelLoad implementations using macro
impl_serialization!(LassoFit, ModelType::Lasso, "Lasso");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_threshold() {
        assert_eq!(soft_threshold(5.0, 2.0), 3.0);
        assert_eq!(soft_threshold(-5.0, 2.0), -3.0);
        assert_eq!(soft_threshold(1.0, 2.0), 0.0);
    }

    #[test]
    fn test_lasso_fit_simple() {
        // Simple test: y = 2*x with perfect linear relationship
        let x_data = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
        let x = Matrix::new(4, 2, x_data);
        let y = vec![2.0, 4.0, 6.0, 8.0];

        let options = LassoFitOptions {
            lambda: 0.01,
            intercept: true,
            standardize: true,
            ..Default::default()
        };

        let fit = lasso_fit(&x, &y, &options).unwrap();

        assert!(fit.converged);
        // Predictions should be close to actual values
        for i in 0..4 {
            assert!((fit.fitted_values[i] - y[i]).abs() < 0.5);
        }
    }
}