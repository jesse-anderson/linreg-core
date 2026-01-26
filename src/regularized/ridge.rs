//! Ridge regression (L2-regularized linear regression).
//!
//! This module provides a wrapper around the elastic net implementation with `alpha=0.0`.

use crate::error::Result;
use crate::linalg::Matrix;
use crate::regularized::elastic_net::{elastic_net_fit, ElasticNetOptions};
use crate::regularized::preprocess::predict;

#[cfg(feature = "wasm")]
use serde::Serialize;

/// Options for ridge regression fitting.
///
/// Configuration options for ridge regression (L2-regularized linear regression).
///
/// # Fields
///
/// - `lambda` - Regularization strength (≥ 0, higher = more shrinkage)
/// - `intercept` - Whether to include an intercept term
/// - `standardize` - Whether to standardize predictors to unit variance
/// - `max_iter` - Maximum coordinate descent iterations
/// - `tol` - Convergence tolerance on coefficient changes
/// - `warm_start` - Optional initial coefficient values for warm starts
/// - `weights` - Optional observation weights
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::ridge::RidgeFitOptions;
/// let options = RidgeFitOptions {
///     lambda: 1.0,
///     intercept: true,
///     standardize: true,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug)]
pub struct RidgeFitOptions {
    pub lambda: f64,
    pub intercept: bool,
    pub standardize: bool,
    pub max_iter: usize, // Added for consistency
    pub tol: f64,        // Added for consistency
    pub warm_start: Option<Vec<f64>>,
    pub weights: Option<Vec<f64>>, // Observation weights
}

impl Default for RidgeFitOptions {
    fn default() -> Self {
        RidgeFitOptions {
            lambda: 1.0,
            intercept: true,
            standardize: true,
            max_iter: 100000,
            tol: 1e-7,
            warm_start: None,
            weights: None,
        }
    }
}

/// Result of a ridge regression fit.
///
/// Contains the fitted model coefficients, predictions, and diagnostic metrics.
///
/// # Fields
///
/// - `lambda` - The regularization strength used
/// - `intercept` - Intercept coefficient (never penalized)
/// - `coefficients` - Slope coefficients (penalized)
/// - `fitted_values` - Predicted values on training data
/// - `residuals` - Residuals (y - fitted_values)
/// - `df` - Approximate effective degrees of freedom
/// - `r_squared` - Coefficient of determination
/// - `adj_r_squared` - Adjusted R²
/// - `mse` - Mean squared error
/// - `rmse` - Root mean squared error
/// - `mae` - Mean absolute error
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::ridge::{ridge_fit, RidgeFitOptions};
/// # use linreg_core::linalg::Matrix;
/// # let y = vec![2.0, 4.0, 6.0, 8.0];
/// # let x = Matrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
/// # let options = RidgeFitOptions { lambda: 0.1, intercept: true, standardize: false, ..Default::default() };
/// let fit = ridge_fit(&x, &y, &options).unwrap();
///
/// // Access model coefficients
/// println!("Intercept: {}", fit.intercept);
/// println!("Slopes: {:?}", fit.coefficients);
///
/// // Access predictions and diagnostics
/// println!("R²: {}", fit.r_squared);
/// println!("RMSE: {}", fit.rmse);
/// # Ok::<(), linreg_core::Error>(())
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(Serialize))]
pub struct RidgeFit {
    pub lambda: f64,
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
    pub df: f64, // Still computed, though approximation
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
}

/// Fits ridge regression for a single lambda value.
///
/// Ridge regression adds an L2 penalty to the coefficients, which helps with
/// multicollinearity and overfitting. The intercept is never penalized.
///
/// # Arguments
///
/// * `x` - Design matrix (n rows × p columns including intercept)
/// * `y` - Response variable (n observations)
/// * `options` - Configuration options for ridge regression
///
/// # Returns
///
/// A `RidgeFit` containing coefficients, fitted values, residuals, and metrics.
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::ridge::{ridge_fit, RidgeFitOptions};
/// # use linreg_core::linalg::Matrix;
/// let y = vec![2.0, 4.0, 6.0, 8.0];
/// let x = Matrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
///
/// let options = RidgeFitOptions {
///     lambda: 0.1,
///     intercept: true,
///     standardize: false,
///     ..Default::default()
/// };
///
/// let fit = ridge_fit(&x, &y, &options).unwrap();
/// assert!(fit.coefficients.len() == 1); // One slope coefficient
/// assert!(fit.r_squared > 0.9); // Good fit for linear data
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn ridge_fit(x: &Matrix, y: &[f64], options: &RidgeFitOptions) -> Result<RidgeFit> {
    // DEBUG: Print lambda info
    // #[cfg(debug_assertions)]
    // {
    //     eprintln!("DEBUG ridge_fit: user_lambda = {}, standardize = {}", options.lambda, options.standardize);
    // }

    let en_options = ElasticNetOptions {
        lambda: options.lambda,
        alpha: 0.0, // Ridge
        intercept: options.intercept,
        standardize: options.standardize,
        max_iter: options.max_iter,
        tol: options.tol,
        penalty_factor: None,
        warm_start: options.warm_start.clone(),
        weights: options.weights.clone(),
        coefficient_bounds: None,
    };

    let fit = elastic_net_fit(x, y, &en_options)?;

    // #[cfg(debug_assertions)]
    // {
    //     eprintln!("DEBUG ridge_fit: fit.intercept = {}, fit.coefficients[0] = {}", fit.intercept,
    //              fit.coefficients.first().unwrap_or(&0.0));
    // }

    // Approximation of degrees of freedom for ridge regression.
    //
    // The true effective df requires SVD: sum(eigenvalues / (eigenvalues + lambda)).
    // Since coordinate descent doesn't compute the SVD, we use a closed-form approximation
    // that works well when X is standardized: df ≈ p / (1 + lambda).
    //
    // This approximation is reasonable for most practical purposes. For exact df,
    // users would need to implement SVD-based calculation separately.
    let p = x.cols;
    let df = (p as f64) / (1.0 + options.lambda);

    Ok(RidgeFit {
        lambda: fit.lambda,
        intercept: fit.intercept,
        coefficients: fit.coefficients,
        fitted_values: fit.fitted_values,
        residuals: fit.residuals,
        df,
        r_squared: fit.r_squared,
        adj_r_squared: fit.adj_r_squared,
        mse: fit.mse,
        rmse: fit.rmse,
        mae: fit.mae,
    })
}

/// Makes predictions using a ridge regression fit.
///
/// Computes predictions for new observations using the fitted ridge regression model.
///
/// # Arguments
///
/// * `fit` - Fitted ridge regression model
/// * `x_new` - New design matrix (same number of columns as training data)
///
/// # Returns
///
/// Vector of predicted values.
///
/// # Example
///
/// ```
/// # use linreg_core::regularized::ridge::{ridge_fit, predict_ridge, RidgeFitOptions};
/// # use linreg_core::linalg::Matrix;
/// // Training data
/// let y = vec![2.0, 4.0, 6.0, 8.0];
/// let x = Matrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
///
/// let options = RidgeFitOptions {
///     lambda: 0.1,
///     intercept: true,
///     standardize: false,
///     ..Default::default()
/// };
/// let fit = ridge_fit(&x, &y, &options).unwrap();
///
/// // Predict on new data
/// let x_new = Matrix::new(2, 2, vec![1.0, 5.0, 1.0, 6.0]);
/// let predictions = predict_ridge(&fit, &x_new);
///
/// assert_eq!(predictions.len(), 2);
/// // Predictions should be close to [10.0, 12.0] for the linear relationship y = 2*x
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn predict_ridge(fit: &RidgeFit, x_new: &Matrix) -> Vec<f64> {
    predict(x_new, fit.intercept, &fit.coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_fit_simple() {
        let x_data = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
        let x = Matrix::new(4, 2, x_data);
        let y = vec![2.0, 4.0, 6.0, 8.0];

        let options = RidgeFitOptions {
            lambda: 0.1,
            intercept: true,
            standardize: false,
            ..Default::default()
        };

        let fit = ridge_fit(&x, &y, &options).unwrap();

        // OLS: intercept ≈ 0, slope ≈ 2
        assert!((fit.coefficients[0] - 2.0).abs() < 0.2);
        assert!(fit.intercept.abs() < 0.5);
    }
}