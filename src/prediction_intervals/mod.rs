//! Prediction Intervals Module
//!
//! Provides prediction interval functionality for OLS regression.
//! Prediction intervals quantify uncertainty around individual future observations,
//! as opposed to confidence intervals which quantify uncertainty around the mean response.
//!
//! # Formula
//!
//! For a prediction at point x₀:
//! ```text
//! PI = ŷ₀ ± t(α/2, df) × SE_pred
//!
//! SE_pred = √(MSE × (1 + h₀))
//!
//! where h₀ = x₀ᵀ(XᵀX)⁻¹x₀ is the leverage of the new point
//! ```

use crate::core::{compute_leverage, ols_regression, RegressionOutput};
use crate::distributions::student_t_inverse_cdf;
use crate::error::{Error, Result};
use crate::linalg::Matrix;
use crate::regularized::elastic_net::ElasticNetFit;
use crate::regularized::lasso::LassoFit;
use crate::regularized::ridge::RidgeFit;
use serde::{Deserialize, Serialize};

/// Output from prediction interval computation.
#[derive(Serialize, Deserialize)]
pub struct PredictionIntervalOutput {
    /// Point predictions (fitted values) for new observations
    pub predicted: Vec<f64>,
    /// Lower bounds of prediction intervals
    pub lower_bound: Vec<f64>,
    /// Upper bounds of prediction intervals
    pub upper_bound: Vec<f64>,
    /// Standard errors for predictions (includes both parameter uncertainty and residual variance)
    pub se_pred: Vec<f64>,
    /// Leverage values for the new observations
    pub leverage: Vec<f64>,
    /// Significance level used (e.g., 0.05 for 95% PI)
    pub alpha: f64,
    /// Residual degrees of freedom from the fitted model
    pub df_residuals: f64,
}

/// Computes prediction intervals for new observations from raw training data.
///
/// Fits an OLS model internally and then computes prediction intervals for the
/// new observations. This follows the same pattern as diagnostic test functions.
///
/// # Arguments
///
/// * `y` - Response variable from training data
/// * `x_vars` - Predictor variables from training data (each inner slice is one variable)
/// * `new_x` - New predictor values to generate predictions for (each inner slice is one variable)
/// * `alpha` - Significance level (e.g., 0.05 for 95% prediction interval)
///
/// # Returns
///
/// A [`PredictionIntervalOutput`] with predictions and interval bounds.
pub fn prediction_intervals(
    y: &[f64],
    x_vars: &[Vec<f64>],
    new_x: &[&[f64]],
    alpha: f64,
) -> Result<PredictionIntervalOutput> {
    // Build names for OLS
    let mut names = vec!["Intercept".to_string()];
    for i in 0..x_vars.len() {
        names.push(format!("X{}", i + 1));
    }

    let x_refs: Vec<Vec<f64>> = x_vars.to_vec();
    let fit = ols_regression(y, &x_refs, &names)?;

    compute_from_fit(&fit, x_vars, new_x, alpha)
}

/// Computes prediction intervals for new observations using a pre-fitted OLS model.
///
/// Requires the original training predictors to reconstruct (X'X)^{-1} for
/// computing leverage of new points.
///
/// # Arguments
///
/// * `fit_result` - Reference to the fitted OLS model
/// * `x_vars` - Original training predictor variables (each inner slice is one variable)
/// * `new_x` - New predictor values to generate predictions for (each inner slice is one variable)
/// * `alpha` - Significance level (e.g., 0.05 for 95% prediction interval)
///
/// # Returns
///
/// A [`PredictionIntervalOutput`] with predictions and interval bounds.
pub fn compute_from_fit(
    fit_result: &RegressionOutput,
    x_vars: &[Vec<f64>],
    new_x: &[&[f64]],
    alpha: f64,
) -> Result<PredictionIntervalOutput> {
    let n = fit_result.n;
    let k = fit_result.k;
    let p = k + 1; // number of coefficients (including intercept)

    // Validate alpha
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(Error::InvalidInput(
            "alpha must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    // Validate new_x dimensions
    if new_x.len() != k {
        return Err(Error::InvalidInput(format!(
            "new_x has {} variables but model has {} predictors",
            new_x.len(),
            k
        )));
    }

    if new_x.is_empty() {
        return Err(Error::InvalidInput("new_x is empty".to_string()));
    }

    // Get number of new observations and validate consistent lengths
    let n_new = new_x[0].len();
    if n_new == 0 {
        return Err(Error::InvalidInput(
            "new_x variables have zero observations".to_string(),
        ));
    }
    for (i, var) in new_x.iter().enumerate() {
        if var.len() != n_new {
            return Err(Error::InvalidInput(format!(
                "new_x variable {} has {} observations but variable 0 has {}",
                i,
                var.len(),
                n_new
            )));
        }
        for val in var.iter() {
            if !val.is_finite() {
                return Err(Error::InvalidInput(
                    "new_x contains non-finite values".to_string(),
                ));
            }
        }
    }

    // Validate x_vars match the model
    if x_vars.len() != k {
        return Err(Error::InvalidInput(format!(
            "x_vars has {} variables but model has {} predictors",
            x_vars.len(),
            k
        )));
    }

    // Build the training design matrix X (n × p) with intercept column
    let mut x_data = Vec::with_capacity(n * p);
    for i in 0..n {
        x_data.push(1.0); // intercept
        for var in x_vars.iter() {
            x_data.push(var[i]);
        }
    }
    let x_matrix = Matrix::new(n, p, x_data);

    // Compute (X'X)^{-1}
    let xtx = x_matrix.transpose().matmul(&x_matrix);
    let xtx_inv = match xtx.invert() {
        Some(inv) => inv,
        None => {
            return Err(Error::InvalidInput(
                "X'X is singular; cannot compute prediction intervals".to_string(),
            ))
        }
    };

    // Build the new observation design matrix (n_new × p)
    let mut new_x_data = Vec::with_capacity(n_new * p);
    for i in 0..n_new {
        new_x_data.push(1.0); // intercept
        for var in new_x.iter() {
            new_x_data.push(var[i]);
        }
    }
    let new_x_matrix = Matrix::new(n_new, p, new_x_data);

    // Compute leverage for new points: h₀ = x₀ᵀ(XᵀX)⁻¹x₀
    let new_leverage = compute_leverage(&new_x_matrix, &xtx_inv);

    // Extract model parameters
    let df_residuals = fit_result.df as f64;
    let mse = fit_result.mse;
    let beta = &fit_result.coefficients;

    // Critical t-value
    let t_critical = student_t_inverse_cdf(1.0 - alpha / 2.0, df_residuals);

    // Compute predictions and intervals
    let mut predicted = Vec::with_capacity(n_new);
    let mut lower_bound = Vec::with_capacity(n_new);
    let mut upper_bound = Vec::with_capacity(n_new);
    let mut se_pred = Vec::with_capacity(n_new);

    for i in 0..n_new {
        // Compute predicted value: ŷ = x₀ᵀβ
        let mut y_hat = 0.0;
        for j in 0..p {
            let x_val = new_x_matrix.get(i, j);
            let b = beta[j];
            if !b.is_nan() {
                y_hat += x_val * b;
            }
        }
        predicted.push(y_hat);

        // Prediction standard error: SE_pred = √(MSE × (1 + h₀))
        let h = new_leverage[i];
        let se = (mse * (1.0 + h)).sqrt();
        se_pred.push(se);

        // Prediction interval bounds
        let margin = t_critical * se;
        lower_bound.push(y_hat - margin);
        upper_bound.push(y_hat + margin);
    }

    Ok(PredictionIntervalOutput {
        predicted,
        lower_bound,
        upper_bound,
        se_pred,
        leverage: new_leverage,
        alpha,
        df_residuals,
    })
}

/// Shared helper for computing prediction intervals from regularized regression fits.
///
/// Uses the conservative approximation: leverage from unpenalized X'X, MSE from the
/// regularized fit, and effective df from the fit.
fn compute_regularized_pi(
    intercept: f64,
    coefficients: &[f64],
    mse: f64,
    df_residual: f64,
    x_vars: &[Vec<f64>],
    new_x: &[&[f64]],
    alpha: f64,
) -> Result<PredictionIntervalOutput> {
    let k = x_vars.len(); // number of predictors (excluding intercept)

    // Validate alpha
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(Error::InvalidInput(
            "alpha must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    // Validate dimensions
    if new_x.len() != k {
        return Err(Error::InvalidInput(format!(
            "new_x has {} variables but model has {} predictors",
            new_x.len(),
            k
        )));
    }
    if k == 0 || new_x.is_empty() {
        return Err(Error::InvalidInput("new_x is empty".to_string()));
    }

    let n_new = new_x[0].len();
    if n_new == 0 {
        return Err(Error::InvalidInput(
            "new_x variables have zero observations".to_string(),
        ));
    }
    for (i, var) in new_x.iter().enumerate() {
        if var.len() != n_new {
            return Err(Error::InvalidInput(format!(
                "new_x variable {} has {} observations but variable 0 has {}",
                i,
                var.len(),
                n_new
            )));
        }
        for val in var.iter() {
            if !val.is_finite() {
                return Err(Error::InvalidInput(
                    "new_x contains non-finite values".to_string(),
                ));
            }
        }
    }

    if coefficients.len() != k {
        return Err(Error::InvalidInput(format!(
            "coefficients has {} values but model has {} predictors",
            coefficients.len(),
            k
        )));
    }

    // Validate df_residual
    if df_residual <= 0.0 {
        return Err(Error::InvalidInput(
            "Effective degrees of freedom must be positive".to_string(),
        ));
    }

    let n = x_vars[0].len();
    let p = k + 1; // intercept + predictors

    // Build training design matrix (n × p) with intercept column
    let mut x_data = Vec::with_capacity(n * p);
    for i in 0..n {
        x_data.push(1.0);
        for var in x_vars.iter() {
            x_data.push(var[i]);
        }
    }
    let x_matrix = Matrix::new(n, p, x_data);

    // Compute (X'X)^{-1}
    let xtx = x_matrix.transpose().matmul(&x_matrix);
    let xtx_inv = match xtx.invert() {
        Some(inv) => inv,
        None => {
            return Err(Error::InvalidInput(
                "X'X is singular; cannot compute prediction intervals".to_string(),
            ))
        }
    };

    // Build new observation design matrix (n_new × p)
    let mut new_x_data = Vec::with_capacity(n_new * p);
    for i in 0..n_new {
        new_x_data.push(1.0);
        for var in new_x.iter() {
            new_x_data.push(var[i]);
        }
    }
    let new_x_matrix = Matrix::new(n_new, p, new_x_data);

    // Compute leverage for new points
    let new_leverage = compute_leverage(&new_x_matrix, &xtx_inv);

    // Critical t-value
    let t_critical = student_t_inverse_cdf(1.0 - alpha / 2.0, df_residual);

    // Compute predictions and intervals
    let mut predicted = Vec::with_capacity(n_new);
    let mut lower_bound = Vec::with_capacity(n_new);
    let mut upper_bound = Vec::with_capacity(n_new);
    let mut se_pred = Vec::with_capacity(n_new);

    for i in 0..n_new {
        // ŷ = intercept + Σ(coef_j × x_j)
        let mut y_hat = intercept;
        for (j, coef) in coefficients.iter().enumerate() {
            y_hat += coef * new_x[j][i];
        }
        predicted.push(y_hat);

        let h = new_leverage[i];
        let se = (mse * (1.0 + h)).sqrt();
        se_pred.push(se);

        let margin = t_critical * se;
        lower_bound.push(y_hat - margin);
        upper_bound.push(y_hat + margin);
    }

    Ok(PredictionIntervalOutput {
        predicted,
        lower_bound,
        upper_bound,
        se_pred,
        leverage: new_leverage,
        alpha,
        df_residuals: df_residual,
    })
}

/// Computes approximate prediction intervals for Ridge regression.
///
/// Uses the conservative approximation with leverage from unpenalized X'X,
/// MSE from the ridge fit, and effective degrees of freedom from `fit.df`.
///
/// # Arguments
///
/// * `fit` - Reference to the fitted Ridge model
/// * `x_vars` - Original training predictor variables (each inner vec is one variable)
/// * `new_x` - New predictor values (each inner slice is one variable)
/// * `alpha` - Significance level (e.g., 0.05 for 95% prediction interval)
pub fn ridge_prediction_intervals(
    fit: &RidgeFit,
    x_vars: &[Vec<f64>],
    new_x: &[&[f64]],
    alpha: f64,
) -> Result<PredictionIntervalOutput> {
    let n = x_vars.get(0).map_or(0, |v| v.len()) as f64;
    // df_residual = n - 1 - effective_df (where fit.df is approximate effective df)
    let df_residual = n - 1.0 - fit.df;
    compute_regularized_pi(fit.intercept, &fit.coefficients, fit.mse, df_residual, x_vars, new_x, alpha)
}

/// Computes approximate prediction intervals for Lasso regression.
///
/// Uses the conservative approximation with leverage from unpenalized X'X,
/// MSE from the lasso fit, and `n_nonzero` as the effective degrees of freedom.
///
/// # Arguments
///
/// * `fit` - Reference to the fitted Lasso model
/// * `x_vars` - Original training predictor variables (each inner vec is one variable)
/// * `new_x` - New predictor values (each inner slice is one variable)
/// * `alpha` - Significance level (e.g., 0.05 for 95% prediction interval)
pub fn lasso_prediction_intervals(
    fit: &LassoFit,
    x_vars: &[Vec<f64>],
    new_x: &[&[f64]],
    alpha: f64,
) -> Result<PredictionIntervalOutput> {
    let n = x_vars.get(0).map_or(0, |v| v.len()) as f64;
    let df_residual = n - 1.0 - fit.n_nonzero as f64;
    compute_regularized_pi(fit.intercept, &fit.coefficients, fit.mse, df_residual, x_vars, new_x, alpha)
}

/// Computes approximate prediction intervals for Elastic Net regression.
///
/// Uses the conservative approximation with leverage from unpenalized X'X,
/// MSE from the elastic net fit, and `n_nonzero` as the effective degrees of freedom.
///
/// # Arguments
///
/// * `fit` - Reference to the fitted Elastic Net model
/// * `x_vars` - Original training predictor variables (each inner vec is one variable)
/// * `new_x` - New predictor values (each inner slice is one variable)
/// * `alpha` - Significance level (e.g., 0.05 for 95% prediction interval)
pub fn elastic_net_prediction_intervals(
    fit: &ElasticNetFit,
    x_vars: &[Vec<f64>],
    new_x: &[&[f64]],
    alpha: f64,
) -> Result<PredictionIntervalOutput> {
    let n = x_vars.get(0).map_or(0, |v| v.len()) as f64;
    let df_residual = n - 1.0 - fit.n_nonzero as f64;
    compute_regularized_pi(fit.intercept, &fit.coefficients, fit.mse, df_residual, x_vars, new_x, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_intervals_simple() {
        // y = 2x + noise
        let y = vec![3.1, 4.9, 7.2, 8.8, 11.1];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let names = vec!["Intercept".to_string(), "X1".to_string()];
        let fit = ols_regression(&y, &[x1.clone()], &names).unwrap();

        let new_x1 = [6.0];
        let result = compute_from_fit(&fit, &[x1], &[&new_x1], 0.05).unwrap();

        assert_eq!(result.predicted.len(), 1);
        // PI bounds should bracket the prediction
        assert!(result.lower_bound[0] < result.predicted[0]);
        assert!(result.upper_bound[0] > result.predicted[0]);
        assert!(result.se_pred[0] > 0.0);
        assert!((result.alpha - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_intervals_multiple_observations() {
        let y = vec![3.1, 4.9, 7.2, 8.8, 11.1];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let names = vec!["Intercept".to_string(), "X1".to_string()];
        let fit = ols_regression(&y, &[x1.clone()], &names).unwrap();

        // Predict at multiple new points
        let new_x1 = [6.0, 7.0, 3.0];
        let result = compute_from_fit(&fit, &[x1], &[&new_x1], 0.05).unwrap();

        assert_eq!(result.predicted.len(), 3);
        assert_eq!(result.lower_bound.len(), 3);
        assert_eq!(result.upper_bound.len(), 3);

        for i in 0..3 {
            assert!(result.lower_bound[i] < result.predicted[i]);
            assert!(result.upper_bound[i] > result.predicted[i]);
        }
    }

    #[test]
    fn test_prediction_intervals_multiple_predictors() {
        let y = vec![3.0, 5.5, 7.0, 9.5, 11.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 5.0, 6.0, 8.0];

        let names = vec![
            "Intercept".to_string(),
            "X1".to_string(),
            "X2".to_string(),
        ];
        let fit = ols_regression(&y, &[x1.clone(), x2.clone()], &names).unwrap();

        let new_x1 = [6.0];
        let new_x2 = [9.0];
        let result =
            compute_from_fit(&fit, &[x1, x2], &[&new_x1, &new_x2], 0.05).unwrap();

        assert_eq!(result.predicted.len(), 1);
        assert!(result.lower_bound[0] < result.predicted[0]);
        assert!(result.upper_bound[0] > result.predicted[0]);
    }

    #[test]
    fn test_wider_pi_for_lower_alpha() {
        // Lower alpha (higher confidence) should give wider intervals
        let y = vec![1.2, 2.1, 2.8, 4.1, 4.9];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let names = vec!["Intercept".to_string(), "X1".to_string()];
        let fit = ols_regression(&y, &[x1.clone()], &names).unwrap();

        let new_x1 = [3.0];

        let result_95 =
            compute_from_fit(&fit, &[x1.clone()], &[&new_x1], 0.05).unwrap();
        let result_99 =
            compute_from_fit(&fit, &[x1], &[&new_x1], 0.01).unwrap();

        let width_95 = result_95.upper_bound[0] - result_95.lower_bound[0];
        let width_99 = result_99.upper_bound[0] - result_99.lower_bound[0];

        // 99% PI should be wider than 95% PI
        assert!(width_99 > width_95);
    }

    #[test]
    fn test_extrapolation_has_higher_leverage() {
        // Points far from the training data center should have higher leverage
        let y = vec![1.2, 2.1, 2.8, 4.1, 4.9];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let names = vec!["Intercept".to_string(), "X1".to_string()];
        let fit = ols_regression(&y, &[x1.clone()], &names).unwrap();

        // x=3 is at the center, x=10 is extrapolation
        let new_center = [3.0];
        let new_extrap = [10.0];

        let result_center =
            compute_from_fit(&fit, &[x1.clone()], &[&new_center], 0.05).unwrap();
        let result_extrap =
            compute_from_fit(&fit, &[x1], &[&new_extrap], 0.05).unwrap();

        // Extrapolation point should have higher leverage and wider PI
        assert!(result_extrap.leverage[0] > result_center.leverage[0]);
        assert!(result_extrap.se_pred[0] > result_center.se_pred[0]);

        let width_center = result_center.upper_bound[0] - result_center.lower_bound[0];
        let width_extrap = result_extrap.upper_bound[0] - result_extrap.lower_bound[0];
        assert!(width_extrap > width_center);
    }

    #[test]
    fn test_prediction_intervals_convenience_function() {
        let y = vec![3.1, 4.9, 7.2, 8.8, 11.1];
        let x_vars = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

        let new_x1 = [6.0];
        let result = prediction_intervals(&y, &x_vars, &[&new_x1], 0.05).unwrap();

        assert_eq!(result.predicted.len(), 1);
        assert!(result.lower_bound[0] < result.predicted[0]);
        assert!(result.upper_bound[0] > result.predicted[0]);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let names = vec!["Intercept".to_string(), "X1".to_string()];
        let fit = ols_regression(&y, &[x1.clone()], &names).unwrap();

        // Wrong number of predictor variables in new_x
        let new_x1 = [6.0];
        let new_x2 = [7.0];
        let result = compute_from_fit(&fit, &[x1], &[&new_x1, &new_x2], 0.05);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_alpha() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let names = vec!["Intercept".to_string(), "X1".to_string()];
        let fit = ols_regression(&y, &[x1.clone()], &names).unwrap();

        let new_x1 = [6.0];
        assert!(compute_from_fit(&fit, &[x1.clone()], &[&new_x1], 0.0).is_err());
        assert!(compute_from_fit(&fit, &[x1.clone()], &[&new_x1], 1.0).is_err());
        assert!(compute_from_fit(&fit, &[x1], &[&new_x1], -0.1).is_err());
    }

    #[test]
    fn test_se_pred_includes_residual_variance() {
        // SE_pred should always be >= sqrt(MSE) since SE_pred = sqrt(MSE * (1 + h))
        // and h >= 0
        let y = vec![1.2, 2.1, 2.8, 4.1, 4.9];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let names = vec!["Intercept".to_string(), "X1".to_string()];
        let fit = ols_regression(&y, &[x1.clone()], &names).unwrap();

        let new_x1 = [3.0];
        let result = compute_from_fit(&fit, &[x1], &[&new_x1], 0.05).unwrap();

        let sqrt_mse = fit.mse.sqrt();
        assert!(result.se_pred[0] >= sqrt_mse);
    }

    // =========================================================================
    // Regularized prediction interval tests
    // =========================================================================

    #[test]
    fn test_ridge_prediction_intervals_simple() {
        use crate::regularized::ridge::{ridge_fit, RidgeFitOptions};

        let y = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        // Build design matrix with intercept
        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0);
            x_data.push(x1[i]);
        }
        let x = Matrix::new(y.len(), 2, x_data);

        let options = RidgeFitOptions {
            lambda: 0.1,
            intercept: true,
            standardize: true,
            ..Default::default()
        };
        let fit = ridge_fit(&x, &y, &options).unwrap();

        let new_x1 = [8.0];
        let result = ridge_prediction_intervals(&fit, &[x1], &[&new_x1], 0.05).unwrap();

        assert_eq!(result.predicted.len(), 1);
        assert!(result.lower_bound[0] < result.predicted[0]);
        assert!(result.upper_bound[0] > result.predicted[0]);
        assert!(result.se_pred[0] > 0.0);
        // Prediction should be roughly 2*8 + 1 = 17
        assert!((result.predicted[0] - 17.0).abs() < 2.0);
    }

    #[test]
    fn test_lasso_prediction_intervals_basic() {
        use crate::regularized::lasso::{lasso_fit, LassoFitOptions};

        let y = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0);
            x_data.push(x1[i]);
        }
        let x = Matrix::new(y.len(), 2, x_data);

        let options = LassoFitOptions {
            lambda: 0.01,
            intercept: true,
            standardize: true,
            ..Default::default()
        };
        let fit = lasso_fit(&x, &y, &options).unwrap();

        let new_x1 = [8.0];
        let result = lasso_prediction_intervals(&fit, &[x1], &[&new_x1], 0.05).unwrap();

        assert_eq!(result.predicted.len(), 1);
        assert!(result.lower_bound[0] < result.predicted[0]);
        assert!(result.upper_bound[0] > result.predicted[0]);
        assert!(result.se_pred[0] > 0.0);
    }

    #[test]
    fn test_elastic_net_prediction_intervals_basic() {
        use crate::regularized::elastic_net::{elastic_net_fit, ElasticNetOptions};

        let y = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0);
            x_data.push(x1[i]);
        }
        let x = Matrix::new(y.len(), 2, x_data);

        let options = ElasticNetOptions {
            lambda: 0.01,
            alpha: 0.5,
            intercept: true,
            standardize: true,
            ..Default::default()
        };
        let fit = elastic_net_fit(&x, &y, &options).unwrap();

        let new_x1 = [8.0];
        let result = elastic_net_prediction_intervals(&fit, &[x1], &[&new_x1], 0.05).unwrap();

        assert_eq!(result.predicted.len(), 1);
        assert!(result.lower_bound[0] < result.predicted[0]);
        assert!(result.upper_bound[0] > result.predicted[0]);
    }

    #[test]
    fn test_regularized_pi_extrapolation_wider() {
        use crate::regularized::ridge::{ridge_fit, RidgeFitOptions};

        let y = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0);
            x_data.push(x1[i]);
        }
        let x = Matrix::new(y.len(), 2, x_data);

        let options = RidgeFitOptions {
            lambda: 0.1,
            intercept: true,
            standardize: true,
            ..Default::default()
        };
        let fit = ridge_fit(&x, &y, &options).unwrap();

        // Center vs far extrapolation
        let center = [4.0];
        let extrap = [20.0];

        let result_center = ridge_prediction_intervals(&fit, &[x1.clone()], &[&center], 0.05).unwrap();
        let result_extrap = ridge_prediction_intervals(&fit, &[x1], &[&extrap], 0.05).unwrap();

        let width_center = result_center.upper_bound[0] - result_center.lower_bound[0];
        let width_extrap = result_extrap.upper_bound[0] - result_extrap.lower_bound[0];

        assert!(width_extrap > width_center, "Extrapolation PI should be wider");
    }

    #[test]
    fn test_regularized_pi_alpha_comparison() {
        use crate::regularized::ridge::{ridge_fit, RidgeFitOptions};

        let y = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let mut x_data = Vec::new();
        for i in 0..y.len() {
            x_data.push(1.0);
            x_data.push(x1[i]);
        }
        let x = Matrix::new(y.len(), 2, x_data);

        let options = RidgeFitOptions {
            lambda: 0.1,
            intercept: true,
            standardize: true,
            ..Default::default()
        };
        let fit = ridge_fit(&x, &y, &options).unwrap();

        let new_x1 = [8.0];
        let result_95 = ridge_prediction_intervals(&fit, &[x1.clone()], &[&new_x1], 0.05).unwrap();
        let result_99 = ridge_prediction_intervals(&fit, &[x1], &[&new_x1], 0.01).unwrap();

        let width_95 = result_95.upper_bound[0] - result_95.lower_bound[0];
        let width_99 = result_99.upper_bound[0] - result_99.lower_bound[0];

        assert!(width_99 > width_95, "99% PI should be wider than 95% PI");
    }
}
