//! SHAP (SHapley Additive exPlanations) values for linear models.
//!
//! SHAP values decompose predictions into the contribution of each feature.
//! For linear models, this has a closed-form solution:
//!
//! ```text
//! SHAPᵢ = coefᵢ × (xᵢ - mean(xᵢ))
//! ```
//!
//! # Properties
//!
//! - **Local accuracy**: Σ SHAPᵢ + base_value = prediction (exact)
//! - **Missingness**: Features not in model have SHAP = 0
//! - **Consistency**: Guaranteed by exact linear solution
//!
//! # Interpretation for Regularized Models
//!
//! For Ridge, Lasso, and Elastic Net, SHAP values are computed using the
//! regularized coefficients. Interpret with caution because:
//!
//! - Coefficients are biased toward zero due to regularization
//! - SHAP values reflect the biased coefficients
//! - This is still useful for understanding the fitted model's behavior
//!
//! # Example
//!
//! ```
//! # use linreg_core::feature_importance::shap_values_linear;
//! let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];
//! let coefficients = vec![1.0, 0.5, -0.3]; // intercept, coef1, coef2
//!
//! let shap = shap_values_linear(&[x1, x2], &coefficients)?;
//!
//! // shap.shap_values[i][j] = contribution of feature j to observation i
//! // shap.base_value = intercept + mean prediction adjustment
//! # Ok::<(), linreg_core::Error>(())
//! ```

use crate::error::{Error, Result};
use crate::feature_importance::types::ShapOutput;
use crate::stats::mean;

/// Computes exact SHAP values for linear models.
///
/// For linear models, SHAP has a closed-form solution:
/// ```text
/// SHAPᵢ = coefᵢ × (xᵢ - mean(xᵢ))
/// ```
///
/// This provides both local (per-observation) and global importance
/// via mean absolute SHAP values.
///
/// # Arguments
///
/// * `x_vars` - Predictor variables (each Vec<f64> is a column)
/// * `coefficients` - Model coefficients including intercept as first element
///
/// # Returns
///
/// A [`ShapOutput`] containing:
/// - SHAP values matrix (n_observations × n_features)
/// - Base value (mean prediction)
/// - Mean absolute SHAP values per feature (global importance)
///
/// # Errors
///
/// * [`Error::InvalidInput`] - if dimensions don't match or data is invalid
///
/// # Local Accuracy Property
///
/// For any observation i:
/// ```text
/// prediction[i] = base_value + Σⱼ SHAP_values[i][j]
/// ```
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::shap_values_linear;
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];
/// let coefficients = vec![1.0, 0.5, -0.3]; // intercept, coef1, coef2
///
/// let shap = shap_values_linear(&[x1, x2], &coefficients)?;
///
/// // Get SHAP values for first observation
/// let obs0_contrib = shap.observation_contribution(0);
/// println!("X1 contributed: {}", obs0_contrib[0].1);
/// println!("X2 contributed: {}", obs0_contrib[1].1);
///
/// // Get global importance ranking
/// let ranking = shap.ranking();
/// println!("Most important: {}", ranking[0].0);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn shap_values_linear(x_vars: &[Vec<f64>], coefficients: &[f64]) -> Result<ShapOutput> {
    let n_predictors = x_vars.len();

    // Validate dimensions
    if coefficients.len() != n_predictors + 1 {
        return Err(Error::InvalidInput(format!(
            "coefficients length ({}) must equal x_vars length + 1 ({})",
            coefficients.len(),
            n_predictors + 1
        )));
    }

    // Get number of observations from first predictor
    let n_observations = if n_predictors > 0 {
        if let Some(first_var) = x_vars.first() {
            first_var.len()
        } else {
            return Err(Error::InsufficientData {
                required: 1,
                available: 0,
            });
        }
    } else {
        return Ok(ShapOutput {
            variable_names: vec![],
            shap_values: vec![],
            base_value: coefficients[0],
            mean_abs_shap: vec![],
        });
    };

    // Validate all predictors have same length
    for (i, var) in x_vars.iter().enumerate() {
        if var.len() != n_observations {
            return Err(Error::InvalidInput(format!(
                "x_vars[{}] has {} elements, expected {}",
                i, var.len(), n_observations
            )));
        }
    }

    // Compute base value (intercept)
    let base_value = coefficients[0];

    // Compute SHAP values for each observation and feature
    let mut shap_values = vec![vec![0.0; n_predictors]; n_observations];
    let mut mean_abs_shap = vec![0.0; n_predictors];
    let mut variable_names = Vec::with_capacity(n_predictors);

    for j in 0..n_predictors {
        let x_col = &x_vars[j];
        let x_mean = mean(x_col);
        let coef = coefficients[j + 1]; // Skip intercept

        variable_names.push(format!("X{}", j + 1));

        let mut sum_abs = 0.0;
        for i in 0..n_observations {
            // SHAP = coef * (x - mean)
            let shap_val = coef * (x_col[i] - x_mean);
            shap_values[i][j] = shap_val;
            sum_abs += shap_val.abs();
        }

        mean_abs_shap[j] = sum_abs / n_observations as f64;
    }

    Ok(ShapOutput {
        variable_names,
        shap_values,
        base_value,
        mean_abs_shap,
    })
}

/// Computes SHAP values with custom variable names.
///
/// This version allows specifying variable names for clearer output.
///
/// # Arguments
///
/// * `x_vars` - Predictor variables (each Vec<f64> is a column)
/// * `coefficients` - Model coefficients including intercept as first element
/// * `variable_names` - Names for each predictor variable
///
/// # Returns
///
/// A [`ShapOutput`] with the specified variable names.
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::shap_values_linear_named;
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0];
/// let coefficients = vec![1.0, 0.5, -0.3];
/// let names = vec!["Temperature".to_string(), "Pressure".to_string()];
///
/// let shap = shap_values_linear_named(&[x1, x2], &coefficients, &names)?;
/// assert_eq!(shap.variable_names, names);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn shap_values_linear_named(
    x_vars: &[Vec<f64>],
    coefficients: &[f64],
    variable_names: &[String],
) -> Result<ShapOutput> {
    let n_predictors = x_vars.len();

    if variable_names.len() != n_predictors {
        return Err(Error::InvalidInput(format!(
            "variable_names length ({}) must equal x_vars length ({})",
            variable_names.len(),
            n_predictors
        )));
    }

    let mut result = shap_values_linear(x_vars, coefficients)?;
    result.variable_names = variable_names.to_vec();
    Ok(result)
}

/// Computes SHAP values for polynomial regression.
///
/// For polynomial regression with degree d, we have d+1 coefficients (including intercept).
/// The SHAP value for each polynomial term is:
///
/// ```text
/// SHAP_linear = coef_linear × (x - mean_x)
/// SHAP_squared = coef_squared × (x² - mean_x²)
/// SHAP_cubed = coef_cubed × (x³ - mean_x³)
/// ...
/// ```
///
/// # Arguments
///
/// * `x` - Predictor variable values
/// * `fit` - Fitted polynomial regression model
///
/// # Returns
///
/// A [`ShapOutput`] containing:
/// - Variable names ("X", "X²", "X³", ...)
/// - SHAP values matrix (n_observations × degree)
/// - Base value (intercept contribution)
/// - Mean absolute SHAP values per polynomial term
///
/// # Note on Centering
///
/// If the polynomial was fit with centering, the SHAP values are computed
/// on the centered scale. The interpretation remains valid: SHAP represents
/// the contribution of each polynomial term to the prediction.
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::shap_values_polynomial;
/// # use linreg_core::polynomial::{polynomial_regression, PolynomialOptions};
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + 0.5 * xi * xi).collect();
///
/// let options = PolynomialOptions {
///     degree: 2,
///     center: false,
///     ..Default::default()
/// };
/// let fit = polynomial_regression(&y, &x, &options).unwrap();
///
/// let shap = shap_values_polynomial(&x, &fit)?;
///
/// // shap.shap_values[i] gives contributions for observation i
/// // shap.shap_values[i][0] = contribution of linear term
/// // shap.shap_values[i][1] = contribution of squared term
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn shap_values_polynomial(
    x: &[f64],
    fit: &crate::polynomial::PolynomialFit,
) -> Result<ShapOutput> {
    let n = x.len();
    let degree = fit.degree;

    // Validate input
    if n == 0 {
        return Err(Error::InsufficientData {
            required: 1,
            available: 0,
        });
    }

    if fit.ols_output.coefficients.len() != degree + 1 {
        return Err(Error::InvalidInput(format!(
            "PolynomialFit has {} coefficients but degree is {}",
            fit.ols_output.coefficients.len(),
            degree
        )));
    }

    // Compute polynomial features
    let mut poly_features: Vec<Vec<f64>> = vec![vec![0.0; n]; degree];
    let mut x_work = x.to_vec();

    // Apply centering if used during fit
    if fit.centered {
        for xi in &mut x_work {
            *xi -= fit.x_mean;
        }
    }

    // Create polynomial features: x, x², x³, ...
    for i in 0..n {
        let mut val = x_work[i];
        for d in 0..degree {
            if d > 0 {
                val *= x_work[i];
            }
            poly_features[d][i] = val;
        }
    }

    // Apply standardization if used during fit
    if fit.standardized && !fit.feature_means.is_empty() {
        for d in 0..degree {
            let mean = fit.feature_means[d];
            let std = fit.feature_stds[d];
            for i in 0..n {
                poly_features[d][i] = (poly_features[d][i] - mean) / std;
            }
        }
    }

    // Compute SHAP values for each polynomial term
    let coefficients = &fit.ols_output.coefficients;
    let base_value = coefficients[0]; // Intercept

    let mut shap_values = vec![vec![0.0; degree]; n];
    let mut mean_abs_shap = vec![0.0; degree];
    let mut variable_names = Vec::with_capacity(degree);

    // Create variable names: "X", "X²", "X³", ...
    let superscripts = &['\u{2070}', '\u{00B9}', '\u{00B2}', '\u{00B3}', '\u{2074}', '\u{2075}', '\u{2076}', '\u{2077}', '\u{2078}', '\u{2079}'];
    for d in 0..degree {
        let superscript = if d < superscripts.len() {
            superscripts[d]
        } else {
            '^'
        };
        variable_names.push(format!("X{}", superscript));
    }

    for d in 0..degree {
        let poly_col = &poly_features[d];
        let poly_mean = mean(poly_col);
        let coef = coefficients[d + 1]; // Skip intercept

        let mut sum_abs = 0.0;
        for i in 0..n {
            // SHAP = coef * (feature - mean_feature)
            let shap_val = coef * (poly_col[i] - poly_mean);
            shap_values[i][d] = shap_val;
            sum_abs += shap_val.abs();
        }

        mean_abs_shap[d] = sum_abs / n as f64;
    }

    Ok(ShapOutput {
        variable_names,
        shap_values,
        base_value,
        mean_abs_shap,
    })
}

/// Computes SHAP values for Ridge regression.
///
/// This function computes SHAP values using the Ridge regression coefficients.
///
/// # Interpretation Caveat
///
/// Ridge regression shrinks coefficients toward zero, so SHAP values
/// reflect these biased coefficients. This is still useful for understanding
/// the fitted model's behavior, but the magnitude of SHAP values will be
/// attenuated compared to OLS.
///
/// # Arguments
///
/// * `x_vars` - Predictor variables (each Vec<f64> is a column)
/// * `fit` - Fitted Ridge regression model
///
/// # Returns
///
/// A [`ShapOutput`] containing SHAP values and global importance
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::shap_values_ridge;
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
/// let shap = shap_values_ridge(&[vec![1.0,2.0,3.0,4.0,5.0]], &fit)?;
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn shap_values_ridge(
    x_vars: &[Vec<f64>],
    fit: &crate::regularized::RidgeFit,
) -> Result<ShapOutput> {
    shap_values_regularized(x_vars, &fit.coefficients, fit.intercept)
}

/// Computes SHAP values for Lasso regression.
///
/// This function computes SHAP values using the Lasso regression coefficients.
///
/// # Interpretation Caveat
///
/// Lasso regression shrinks coefficients toward zero and can set some
/// coefficients to exactly zero. Features with zero coefficients will have
/// SHAP values of zero (they contribute nothing to predictions).
///
/// # Arguments
///
/// * `x_vars` - Predictor variables (each Vec<f64> is a column)
/// * `fit` - Fitted Lasso regression model
///
/// # Returns
///
/// A [`ShapOutput`] containing SHAP values and global importance
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::shap_values_lasso;
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
/// let shap = shap_values_lasso(&[vec![1.0,2.0,3.0,4.0,5.0]], &fit)?;
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn shap_values_lasso(
    x_vars: &[Vec<f64>],
    fit: &crate::regularized::LassoFit,
) -> Result<ShapOutput> {
    shap_values_regularized(x_vars, &fit.coefficients, fit.intercept)
}

/// Computes SHAP values for Elastic Net regression.
///
/// This function computes SHAP values using the Elastic Net regression coefficients.
///
/// # Interpretation Caveat
///
/// Elastic Net regression combines L1 and L2 regularization, shrinking coefficients
/// toward zero. SHAP values reflect these biased coefficients. This is still useful
/// for understanding the fitted model's behavior.
///
/// # Arguments
///
/// * `x_vars` - Predictor variables (each Vec<f64> is a column)
/// * `fit` - Fitted Elastic Net regression model
///
/// # Returns
///
/// A [`ShapOutput`] containing SHAP values and global importance
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::shap_values_elastic_net;
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
/// let shap = shap_values_elastic_net(&[vec![1.0,2.0,3.0,4.0,5.0]], &fit)?;
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn shap_values_elastic_net(
    x_vars: &[Vec<f64>],
    fit: &crate::regularized::ElasticNetFit,
) -> Result<ShapOutput> {
    shap_values_regularized(x_vars, &fit.coefficients, fit.intercept)
}

/// Helper function for computing SHAP values for regularized models.
///
/// For regularized regression (Ridge/Lasso/ElasticNet), we use the same
/// closed-form SHAP formula as OLS, but with the regularized coefficients.
///
/// Note: This assumes the original (unscaled) data is provided. The SHAP
/// values are computed on the original scale for interpretability.
fn shap_values_regularized(
    x_vars: &[Vec<f64>],
    coefficients: &[f64],
    intercept: f64,
) -> Result<ShapOutput> {
    let n_predictors = x_vars.len();

    // Validate dimensions
    if coefficients.len() != n_predictors {
        return Err(Error::InvalidInput(format!(
            "coefficients length ({}) must equal x_vars length ({})",
            coefficients.len(),
            n_predictors
        )));
    }

    // Get number of observations from first predictor
    let n_observations = if n_predictors > 0 {
        if let Some(first_var) = x_vars.first() {
            first_var.len()
        } else {
            return Ok(ShapOutput {
                variable_names: vec![],
                shap_values: vec![],
                base_value: intercept,
                mean_abs_shap: vec![],
            });
        }
    } else {
        return Ok(ShapOutput {
            variable_names: vec![],
            shap_values: vec![],
            base_value: intercept,
            mean_abs_shap: vec![],
        });
    };

    // Validate all predictors have same length
    for (i, var) in x_vars.iter().enumerate() {
        if var.len() != n_observations {
            return Err(Error::InvalidInput(format!(
                "x_vars[{}] has {} elements, expected {}",
                i, var.len(), n_observations
            )));
        }
    }

    // Compute SHAP values for each observation and feature
    let mut shap_values = vec![vec![0.0; n_predictors]; n_observations];
    let mut mean_abs_shap = vec![0.0; n_predictors];
    let mut variable_names = Vec::with_capacity(n_predictors);

    for j in 0..n_predictors {
        let x_col = &x_vars[j];
        let x_mean = mean(x_col);
        let coef = coefficients[j];

        variable_names.push(format!("X{}", j + 1));

        let mut sum_abs = 0.0;
        for i in 0..n_observations {
            // SHAP = coef * (x - mean)
            let shap_val = coef * (x_col[i] - x_mean);
            shap_values[i][j] = shap_val;
            sum_abs += shap_val.abs();
        }

        mean_abs_shap[j] = sum_abs / n_observations as f64;
    }

    Ok(ShapOutput {
        variable_names,
        shap_values,
        base_value: intercept,
        mean_abs_shap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shap_values_linear_basic() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let coefficients = vec![1.0, 2.0, 3.0];

        let shap = shap_values_linear(&[x1.clone(), x2.clone()], &coefficients).unwrap();

        assert_eq!(shap.variable_names, vec!["X1", "X2"]);
        assert_eq!(shap.shap_values.len(), 5);
        assert_eq!(shap.shap_values[0].len(), 2);
        assert!(shap.base_value.is_finite());
    }

    #[test]
    fn test_shap_values_constant_feature() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let coefficients = vec![1.0, 0.5, -0.3];

        let shap = shap_values_linear(&[x1.clone(), x2.clone()], &coefficients).unwrap();

        for obs in &shap.shap_values {
            if obs[1].is_finite() {
                assert_eq!(obs[1], 0.0);
            }
        }
    }

    #[test]
    fn test_shap_ranking() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let coefficients = vec![1.0, 2.0, 3.0];

        let shap = shap_values_linear(&[x1, x2], &coefficients).unwrap();
        let ranking = shap.ranking();

        assert_eq!(ranking[0].0, "X1");
        assert!(ranking[0].1 > ranking[1].1);
    }
}
