use super::types::PolynomialFit;
use crate::error::{Error, Result};

/// Predict using a fitted polynomial regression model.
///
/// All centering and standardization parameters are read from `fit` directly.
/// The same transformations applied during training are automatically re-applied
/// to `x_new`.
///
/// # Arguments
///
/// * `fit` - Fitted polynomial model produced by [`polynomial_regression`]
/// * `x_new` - New predictor values to predict at
///
/// # Returns
///
/// Vector of predicted values with the same length as `x_new`.
///
/// # Example
///
/// ```
/// use linreg_core::polynomial::{polynomial_regression, predict, PolynomialOptions};
///
/// let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let options = PolynomialOptions::default();
/// let fit = polynomial_regression(&y, &x, &options).unwrap();
///
/// let predictions = predict(&fit, &[6.0, 7.0]).unwrap();
/// assert!((predictions[0] - 36.0).abs() < 0.1); // x=6 → ~36
/// ```
///
/// [`polynomial_regression`]: crate::polynomial::polynomial_regression
pub fn predict(fit: &PolynomialFit, x_new: &[f64]) -> Result<Vec<f64>> {
    let n = x_new.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Sanity-check: if standardized, we must have stored means/stds for every term
    if fit.standardized && fit.feature_stds.len() != fit.degree {
        return Err(Error::InvalidInput(
            "PolynomialFit has inconsistent standardization state: \
             feature_stds length does not match degree"
                .into(),
        ));
    }

    let coeffs = &fit.ols_output.coefficients;
    let expected_n_coeffs = fit.degree + 1; // intercept + degree terms
    if coeffs.len() != expected_n_coeffs {
        return Err(Error::InvalidInput(format!(
            "PolynomialFit has {} coefficients but expected {}",
            coeffs.len(),
            expected_n_coeffs
        )));
    }

    let mut predictions = Vec::with_capacity(n);

    for &xi in x_new {
        // Step 1: Center using the training mean
        let xi_centered = if fit.centered {
            xi - fit.x_mean
        } else {
            xi
        };

        // Step 2: Build feature vector [1, x, x², x³, …]
        let mut features = Vec::with_capacity(fit.degree + 1);
        features.push(1.0); // intercept

        // Linear x term (feature_means/stds index 0)
        let xi_linear = if fit.standardized {
            (xi_centered - fit.feature_means[0]) / fit.feature_stds[0]
        } else {
            xi_centered
        };
        features.push(xi_linear);

        // Higher-order terms: raise the *centered* x to the power, then standardize
        for d in 2..=fit.degree {
            let xi_poly = xi_centered.powi(d as i32);
            let xi_poly_final = if fit.standardized {
                (xi_poly - fit.feature_means[d - 1]) / fit.feature_stds[d - 1]
            } else {
                xi_poly
            };
            features.push(xi_poly_final);
        }

        // Step 3: ŷ = β₀ + β₁x + β₂x² + …
        let pred: f64 = features
            .iter()
            .zip(coeffs.iter())
            .map(|(&f, &b)| f * b)
            .sum();

        predictions.push(pred);
    }

    Ok(predictions)
}
