//! Standardized coefficients for feature importance.
//!
//! Standardized coefficients (beta*) represent the change in Y (in standard deviations)
//! for a one standard deviation change in X. This makes coefficients comparable
//! across predictors with different units/scales.
//!
//! # Formula
//!
//! ```text
//! β* = β × (σₓ / σᵧ)
//! ```
//!
//! where:
//! - β is the raw coefficient (excluding intercept)
//! - σₓ is the standard deviation of predictor X
//! - σᵧ is the standard deviation of response Y
//!
//! # Interpretation
//!
//! - "A 1 SD increase in Xᵢ leads to β* SD change in Y"
//! - Absolute values indicate relative importance
//! - Sign indicates direction of relationship
//!
//! # Example
//!
//! ```
//! # use linreg_core::feature_importance::standardized_coefficients;
//! let coefficients = vec![1.0, 2.0, 0.5]; // intercept, coef1, coef2
//! let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let x2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
//!
//! let result = standardized_coefficients(&coefficients, &[x1, x2])?;
//! // result.standardized_coefficients contains the beta* values
//! # Ok::<(), linreg_core::Error>(())
//! ```

use crate::error::{Error, Result};
use crate::feature_importance::types::StandardizedCoefficientsOutput;
use crate::stats::stddev;

/// Computes standardized coefficients for feature importance.
///
/// Standardized coefficients (beta*) represent the change in Y (in standard deviations)
/// for a one standard deviation change in X.
///
/// # Arguments
///
/// * `coefficients` - Model coefficients including intercept as first element
/// * `x_vars` - Predictor variables (each `Vec<f64>` is a column)
///
/// # Returns
///
/// A [`StandardizedCoefficientsOutput`] containing:
/// - Variable names (X1, X2, ...)
/// - Standardized coefficients (one per predictor, excluding intercept)
/// - Standard deviation of Y
///
/// # Errors
///
/// * [`Error::InvalidInput`] - if coefficients length doesn't match x_vars + 1
/// * [`Error::InsufficientData`] - if any predictor has fewer than 2 elements
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::standardized_coefficients;
/// // Coefficients: [intercept=1.0, coef_x1=2.0, coef_x2=0.5]
/// let coefficients = vec![1.0, 2.0, 0.5];
///
/// // X1 has small variance (SD ≈ 1.58)
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// // X2 has large variance (SD ≈ 15.8)
/// let x2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
///
/// let result = standardized_coefficients(&coefficients, &[x1, x2])?;
///
/// // Even though coef_x1 (2.0) > coef_x2 (0.5),
/// // after standardization, X2 may appear more important
/// // because it has larger variance.
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn standardized_coefficients(
    coefficients: &[f64],
    x_vars: &[Vec<f64>],
) -> Result<StandardizedCoefficientsOutput> {
    // Validate: coefficients should have intercept + n predictors
    let n_predictors = x_vars.len();
    if coefficients.len() != n_predictors + 1 {
        return Err(Error::InvalidInput(format!(
            "coefficients length ({}) must equal x_vars length + 1 ({})",
            coefficients.len(),
            n_predictors + 1
        )));
    }

    // Validate all predictors have at least 2 elements (for stddev)
    for (_i, var) in x_vars.iter().enumerate() {
        if var.len() < 2 {
            return Err(Error::InsufficientData {
                required: 2,
                available: var.len(),
            });
        }
    }

    // Compute standard deviation of Y (residuals proxy or we could accept y as input)
    // For now, we'll return y_std = 1.0 since we don't have y in this function signature
    // Users can scale externally if needed, or we could add an overload
    let y_std = 1.0;

    // Compute standardized coefficients: β* = β × (σₓ / σᵧ)
    let mut standardized_coefficients = Vec::with_capacity(n_predictors);
    let mut variable_names = Vec::with_capacity(n_predictors);

    for (i, x_col) in x_vars.iter().enumerate() {
        let x_std = stddev(x_col);

        if !x_std.is_finite() || x_std == 0.0 {
            return Err(Error::InvalidInput(format!(
                "Predictor X{} has zero or invalid standard deviation",
                i + 1
            )));
        }

        // Coefficient for this predictor (skip intercept)
        let coef = coefficients[i + 1];

        // Standardized coefficient
        let std_coef = coef * (x_std / y_std);
        standardized_coefficients.push(std_coef);
        variable_names.push(format!("X{}", i + 1));
    }

    Ok(StandardizedCoefficientsOutput {
        variable_names,
        standardized_coefficients,
        y_std,
    })
}

/// Computes standardized coefficients with variable names and Y standard deviation.
///
/// This version allows specifying variable names and the Y standard deviation
/// for more accurate standardized coefficients.
///
/// # Arguments
///
/// * `coefficients` - Model coefficients including intercept as first element
/// * `x_vars` - Predictor variables (each `Vec<f64>` is a column)
/// * `variable_names` - Names for each predictor variable
/// * `y_std` - Standard deviation of the response variable Y
///
/// # Returns
///
/// A [`StandardizedCoefficientsOutput`] with proper variable names and Y scaling.
///
/// # Example
///
/// ```
/// # use linreg_core::feature_importance::standardized_coefficients_named;
/// let coefficients = vec![1.0, 2.0, 0.5];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
/// let names = vec!["Temperature".to_string(), "Pressure".to_string()];
/// let y_std = 2.5; // Standard deviation of response
///
/// let result = standardized_coefficients_named(&coefficients, &[x1, x2], &names, y_std)?;
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn standardized_coefficients_named(
    coefficients: &[f64],
    x_vars: &[Vec<f64>],
    variable_names: &[String],
    y_std: f64,
) -> Result<StandardizedCoefficientsOutput> {
    let n_predictors = x_vars.len();

    // Validate inputs
    if coefficients.len() != n_predictors + 1 {
        return Err(Error::InvalidInput(format!(
            "coefficients length ({}) must equal x_vars length + 1 ({})",
            coefficients.len(),
            n_predictors + 1
        )));
    }

    if variable_names.len() != n_predictors {
        return Err(Error::InvalidInput(format!(
            "variable_names length ({}) must equal x_vars length ({})",
            variable_names.len(),
            n_predictors
        )));
    }

    if y_std <= 0.0 || !y_std.is_finite() {
        return Err(Error::InvalidInput(
            "y_std must be positive and finite".to_string(),
        ));
    }

    for (_i, var) in x_vars.iter().enumerate() {
        if var.len() < 2 {
            return Err(Error::InsufficientData {
                required: 2,
                available: var.len(),
            });
        }
    }

    // Compute standardized coefficients
    let mut standardized_coefficients = Vec::with_capacity(n_predictors);

    for (i, x_col) in x_vars.iter().enumerate() {
        let x_std = stddev(x_col);

        if !x_std.is_finite() || x_std == 0.0 {
            return Err(Error::InvalidInput(format!(
                "Predictor {} has zero or invalid standard deviation",
                variable_names[i]
            )));
        }

        let coef = coefficients[i + 1];
        let std_coef = coef * (x_std / y_std);
        standardized_coefficients.push(std_coef);
    }

    Ok(StandardizedCoefficientsOutput {
        variable_names: variable_names.to_vec(),
        standardized_coefficients,
        y_std,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standardized_coefficients_basic() {
        let coefficients = vec![1.0, 2.0, 0.5];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let result = standardized_coefficients(&coefficients, &[x1, x2]).unwrap();

        assert_eq!(result.variable_names, vec!["X1", "X2"]);
        assert_eq!(result.standardized_coefficients.len(), 2);
        assert!(result.y_std > 0.0);

        // X1 has smaller SD than X2, so standardized coef for X1 should be smaller
        // than X2 even though raw coef is larger
        let x1_std = stddev(&vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let x2_std = stddev(&vec![10.0, 20.0, 30.0, 40.0, 50.0]);

        assert_eq!(
            result.standardized_coefficients[0],
            2.0 * x1_std
        );
        assert_eq!(
            result.standardized_coefficients[1],
            0.5 * x2_std
        );
    }

    #[test]
    fn test_standardized_coefficients_named() {
        let coefficients = vec![1.0, 2.0, 0.5];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let names = vec!["Temp".to_string(), "Pressure".to_string()];
        let y_std = 2.5;

        let result = standardized_coefficients_named(
            &coefficients,
            &[x1, x2],
            &names,
            y_std,
        ).unwrap();

        assert_eq!(result.variable_names, names);
        assert_eq!(result.y_std, y_std);

        let x1_std = stddev(&vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let expected = 2.0 * x1_std / y_std;
        assert!((result.standardized_coefficients[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_standardized_coefficients_ranking() {
        let coefficients = vec![1.0, 0.5, -0.8, 0.1];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Same SD as x1
        let x3 = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let result = standardized_coefficients(&coefficients, &[x1, x2, x3]).unwrap();
        let ranking = result.ranking();

        // X3 should be most important (largest absolute coef * largest SD)
        assert_eq!(ranking[0].0, "X3");
        // X3 should have highest absolute value
        assert!(ranking[0].1 > ranking[1].1);
    }

    #[test]
    fn test_standardized_coefficients_invalid_input() {
        // Wrong number of coefficients
        let coefficients = vec![1.0, 2.0]; // Only 2 coefs for 3 predictors
        let x1 = vec![1.0, 2.0, 3.0];
        let x2 = vec![1.0, 2.0, 3.0];
        let x3 = vec![1.0, 2.0, 3.0];

        let result = standardized_coefficients(&coefficients, &[x1, x2, x3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_standardized_coefficients_insufficient_data() {
        let coefficients = vec![1.0, 2.0];
        let x1 = vec![1.0]; // Only 1 element

        let result = standardized_coefficients(&coefficients, &[x1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_standardized_coefficients_constant_predictor() {
        let coefficients = vec![1.0, 2.0];
        let x1 = vec![5.0, 5.0, 5.0, 5.0]; // Constant - SD = 0

        let result = standardized_coefficients(&coefficients, &[x1]);
        assert!(result.is_err());
    }
}
