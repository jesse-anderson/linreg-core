use super::features::{compute_mean, polynomial_features};
use crate::error::{Error, Result};
use crate::linalg::Matrix;
use crate::regularized::elastic_net::{elastic_net_fit, ElasticNetFit, ElasticNetOptions};
use crate::regularized::lasso::{lasso_fit, LassoFit, LassoFitOptions};
use crate::regularized::ridge::{ridge_fit, RidgeFit, RidgeFitOptions};

/// Build the polynomial design matrix (with intercept column first).
///
/// Layout: `[1, x, x², x³, …, x^degree]` (n rows × (degree+1) columns).
fn build_poly_matrix(
    x: &[f64],
    degree: usize,
    center: bool,
    x_mean: f64,
) -> Result<Matrix> {
    let n = x.len();
    let cols = degree + 1; // intercept + degree predictor columns

    // polynomial_features returns degrees 2..=degree
    let higher_features = polynomial_features(x, degree, center, x_mean)?;

    let mut data = vec![0.0f64; n * cols];

    for row in 0..n {
        // Intercept column (index 0)
        data[row * cols] = 1.0;

        // Linear x term (index 1)
        let xi_c = if center { x[row] - x_mean } else { x[row] };
        data[row * cols + 1] = xi_c;

        // Higher-order terms (index 2 .. degree)
        for (col_idx, feature) in higher_features.iter().enumerate() {
            data[row * cols + col_idx + 2] = feature[row];
        }
    }

    Ok(Matrix::new(n, cols, data))
}

/// Fit polynomial Ridge regression (L2 penalty).
///
/// Ridge regularization helps with multicollinearity in polynomial features.
///
/// # Arguments
///
/// * `y` - Response variable
/// * `x` - Single predictor
/// * `degree` - Polynomial degree (≥ 1)
/// * `lambda` - Regularization strength (≥ 0)
/// * `center` - Whether to center `x` before expansion
/// * `standardize` - Whether to standardize features (recommended)
///
/// # Returns
///
/// [`RidgeFit`] with coefficients on the original (or centered) scale.
pub fn polynomial_ridge(
    y: &[f64],
    x: &[f64],
    degree: usize,
    lambda: f64,
    center: bool,
    standardize: bool,
) -> Result<RidgeFit> {
    if degree < 1 {
        return Err(Error::InvalidInput(
            "Polynomial degree must be at least 1".into(),
        ));
    }
    if y.len() != x.len() {
        return Err(Error::DimensionMismatch(format!(
            "Length of y ({}) must match length of x ({})",
            y.len(),
            x.len()
        )));
    }

    let x_mean = if center { compute_mean(x) } else { 0.0 };
    let x_matrix = build_poly_matrix(x, degree, center, x_mean)?;

    let options = RidgeFitOptions {
        lambda,
        intercept: true,
        standardize,
        ..Default::default()
    };

    ridge_fit(&x_matrix, y, &options)
}

/// Fit polynomial Lasso regression (L1 penalty).
///
/// Lasso can perform variable selection among polynomial terms,
/// potentially eliminating higher-order terms.
///
/// # Arguments
///
/// * `y` - Response variable
/// * `x` - Single predictor
/// * `degree` - Polynomial degree (≥ 1)
/// * `lambda` - Regularization strength (≥ 0)
/// * `center` - Whether to center `x` before expansion
/// * `standardize` - Whether to standardize features (recommended)
pub fn polynomial_lasso(
    y: &[f64],
    x: &[f64],
    degree: usize,
    lambda: f64,
    center: bool,
    standardize: bool,
) -> Result<LassoFit> {
    if degree < 1 {
        return Err(Error::InvalidInput(
            "Polynomial degree must be at least 1".into(),
        ));
    }
    if y.len() != x.len() {
        return Err(Error::DimensionMismatch(format!(
            "Length of y ({}) must match length of x ({})",
            y.len(),
            x.len()
        )));
    }

    let x_mean = if center { compute_mean(x) } else { 0.0 };
    let x_matrix = build_poly_matrix(x, degree, center, x_mean)?;

    let options = LassoFitOptions {
        lambda,
        intercept: true,
        standardize,
        ..Default::default()
    };

    lasso_fit(&x_matrix, y, &options)
}

/// Fit polynomial Elastic Net regression (L1 + L2 penalty).
///
/// # Arguments
///
/// * `y` - Response variable
/// * `x` - Single predictor
/// * `degree` - Polynomial degree (≥ 1)
/// * `lambda` - Regularization strength (≥ 0)
/// * `alpha` - Mixing parameter: 0 = Ridge, 1 = Lasso
/// * `center` - Whether to center `x` before expansion
/// * `standardize` - Whether to standardize features (recommended)
pub fn polynomial_elastic_net(
    y: &[f64],
    x: &[f64],
    degree: usize,
    lambda: f64,
    alpha: f64,
    center: bool,
    standardize: bool,
) -> Result<ElasticNetFit> {
    if degree < 1 {
        return Err(Error::InvalidInput(
            "Polynomial degree must be at least 1".into(),
        ));
    }
    if y.len() != x.len() {
        return Err(Error::DimensionMismatch(format!(
            "Length of y ({}) must match length of x ({})",
            y.len(),
            x.len()
        )));
    }

    let x_mean = if center { compute_mean(x) } else { 0.0 };
    let x_matrix = build_poly_matrix(x, degree, center, x_mean)?;

    let options = ElasticNetOptions {
        lambda,
        alpha,
        intercept: true,
        standardize,
        ..Default::default()
    };

    elastic_net_fit(&x_matrix, y, &options)
}
