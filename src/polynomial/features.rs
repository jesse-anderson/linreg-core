use crate::error::{Error, Result};

/// Expand a single predictor into higher-order polynomial features.
///
/// Returns feature vectors for degrees **2 through `degree`** only. The linear
/// x term (degree 1) is intentionally excluded; the caller adds it separately.
///
/// # Arguments
///
/// * `x` - Predictor variable (n observations, already centered if desired)
/// * `degree` - Polynomial degree (must be ≥ 1)
/// * `center` - Whether to center `x` before raising to each power
/// * `x_mean` - Pre-computed mean of `x` (used only when `center = true`)
///
/// # Returns
///
/// A `Vec` of length `degree - 1` where element `i` corresponds to `x^(i+2)`.
///
/// # Example
///
/// ```
/// use linreg_core::polynomial::features::polynomial_features;
///
/// let x = vec![1.0, 2.0, 3.0];
/// let features = polynomial_features(&x, 3, false, 0.0).unwrap();
/// // features[0] = [1.0, 4.0, 9.0]    (x²)
/// // features[1] = [1.0, 8.0, 27.0]   (x³)
/// ```
pub fn polynomial_features(
    x: &[f64],
    degree: usize,
    center: bool,
    x_mean: f64,
) -> Result<Vec<Vec<f64>>> {
    if degree < 1 {
        return Err(Error::InvalidInput(
            "Polynomial degree must be at least 1".into(),
        ));
    }

    let n = x.len();
    if n < 2 {
        return Err(Error::InsufficientData {
            required: 2,
            available: n,
        });
    }

    for (i, &xi) in x.iter().enumerate() {
        if !xi.is_finite() {
            return Err(Error::InvalidInput(format!(
                "x[{}] is not finite: {}",
                i, xi
            )));
        }
    }

    let x_centered: Vec<f64> = if center {
        x.iter().map(|&xi| xi - x_mean).collect()
    } else {
        x.to_vec()
    };

    // Create polynomial features: x², x³, …, x^degree
    let mut features = Vec::with_capacity(degree.saturating_sub(1));
    for d in 2..=degree {
        let power = d as i32;
        let poly_feature: Vec<f64> = x_centered.iter().map(|&xi| xi.powi(power)).collect();

        for (i, &val) in poly_feature.iter().enumerate() {
            if !val.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "x^{} at index {} is not finite: {}",
                    d, i, val
                )));
            }
        }

        features.push(poly_feature);
    }

    Ok(features)
}

/// Generate feature names for polynomial regression terms.
///
/// Returns names for the linear term and all higher-order terms:
/// `["x", "x^2", "x^3", …]` (or centered variants).
pub fn polynomial_feature_names(degree: usize, centered: bool) -> Vec<String> {
    let mut names = Vec::with_capacity(degree);

    if centered {
        names.push("x_centered".to_string());
        for d in 2..=degree {
            names.push(format!("x^{}_centered", d));
        }
    } else {
        names.push("x".to_string());
        for d in 2..=degree {
            names.push(format!("x^{}", d));
        }
    }

    names
}

/// Compute the mean of a slice.
pub fn compute_mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Compute the sample standard deviation of a slice given a pre-computed mean.
pub fn compute_std(x: &[f64], x_mean: f64) -> f64 {
    let variance = x
        .iter()
        .map(|&xi| (xi - x_mean).powi(2))
        .sum::<f64>()
        / (x.len() - 1) as f64;
    variance.sqrt()
}
