use super::features::{compute_mean, compute_std, polynomial_feature_names, polynomial_features};
use super::types::{PolynomialFit, PolynomialOptions};
use crate::core::ols_regression;
use crate::error::{Error, Result};

/// Fit polynomial regression for a single predictor.
///
/// Polynomial regression models the relationship between `y` and `x` as:
///
/// ```text
/// y = β₀ + β₁x + β₂x² + … + β_d·x^d + ε
/// ```
///
/// This is still linear in the parameters `β`, so the existing OLS solver is used
/// after transforming `x` into polynomial features.
///
/// # Arguments
///
/// * `y` - Response variable (n observations)
/// * `x` - Single predictor variable (n observations)
/// * `options` - Fitting options (degree, centering, standardization)
///
/// # Returns
///
/// [`PolynomialFit`] containing the OLS output plus polynomial metadata.
///
/// # Example
///
/// ```
/// use linreg_core::polynomial::{polynomial_regression, PolynomialOptions};
///
/// let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// let options = PolynomialOptions { degree: 2, ..Default::default() };
/// let fit = polynomial_regression(&y, &x, &options).unwrap();
/// assert!((fit.ols_output.r_squared - 1.0).abs() < 1e-6);
/// ```
pub fn polynomial_regression(
    y: &[f64],
    x: &[f64],
    options: &PolynomialOptions,
) -> Result<PolynomialFit> {
    // --- Input validation ---
    if y.len() != x.len() {
        return Err(Error::DimensionMismatch(format!(
            "Length of y ({}) must match length of x ({})",
            y.len(),
            x.len()
        )));
    }

    if y.len() < 2 {
        return Err(Error::InsufficientData {
            required: 2,
            available: y.len(),
        });
    }

    if options.degree < 1 {
        return Err(Error::InvalidInput(
            "Polynomial degree must be at least 1".into(),
        ));
    }

    // --- Step 1: Center x if requested ---
    let x_mean = if options.center {
        compute_mean(x)
    } else {
        0.0
    };

    let x_centered: Vec<f64> = if options.center {
        x.iter().map(|&xi| xi - x_mean).collect()
    } else {
        x.to_vec()
    };

    // --- Step 2: Build higher-order features (x², x³, …) from centered x ---
    // polynomial_features returns degrees 2..=degree; the linear term is added separately.
    let poly_features_raw = polynomial_features(
        &x_centered,
        options.degree,
        false, // already centered above
        0.0,
    )?;

    // --- Step 3: Optionally standardize all features, recording per-feature stats ---
    let mut feature_means: Vec<f64> = Vec::new();
    let mut feature_stds: Vec<f64> = Vec::new();

    // Standardize (or keep as-is) the linear x term
    let x_feature: Vec<f64> = if options.standardize {
        let mean = compute_mean(&x_centered);
        let std_val = compute_std(&x_centered, mean);
        if std_val < 1e-10 {
            return Err(Error::InvalidInput(
                "Cannot standardize x: standard deviation is too small or zero".into(),
            ));
        }
        feature_means.push(mean);
        feature_stds.push(std_val);
        x_centered.iter().map(|&v| (v - mean) / std_val).collect()
    } else {
        x_centered.clone()
    };

    // Standardize (or keep as-is) each higher-order polynomial feature
    let mut poly_features_processed: Vec<Vec<f64>> =
        Vec::with_capacity(poly_features_raw.len());
    for feature in &poly_features_raw {
        if options.standardize {
            let mean = compute_mean(feature);
            let std_val = compute_std(feature, mean);
            if std_val < 1e-10 {
                // If variance is zero the feature is constant; push zeros to avoid NaN
                feature_means.push(mean);
                feature_stds.push(1.0);
                poly_features_processed
                    .push(feature.iter().map(|_| 0.0).collect());
            } else {
                feature_means.push(mean);
                feature_stds.push(std_val);
                poly_features_processed
                    .push(feature.iter().map(|&v| (v - mean) / std_val).collect());
            }
        } else {
            poly_features_processed.push(feature.clone());
        }
    }

    // --- Step 4: Assemble predictor matrix as Vec<Vec<f64>> for ols_regression ---
    let mut all_predictor_vecs: Vec<Vec<f64>> = Vec::with_capacity(options.degree);
    all_predictor_vecs.push(x_feature);
    all_predictor_vecs.extend(poly_features_processed);

    // --- Step 5: Build variable names (Intercept + one name per polynomial term) ---
    let feature_names = polynomial_feature_names(options.degree, options.center);
    let mut all_names = vec!["Intercept".to_string()];
    all_names.extend(feature_names.iter().cloned());

    // --- Step 6: Fit OLS ---
    let ols_output = ols_regression(y, &all_predictor_vecs, &all_names)?;

    let x_std = feature_stds.first().copied().unwrap_or(1.0);

    Ok(PolynomialFit {
        ols_output,
        degree: options.degree,
        centered: options.center,
        x_mean,
        x_std,
        standardized: options.standardize,
        n_features: options.degree,
        feature_names: all_names,
        feature_means,
        feature_stds,
    })
}
