// ============================================================================
// Rainbow Test for Linearity
// ============================================================================
//
// H0: The relationship is linear
// H1: The relationship is non-linear (functional misspecification)
//
// Reference:
// - Utts, J. M. (1982). "The Rainbow Test for Lack of Fit in Regression".
//   Communications in Statistics - Theory and Methods, 11(24), 2801-2815.
//
// Supports three methods:
// - R: Uses lmtest::raintest algorithm (Type 7 quantile with interpolation)
// - Python: Uses statsmodels algorithm (direct formula)
// - Both: Returns both R and Python results

use super::helpers::{compute_rss, f_p_value, fit_ols};
use super::types::{RainbowMethod, RainbowSingleResult, RainbowTestOutput};
use crate::error::{Error, Result};
use crate::linalg::Matrix;

/// R's lmtest::raintest subset selection using Type 7 quantile
///
/// R uses 1-based indexing and Type 7 quantile (linear interpolation):
/// - Q(p) = x\[j\] + gamma * (x\[j+1\] - x\[j\])
/// - where j = floor(p * (n-1) + 1), gamma = p * (n-1) + 1 - j
/// - For sequence 1:n, this simplifies to: Q(p) = j + gamma
fn raintest_subset_r(n: usize, fraction: f64, center: f64) -> (usize, usize) {
    let n_f = n as f64;

    // Type 7 quantile computation for sequence 1:n
    // p = center - fraction/2 (lower bound probability)
    let p = center - fraction / 2.0;

    // For Type 7 quantile: j = floor(p * (n-1) + 1), gamma = p * (n-1) + 1 - j
    // Since x[i] = i for sequence 1:n, Q(p) = j + gamma = p * (n-1) + 1
    let idx_float = p * (n_f - 1.0) + 1.0;

    // R uses ceiling() on the result
    let from_1based = idx_float.ceil() as usize;

    // Upper bound: from + floor(fraction * n) - 1
    let to_1based = from_1based + (fraction * n_f).floor() as usize - 1;

    // Convert to 0-based for Rust
    (from_1based - 1, to_1based - 1)
}

/// Python's statsmodels subset selection using direct formula
///
/// Python statsmodels uses ceil for lower bound (not floor):
/// - lowidx = ceil(0.5 * (1 - frac) * nobs)
/// - uppidx = floor(lowidx + frac * nobs)
///   The uppidx is exclusive (not inclusive) in Python
fn raintest_subset_python(n: usize, fraction: f64) -> (usize, usize) {
    let n_f = n as f64;

    // Python statsmodels uses ceil for lowidx (not floor)
    let lowidx = (0.5 * (1.0 - fraction) * n_f).ceil() as usize;
    // uppidx is exclusive, so we subtract 1 for Rust's inclusive range
    let uppidx_excl = (lowidx as f64 + fraction * n_f).floor() as usize;

    (lowidx, uppidx_excl - 1)
}

/// Internal function to compute the Rainbow test statistic given data indices
///
/// For R method: indices are sorted by fitted values, subset is central portion
/// For Python method: indices are in original order (NOT sorted), subset is central portion
fn rainbow_test_internal(
    y: &[f64],
    x_data: &[f64],
    x_full: &Matrix,
    beta_full: &[f64],
    indices: &[usize],
    subset_range: (usize, usize),
    _method: RainbowMethod,
) -> Result<(f64, f64)> {
    let n = y.len();
    let p = x_full.cols;

    let rss_full = compute_rss(y, x_full, beta_full)?;

    // Get subset indices from indices using the range
    let (start, end) = subset_range;
    let subset_indices = &indices[start..=end];

    if subset_indices.len() < p {
        return Err(Error::InsufficientData {
            required: p,
            available: subset_indices.len(),
        });
    }

    // Create subset data
    let mut y_subset = Vec::with_capacity(subset_indices.len());
    let mut x_subset_data = Vec::with_capacity(subset_indices.len() * p);

    for &idx in subset_indices {
        y_subset.push(y[idx]);
        for j in 0..p {
            x_subset_data.push(x_data[idx * p + j]);
        }
    }

    let x_subset = Matrix::new(subset_indices.len(), p, x_subset_data);

    // Fit OLS on subset
    let beta_subset = fit_ols(&y_subset, &x_subset)?;

    // FIX: Compute RSS of subset model on SUBSET data only (not full data)
    // This matches both R's lmtest::raintest and Python's statsmodels.linear_rainbow
    let predictions_subset = x_subset.mul_vec(&beta_subset);
    let residuals_subset: Vec<f64> = y_subset
        .iter()
        .zip(predictions_subset.iter())
        .map(|(&yi, &yi_hat)| yi - yi_hat)
        .collect();
    let rss_subset = residuals_subset.iter().map(|&r| r * r).sum::<f64>();

    // DEBUG: Print RSS values to diagnose issues
    #[cfg(test)]
    {
        eprintln!(
            "DEBUG Rainbow: rss_full = {:.6}, rss_subset = {:.6}, diff = {:.6}",
            rss_full,
            rss_subset,
            rss_subset - rss_full
        );
        eprintln!(
            "DEBUG Rainbow: subset_size = {}, n = {}",
            subset_indices.len(),
            n
        );
        eprintln!("DEBUG Rainbow: method = {:?}", _method);
    }

    // Calculate F-statistic using the formula from R's lmtest::raintest
    // F = ((RSS_full - RSS_subset) / df1) / (RSS_subset / df2)
    // where df1 = n - subset_size, df2 = subset_size - p
    let subset_size = subset_indices.len() as f64;
    let df1 = (n - subset_indices.len()) as f64;
    let df2 = subset_size - (p as f64);

    let numerator = (rss_full - rss_subset).max(0.0) / df1;
    let denominator = rss_subset / df2;

    let f_stat = if denominator > 1e-10 {
        numerator / denominator
    } else {
        return Err(Error::InvalidInput(
            "Invalid denominator in Rainbow test".to_string(),
        ));
    };

    // Calculate p-value
    let p_value = f_p_value(f_stat, df1, df2);

    Ok((f_stat, p_value))
}

/// Rainbow test with method selection (R, Python, or Both).
///
/// The Rainbow test checks for linearity by comparing the fit on a subset of
/// central observations against the fit on all observations. If the relationship
/// is truly linear, both fits should be similar.
///
/// # Arguments
///
/// * `y` - Dependent variable values
/// * `x_vars` - Independent variables (each vec is a column)
/// * `fraction` - Fraction of data to use in central subset (0.0 to 1.0, default: 0.5)
/// * `method` - Which implementation to use: R, Python, or Both
///
/// # Returns
///
/// A [`RainbowTestOutput`] containing F-statistics, p-values, and interpretation
/// for each requested method.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if n ≤ k + 2.
///
/// # Note
///
/// The subset center is fixed at 0.5 (midpoint of ordered data).
pub fn rainbow_test(
    y: &[f64],
    x_vars: &[Vec<f64>],
    fraction: f64,
    method: RainbowMethod,
) -> Result<RainbowTestOutput> {
    let n = y.len();
    let k = x_vars.len();
    let p = k + 1;

    // Validate inputs
    if n <= p + 1 {
        return Err(Error::InsufficientData {
            required: p + 2,
            available: n,
        });
    }

    // Validate dimensions and finite values using shared helper
    super::helpers::validate_regression_data(y, x_vars)?;

    let fraction = if fraction <= 0.0 || fraction > 1.0 {
        0.5
    } else {
        fraction
    };
    let center = 0.5; // Default center value

    // Create design matrix
    let mut x_data = vec![0.0; n * p];
    for (row, _yi) in y.iter().enumerate() {
        x_data[row * p] = 1.0; // intercept
        for (col, x_var) in x_vars.iter().enumerate() {
            x_data[row * p + col + 1] = x_var[row];
        }
    }

    let x_full = Matrix::new(n, p, x_data.clone());

    // Fit OLS on full data
    let beta_full = fit_ols(y, &x_full)?;

    // Sort indices by fitted values
    let fitted_full = x_full.mul_vec(&beta_full);
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        fitted_full[a]
            .partial_cmp(&fitted_full[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Calculate results based on method
    let r_result = match method {
        RainbowMethod::R | RainbowMethod::Both => {
            // R's lmtest::raintest with order.by=NULL uses ORIGINAL data order (NOT sorted)
            let original_indices: Vec<usize> = (0..n).collect();
            let range = raintest_subset_r(n, fraction, center);
            match rainbow_test_internal(
                y,
                &x_data,
                &x_full,
                &beta_full,
                &original_indices,
                range,
                RainbowMethod::R,
            ) {
                Ok((f_stat, p_value)) => {
                    let alpha = 0.05;
                    Some(RainbowSingleResult {
                        method: "R (lmtest::raintest)".to_string(),
                        statistic: f_stat,
                        p_value,
                        passed: p_value > alpha,
                    })
                },
                Err(_) => None,
            }
        },
        _ => None,
    };

    let python_result = match method {
        RainbowMethod::Python | RainbowMethod::Both => {
            // Python's statsmodels uses ORIGINAL data order (NOT sorted by fitted values)
            let original_indices: Vec<usize> = (0..n).collect();
            let range = raintest_subset_python(n, fraction);
            match rainbow_test_internal(
                y,
                &x_data,
                &x_full,
                &beta_full,
                &original_indices,
                range,
                RainbowMethod::Python,
            ) {
                Ok((f_stat, p_value)) => {
                    let alpha = 0.05;
                    Some(RainbowSingleResult {
                        method: "Python (statsmodels)".to_string(),
                        statistic: f_stat,
                        p_value,
                        passed: p_value > alpha,
                    })
                },
                Err(_) => None,
            }
        },
        _ => None,
    };

    // Build interpretation based on primary result (R if available, else Python)
    let primary_result = r_result.as_ref().or(python_result.as_ref());

    let (interpretation, guidance) = if let Some(result) = primary_result {
        let alpha = 0.05;
        if result.passed {
            (
                format!("p-value = {:.4} is greater than {:.2}. Cannot reject H0. No significant evidence of non-linearity.", result.p_value, alpha),
                "The linear model appears appropriate. Consider other diagnostic tests.".to_string()
            )
        } else {
            (
                format!("p-value = {:.4} is less than or equal to {:.2}. Reject H0. Significant evidence of non-linearity detected.", result.p_value, alpha),
                "Consider adding polynomial terms, transforming variables, or using non-linear modeling.".to_string()
            )
        }
    } else {
        (
            "Unable to compute Rainbow test.".to_string(),
            "Check your data and try again.".to_string(),
        )
    };

    Ok(RainbowTestOutput {
        test_name: "Rainbow Test for Linearity".to_string(),
        r_result,
        python_result,
        interpretation,
        guidance,
    })
}
