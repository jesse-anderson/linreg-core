//! Robust fitting using biweight function

/// Compute biweight robustness weights
///
/// Computes robustness weights using the biweight function:
/// - w = (1 - (r/c)^2)^2 for |r| < c
/// - w = 1 for |r| <= c1
/// - w = 0 for |r| >= c9
///   where c = 6 * MAD (median absolute deviation)
///
/// Note: The caller should check if consistent_mad (6 * MAD) is too small relative to
/// the mean absolute residual before calling this function.
///
/// # Arguments
///
/// * `residuals` - Residuals from previous fit
///
/// # Returns
///
/// Vector of robustness weights
///
/// # Example
///
/// ```
/// use linreg_core::loess::robust::compute_biweight_weights;
///
/// // Residuals with one clear outlier
/// let residuals = vec![0.0, 0.1, 0.2, -0.1, 5.0];
///
/// let weights = compute_biweight_weights(&residuals);
///
/// // All weights should be in [0, 1]
/// assert!(weights.iter().all(|w| *w >= 0.0 && *w <= 1.0));
///
/// // Zero residual should give weight 1.0
/// assert_eq!(weights[0], 1.0);
///
/// // Large residual (5.0) should have smaller weight
/// assert!(weights[4] < weights[1]);
/// ```
pub fn compute_biweight_weights(residuals: &[f64]) -> Vec<f64> {
    let n = residuals.len();

    if n == 0 {
        return vec![];
    }

    // Compute median absolute deviation (MAD)
    let mut abs_residuals: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
    abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_index = n / 2;
    let mad = if n.is_multiple_of(2) {
        (abs_residuals[median_index - 1] + abs_residuals[median_index]) / 2.0
    } else {
        abs_residuals[median_index]
    };

    // robustness_scale = 6 * MAD
    let robustness_scale = 6.0 * mad;

    // Boundary thresholds for numerical stability
    let outer_threshold = 0.999 * robustness_scale;  // Beyond this, weight = 0
    let inner_threshold = 0.001 * robustness_scale;  // Within this, weight = 1

    // Compute biweight weights
    residuals
        .iter()
        .map(|&r| {
            let abs_r = r.abs();
            if abs_r <= inner_threshold {
                1.0
            } else if abs_r >= outer_threshold {
                0.0
            } else {
                let normalized_residual = abs_r / robustness_scale;
                (1.0 - normalized_residual * normalized_residual).powi(2)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_biweight_weights() {
        // Test biweight weight computation
        let residuals = vec![0.0, 0.5, 1.0, 2.0, 5.0];

        let weights = compute_biweight_weights(&residuals);

        // All weights should be in [0, 1]
        for w in &weights {
            assert!(*w >= 0.0 && *w <= 1.0);
        }

        // Zero residual should give weight 1.0
        assert_eq!(weights[0], 1.0);

        // Large residuals should have smaller weights
        assert!(weights[4] < weights[1]);
    }
}
