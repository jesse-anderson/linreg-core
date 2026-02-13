// ============================================================================
// Metrics for Cross Validation
// ============================================================================

//! Metric calculations for cross-validation performance evaluation.
//!
//! This module provides functions to compute common regression metrics
//! for evaluating model performance on train and test sets.

/// Computes the mean of a metric applied element-wise to two slices.
///
/// Helper function that avoids repetition in metric calculations.
///
/// # Arguments
///
/// * `y_true` — Actual/observed values
/// * `y_pred` — Predicted values
/// * `f` — Function to compute the error metric for a single pair
///
/// # Returns
///
/// The mean of the metric values. Returns 0.0 for empty slices.
fn compute_mean_metric<F>(y_true: &[f64], y_pred: &[f64], f: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    if y_true.is_empty() {
        return 0.0;
    }
    let n = y_true.len();
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| f(yt, yp))
        .sum::<f64>() / n as f64
}

/// Computes the Mean Squared Error (MSE).
///
/// MSE = (1/n) * Σ(y_true - y_pred)²
///
/// # Arguments
///
/// * `y_true` — Actual/observed values
/// * `y_pred` — Predicted values
///
/// # Returns
///
/// The mean squared error. Returns 0.0 for empty slices.
///
/// # Example
///
/// ```rust
/// use linreg_core::cross_validation::metrics::compute_mse;
///
/// let y_true = vec![3.0, -0.5, 2.0, 7.0];
/// let y_pred = vec![2.5, 0.0, 2.0, 8.0];
///
/// let mse = compute_mse(&y_true, &y_pred);
/// assert!((mse - 0.375).abs() < 1e-10);
/// ```
pub fn compute_mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    compute_mean_metric(y_true, y_pred, |yt, yp| (yt - yp).powi(2))
}

/// Computes the Root Mean Squared Error (RMSE).
///
/// RMSE = sqrt(MSE)
///
/// # Arguments
///
/// * `y_true` — Actual/observed values
/// * `y_pred` — Predicted values
///
/// # Returns
///
/// The root mean squared error.
///
/// # Example
///
/// ```rust
/// use linreg_core::cross_validation::metrics::compute_rmse;
///
/// let y_true = vec![3.0, -0.5, 2.0, 7.0];
/// let y_pred = vec![2.5, 0.0, 2.0, 8.0];
///
/// let rmse = compute_rmse(&y_true, &y_pred);
/// assert!((rmse - 0.61237).abs() < 1e-5);
/// ```
pub fn compute_rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    compute_mse(y_true, y_pred).sqrt()
}

/// Computes the Mean Absolute Error (MAE).
///
/// MAE = (1/n) * Σ|y_true - y_pred|
///
/// # Arguments
///
/// * `y_true` — Actual/observed values
/// * `y_pred` — Predicted values
///
/// # Returns
///
/// The mean absolute error. Returns 0.0 for empty slices.
///
/// # Example
///
/// ```rust
/// use linreg_core::cross_validation::metrics::compute_mae;
///
/// let y_true = vec![3.0, -0.5, 2.0, 7.0];
/// let y_pred = vec![2.5, 0.0, 2.0, 8.0];
///
/// let mae = compute_mae(&y_true, &y_pred);
/// assert!((mae - 0.5).abs() < 1e-10);
/// ```
pub fn compute_mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    compute_mean_metric(y_true, y_pred, |yt, yp| (yt - yp).abs())
}

/// Computes the R-squared (coefficient of determination).
///
/// R² = 1 - (SS_res / SS_tot)
///
/// Where:
/// - SS_res = Σ(y_true - y_pred)² (residual sum of squares)
/// - SS_tot = Σ(y_true - mean(y_true))² (total sum of squares)
///
/// R² represents the proportion of variance in the dependent variable
/// that is predictable from the independent variable(s).
///
/// # Arguments
///
/// * `y_true` — Actual/observed values
/// * `y_pred` — Predicted values
///
/// # Returns
///
/// The R-squared value. Returns 0.0 for empty or single-element slices.
/// Can be negative if the model is worse than just predicting the mean.
///
/// # Example
///
/// ```rust
/// use linreg_core::cross_validation::metrics::compute_r_squared;
///
/// let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y_pred = vec![1.1, 1.9, 3.1, 3.9, 5.1];
///
/// let r2 = compute_r_squared(&y_true, &y_pred);
/// assert!(r2 > 0.99); // Very good fit
/// ```
pub fn compute_r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.len() <= 1 {
        return 0.0;
    }

    let n = y_true.len();
    let mean_true: f64 = y_true.iter().sum::<f64>() / n as f64;

    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(yt, yp)| (yt - yp).powi(2))
        .sum();

    let ss_tot: f64 = y_true.iter().map(|yt| (yt - mean_true).powi(2)).sum();

    if ss_tot == 0.0 {
        // All y_true values are the same
        return if ss_res == 0.0 { 1.0 } else { 0.0 };
    }

    1.0 - (ss_res / ss_tot)
}

/// Computes the mean and standard deviation of a slice of values.
///
/// # Arguments
///
/// * `values` — Slice of f64 values
///
/// # Returns
///
/// A tuple `(mean, std_dev)` where:
/// - `mean` is the arithmetic mean
/// - `std_dev` is the sample standard deviation (using n-1 denominator)
///
/// Returns `(0.0, 0.0)` for empty or single-element slices.
///
/// # Example
///
/// ```rust
/// use linreg_core::cross_validation::metrics::mean_std;
///
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let (mean, std) = mean_std(&values);
///
/// assert_eq!(mean, 3.0);
/// assert!((std - 1.5811).abs() < 1e-4);
/// ```
pub fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;

    if n == 1 {
        return (mean, 0.0);
    }

    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std_dev = variance.sqrt();

    (mean, std_dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mse_basic() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];

        let mse = compute_mse(&y_true, &y_pred);
        // ((3-2.5)² + (-0.5-0)² + (2-2)² + (7-8)²) / 4
        // = (0.25 + 0.25 + 0 + 1) / 4 = 1.5 / 4 = 0.375
        assert!((mse - 0.375).abs() < 1e-10);
    }

    #[test]
    fn test_compute_mse_perfect_prediction() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 3.0];

        assert_eq!(compute_mse(&y_true, &y_pred), 0.0);
    }

    #[test]
    fn test_compute_mse_empty() {
        assert_eq!(compute_mse(&[], &[]), 0.0);
    }

    #[test]
    fn test_compute_rmse_basic() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];

        let rmse = compute_rmse(&y_true, &y_pred);
        // sqrt(0.375) ≈ 0.61237
        assert!((rmse - 0.61237).abs() < 1e-5);
    }

    #[test]
    fn test_compute_mae_basic() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];

        let mae = compute_mae(&y_true, &y_pred);
        // (|3-2.5| + |-0.5-0| + |2-2| + |7-8|) / 4
        // = (0.5 + 0.5 + 0 + 1) / 4 = 2.0 / 4 = 0.5
        assert_eq!(mae, 0.5);
    }

    #[test]
    fn test_compute_mae_empty() {
        assert_eq!(compute_mae(&[], &[]), 0.0);
    }

    #[test]
    fn test_compute_r_squared_perfect_fit() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(compute_r_squared(&y_true, &y_pred), 1.0);
    }

    #[test]
    fn test_compute_r_squared_good_fit() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 1.9, 3.1, 3.9, 5.1];

        let r2 = compute_r_squared(&y_true, &y_pred);
        assert!(r2 > 0.99);
    }

    #[test]
    fn test_compute_r_squared_negative() {
        // Predictions worse than just using the mean
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let r2 = compute_r_squared(&y_true, &y_pred);
        assert!(r2 < 0.0);
    }

    #[test]
    fn test_compute_r_squared_constant_values() {
        // When all y_true are the same
        let y_true = vec![5.0, 5.0, 5.0, 5.0];
        let y_pred = vec![5.0, 5.0, 5.0, 5.0];

        assert_eq!(compute_r_squared(&y_true, &y_pred), 1.0);
    }

    #[test]
    fn test_compute_r_squared_empty() {
        assert_eq!(compute_r_squared(&[], &[]), 0.0);
    }

    #[test]
    fn test_compute_r_squared_single_element() {
        assert_eq!(compute_r_squared(&[1.0], &[1.0]), 0.0);
    }

    #[test]
    fn test_mean_std_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std) = mean_std(&values);

        assert_eq!(mean, 3.0);
        assert!((std - 1.5811).abs() < 1e-4);
    }

    #[test]
    fn test_mean_std_constant_values() {
        let values = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let (mean, std) = mean_std(&values);

        assert_eq!(mean, 5.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_mean_std_empty() {
        let (mean, std) = mean_std(&[]);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_mean_std_single_element() {
        let (mean, std) = mean_std(&[42.0]);
        assert_eq!(mean, 42.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_metrics_consistency() {
        // Verify RMSE = sqrt(MSE)
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.2, 1.8, 3.1, 4.2, 4.8];

        let mse = compute_mse(&y_true, &y_pred);
        let rmse = compute_rmse(&y_true, &y_pred);

        assert!((rmse - mse.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_different_length_arrays() {
        // Different lengths should still work (zip handles this)
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0]; // Shorter

        let mse = compute_mse(&y_true, &y_pred);
        // Only first 2 elements are compared
        // ((1-1)² + (2-2)²) / 2 = 0
        assert_eq!(mse, 0.0);
    }
}
