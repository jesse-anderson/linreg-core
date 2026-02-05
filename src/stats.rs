//! Basic statistical utility functions.
//!
//! This module provides fundamental descriptive statistics operations
//! including measures of central tendency, dispersion, and position.

#![allow(clippy::needless_range_loop)]

/// Calculates the arithmetic mean (average) of a slice of f64 values.
///
/// # Arguments
///
/// * `data` - Slice of f64 values
///
/// # Returns
///
/// The mean as f64, or NaN if the slice is empty
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::mean;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(mean(&data), 3.0);
/// ```
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

/// Calculates the sample variance of a slice of f64 values.
///
/// Uses the (n-1) denominator for unbiased sample variance estimation.
///
/// This implementation uses **Welford's online algorithm** for numerical
/// stability, avoiding catastrophic cancellation that can occur with the
/// two-pass approach when values have large magnitude.
///
/// # Arguments
///
/// * `data` - Slice of f64 values
///
/// # Returns
///
/// The variance as f64, or NaN if the slice has fewer than 2 elements
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::variance;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let v = variance(&data);
/// assert!((v - 2.5).abs() < 1e-10);
/// ```
pub fn variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return f64::NAN;
    }

    // Welford's online algorithm for numerical stability
    let mut mean = data[0];
    let mut m2 = 0.0;

    for i in 1..n {
        let x = data[i];
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        let delta_new = x - mean;
        m2 += delta * delta_new;
    }

    m2 / (n - 1) as f64
}

/// Calculates the population variance of a slice of f64 values.
///
/// Uses the n denominator (for when data represents the entire population).
///
/// This implementation uses **Welford's online algorithm** for numerical
/// stability, avoiding catastrophic cancellation that can occur with the
/// two-pass approach when values have large magnitude.
///
/// # Arguments
///
/// * `data` - Slice of f64 values
///
/// # Returns
///
/// The population variance as f64, or NaN if the slice is empty
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::variance_population;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let v = variance_population(&data);
/// assert!((v - 2.0).abs() < 1e-10);
/// ```
pub fn variance_population(data: &[f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return f64::NAN;
    }

    // Welford's online algorithm for numerical stability
    let mut mean = data[0];
    let mut m2 = 0.0;

    for i in 1..n {
        let x = data[i];
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        let delta_new = x - mean;
        m2 += delta * delta_new;
    }

    m2 / n as f64
}

/// Calculates the sample standard deviation of a slice of f64 values.
///
/// Uses the (n-1) denominator for unbiased estimation.
///
/// # Arguments
///
/// * `data` - Slice of f64 values
///
/// # Returns
///
/// The standard deviation as f64, or NaN if the slice has fewer than 2 elements
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::stddev;
///
/// let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let s = stddev(&data);
/// assert!((s - 2.138089935).abs() < 1e-9);
/// ```
pub fn stddev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Calculates the population standard deviation of a slice of f64 values.
///
/// Uses the n denominator (for when data represents the entire population).
///
/// # Arguments
///
/// * `data` - Slice of f64 values
///
/// # Returns
///
/// The population standard deviation as f64, or NaN if the slice is empty
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::stddev_population;
///
/// let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let s = stddev_population(&data);
/// assert!((s - 2.0).abs() < 1e-9);
/// ```
pub fn stddev_population(data: &[f64]) -> f64 {
    variance_population(data).sqrt()
}

/// Calculates the median of a slice of f64 values.
///
/// # Arguments
///
/// * `data` - Slice of f64 values
///
/// # Returns
///
/// The median as f64, or NaN if the slice is empty
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::median;
///
/// let odd = vec![1.0, 3.0, 5.0];
/// assert_eq!(median(&odd), 3.0);
///
/// let even = vec![1.0, 2.0, 3.0, 4.0];
/// assert_eq!(median(&even), 2.5);
/// ```
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = sorted.len();
    if len.is_multiple_of(2) {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    }
}

/// Calculates a quantile of a slice of f64 values using linear interpolation.
///
/// # Arguments
///
/// * `data` - Slice of f64 values
/// * `q` - Quantile to calculate (0.0 to 1.0)
///
/// # Returns
///
/// The quantile value as f64, or NaN if the slice is empty or q is out of range
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::quantile;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// let q25 = quantile(&data, 0.25);
/// let q50 = quantile(&data, 0.50);
/// let q75 = quantile(&data, 0.75);
/// assert_eq!(q50, 5.0);
/// ```
pub fn quantile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let index = q * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted[lower]
    } else {
        let weight = index - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

/// Calculates the sum of a slice of f64 values.
///
/// # Arguments
///
/// * `data` - Slice of f64 values
///
/// # Returns
///
/// The sum as f64 (0.0 for empty slice)
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::sum;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(sum(&data), 15.0);
/// ```
pub fn sum(data: &[f64]) -> f64 {
    data.iter().sum()
}

/// Finds the minimum value in a slice of f64 values.
///
/// # Arguments
///
/// * `data` - Slice of f64 values
///
/// # Returns
///
/// The minimum value as f64, or NaN if the slice is empty
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::min;
///
/// let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
/// assert_eq!(min(&data), 1.0);
/// ```
pub fn min(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}

/// Finds the maximum value in a slice of f64 values.
///
/// # Arguments
///
/// * `data` - Slice of f64 values
///
/// # Returns
///
/// The maximum value as f64, or NaN if the slice is empty
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::max;
///
/// let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
/// assert_eq!(max(&data), 9.0);
/// ```
pub fn max(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

/// Calculates the range (max - min) of a slice of f64 values.
///
/// # Arguments
///
/// * `data` - Slice of f64 values
///
/// # Returns
///
/// The range as f64, or NaN if the slice is empty
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::range;
///
/// let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
/// assert_eq!(range(&data), 8.0);  // 9.0 - 1.0
/// ```
pub fn range(data: &[f64]) -> f64 {
    max(data) - min(data)
}

/// Calculates the correlation coefficient (Pearson's r) between two slices.
///
/// This implementation uses a **numerically stable single-pass algorithm**
/// that avoids catastrophic cancellation, similar to Welford's method.
/// It computes mean, variance, and covariance in one pass.
///
/// # Arguments
///
/// * `x` - First slice of f64 values
/// * `y` - Second slice of f64 values (must be same length as x)
///
/// # Returns
///
/// The correlation coefficient as f64, or NaN if inputs are invalid
///
/// # Examples
///
/// ```rust
/// use linreg_core::stats::correlation;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
/// let r = correlation(&x, &y);
/// assert!((r - 0.7746).abs() < 1e-4);
/// ```
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }

    let n = x.len();

    // Numerically stable single-pass algorithm
    // Tracks: mean_x, mean_y, m2_x, m2_y (variances), and covariance
    let mut mean_x = x[0];
    let mut mean_y = y[0];
    let mut m2_x = 0.0;
    let mut m2_y = 0.0;
    let mut cov = 0.0;

    for i in 1..n {
        let xi = x[i];
        let yi = y[i];
        let delta_x = xi - mean_x;
        let delta_y = yi - mean_y;

        let i_inv = 1.0 / (i + 1) as f64;
        mean_x += delta_x * i_inv;
        mean_y += delta_y * i_inv;

        let delta_x_new = xi - mean_x;
        let delta_y_new = yi - mean_y;

        m2_x += delta_x * delta_x_new;
        m2_y += delta_y * delta_y_new;
        cov += delta_x * delta_y_new;
    }

    let denom = (m2_x * m2_y).sqrt();
    if denom == 0.0 {
        return f64::NAN;
    }

    cov / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
        assert!(mean(&[]).is_nan());
    }

    #[test]
    fn test_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = variance(&data);
        assert!((v - 2.5).abs() < 1e-10);
        assert!(variance(&[1.0]).is_nan());
    }

    #[test]
    fn test_variance_population() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = variance_population(&data);
        assert!((v - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_stddev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = stddev(&data);
        assert!((s - 2.138089935).abs() < 1e-9);
    }

    #[test]
    fn test_median() {
        assert_eq!(median(&[1.0, 3.0, 5.0]), 3.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn test_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(quantile(&data, 0.0), 1.0);
        assert_eq!(quantile(&data, 0.5), 5.0);
        assert_eq!(quantile(&data, 1.0), 9.0);
    }

    #[test]
    fn test_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        assert_eq!(min(&data), 1.0);
        assert_eq!(max(&data), 9.0);
    }

    #[test]
    fn test_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(sum(&data), 15.0);
        // Empty slice returns 0.0 (standard Rust behavior)
        assert_eq!(sum(&[]), 0.0);
    }

    #[test]
    fn test_range() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        assert_eq!(range(&data), 8.0);  // 9.0 - 1.0
        // Empty slice returns NaN (max - min = NaN - NaN)
        assert!(range(&[]).is_nan());
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let r = correlation(&x, &y);
        // Correct value: r = 6/sqrt(60) ≈ 0.7746
        assert!((r - 0.7746).abs() < 1e-4);
    }

    // ============================================================================
    // NaN/Inf Handling Tests
    // ============================================================================

    #[test]
    fn test_mean_with_nan() {
        let data = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        // mean propagates NaN (standard Rust behavior for sum)
        assert!(mean(&data).is_nan());
    }

    #[test]
    fn test_mean_with_inf() {
        let data = vec![1.0, 2.0, f64::INFINITY, 4.0, 5.0];
        assert_eq!(mean(&data), f64::INFINITY);

        let data2 = vec![1.0, 2.0, f64::NEG_INFINITY, 4.0, 5.0];
        assert_eq!(mean(&data2), f64::NEG_INFINITY);
    }

    #[test]
    fn test_variance_with_nan() {
        let data = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        assert!(variance(&data).is_nan());
    }

    #[test]
    fn test_variance_with_inf() {
        let data = vec![1.0, 2.0, f64::INFINITY, 4.0, 5.0];
        // Variance with INF should be NaN or INF depending on calculation
        let v = variance(&data);
        assert!(v.is_nan() || v.is_infinite());
    }

    #[test]
    fn test_correlation_with_nan() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, f64::NAN, 5.0, 4.0, 5.0];
        // correlation with NaN in data returns NaN
        assert!(correlation(&x, &y).is_nan());
    }

    #[test]
    fn test_correlation_with_inf() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, f64::INFINITY, 5.0, 4.0, 5.0];
        // correlation with INF in data returns NaN
        let r = correlation(&x, &y);
        assert!(r.is_nan() || r.is_infinite());
    }

    #[test]
    fn test_correlation_single_value() {
        let x = vec![1.0];
        let y = vec![2.0];
        // Single value arrays should return NaN (undefined correlation)
        assert!(correlation(&x, &y).is_nan());
    }

    #[test]
    fn test_correlation_mismatched_lengths() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        // Mismatched lengths should return NaN
        assert!(correlation(&x, &y).is_nan());
    }

    #[test]
    fn test_min_max_with_nan() {
        let data = vec![3.0, 1.0, f64::NAN, 4.0, 5.0];
        // min/max ignore NaN in fold comparison
        assert_eq!(min(&data), 1.0);
        assert_eq!(max(&data), 5.0);
    }

    #[test]
    fn test_min_max_with_inf() {
        let data = vec![3.0, 1.0, f64::INFINITY, 4.0, 5.0];
        assert_eq!(min(&data), 1.0);
        assert_eq!(max(&data), f64::INFINITY);
    }

    #[test]
    fn test_median_with_inf() {
        let data = vec![1.0, 2.0, f64::INFINITY, 4.0, 5.0];
        // INF sorts to the end
        assert_eq!(median(&data), 4.0);
    }

    #[test]
    fn test_stddev_single_value() {
        // Single element should return NaN (undefined sample stddev)
        assert!(stddev(&[1.0]).is_nan());
    }

    #[test]
    fn test_stddev_population_single_value() {
        // Single element population stddev should be 0
        assert_eq!(stddev_population(&[1.0]), 0.0);
    }
}
