//! Predictor normalization utilities

use crate::linalg::Matrix;

/// Normalization information for predictors
///
/// Stores the minimum and range values needed to normalize/denormalize
/// predictor variables to/from the [0, 1] range.
#[derive(Debug, Clone)]
pub struct NormalizationInfo {
    /// Minimum values for each predictor
    pub min: Vec<f64>,
    /// Range (max - min) for each predictor
    pub range: Vec<f64>,
}

/// Normalize predictor matrix to [0, 1] range
///
/// For each column, applies min-max normalization: `(x - min) / (max - min)`.
/// This ensures all predictors are on the same scale for distance calculations.
///
/// # Arguments
///
/// * `x` - Predictor matrix (n obs × p predictors)
///
/// # Returns
///
/// A tuple of:
/// - Normalized matrix with values in [0, 1]
/// - NormalizationInfo containing min and range for each predictor
///
/// # Panics
///
/// Panics if the matrix has zero columns.
///
/// # Example
///
/// ```
/// use linreg_core::loess::normalize::normalize_predictors;
/// use linreg_core::linalg::Matrix;
///
/// // Simple test: normalize [0, 10] to [0, 1]
/// let data = vec![0.0, 2.5, 5.0, 7.5, 10.0];
/// let x = Matrix::new(5, 1, data);
///
/// let (normalized, info) = normalize_predictors(&x);
///
/// // Check normalization info
/// assert_eq!(info.min[0], 0.0);
/// assert_eq!(info.range[0], 10.0);
///
/// // Check endpoints are normalized to 0 and 1
/// assert_eq!(normalized.get(0, 0), 0.0);
/// assert_eq!(normalized.get(4, 0), 1.0);
/// ```
pub fn normalize_predictors(x: &Matrix) -> (Matrix, NormalizationInfo) {
    let n = x.rows;
    let p = x.cols;

    assert!(p > 0, "Predictor matrix must have at least one column");

    let mut min = Vec::with_capacity(p);
    let mut range = Vec::with_capacity(p);

    // First pass: compute min and range for each column
    for j in 0..p {
        let mut col_min = f64::INFINITY;
        let mut col_max = f64::NEG_INFINITY;

        for i in 0..n {
            let val = x.get(i, j);
            if val < col_min {
                col_min = val;
            }
            if val > col_max {
                col_max = val;
            }
        }

        let col_range = col_max - col_min;

        if col_range <= f64::EPSILON {
            // Constant column - avoid division by zero
            min.push(col_min);
            range.push(1.0);
        } else {
            min.push(col_min);
            range.push(col_range);
        }
    }

    // Second pass: fill normalized data in row-major order
    let mut normalized_data = Vec::with_capacity(n * p);
    for i in 0..n {
        for j in 0..p {
            let val = x.get(i, j);
            let col_min = min[j];
            let col_range = range[j];

            // Check if this is a constant column (we set range to 1.0 for those)
            if col_range == 1.0 && (val - col_min).abs() < f64::EPSILON {
                // Constant column - use 0.5 as normalized value
                normalized_data.push(0.5);
            } else {
                normalized_data.push((val - col_min) / col_range);
            }
        }
    }

    let normalized = Matrix::new(n, p, normalized_data);
    let info = NormalizationInfo { min, range };

    (normalized, info)
}

/// Denormalize a value from \[0,1\] back to original scale
///
/// # Arguments
///
/// * `value` - Normalized value in \[0, 1\]
/// * `min` - Minimum of original data
/// * `range` - Range (max - min) of original data
///
/// # Returns
///
/// Denormalized value on the original scale
///
/// # Example
///
/// ```
/// use linreg_core::loess::normalize::denormalize;
///
/// let min = 10.0;
/// let range = 50.0;
///
/// // 0.0 -> min
/// assert_eq!(denormalize(0.0, min, range), min);
/// // 1.0 -> min + range = max
/// assert_eq!(denormalize(1.0, min, range), min + range);
/// // 0.5 -> midpoint
/// assert_eq!(denormalize(0.5, min, range), min + range / 2.0);
/// ```
#[inline]
pub fn denormalize(value: f64, min: f64, range: f64) -> f64 {
    value * range + min
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_simple() {
        // Simple test: normalize [0, 10] to [0, 1]
        let data = vec![0.0, 2.5, 5.0, 7.5, 10.0];
        let x = Matrix::new(5, 1, data.clone());

        let (normalized, info) = normalize_predictors(&x);

        assert_eq!(info.min[0], 0.0);
        assert_eq!(info.range[0], 10.0);

        // Check endpoints
        assert_eq!(normalized.get(0, 0), 0.0);
        assert_eq!(normalized.get(4, 0), 1.0);

        // Check midpoint
        assert!((normalized.get(2, 0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_multiple_columns() {
        // Test with multiple predictors
        // Data in row-major order:
        // Row 0: col0=0.0, col1=100.0
        // Row 1: col0=5.0, col1=150.0
        // Row 2: col0=10.0, col1=200.0
        let data = vec![
            0.0, 100.0,  // Row 0
            5.0, 150.0,  // Row 1
            10.0, 200.0, // Row 2
        ];
        let x = Matrix::new(3, 2, data);

        let (normalized, info) = normalize_predictors(&x);

        // Column 0: values [0, 5, 10]
        assert_eq!(info.min[0], 0.0);
        assert_eq!(info.range[0], 10.0);
        // Column 1: values [100, 150, 200]
        assert_eq!(info.min[1], 100.0);
        assert_eq!(info.range[1], 100.0);

        // Check normalized values are in [0, 1]
        for i in 0..3 {
            for j in 0..2 {
                let val = normalized.get(i, j);
                assert!(val >= 0.0 && val <= 1.0);
            }
        }

        // Check specific values
        assert_eq!(normalized.get(0, 0), 0.0);  // (0-0)/10 = 0
        assert_eq!(normalized.get(2, 0), 1.0);  // (10-0)/10 = 1
        assert_eq!(normalized.get(0, 1), 0.0);  // (100-100)/100 = 0
        assert_eq!(normalized.get(2, 1), 1.0);  // (200-100)/100 = 1
    }

    #[test]
    fn test_denormalize() {
        // Test denormalization
        let min = 10.0;
        let range = 50.0;

        // 0.0 -> min
        assert_eq!(denormalize(0.0, min, range), min);
        // 1.0 -> min + range = max
        assert_eq!(denormalize(1.0, min, range), min + range);
        // 0.5 -> midpoint
        assert_eq!(denormalize(0.5, min, range), min + range / 2.0);
    }

    #[test]
    fn test_normalize_roundtrip() {
        // Test that normalize -> denormalize preserves values
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let x = Matrix::new(5, 1, data.clone());

        let (normalized, info) = normalize_predictors(&x);

        // Denormalize each value and check we get back the original
        for i in 0..5 {
            let denorm = denormalize(normalized.get(i, 0), info.min[0], info.range[0]);
            assert!((denorm - data[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalize_constant_column() {
        // Test with constant values (all same)
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let x = Matrix::new(5, 1, data);

        let (normalized, info) = normalize_predictors(&x);

        // All normalized values should be 0.5 (midpoint)
        for i in 0..5 {
            assert!((normalized.get(i, 0) - 0.5).abs() < 1e-10);
        }
        assert_eq!(info.min[0], 5.0);
        assert_eq!(info.range[0], 1.0); // Should be set to 1.0 to avoid division by zero
    }

    #[test]
    fn test_normalize_negative_values() {
        // Test with negative values
        let data = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
        let x = Matrix::new(5, 1, data);

        let (normalized, info) = normalize_predictors(&x);

        assert_eq!(info.min[0], -10.0);
        assert_eq!(info.range[0], 20.0);

        // Check endpoints and midpoint
        assert_eq!(normalized.get(0, 0), 0.0);  // -10 -> 0
        assert_eq!(normalized.get(4, 0), 1.0);  // 10 -> 1
        assert!((normalized.get(2, 0) - 0.5).abs() < 1e-10);  // 0 -> 0.5
    }
}
