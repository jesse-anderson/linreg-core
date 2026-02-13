// ============================================================================
// Type Definitions for Cross Validation
// ============================================================================

//! Configuration options and result types for K-Fold Cross Validation.

use crate::error::{Error, Result};

/// Configuration options for K-Fold Cross Validation.
///
/// # Fields
///
/// - `n_folds` — Number of folds (default: 5, must be >= 2)
/// - `shuffle` — Whether to shuffle data before splitting
/// - `seed` — Optional random seed for reproducible shuffling
///
/// # Example
///
/// ```rust
/// use linreg_core::cross_validation::KFoldOptions;
///
/// let options = KFoldOptions {
///     n_folds: 10,
///     shuffle: true,
///     seed: Some(42),
/// };
/// ```
#[derive(Clone, Debug)]
pub struct KFoldOptions {
    /// Number of folds (default: 5, must be >= 2)
    pub n_folds: usize,

    /// Whether to shuffle data before splitting
    pub shuffle: bool,

    /// Optional random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for KFoldOptions {
    fn default() -> Self {
        KFoldOptions {
            n_folds: 5,
            shuffle: false,
            seed: None,
        }
    }
}

impl KFoldOptions {
    /// Creates a new KFoldOptions with the specified number of folds.
    ///
    /// # Arguments
    ///
    /// * `n_folds` — Number of folds (must be >= 2)
    pub fn new(n_folds: usize) -> Self {
        KFoldOptions {
            n_folds,
            shuffle: false,
            seed: None,
        }
    }

    /// Sets whether to shuffle data before splitting.
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets the random seed for reproducible shuffling.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the options.
    pub(crate) fn validate(&self, n_samples: usize) -> Result<()> {
        if self.n_folds < 2 {
            return Err(Error::InvalidInput(
                "n_folds must be at least 2".to_string(),
            ));
        }
        if n_samples < self.n_folds {
            return Err(Error::InsufficientData {
                required: self.n_folds,
                available: n_samples,
            });
        }
        Ok(())
    }
}

/// Results from a single fold in cross-validation.
///
/// Contains the performance metrics for a single train/test split.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize))]
pub struct FoldResult {
    /// Fold index (1-based)
    pub fold_index: usize,

    /// Number of training observations
    pub train_size: usize,

    /// Number of test observations
    pub test_size: usize,

    /// Mean Squared Error on test set
    pub mse: f64,

    /// Root Mean Squared Error on test set
    pub rmse: f64,

    /// Mean Absolute Error on test set
    pub mae: f64,

    /// R-squared on test set
    pub r_squared: f64,

    /// Training R-squared (for detecting overfitting)
    pub train_r_squared: f64,
}

impl FoldResult {
    /// Creates a new FoldResult.
    pub(crate) fn new(
        fold_index: usize,
        train_size: usize,
        test_size: usize,
        mse: f64,
        rmse: f64,
        mae: f64,
        r_squared: f64,
        train_r_squared: f64,
    ) -> Self {
        FoldResult {
            fold_index,
            train_size,
            test_size,
            mse,
            rmse,
            mae,
            r_squared,
            train_r_squared,
        }
    }
}

/// Aggregated cross-validation results.
///
/// Contains the mean and standard deviation of metrics across all folds,
/// as well as individual fold results for detailed analysis.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize))]
pub struct CVResult {
    /// Number of folds used
    pub n_folds: usize,

    /// Total number of observations
    pub n_samples: usize,

    /// Mean MSE across all folds
    pub mean_mse: f64,

    /// Standard deviation of MSE across folds
    pub std_mse: f64,

    /// Mean RMSE across all folds
    pub mean_rmse: f64,

    /// Standard deviation of RMSE across folds
    pub std_rmse: f64,

    /// Mean MAE across all folds
    pub mean_mae: f64,

    /// Standard deviation of MAE across folds
    pub std_mae: f64,

    /// Mean R-squared across all folds
    pub mean_r_squared: f64,

    /// Standard deviation of R-squared across folds
    pub std_r_squared: f64,

    /// Mean training R-squared (for overfitting detection)
    pub mean_train_r_squared: f64,

    /// Results for each individual fold
    pub fold_results: Vec<FoldResult>,

    /// Coefficient estimates from each fold (for stability analysis)
    pub fold_coefficients: Vec<Vec<f64>>,
}

impl CVResult {
    /// Creates a new CVResult from fold results.
    pub(crate) fn from_folds(
        n_samples: usize,
        n_folds: usize,
        fold_results: Vec<FoldResult>,
        fold_coefficients: Vec<Vec<f64>>,
    ) -> Self {
        let _n = fold_results.len(); // Used for validation, currently unused

        let mean_mse = mean(&fold_results, |f| f.mse);
        let std_mse = std_dev(&fold_results, |f| f.mse, mean_mse);

        let mean_rmse = mean(&fold_results, |f| f.rmse);
        let std_rmse = std_dev(&fold_results, |f| f.rmse, mean_rmse);

        let mean_mae = mean(&fold_results, |f| f.mae);
        let std_mae = std_dev(&fold_results, |f| f.mae, mean_mae);

        let mean_r_squared = mean(&fold_results, |f| f.r_squared);
        let std_r_squared = std_dev(&fold_results, |f| f.r_squared, mean_r_squared);

        let mean_train_r_squared = mean(&fold_results, |f| f.train_r_squared);

        CVResult {
            n_folds,
            n_samples,
            mean_mse,
            std_mse,
            mean_rmse,
            std_rmse,
            mean_mae,
            std_mae,
            mean_r_squared,
            std_r_squared,
            mean_train_r_squared,
            fold_results,
            fold_coefficients,
        }
    }

    /// Returns the coefficient with the smallest standard deviation across folds.
    ///
    /// This is a measure of coefficient stability - more stable coefficients
    /// are less sensitive to the training data.
    pub fn most_stable_coefficient(&self) -> Option<usize> {
        if self.fold_coefficients.is_empty() || self.fold_coefficients[0].is_empty() {
            return None;
        }

        let n_coef = self.fold_coefficients[0].len();
        let mut min_std = f64::INFINITY;
        let mut most_stable = 0;

        for i in 0..n_coef {
            let values: Vec<f64> = self.fold_coefficients.iter().map(|c| c[i]).collect();
            let mean_val = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&v| (v - mean_val).powi(2)).sum::<f64>() / values.len() as f64;
            let std_val = variance.sqrt();

            if std_val < min_std {
                min_std = std_val;
                most_stable = i;
            }
        }

        Some(most_stable)
    }
}

/// Computes the mean of a value extracted from a slice.
fn mean<F>(values: &[FoldResult], f: F) -> f64
where
    F: Fn(&FoldResult) -> f64,
{
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(&f).sum::<f64>() / values.len() as f64
}

/// Computes the standard deviation of a value extracted from a slice.
fn std_dev<F>(values: &[FoldResult], f: F, mean: f64) -> f64
where
    F: Fn(&FoldResult) -> f64,
{
    if values.len() <= 1 {
        return 0.0;
    }
    let variance = values.iter().map(&f).map(|v| (v - mean).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kfold_options_default() {
        let options = KFoldOptions::default();
        assert_eq!(options.n_folds, 5);
        assert_eq!(options.shuffle, false);
        assert_eq!(options.seed, None);
    }

    #[test]
    fn test_kfold_options_builder() {
        let options = KFoldOptions::new(10).with_shuffle(true).with_seed(42);
        assert_eq!(options.n_folds, 10);
        assert_eq!(options.shuffle, true);
        assert_eq!(options.seed, Some(42));
    }

    #[test]
    fn test_kfold_options_validate_valid() {
        let options = KFoldOptions::new(5);
        assert!(options.validate(100).is_ok());
    }

    #[test]
    fn test_kfold_options_validate_too_few_folds() {
        let options = KFoldOptions::new(1);
        assert!(options.validate(100).is_err());
    }

    #[test]
    fn test_kfold_options_validate_insufficient_samples() {
        let options = KFoldOptions::new(10);
        let result = options.validate(5);
        assert!(result.is_err());
        match result {
            Err(Error::InsufficientData { required, available }) => {
                assert_eq!(required, 10);
                assert_eq!(available, 5);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_fold_result_new() {
        let result = FoldResult::new(1, 80, 20, 1.5, 1.225, 0.8, 0.85, 0.90);
        assert_eq!(result.fold_index, 1);
        assert_eq!(result.train_size, 80);
        assert_eq!(result.test_size, 20);
        assert_eq!(result.mse, 1.5);
        assert_eq!(result.rmse, 1.225);
        assert_eq!(result.mae, 0.8);
        assert_eq!(result.r_squared, 0.85);
        assert_eq!(result.train_r_squared, 0.90);
    }

    #[test]
    fn test_cv_result_from_folds() {
        let fold_results = vec![
            FoldResult::new(1, 80, 20, 1.0, 1.0, 0.8, 0.85, 0.90),
            FoldResult::new(2, 80, 20, 2.0, 1.414, 0.9, 0.80, 0.88),
        ];

        let fold_coefficients = vec![vec![1.0, 2.0], vec![1.1, 1.9]];

        let cv_result = CVResult::from_folds(100, 2, fold_results, fold_coefficients);

        assert_eq!(cv_result.n_folds, 2);
        assert_eq!(cv_result.n_samples, 100);
        assert_eq!(cv_result.mean_mse, 1.5);
        assert!((cv_result.std_mse - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_cv_result_most_stable_coefficient() {
        let fold_coefficients = vec![
            vec![1.0, 10.0, 5.0],  // std: 0.0 for first (same values)
            vec![1.0, 12.0, 6.0],  // std: ~1.41 for second
            vec![1.0, 8.0, 4.0],   // std: ~1.0 for third
        ];

        let fold_results = vec![
            FoldResult::new(1, 80, 20, 1.0, 1.0, 0.8, 0.85, 0.90),
            FoldResult::new(2, 80, 20, 2.0, 1.414, 0.9, 0.80, 0.88),
            FoldResult::new(3, 80, 20, 1.5, 1.225, 0.85, 0.83, 0.89),
        ];

        let cv_result = CVResult::from_folds(100, 3, fold_results, fold_coefficients);

        // First coefficient is most stable (std = 0)
        assert_eq!(cv_result.most_stable_coefficient(), Some(0));
    }

    #[test]
    fn test_cv_result_most_stable_coefficient_empty() {
        let fold_results = vec![FoldResult::new(1, 80, 20, 1.0, 1.0, 0.8, 0.85, 0.90)];
        let cv_result = CVResult::from_folds(100, 1, fold_results, vec![]);
        assert_eq!(cv_result.most_stable_coefficient(), None);
    }
}
