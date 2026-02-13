// ============================================================================
// Data Splitting for Cross Validation
// ============================================================================

//! Utilities for creating K-Fold train/test splits.
//!
//! This module provides functions to split data into K folds for cross-validation,
//! with optional shuffling for reproducibility.

use crate::error::{Error, Result};

/// Creates K-Fold train/test splits for cross-validation.
///
/// # Arguments
///
/// * `n_samples` — Total number of observations
/// * `n_folds` — Number of folds to create
/// * `shuffle` — Whether to shuffle indices before splitting
/// * `seed` — Optional random seed for reproducible shuffling
///
/// # Returns
///
/// A vector of tuples where each tuple contains:
/// - `train_indices` — Indices for training data
/// - `test_indices` — Indices for test data
///
/// # Algorithm
///
/// 1. Create indices 0..n_samples
/// 2. If shuffle is enabled, apply Fisher-Yates shuffle with the provided seed
/// 3. Partition indices into n_folds groups
///    - First `n_samples % n_folds` folds get one extra sample
/// 4. For each fold, test = fold indices, train = all other indices
///
/// # Example
///
/// ```rust
/// use linreg_core::cross_validation::splits::create_kfold_splits;
///
/// let splits = create_kfold_splits(10, 3, false, None)?;
/// assert_eq!(splits.len(), 3);
///
/// // First fold: test indices are [0, 1, 2, 3], train is the rest
/// let (train, test) = &splits[0];
/// assert_eq!(test.len(), 4);  // First fold gets extra sample (10 % 3 = 1)
/// assert_eq!(train.len(), 6);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn create_kfold_splits(
    n_samples: usize,
    n_folds: usize,
    shuffle: bool,
    seed: Option<u64>,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if n_folds < 2 {
        return Err(Error::InvalidInput("n_folds must be at least 2".to_string()));
    }
    if n_samples < n_folds {
        return Err(Error::InsufficientData {
            required: n_folds,
            available: n_samples,
        });
    }

    // Create indices 0..n_samples
    let mut indices: Vec<usize> = (0..n_samples).collect();

    // Shuffle if requested
    if shuffle {
        let seed_val = seed.unwrap_or(42);
        fisher_yates_shuffle(&mut indices, seed_val);
    }

    // Partition into folds
    let fold_size = n_samples / n_folds;
    let remainder = n_samples % n_folds;

    let mut folds: Vec<Vec<usize>> = Vec::with_capacity(n_folds);
    let mut start = 0;

    for fold_idx in 0..n_folds {
        let size = fold_size + if fold_idx < remainder { 1 } else { 0 };
        let end = start + size;
        folds.push(indices[start..end].to_vec());
        start = end;
    }

    // Create train/test splits
    let mut splits: Vec<(Vec<usize>, Vec<usize>)> = Vec::with_capacity(n_folds);

    for test_indices in folds {
        let train_indices: Vec<usize> = indices
            .iter()
            .filter(|&i| !test_indices.contains(i))
            .copied()
            .collect();

        splits.push((train_indices, test_indices));
    }

    Ok(splits)
}

/// Fisher-Yates shuffle with a simple LCG for reproducibility.
///
/// This is a self-contained implementation that doesn't rely on external RNG.
/// The LCG uses the constants from glibc: a = 1103515245, c = 12345, m = 2^32.
///
/// # Arguments
///
/// * `indices` — The slice to shuffle in-place
/// * `seed` — Random seed for the LCG
///
/// # Algorithm
///
/// The Fisher-Yates shuffle works by iterating from the last element to the first,
/// and swapping each element with a randomly chosen element from the remaining
/// unshuffled portion.
///
/// # Example
///
/// ```rust
/// use linreg_core::cross_validation::splits::fisher_yates_shuffle;
///
/// let mut indices = vec![0, 1, 2, 3, 4];
/// fisher_yates_shuffle(&mut indices, 42);
/// // Indices are now in a deterministic but shuffled order
/// ```
pub fn fisher_yates_shuffle(indices: &mut [usize], seed: u64) {
    if indices.is_empty() {
        return;
    }

    let mut rng = Lcg::new(seed);
    let n = indices.len();

    for i in (1..n).rev() {
        // Generate random index in [0, i]
        let j = rng.next_usize(0, i);
        indices.swap(i, j);
    }
}

/// Simple Linear Congruential Generator for reproducible randomness.
///
/// Uses glibc constants:
/// - a = 1103515245
/// - c = 12345
/// - m = 2^32 (implicit due to u32 wrapping)
///
/// This provides sufficient randomness for shuffling while remaining
/// completely self-contained.
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Creates a new LCG with the given seed.
    fn new(seed: u64) -> Self {
        Lcg { state: seed.wrapping_add(1) }
    }

    /// Generates the next random number in the sequence.
    #[inline]
    fn next(&mut self) -> u32 {
        // glibc LCG: state = (a * state + c) mod 2^32
        const A: u64 = 1103515245;
        const C: u64 = 12345;

        self.state = self.state.wrapping_mul(A).wrapping_add(C);
        (self.state & 0xFFFFFFFF) as u32
    }

    /// Generates a random usize in the range [min, max] inclusive.
    #[inline]
    fn next_usize(&mut self, min: usize, max: usize) -> usize {
        let range = max - min + 1;

        if range <= 1 {
            return min;
        }

        // Use a unbiased approach: reject samples outside the largest
        // multiple of `range` that fits in u32
        let threshold = (u32::MAX / range as u32) * range as u32;

        loop {
            let val = self.next();
            if val < threshold {
                return min + (val as usize) % range;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_kfold_splits_basic() {
        let splits = create_kfold_splits(10, 3, false, None).unwrap();
        assert_eq!(splits.len(), 3);

        // Check that each split has correct sizes
        // First fold gets extra sample (10 % 3 = 1)
        assert_eq!(splits[0].1.len(), 4); // test
        assert_eq!(splits[0].0.len(), 6); // train

        // Other folds have 3 samples each
        assert_eq!(splits[1].1.len(), 3);
        assert_eq!(splits[1].0.len(), 7);

        assert_eq!(splits[2].1.len(), 3);
        assert_eq!(splits[2].0.len(), 7);
    }

    #[test]
    fn test_create_kfold_splits_no_remainder() {
        let splits = create_kfold_splits(9, 3, false, None).unwrap();
        assert_eq!(splits.len(), 3);

        // All folds have equal size
        for (train, test) in &splits {
            assert_eq!(test.len(), 3);
            assert_eq!(train.len(), 6);
        }
    }

    #[test]
    fn test_create_kfold_splits_all_samples_used() {
        let n_samples = 20;
        let splits = create_kfold_splits(n_samples, 5, false, None).unwrap();

        let mut all_test_indices: Vec<usize> = Vec::new();
        for (_, test) in &splits {
            all_test_indices.extend(test);
        }

        all_test_indices.sort();
        let expected: Vec<usize> = (0..n_samples).collect();
        assert_eq!(all_test_indices, expected);
    }

    #[test]
    fn test_create_kfold_splits_no_overlap() {
        let splits = create_kfold_splits(15, 5, false, None).unwrap();

        for (train, test) in &splits {
            // No index should be in both train and test
            for &t in test {
                assert!(!train.contains(&t));
            }
        }
    }

    #[test]
    fn test_create_kfold_splits_with_shuffle() {
        let splits1 = create_kfold_splits(20, 5, true, Some(42)).unwrap();
        let splits2 = create_kfold_splits(20, 5, true, Some(42)).unwrap();

        // Same seed should produce identical splits
        assert_eq!(splits1.len(), splits2.len());
        for (s1, s2) in splits1.iter().zip(splits2.iter()) {
            assert_eq!(s1.0, s2.0);
            assert_eq!(s1.1, s2.1);
        }

        // Different seed should produce different splits (very likely)
        let splits3 = create_kfold_splits(20, 5, true, Some(123)).unwrap();
        assert_ne!(splits1[0].1, splits3[0].1);
    }

    #[test]
    fn test_create_kfold_splits_no_shuffle_reproducible() {
        let splits1 = create_kfold_splits(10, 3, false, None).unwrap();
        let splits2 = create_kfold_splits(10, 3, false, None).unwrap();

        // Without shuffling, indices should be in order
        // First fold test indices: [0, 1, 2, 3] (with remainder)
        assert_eq!(splits1[0].1, vec![0, 1, 2, 3]);
        assert_eq!(splits1, splits2);
    }

    #[test]
    fn test_create_kfold_splits_invalid_folds() {
        let result = create_kfold_splits(10, 1, false, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_kfold_splits_insufficient_samples() {
        let result = create_kfold_splits(5, 10, false, None);
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
    fn test_fisher_yates_shuffle_deterministic() {
        let mut indices1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut indices2 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        fisher_yates_shuffle(&mut indices1, 42);
        fisher_yates_shuffle(&mut indices2, 42);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_fisher_yates_shuffle_different_seeds() {
        let mut indices1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut indices2 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        fisher_yates_shuffle(&mut indices1, 42);
        fisher_yates_shuffle(&mut indices2, 123);

        assert_ne!(indices1, indices2);
    }

    #[test]
    fn test_fisher_yates_shuffle_permutation() {
        let mut indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let original = indices.clone();

        fisher_yates_shuffle(&mut indices, 42);

        // Should contain same elements, just reordered
        let mut sorted1 = indices.clone();
        let mut sorted2 = original.clone();
        sorted1.sort();
        sorted2.sort();
        assert_eq!(sorted1, sorted2);
    }

    #[test]
    fn test_fisher_yates_shuffle_empty() {
        let mut indices: Vec<usize> = vec![];
        fisher_yates_shuffle(&mut indices, 42);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_fisher_yates_shuffle_single() {
        let mut indices = vec![42];
        fisher_yates_shuffle(&mut indices, 42);
        assert_eq!(indices, vec![42]);
    }

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(42);

        // Test that values are within expected range
        for _ in 0..1000 {
            let val = rng.next_usize(0, 99);
            assert!(val <= 99);
        }
    }

    #[test]
    fn test_lcg_uniform_distribution() {
        let mut rng = Lcg::new(42);
        const N: usize = 10000;
        const RANGE: usize = 10;

        let mut counts = [0usize; RANGE];

        for _ in 0..N {
            let val = rng.next_usize(0, RANGE - 1);
            counts[val] += 1;
        }

        // Each bucket should have roughly N/RANGE counts
        // Allow for some variation (20% tolerance)
        let expected = N / RANGE;
        for count in counts {
            assert!(
                count > expected * 80 / 100 && count < expected * 120 / 100,
                "Count {} is outside expected range {}",
                count,
                expected
            );
        }
    }
}
