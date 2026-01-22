// ============================================================================
// Input Validation Unit Tests
// ============================================================================
//
// Tests for validating edge cases and invalid inputs across the library.
// These tests ensure the library handles malformed data gracefully.

use linreg_core::{core::ols_regression, diagnostics::*, Error};

// ============================================================================
// NaN Value Tests
// ============================================================================

#[test]
fn test_ols_rejects_nan_in_y() {
    let y = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    // Should fail gracefully - either with InvalidInput or through QR decomposition
    assert!(result.is_err(), "OLS should reject NaN values in y");
}

#[test]
fn test_ols_rejects_nan_in_x() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    assert!(result.is_err(), "OLS should reject NaN values in x");
}

#[test]
fn test_jarque_bera_rejects_nan_in_y() {
    let y = vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let result = jarque_bera_test(&y, &[x1]);

    assert!(result.is_err(), "Jarque-Bera should reject NaN values");
}

#[test]
fn test_shapiro_wilk_raw_rejects_nan() {
    let sample = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];

    let result = shapiro_wilk_test_raw(&sample);

    // NaN values should cause the test to fail
    // The sort will place NaN at the end, but the range check should fail
    assert!(result.is_err(), "Shapiro-Wilk should reject NaN values");
}

#[test]
fn test_anderson_darling_raw_rejects_nan() {
    let sample = vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0, 8.0];

    let result = anderson_darling_test_raw(&sample);

    assert!(result.is_err(), "Anderson-Darling should reject NaN values");
}

// ============================================================================
// Infinite Value Tests
// ============================================================================

#[test]
fn test_ols_rejects_infinity_in_y() {
    let y = vec![1.0, 2.0, f64::INFINITY, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    assert!(result.is_err(), "OLS should reject infinite values in y");
}

#[test]
fn test_ols_rejects_negative_infinity_in_y() {
    let y = vec![1.0, 2.0, f64::NEG_INFINITY, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    assert!(
        result.is_err(),
        "OLS should reject negative infinite values in y"
    );
}

#[test]
fn test_durbin_watson_rejects_infinity() {
    let y = vec![1.0, 2.0, f64::INFINITY, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = durbin_watson_test(&y, &[x1]);

    assert!(
        result.is_err(),
        "Durbin-Watson should reject infinite values"
    );
}

// ============================================================================
// Dimension Mismatch Tests
// ============================================================================

#[test]
fn test_ols_rejects_mismatched_lengths() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0]; // Different length

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    // May fail with IndexOutOfBounds during QR or with DimensionMismatch
    assert!(
        result.is_err(),
        "OLS should reject mismatched vector lengths"
    );
}

#[test]
fn test_ols_rejects_multiple_x_with_mismatched_lengths() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 4.0, 6.0]; // Different length

    let result = ols_regression(
        &y,
        &[x1, x2],
        &["Intercept".to_string(), "X1".to_string(), "X2".to_string()],
    );

    assert!(
        result.is_err(),
        "OLS should reject mismatched x variable lengths"
    );
}

#[test]
fn test_diagnostics_rejects_mismatched_lengths() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0]; // Different length

    let test_results: Vec<Result<DiagnosticTestResult, Error>> = vec![
        jarque_bera_test(&y, &[x1.clone()]),
        breusch_pagan_test(&y, &[x1.clone()]),
    ];

    for result in test_results {
        assert!(
            result.is_err(),
            "Diagnostic tests should reject mismatched lengths"
        );
    }

    // Durbin-Watson returns a different result type, test separately
    let dw_result = durbin_watson_test(&y, &[x1]);
    assert!(
        dw_result.is_err(),
        "Durbin-Watson should reject mismatched lengths"
    );
}

// ============================================================================
// Empty Vector Tests
// ============================================================================

#[test]
fn test_ols_rejects_empty_y() {
    let y: Vec<f64> = vec![];
    let x1: Vec<f64> = vec![];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    match result {
        Err(Error::InsufficientData { .. }) => (),
        _ => panic!(
            "Expected InsufficientData error for empty y, got {:?}",
            result
        ),
    }
}

#[test]
fn test_ols_rejects_single_observation() {
    let y = vec![5.0];
    let x1 = vec![2.0];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    match result {
        Err(Error::InsufficientData { .. }) => (),
        _ => panic!("Expected InsufficientData error for single observation"),
    }
}

#[test]
fn test_rainbow_rejects_insufficient_data() {
    let y = vec![1.0, 2.0];
    let x1 = vec![1.0, 2.0];

    let result = rainbow_test(&y, &[x1], 0.5, linreg_core::diagnostics::RainbowMethod::R);

    assert!(
        result.is_err(),
        "Rainbow test should reject insufficient data"
    );
}

// ============================================================================
// Constant Value Tests (Zero Variance)
// ============================================================================

#[test]
fn test_ols_with_constant_y() {
    let y = vec![5.0; 10];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    // Should work - constant y with varying x gives flat regression
    assert!(
        result.is_ok(),
        "OLS should handle constant y with varying x"
    );

    let r = result.unwrap();
    // X1 coefficient should be near zero since y is constant
    assert!(r.coefficients[1].abs() < 1e-10);
}

#[test]
fn test_ols_with_constant_x() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![5.0; 5]; // Constant x

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    // Constant x creates perfect multicollinearity with intercept
    // The library may use ridge fallback or return SingularMatrix error
    match result {
        Ok(_) => {},                      // Ridge fallback succeeded
        Err(Error::SingularMatrix) => {}, // Correctly detected multicollinearity
        Err(other) => panic!("Unexpected error: {:?}", other),
    }
}

#[test]
fn test_shapiro_wilk_raw_rejects_constant_sample() {
    let sample = vec![5.0; 100];

    let result = shapiro_wilk_test_raw(&sample);

    assert!(
        result.is_err(),
        "Shapiro-Wilk should reject constant sample"
    );
}

#[test]
fn test_anderson_darling_raw_rejects_constant_sample() {
    let sample = vec![5.0; 100];

    let result = anderson_darling_test_raw(&sample);

    assert!(
        result.is_err(),
        "Anderson-Darling should reject constant sample"
    );
}

// ============================================================================
// Very Small Sample Tests
// ============================================================================

#[test]
fn test_shapiro_wilk_minimum_sample_size() {
    let sample = vec![1.0, 2.0, 3.0]; // Minimum valid size

    let result = shapiro_wilk_test_raw(&sample);

    assert!(result.is_ok(), "Shapiro-Wilk should accept n=3");

    let r = result.unwrap();
    assert!(r.statistic > 0.0 && r.statistic <= 1.0);
}

#[test]
fn test_shapiro_wilk_below_minimum() {
    let sample = vec![1.0, 2.0];

    let result = shapiro_wilk_test_raw(&sample);

    match result {
        Err(Error::InsufficientData {
            required,
            available,
        }) => {
            assert_eq!(required, 3);
            assert_eq!(available, 2);
        },
        _ => panic!("Expected InsufficientData error for n=2"),
    }
}

#[test]
fn test_anderson_darling_minimum_sample_size() {
    let sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // Minimum valid size

    let result = anderson_darling_test_raw(&sample);

    assert!(result.is_ok(), "Anderson-Darling should accept n=8");
}

#[test]
fn test_anderson_darling_below_minimum() {
    let sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    let result = anderson_darling_test_raw(&sample);

    match result {
        Err(Error::InsufficientData {
            required,
            available,
        }) => {
            assert_eq!(required, 8);
            assert_eq!(available, 7);
        },
        _ => panic!("Expected InsufficientData error for n=7"),
    }
}

// ============================================================================
// Extreme Value Tests (Large but Valid)
// ============================================================================

#[test]
fn test_ols_handles_large_values() {
    let y = vec![1e10, 2e10, 3e10, 4e10, 5e10];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    assert!(result.is_ok(), "OLS should handle large values");

    let r = result.unwrap();
    // Coefficient should be approximately 1e10
    assert!((r.coefficients[1] - 1e10).abs() < 1e8);
}

#[test]
fn test_ols_handles_small_values() {
    let y = vec![1e-10, 2e-10, 3e-10, 4e-10, 5e-10];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    assert!(result.is_ok(), "OLS should handle small values");

    let r = result.unwrap();
    // Coefficient should be approximately 1e-10
    assert!((r.coefficients[1] - 1e-10).abs() < 1e-12);
}

#[test]
fn test_shapiro_wilk_maximum_sample_size() {
    // Create sample at the maximum allowed size (5000)
    let sample: Vec<f64> = (0..5000).map(|i| (i as f64) / 5000.0).collect();

    let result = shapiro_wilk_test_raw(&sample);

    assert!(result.is_ok(), "Shapiro-Wilk should accept n=5000");
}

#[test]
fn test_shapiro_wilk_above_maximum() {
    // Create sample exceeding the maximum allowed size
    let sample: Vec<f64> = (0..6000).map(|i| i as f64).collect();

    let result = shapiro_wilk_test_raw(&sample);

    assert!(result.is_err(), "Shapiro-Wilk should reject n > 5000");
}

// ============================================================================
// Zero Value Tests
// ============================================================================

#[test]
fn test_ols_handles_all_zeros_in_x() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![0.0; 5];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    // All zeros in x creates perfect multicollinearity with intercept
    // This should return SingularMatrix error (correct behavior)
    // The library may use ridge fallback, so we accept either outcome
    match result {
        Ok(_) => {},                      // Ridge fallback succeeded
        Err(Error::SingularMatrix) => {}, // Correctly detected multicollinearity
        Err(other) => panic!("Unexpected error: {:?}", other),
    }
}

#[test]
fn test_ols_handles_all_zeros_in_y() {
    let y = vec![0.0; 5];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    assert!(result.is_ok(), "OLS should handle all zeros in y");

    let r = result.unwrap();
    // All coefficients should be near zero
    assert!(r.coefficients.iter().all(|c| c.abs() < 1e-10));
}

// ============================================================================
// Perfect Multicollinearity Tests
// ============================================================================

#[test]
fn test_ols_detects_perfect_multicollinearity() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Exactly 2 * x1

    let result = ols_regression(
        &y,
        &[x1, x2],
        &["Intercept".to_string(), "X1".to_string(), "X2".to_string()],
    );

    // Should detect multicollinearity
    match result {
        Err(Error::SingularMatrix) => {
            // Expected - perfect multicollinearity
        },
        Ok(r) => {
            // Some implementations may use ridge fallback
            // Check that VIF is high for collinear variables
            assert!(r.vif.iter().any(|v| v.vif > 100.0));
        },
        _ => panic!("Unexpected result: {:?}", result),
    }
}

// ============================================================================
// Mixed Valid/Invalid Data Tests
// ============================================================================

#[test]
fn test_ols_with_one_zero_value() {
    let y = vec![1.0, 2.0, 0.0, 4.0, 5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    assert!(result.is_ok(), "OLS should handle data with zero values");
}

#[test]
fn test_ols_with_negative_values() {
    let y = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = ols_regression(&y, &[x1], &["Intercept".to_string(), "X1".to_string()]);

    assert!(result.is_ok(), "OLS should handle negative values");

    let r = result.unwrap();
    // Coefficient should be negative
    assert!(r.coefficients[1] < 0.0);
}
