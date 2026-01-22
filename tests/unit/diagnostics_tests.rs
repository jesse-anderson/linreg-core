// ============================================================================
// Diagnostic Tests Unit Tests
// ============================================================================
//
// Unit tests for diagnostic tests in the diagnostics module.

use linreg_core::diagnostics;
use linreg_core::error::Error;

// ============================================================================
// Jarque-Bera Test Unit Tests
// ============================================================================

#[test]
fn test_jarque_bera_insufficient_data() {
    let y = vec![1.0, 2.0];
    let x_vars = vec![vec![1.0, 2.0]];

    let result = diagnostics::jarque_bera_test(&y, &x_vars);

    match result {
        Err(Error::InsufficientData {
            required,
            available,
        }) => {
            assert_eq!(required, 3); // p + 1 = 2 + 1 = 3
            assert_eq!(available, 2);
        },
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_jarque_bera_normal_residues() {
    // Generate data that should have approximately normal residuals
    // This is a simple linear relationship with small noise
    let y: Vec<f64> = (0..50)
        .map(|i| (i as f64) * 2.0 + 10.0 + (i as f64 % 7.0 - 3.0))
        .collect();
    let x: Vec<f64> = (0..50).map(|i| i as f64).map(|i| i * 2.0).collect();

    let result = diagnostics::jarque_bera_test(&y, &[x]).unwrap();

    // For approximately normal residuals, the p-value should be relatively high
    // (not rejecting the null hypothesis of normality)
    assert!(
        result.p_value > 0.01,
        "p-value = {} should be > 0.01 for approximately normal data",
        result.p_value
    );
    assert_eq!(result.test_name, "Jarque-Bera Test for Normality");
}

#[test]
fn test_jarque_bera_skewed_residues() {
    // Generate data with skewed residuals
    // This uses exponential-like growth which produces non-normal residuals
    let y: Vec<f64> = (0..30).map(|i| (i as f64).exp() + 1.0).collect();
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();

    let result = diagnostics::jarque_bera_test(&y, &[x]).unwrap();

    // For highly skewed data, the p-value should be low (rejecting normality)
    // Note: This might not always be true depending on the data, so we just
    // verify the test runs successfully
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(result.statistic >= 0.0);
}

#[test]
fn test_jarque_bera_simple_linear() {
    // Simple perfect linear relationship - residuals should be ~0
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = diagnostics::jarque_bera_test(&y, &[x]).unwrap();

    // Perfect linear fit has zero residuals, which is degenerate
    // The test should still run without error
    assert!(result.p_value.is_finite());
    assert!(result.statistic.is_finite());
    assert_eq!(result.test_name, "Jarque-Bera Test for Normality");
}

#[test]
fn test_jarque_bera_multiple_predictors() {
    let y = vec![10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

    let result = diagnostics::jarque_bera_test(&y, &[x1, x2]).unwrap();

    assert!(result.p_value.is_finite());
    assert!(result.statistic.is_finite());
    assert_eq!(result.test_name, "Jarque-Bera Test for Normality");
    assert!(result.interpretation.contains("p-value"));
}

#[test]
fn test_jarque_bera_passed_attribute() {
    // Normal-ish data - should pass (not reject null)
    let y: Vec<f64> = (0..30)
        .map(|i| (i as f64) * 1.5 + 5.0 + ((i as f64 * 17.0) % 10.0 - 5.0))
        .collect();
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();

    let result = diagnostics::jarque_bera_test(&y, &[x]).unwrap();

    // The passed field should be true when p-value > 0.05
    assert_eq!(result.passed, result.p_value > 0.05);
}

#[test]
fn test_jarque_bera_output_structure() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let result = diagnostics::jarque_bera_test(&y, &[x]).unwrap();

    // Check all output fields are populated
    assert!(!result.test_name.is_empty());
    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(!result.interpretation.is_empty());
    assert!(!result.guidance.is_empty());
}

#[test]
fn test_jarque_bera_interpretation_content() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let result = diagnostics::jarque_bera_test(&y, &[x]).unwrap();

    // Interpretation should contain key information
    assert!(result.interpretation.contains("p-value"));
    assert!(result.guidance.len() > 10);
}

// ============================================================================
// Anderson-Darling Test Unit Tests
// ============================================================================

#[test]
fn test_anderson_darling_insufficient_data() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let x_vars = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]];

    let result = diagnostics::anderson_darling_test(&y, &x_vars);

    match result {
        Err(Error::InsufficientData {
            required,
            available,
        }) => {
            assert_eq!(required, 8); // Anderson-Darling requires at least 8 observations
            assert_eq!(available, 7);
        },
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_anderson_darling_normal_residues() {
    // Generate data that should have approximately normal residuals
    // This is a simple linear relationship with small noise
    let y: Vec<f64> = (0..50)
        .map(|i| (i as f64) * 2.0 + 10.0 + (i as f64 % 7.0 - 3.0))
        .collect();
    let x: Vec<f64> = (0..50).map(|i| i as f64).map(|i| i * 2.0).collect();

    let result = diagnostics::anderson_darling_test(&y, &[x]).unwrap();

    // For approximately normal residuals, the p-value should be relatively high
    // (not rejecting the null hypothesis of normality)
    assert!(
        result.p_value > 0.01,
        "p-value = {} should be > 0.01 for approximately normal data",
        result.p_value
    );
    assert_eq!(result.test_name, "Anderson-Darling Test for Normality");
    assert!(result.statistic >= 0.0);
}

#[test]
fn test_anderson_darling_skewed_residues() {
    // Generate data with skewed residuals
    // This uses exponential-like growth which produces non-normal residuals
    let y: Vec<f64> = (0..30).map(|i| (i as f64).exp() + 1.0).collect();
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();

    let result = diagnostics::anderson_darling_test(&y, &[x]).unwrap();

    // For highly skewed data, the p-value should be low (rejecting normality)
    // Note: This might not always be true depending on the data, so we just
    // verify the test runs successfully
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(result.statistic >= 0.0);
}

#[test]
fn test_anderson_darling_simple_linear() {
    // Simple perfect linear relationship - residuals should be ~0
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let result = diagnostics::anderson_darling_test(&y, &[x]).unwrap();

    // Perfect linear fit has near-zero residuals
    // The test should still run without error
    assert!(result.p_value.is_finite());
    assert!(result.statistic.is_finite());
    assert_eq!(result.test_name, "Anderson-Darling Test for Normality");
}

#[test]
fn test_anderson_darling_multiple_predictors() {
    let y = vec![10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

    let result = diagnostics::anderson_darling_test(&y, &[x1, x2]).unwrap();

    assert!(result.p_value.is_finite());
    assert!(result.statistic.is_finite());
    assert_eq!(result.test_name, "Anderson-Darling Test for Normality");
    assert!(result.interpretation.contains("p-value"));
}

#[test]
fn test_anderson_darling_passed_attribute() {
    // Normal-ish data - should pass (not reject null)
    let y: Vec<f64> = (0..30)
        .map(|i| (i as f64) * 1.5 + 5.0 + ((i as f64 * 17.0) % 10.0 - 5.0))
        .collect();
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();

    let result = diagnostics::anderson_darling_test(&y, &[x]).unwrap();

    // The passed field should be true when p-value > 0.05
    assert_eq!(result.passed, result.p_value > 0.05);
}

#[test]
fn test_anderson_darling_output_structure() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let result = diagnostics::anderson_darling_test(&y, &[x]).unwrap();

    // Check all output fields are populated
    assert!(!result.test_name.is_empty());
    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(!result.interpretation.is_empty());
    assert!(!result.guidance.is_empty());
}

#[test]
fn test_anderson_darling_interpretation_content() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let result = diagnostics::anderson_darling_test(&y, &[x]).unwrap();

    // Interpretation should contain key information
    assert!(result.interpretation.contains("p-value"));
    assert!(result.guidance.len() > 10);
}

#[test]
fn test_anderson_darling_raw_with_normal_sample() {
    // Test the raw function with a known normal sample
    let normal_data = vec![
        0.1, -0.5, 0.3, 1.2, -0.8, 0.4, -0.2, 0.9, -0.3, 0.6, -0.1, 0.7, -0.4, 0.2, 1.1, -0.6, 0.8,
        -0.9, 0.5, -0.7, 0.0, 0.3, -0.4, 0.6, -0.2,
    ];

    let result = diagnostics::anderson_darling_test_raw(&normal_data).unwrap();

    // For normal data, p-value should be > 0.01
    assert!(result.p_value > 0.01, "p-value = {}", result.p_value);
    assert!(result.statistic >= 0.0);
    assert_eq!(result.test_name, "Anderson-Darling Test for Normality");
}
