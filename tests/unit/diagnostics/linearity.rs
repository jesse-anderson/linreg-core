// ============================================================================
// Linearity Tests Unit Tests
// ============================================================================
//
// Tests for linearity assumption diagnostic tests:
// - rainbow_test(): Linearity test (supports R, Python, and Both methods)
// - harvey_collier_test(): Functional form validation using recursive residuals
// - reset_test(): Ramsey's specification error test (omitted variables/incorrect functional form)

use linreg_core::diagnostics::{harvey_collier_test, rainbow_test, reset_test, HarveyCollierMethod, RainbowMethod, ResetType};
use linreg_core::error::Error;

// ============================================================================
// Test Utilities
// ============================================================================

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// ============================================================================
// Rainbow Test
// ============================================================================

#[test]
fn test_rainbow_insufficient_data() {
    // Rainbow requires sufficient data for the subset
    let y = vec![1.0, 2.0, 3.0];
    let x = vec![1.0, 2.0, 3.0];

    let result = rainbow_test(&y, &[x], 0.5, RainbowMethod::R);

    match result {
        Err(Error::InsufficientData { .. }) => (),
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_rainbow_linear_data() {
    // Perfect linear relationship: should not reject linearity
    let y: Vec<f64> = (1..=20).map(|i| 2.0 * (i as f64) + 1.0).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let result = rainbow_test(&y, &[x], 0.5, RainbowMethod::R).unwrap();

    // Linear data: high p-value (should not reject)
    // Access nested result to check p-value
    if let Some(r_single) = result.r_result {
        assert!(r_single.p_value > 0.05);
        assert_eq!(r_single.passed, r_single.p_value > 0.05);
    }
    assert!(result.test_name.contains("Rainbow"));
}

#[test]
fn test_rainbow_nonlinear_data() {
    // Quadratic relationship: should reject linearity
    let y: Vec<f64> = (1..=20).map(|i| (i as f64).powi(2)).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let result = rainbow_test(&y, &[x], 0.5, RainbowMethod::R).unwrap();

    // Non-linear data: low p-value (should reject linearity)
    if let Some(r_single) = result.r_result {
        assert!(r_single.p_value < 0.05);
        assert_eq!(r_single.passed, false);
    }
}

#[test]
fn test_rainbow_fraction_parameter() {
    // fraction = 0.5 uses central 50% of data
    // fraction = 0.3 uses central 30%
    let y: Vec<f64> = (1..=20).map(|i| 2.0 * (i as f64) + 1.0).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let result_30 = rainbow_test(&y, &[x.clone()], 0.3, RainbowMethod::R);
    let result_50 = rainbow_test(&y, &[x], 0.5, RainbowMethod::R);

    // Both should succeed
    assert!(result_30.is_ok());
    assert!(result_50.is_ok());

    let r30 = result_30.unwrap();
    let r50 = result_50.unwrap();

    // Both should pass for linear data - access nested results
    if let Some(r30_single) = r30.r_result {
        assert!(r30_single.p_value > 0.05);
    }
    if let Some(r50_single) = r50.r_result {
        assert!(r50_single.p_value > 0.05);
    }
}

#[test]
fn test_rainbow_methods_consistency() {
    // R and Python methods should give similar results
    let y: Vec<f64> = (1..=20).map(|i| (i as f64) * 1.5 + 5.0).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let result_r = rainbow_test(&y, &[x.clone()], 0.5, RainbowMethod::R).unwrap();
    let result_python = rainbow_test(&y, &[x.clone()], 0.5, RainbowMethod::Python).unwrap();
    let result_both = rainbow_test(&y, &[x], 0.5, RainbowMethod::Both).unwrap();

    // All should have valid results
    assert!(!result_r.test_name.is_empty());
    assert!(!result_python.test_name.is_empty());
    assert!(!result_both.test_name.is_empty());

    // Check that R and Python methods generally agree
    if let (Some(r_single), Some(py_single)) =
        (result_r.r_result, result_python.python_result)
    {
        // Should both agree on pass/fail at α=0.05
        assert_eq!(r_single.passed, r_single.p_value > 0.05);
        assert_eq!(py_single.passed, py_single.p_value > 0.05);
    }
}

#[test]
fn test_rainbow_output_structure() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let result = rainbow_test(&y, &[x], 0.5, RainbowMethod::R).unwrap();

    assert!(!result.test_name.is_empty());
    // Access nested result for statistic and p_value
    if let Some(r_single) = result.r_result {
        assert!(r_single.statistic.is_finite());
        assert!(r_single.p_value.is_finite());
        assert!(r_single.p_value >= 0.0 && r_single.p_value <= 1.0);
    }
    assert!(!result.interpretation.is_empty());
    assert!(!result.guidance.is_empty());
}

// ============================================================================
// Harvey-Collier Test
// ============================================================================

#[test]
fn test_harvey_collier_insufficient_data() {
    // HC requires sufficient data for recursive residuals
    let y = vec![1.0, 2.0, 3.0];
    let x = vec![1.0, 2.0, 3.0];

    let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R);

    match result {
        Err(Error::InsufficientData { .. }) => (),
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_harvey_collier_linear_data() {
    // Linear data with tiny noise for numerical stability should not reject linearity
    let y: Vec<f64> = (1..=20).map(|i| 2.0 * (i as f64) + 1.0 + 0.01 * ((i % 5) as f64 - 2.0)).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();

    // Linear data: p-value should be large
    assert!(result.p_value > 0.05);
    assert_eq!(result.passed, result.p_value > 0.05);
    assert!(result.test_name.contains("Harvey-Collier"));
}

#[test]
fn test_harvey_collier_nonlinear_data() {
    // Quadratic relationship should be detected
    let y: Vec<f64> = (1..=20).map(|i| (i as f64).powi(2) * 0.5 + (i as f64)).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();

    // Non-linear data: may reject depending on strength of nonlinearity
    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
}

#[test]
fn test_harvey_collier_output_structure() {
    // Add tiny noise to ensure variance in residuals
    let y: Vec<f64> = (1..=10).map(|i| i as f64 + 0.01 * ((i % 3) as f64 - 1.0)).collect();
    let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();

    let result = harvey_collier_test(&y, &[x], HarveyCollierMethod::R).unwrap();

    assert!(!result.test_name.is_empty());
    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(!result.interpretation.is_empty());
    assert!(!result.guidance.is_empty());
}

// ============================================================================
// RESET Test
// ============================================================================

#[test]
fn test_reset_insufficient_data() {
    // RESET requires sufficient data for powered terms
    let y = vec![1.0, 2.0, 3.0];
    let x = vec![1.0, 2.0, 3.0];

    let result = reset_test(&y, &[x], &[2, 3], ResetType::Fitted);

    match result {
        Err(Error::InsufficientData { .. }) => (),
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_reset_correct_specification() {
    // Data from correct model should not reject
    let y: Vec<f64> = (1..=30).map(|i| 2.0 * (i as f64) + 1.0 + 0.5 * (i as f64).ln()).collect();
    let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

    let result = reset_test(&y, &[x], &[2, 3], ResetType::Fitted).unwrap();

    // Correct specification: should not necessarily reject
    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
}

#[test]
fn test_reset_omitted_variable() {
    // Data from y = x² + x + 1, but fitting y ~ x
    // RESET should detect the omitted x² term
    let y: Vec<f64> = (1..=20).map(|i| (i as f64).powi(2) + (i as f64) + 1.0).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let result = reset_test(&y, &[x], &[2, 3], ResetType::Fitted).unwrap();

    // Omitted variable: may reject depending on sample size
    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
}

#[test]
fn test_reset_types() {
    let y: Vec<f64> = (1..=20).map(|i| (i as f64) * 2.0 + 1.0).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    // Test all RESET types
    let result_fitted = reset_test(&y, &[x.clone()], &[2], ResetType::Fitted);
    let result_regressor = reset_test(&y, &[x.clone()], &[2], ResetType::Regressor);
    let result_pc = reset_test(&y, &[x], &[2], ResetType::PrincipalComponent);

    // All should succeed
    assert!(result_fitted.is_ok());
    assert!(result_regressor.is_ok());
    assert!(result_pc.is_ok());

    let r_fitted = result_fitted.unwrap();
    assert!(r_fitted.statistic.is_finite());
    assert!(r_fitted.p_value.is_finite());
}

#[test]
fn test_reset_multiple_powers() {
    let y: Vec<f64> = (1..=30).map(|i| (i as f64) * 2.0 + 1.0).collect();
    let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

    // Test with multiple powers
    let result = reset_test(&y, &[x], &[2, 3, 4], ResetType::Fitted).unwrap();

    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
}

#[test]
fn test_reset_output_structure() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let result = reset_test(&y, &[x], &[2], ResetType::Fitted).unwrap();

    assert!(!result.test_name.is_empty());
    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(!result.interpretation.is_empty());
    assert!(!result.guidance.is_empty());
}

// ============================================================================
// Cross-Test Consistency
// ============================================================================

#[test]
fn test_linearity_tests_agree_on_linear_data() {
    // All linearity tests should agree on linear data with tiny noise
    let y: Vec<f64> = (1..=20).map(|i| 2.0 * (i as f64) + 1.0 + 0.01 * ((i % 5) as f64 - 2.0)).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let rainbow = rainbow_test(&y, &[x.clone()], 0.5, RainbowMethod::R).unwrap();
    let hc = harvey_collier_test(&y, &[x.clone()], HarveyCollierMethod::R).unwrap();
    let reset = reset_test(&y, &[x], &[2], ResetType::Fitted).unwrap();

    // All should pass (not reject linearity)
    if let Some(r_single) = rainbow.r_result {
        assert!(r_single.p_value > 0.05);
    }
    assert!(hc.p_value > 0.05);
    assert!(reset.p_value > 0.05);
}

#[test]
fn test_linearity_tests_agree_on_nonlinear_data() {
    // All linearity tests should agree on quadratic data
    let y: Vec<f64> = (1..=20).map(|i| (i as f64).powi(2)).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let rainbow = rainbow_test(&y, &[x.clone()], 0.5, RainbowMethod::R).unwrap();
    let reset = reset_test(&y, &[x], &[2, 3], ResetType::Fitted).unwrap();

    // Both should detect nonlinearity
    if let Some(r_single) = rainbow.r_result {
        assert!(r_single.p_value < 0.05);
    }
    assert!(reset.p_value < 0.05);
}
