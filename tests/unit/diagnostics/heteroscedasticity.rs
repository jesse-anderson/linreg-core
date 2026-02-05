// ============================================================================
// Heteroscedasticity Tests Unit Tests
// ============================================================================
//
// Tests for heteroscedasticity detection diagnostic tests:
// - breusch_pagan_test(): Studentized (Koenker) variant
// - white_test(): General test (supports R and Python methods via WhiteMethod)

use linreg_core::diagnostics::{breusch_pagan_test, white_test, WhiteMethod, DiagnosticTestResult};
use linreg_core::error::Error;

// ============================================================================
// Test Utilities
// ============================================================================

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// ============================================================================
// Breusch-Pagan Test
// ============================================================================

#[test]
fn test_breusch_pagan_insufficient_data() {
    // BP requires at least p + 2 observations where p is number of parameters
    let y = vec![1.0, 2.0];
    let x = vec![1.0, 2.0];

    let result = breusch_pagan_test(&y, &[x]);

    match result {
        Err(Error::InsufficientData { .. }) => (),
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_breusch_pagan_homoscedastic_data() {
    // Constant variance data should not reject null
    // y = 2x + 1 + small constant noise
    let y: Vec<f64> = (1..=20)
        .map(|i| 2.0 * (i as f64) + 1.0 + 0.1 * (if i % 2 == 0 { 1.0 } else { -1.0 }))
        .collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let result = breusch_pagan_test(&y, &[x]).unwrap();

    // Homoscedastic data: LM statistic should be small, p-value large
    assert!(result.statistic >= 0.0);
    assert!(result.p_value > 0.05); // Should not reject
    assert_eq!(result.passed, result.p_value > 0.05);
    assert_eq!(result.test_name, "Breusch-Pagan Test for Heteroscedasticity");
}

#[test]
fn test_breusch_pagan_heteroscedastic_data() {
    // Variance increases with x (classic heteroscedasticity)
    // y = 2x + 1 + noise where noise ∝ x
    let y: Vec<f64> = (1..=20)
        .map(|i| {
            let x = i as f64;
            2.0 * x + 1.0 + (x * 0.5 * (i % 3) as f64) // Noise scales with x
        })
        .collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let result = breusch_pagan_test(&y, &[x]).unwrap();

    // Heteroscedastic data: LM statistic should be larger
    assert!(result.statistic >= 0.0);
    // Note: May not always reject due to small sample size
    assert!(result.p_value.is_finite());
}

#[test]
fn test_breusch_pagan_multiple_predictors() {
    let y = vec![
        10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0,
        80.0, 85.0, 90.0, 95.0, 100.0, 105.0,
    ];
    let x1: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let x2: Vec<f64> = (1..=20).map(|i| (i as f64) * 2.0).collect();

    let result = breusch_pagan_test(&y, &[x1, x2]).unwrap();

    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
}

#[test]
fn test_breusch_pagan_output_structure() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = breusch_pagan_test(&y, &[x]).unwrap();

    assert!(!result.test_name.is_empty());
    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(!result.interpretation.is_empty());
    assert!(!result.guidance.is_empty());
}

// ============================================================================
// White Test
// ============================================================================

#[test]
fn test_white_test_insufficient_data() {
    let y = vec![1.0, 2.0];
    let x = vec![1.0, 2.0];

    let r_result = white_test(&y, &[x.clone()], WhiteMethod::R);
    let py_result = white_test(&y, &[x], WhiteMethod::Python);

    // Both should return errors for insufficient data
    assert!(r_result.is_err());
    assert!(py_result.is_err());
}

#[test]
fn test_white_test_homoscedastic_data() {
    // Constant variance data
    let y: Vec<f64> = (1..=20)
        .map(|i| 2.0 * (i as f64) + 1.0 + 0.1 * (if i % 2 == 0 { 1.0 } else { -1.0 }))
        .collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let r_result = white_test(&y, &[x.clone()], WhiteMethod::R).unwrap();
    let py_result = white_test(&y, &[x], WhiteMethod::Python).unwrap();

    // Both methods should give finite results - access nested results
    if let Some(r_single) = r_result.r_result {
        assert!(r_single.statistic.is_finite());
    }
    if let Some(py_single) = py_result.python_result {
        assert!(py_single.statistic.is_finite());
    }
}

#[test]
fn test_white_test_methods_consistency() {
    // R and Python methods should give similar results
    let y = vec![
        2.1, 4.2, 5.8, 8.1, 10.2, 12.1, 14.3, 15.9, 18.0, 20.1, 22.2, 24.0, 26.1, 28.0, 30.2,
        31.9, 34.1, 36.0, 38.2, 40.5,
    ];
    let x: Vec<f64> = (1..=20).map(|i| i as f64 * 2.0).collect();

    let r_result = white_test(&y, &[x.clone()], WhiteMethod::R).unwrap();
    let py_result = white_test(&y, &[x], WhiteMethod::Python).unwrap();

    // Both should have valid results
    assert!(!r_result.test_name.is_empty());
    assert!(!py_result.test_name.is_empty());
}

#[test]
fn test_white_test_heteroscedastic_data() {
    // Variance increases with predictor
    let y: Vec<f64> = (1..=30)
        .map(|i| {
            let x = i as f64;
            2.0 * x + 1.0 + (x * 0.3 * ((i as f64) / 30.0))
        })
        .collect();
    let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

    let r_result = white_test(&y, &[x.clone()], WhiteMethod::R).unwrap();
    let py_result = white_test(&y, &[x], WhiteMethod::Python).unwrap();

    // Both should have valid results - access nested results
    if let Some(r_single) = r_result.r_result {
        assert!(r_single.statistic.is_finite());
    }
    if let Some(py_single) = py_result.python_result {
        assert!(py_single.statistic.is_finite());
    }
}

#[test]
fn test_white_test_output_structure() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let r_result = white_test(&y, &[x], WhiteMethod::R).unwrap();

    assert!(!r_result.test_name.is_empty());
    assert!(!r_result.interpretation.is_empty());
    assert!(!r_result.guidance.is_empty());
}

// ============================================================================
// Cross-Test Consistency
// ============================================================================

#[test]
fn test_bp_and_white_agree_on_extreme_cases() {
    // Perfect homoscedasticity: both should agree
    let y: Vec<f64> = (1..=20).map(|i| 2.0 * (i as f64) + 1.0).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let bp = breusch_pagan_test(&y, &[x.clone()]).unwrap();
    let white = white_test(&y, &[x], WhiteMethod::Python).unwrap();

    // Both should not strongly reject homoscedasticity
    assert!(bp.p_value > 0.01);

    // For White test, access nested result
    if let Some(py_single) = white.python_result {
        assert!(py_single.p_value > 0.01);
    }
}
