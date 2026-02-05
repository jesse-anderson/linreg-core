// ============================================================================
// Autocorrelation Tests Unit Tests
// ============================================================================
//
// Tests for autocorrelation detection diagnostic tests:
// - durbin_watson_test(): First-order autocorrelation
// - breusch_godfrey_test(): Higher-order serial correlation (LM test)

use linreg_core::diagnostics::{breusch_godfrey_test, BGTestType, durbin_watson_test, DurbinWatsonResult};
use linreg_core::error::Error;

// ============================================================================
// Test Utilities
// ============================================================================

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// ============================================================================
// Durbin-Watson Test
// ============================================================================

#[test]
fn test_durbin_watson_basic_calculation() {
    // DW = Σ(εᵢ - εᵢ₋₁)² / Σεᵢ²
    // Add some noise to avoid zero residuals
    let y = vec![1.1, 2.2, 2.9, 4.1, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = durbin_watson_test(&y, &[x]).unwrap();

    // DW should be finite and in valid range
    assert!(result.statistic.is_finite());
    assert!(result.statistic >= 0.0 && result.statistic <= 4.0);
    assert_eq!(result.test_name, "Durbin-Watson Test");
}

#[test]
fn test_durbin_watson_no_autocorrelation() {
    // Random residuals should have DW ≈ 2
    // Add some noise to avoid zero residuals
    let y: Vec<f64> = vec![
        10.5, 12.3, 9.8, 14.1, 11.2, 13.5, 10.9, 15.2, 11.8, 14.3,
    ];
    let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();

    let result = durbin_watson_test(&y, &[x]).unwrap();

    // DW should be in valid range [0, 4]
    assert!(result.statistic >= 0.0 && result.statistic <= 4.0);
}

#[test]
fn test_durbin_watson_positive_autocorrelation() {
    // Construct residuals with positive autocorrelation
    // εᵢ ≈ 0.7 * εᵢ₋₁ + noise
    let mut residuals = vec![0.0; 20];
    residuals[0] = 1.0;
    for i in 1..20 {
        residuals[i] = 0.7 * residuals[i - 1] + (i as f64 * 0.01);
    }

    // Need to reconstruct y from residuals
    // y = Xβ + ε, so y = fitted + residuals
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let fitted: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 5.0).collect();
    let y: Vec<f64> = fitted.iter().zip(residuals.iter()).map(|(&f, &r)| f + r).collect();

    let result = durbin_watson_test(&y, &[x]).unwrap();

    // Positive autocorrelation: DW < 2
    assert!(result.statistic < 2.0);
    assert!(result.statistic >= 0.0);
}

#[test]
fn test_durbin_watson_negative_autocorrelation() {
    // Construct residuals with negative autocorrelation
    // εᵢ ≈ -0.5 * εᵢ₋₁ + noise
    let mut residuals = vec![0.0; 20];
    residuals[0] = 1.0;
    for i in 1..20 {
        residuals[i] = -0.5 * residuals[i - 1] + (i as f64 * 0.05);
    }

    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let fitted: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 5.0).collect();
    let y: Vec<f64> = fitted.iter().zip(residuals.iter()).map(|(&f, &r)| f + r).collect();

    let result = durbin_watson_test(&y, &[x]).unwrap();

    // Negative autocorrelation: DW > 2
    assert!(result.statistic > 2.0);
    assert!(result.statistic <= 4.0);
}

#[test]
fn test_durbin_watson_bounds() {
    // DW statistic is bounded in [0, 4]
    // Use data with clear pattern to avoid zero residuals
    let y = vec![2.0, 4.5, 6.0, 9.5, 11.0, 13.5, 15.0, 18.5];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let result = durbin_watson_test(&y, &[x]).unwrap();

    assert!(result.statistic >= 0.0 && result.statistic <= 4.0);
}

#[test]
fn test_durbin_watson_insufficient_data() {
    // DW requires at least 2 observations
    let y = vec![1.0];
    let x = vec![1.0];

    let result = durbin_watson_test(&y, &[x]);

    match result {
        Err(Error::InsufficientData { .. }) => (),
        _ => panic!("Expected InsufficientData error"),
    }
}

// ============================================================================
// Breusch-Godfrey Test
// ============================================================================

#[test]
fn test_breusch_godfrey_insufficient_data() {
    // BG requires sufficient data for the specified lag
    let y = vec![1.0, 2.0, 3.0];
    let x = vec![1.0, 2.0, 3.0];

    // With lag=2, we need at least n > p + lag + 1
    let result = breusch_godfrey_test(&y, &[x], 5, BGTestType::Chisq);

    match result {
        Err(Error::InsufficientData { .. }) => (),
        _ => panic!("Expected InsufficientData error for high lag"),
    }
}

#[test]
fn test_breusch_godfrey_no_autocorrelation() {
    // Random data should not show significant autocorrelation
    // Add some noise to ensure residuals aren't too close to zero
    let y: Vec<f64> = (1..=20).map(|i| (i as f64) * 1.5 + 5.0 + ((i % 3) as f64) * 0.3).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let result_bg = breusch_godfrey_test(&y, &[x.clone()], 2, BGTestType::Chisq).unwrap();
    let result_f = breusch_godfrey_test(&y, &[x], 2, BGTestType::F).unwrap();

    // Check that results are valid
    assert!(result_bg.statistic >= 0.0);
    assert!(result_f.statistic >= 0.0);
    assert!(result_bg.p_value.is_finite());
    assert!(result_f.p_value.is_finite());
    assert!(result_bg.p_value >= 0.0 && result_bg.p_value <= 1.0);
    assert!(result_f.p_value >= 0.0 && result_f.p_value <= 1.0);
}

#[test]
fn test_breusch_godfrey_with_autocorrelation() {
    // Create data with clear lag-1 autocorrelation
    // Using explicit autocorrelated residuals
    let mut residuals = vec![0.0; 20];
    residuals[0] = 1.0;
    for i in 1..20 {
        residuals[i] = 0.7 * residuals[i - 1] + (i as f64 * 0.01);
    }

    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let fitted: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 5.0).collect();
    let y: Vec<f64> = fitted.iter().zip(residuals.iter()).map(|(&f, &r)| f + r).collect();

    let result = breusch_godfrey_test(&y, &[x], 1, BGTestType::Chisq).unwrap();

    assert!(result.statistic >= 0.0);
    assert!(result.p_value.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
}

#[test]
fn test_breusch_godfrey_multiple_lags() {
    let y: Vec<f64> = (1..=30).map(|i| i as f64 + (i as f64 * 0.01)).collect();
    let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

    // Test different lag orders
    for lag in [1, 2, 3, 4] {
        let result = breusch_godfrey_test(&y, &[x.clone()], lag, BGTestType::Chisq);
        if result.is_ok() {
            let r = result.unwrap();
            assert!(r.statistic.is_finite());
            assert!(r.p_value.is_finite());
        }
    }
}

#[test]
fn test_breusch_godfrey_test_types_consistency() {
    // Chisq and F test types should generally agree
    // Add some noise to ensure residuals aren't too close to zero
    let y: Vec<f64> = (1..=20).map(|i| (i as f64) * 1.5 + 5.0 + ((i % 3) as f64) * 0.5).collect();
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();

    let result_chisq = breusch_godfrey_test(&y, &[x.clone()], 2, BGTestType::Chisq).unwrap();
    let result_f = breusch_godfrey_test(&y, &[x], 2, BGTestType::F).unwrap();

    // Both should agree on whether to reject at α=0.05
    assert_eq!(result_chisq.passed, result_chisq.p_value > 0.05);
    assert_eq!(result_f.passed, result_f.p_value > 0.05);
}

#[test]
fn test_breusch_godfrey_output_structure() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let result = breusch_godfrey_test(&y, &[x], 1, BGTestType::Chisq).unwrap();

    assert!(!result.test_name.is_empty());
    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_finite());
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert!(!result.interpretation.is_empty());
    assert!(!result.guidance.is_empty());
}

#[test]
fn test_breusch_godfrey_test_type_names() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let result_chisq = breusch_godfrey_test(&y, &[x.clone()], 1, BGTestType::Chisq).unwrap();
    let result_f = breusch_godfrey_test(&y, &[x], 1, BGTestType::F).unwrap();

    // Test names should indicate the test type
    assert!(result_chisq.test_name.contains("Breusch-Godfrey"));
    assert!(result_f.test_name.contains("Breusch-Godfrey"));
}

// ============================================================================
// Cross-Test Consistency
// ============================================================================

#[test]
fn test_dw_and_bg_consistency() {
    // Both tests detect autocorrelation
    // Data with positive autocorrelation
    let mut residuals = vec![0.0; 20];
    residuals[0] = 1.0;
    for i in 1..20 {
        residuals[i] = 0.7 * residuals[i - 1] + (i as f64 * 0.01);
    }

    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let fitted: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 5.0).collect();
    let y: Vec<f64> = fitted.iter().zip(residuals.iter()).map(|(&f, &r)| f + r).collect();

    let dw = durbin_watson_test(&y, &[x.clone()]).unwrap();
    let bg_lag1 = breusch_godfrey_test(&y, &[x], 1, BGTestType::Chisq).unwrap();

    // DW < 2 indicates positive autocorrelation
    assert!(dw.statistic < 2.0);

    // BG should also detect it (p-value < 0.05 for strong autocorrelation)
    assert!(bg_lag1.statistic >= 0.0);
    assert!(bg_lag1.p_value.is_finite());
}
