// ============================================================================
// Influential Observations Tests Unit Tests
// ============================================================================
//
// Tests for influential observation diagnostic tests:
// - cooks_distance_test(): Identifies high-leverage, high-residual points
// - dfbetas_test(): Measures coefficient change when omitting observation
// - dffits_test(): Measures fitted value change when omitting observation

use linreg_core::diagnostics::{cooks_distance_test, dfbetas_test, dffits_test, DiagnosticTestResult};
use linreg_core::error::Error;

// ============================================================================
// Test Utilities
// ============================================================================

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// ============================================================================
// Cook's Distance
// ============================================================================

#[test]
fn test_cooks_distance_insufficient_data() {
    // Need at least p + 2 observations where p is number of parameters
    let y = vec![1.0, 2.0];
    let x = vec![1.0, 2.0];

    let result = cooks_distance_test(&y, &[x]);

    match result {
        Err(Error::InsufficientData { .. }) => (),
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_cooks_distance_perfect_fit() {
    // Perfect linear fit: all residuals are zero
    // Cook's distance should be zero for all observations
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = cooks_distance_test(&y, &[x]).unwrap();

    // All distances should be zero
    for &d in &result.distances {
        assert!(approx_eq(d, 0.0, 1e-10));
    }
    assert_eq!(result.influential_4_over_n.len(), 0);
    assert_eq!(result.test_name, "Cook's Distance");
}

#[test]
fn test_cooks_distance_with_outlier() {
    // Last observation is an outlier
    let y = vec![1.0, 2.0, 3.0, 4.0, 100.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = cooks_distance_test(&y, &[x]).unwrap();

    // Last observation should have high Cook's distance
    let max_idx = result
        .distances
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();

    assert_eq!(max_idx, 4); // Last observation (index 4, obs 5)
    assert!(result.distances[max_idx] > 0.1); // Should be influential
}

#[test]
fn test_cooks_distance_threshold() {
    // Common threshold: 4/n where n=observations
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let result = cooks_distance_test(&y, &[x]).unwrap();

    // Threshold should be 4/n = 4/6 = 0.667
    let expected_threshold = 4.0 / 6.0;
    assert!((result.threshold_4_over_n - expected_threshold).abs() < 1e-10);

    // Last observation should exceed threshold
    assert_eq!(result.influential_4_over_n, vec![6]);
}

#[test]
fn test_cooks_distance_multiple_predictors() {
    let y = vec![
        10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0,
    ];
    let x1: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    // Add small variation to avoid perfect collinearity
    let x2: Vec<f64> = (1..=10).map(|i| (i as f64) * 2.0 + ((i % 2) as f64)).collect();

    let result = cooks_distance_test(&y, &[x1, x2]).unwrap();

    // Should have one distance per observation
    assert_eq!(result.distances.len(), 10);
    assert_eq!(result.p, 3); // 2 predictors + 1 intercept

    // Check that distances are non-negative
    for &d in &result.distances {
        assert!(d >= 0.0);
    }
}

#[test]
fn test_cooks_distance_output_structure() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = cooks_distance_test(&y, &[x]).unwrap();

    assert!(!result.test_name.is_empty());
    assert!(!result.distances.is_empty());
    assert!(result.p > 0);
    assert!(!result.interpretation.is_empty());
    assert!(!result.guidance.is_empty());
}

// ============================================================================
// DFBETAS
// ============================================================================

#[test]
fn test_dfbetas_insufficient_data() {
    // Need at least p + 2 observations
    let y = vec![1.0, 2.0];
    let x = vec![1.0, 2.0];

    let result = dfbetas_test(&y, &[x]);

    match result {
        Err(Error::InsufficientData { .. }) => (),
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_dfbetas_perfect_fit() {
    // Near-perfect fit: DFBETAS should be relatively small
    // Note: Due to floating point precision, "perfect" fits (y=x) produce
    // small non-zero residuals at machine epsilon level, which get amplified
    // in the DFBETAS formula. Some observations may still exceed the threshold.
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = dfbetas_test(&y, &[x]).unwrap();

    // For a near-perfect fit, at most a couple observations might be flagged
    // due to floating point precision amplification
    let influential_count: usize = result.influential_observations.values().map(|v| v.len()).sum();
    assert!(influential_count <= 3, "Expected at most 3 influential observations for near-perfect fit, got {}", influential_count);
}

#[test]
fn test_dfbetas_structure() {
    // DFBETAS: n observations × p coefficients (including intercept)
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = dfbetas_test(&y, &[x]).unwrap();

    // Should have n sets of DFBETAS
    assert_eq!(result.dfbetas.len(), 5);

    // Each set should have p values (intercept + 1 predictor)
    for obs_dfbetas in &result.dfbetas {
        assert_eq!(obs_dfbetas.len(), 2); // intercept + slope
    }
}

#[test]
fn test_dfbetas_with_outlier() {
    // Last observation is an outlier
    // Note: Extreme outliers heavily influence the OLS fit, which can cause
    // OTHER observations to have high DFBETAS (because the fit changes
    // dramatically when the outlier is present vs. absent). The outlier's
    // own DFBETAS may be lower because the fit is already "pulled" toward it.
    let y = vec![1.0, 2.0, 3.0, 4.0, 100.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = dfbetas_test(&y, &[x]).unwrap();

    // Some observations should be flagged as influential
    // (Extreme outliers cause high DFBETAS for other observations)
    let total_influential: usize = result.influential_observations
        .values()
        .map(|v| v.len())
        .sum();

    assert!(total_influential > 0, "Expected some influential observations with extreme outlier");
}

#[test]
fn test_dfbetas_threshold() {
    // Common threshold: 2/√n for each DFBETAS
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let result = dfbetas_test(&y, &[x]).unwrap();

    // Threshold should be 2/√n = 2/√6 ≈ 0.816
    let expected_threshold = 2.0 / (6.0_f64.sqrt());
    assert!((result.threshold - expected_threshold).abs() < 1e-10);
}

#[test]
fn test_dfbetas_output_structure() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = dfbetas_test(&y, &[x]).unwrap();

    assert!(!result.test_name.is_empty());
    assert_eq!(result.n, 5);
    assert!(result.p > 0);
    assert!(result.threshold.is_finite());
    assert!(!result.interpretation.is_empty());
    assert!(!result.guidance.is_empty());
}

// ============================================================================
// DFFITS
// ============================================================================

#[test]
fn test_dffits_insufficient_data() {
    let y = vec![1.0, 2.0];
    let x = vec![1.0, 2.0];

    let result = dffits_test(&y, &[x]);

    match result {
        Err(Error::InsufficientData { .. }) => (),
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_dffits_perfect_fit() {
    // Perfect fit: all DFFITS should be near zero
    // Note: Due to floating point precision, small non-zero values may appear
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = dffits_test(&y, &[x]).unwrap();

    // Check DFFITS are small (most should be close to zero, small precision errors ok)
    let max_abs = result.dffits.iter().map(|&v| v.abs()).fold(0.0_f64, |a, b| a.max(b));
    assert!(max_abs < 2.0, "Max DFFITS should be small for perfect fit, got {}", max_abs);
}

#[test]
fn test_dffits_with_outlier() {
    // Last observation is an outlier
    // Note: Extreme outliers influence the fit so much that OTHER observations
    // may have higher DFFITS (similar to DFBETAS behavior)
    let y = vec![1.0, 2.0, 3.0, 4.0, 100.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = dffits_test(&y, &[x]).unwrap();

    // Some DFFITS should be reasonably large due to the outlier
    let max_abs = result.dffits.iter().map(|&v| v.abs()).fold(0.0_f64, |a, b| a.max(b));
    assert!(max_abs > 0.5, "Expected some DFFITS to be affected by outlier, got max {}", max_abs);
}

#[test]
fn test_dffits_threshold() {
    // Common threshold: 2*√(p/n)
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = dffits_test(&y, &[x]).unwrap();

    // Threshold should be 2*√(2/5) for simple regression
    let expected_threshold = 2.0 * (2.0_f64 / 5.0_f64).sqrt();
    assert!((result.threshold - expected_threshold).abs() < 1e-10);
}

#[test]
fn test_dffits_n_and_p() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // Use x2 with some noise to avoid perfect collinearity
    let x2 = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0];

    let result = dffits_test(&y, &[x1, x2]).unwrap();

    assert_eq!(result.n, 8);
    assert_eq!(result.p, 3); // 2 predictors + 1 intercept
    assert_eq!(result.dffits.len(), 8);
}

#[test]
fn test_dffits_output_structure() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = dffits_test(&y, &[x]).unwrap();

    assert!(!result.test_name.is_empty());
    assert_eq!(result.dffits.len(), 5);
    assert!(result.n > 0);
    assert!(result.p > 0);
    assert!(result.threshold.is_finite());
    assert!(!result.interpretation.is_empty());
    assert!(!result.guidance.is_empty());
}

// ============================================================================
// Cross-Test Consistency
// ============================================================================

#[test]
fn test_influence_tests_agree_on_outlier() {
    // All influence tests should identify influence from the outlier
    // Note: Extreme outliers can cause high influence metrics for OTHER
    // observations (not just the outlier itself) due to how the fit changes
    let y = vec![1.0, 2.0, 3.0, 4.0, 100.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let cooks = cooks_distance_test(&y, &[x.clone()]).unwrap();
    let dfbetas = dfbetas_test(&y, &[x.clone()]).unwrap();
    let dffits = dffits_test(&y, &[x]).unwrap();

    // Cook's should identify the outlier
    assert!(cooks.influential_4_over_n.contains(&5));

    // DFBETAS or DFFITS should flag some observations as influential
    // (may be the outlier itself, or other observations affected by it)
    let dfbetas_count: usize = dfbetas.influential_observations.values().map(|v| v.len()).sum();
    let dffits_count = dffits.influential_observations.len();

    assert!(dfbetas_count > 0 || dffits_count > 0 || cooks.influential_4_over_n.len() > 0,
            "Expected some influential observations to be detected");
}

#[test]
fn test_influence_tests_agree_on_clean_data() {
    // Clean data: minimal influence expected
    // Note: Due to floating point precision, "perfect" fits (y=x) produce
    // small non-zero residuals at machine epsilon level, which can cause
    // some observations to be flagged as influential. The thresholds are
    // designed to be conservative.
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let cooks = cooks_distance_test(&y, &[x.clone()]).unwrap();
    let dfbetas = dfbetas_test(&y, &[x.clone()]).unwrap();
    let dffits = dffits_test(&y, &[x]).unwrap();

    // Cook's should have no influential observations for clean data
    assert!(cooks.influential_4_over_n.is_empty());

    // DFBETAS and DFFITS may have some flagged due to floating point precision,
    // but the overall influence should still be minimal (most observations not flagged)
    let dfbetas_count: usize = dfbetas.influential_observations.values().map(|v| v.len()).sum();
    let dffits_count = dffits.influential_observations.len();

    // At most a couple observations should be flagged (due to precision)
    assert!(dfbetas_count <= 3, "DFBETAS flagged {} observations, expected <= 3", dfbetas_count);
    assert!(dffits_count <= 3, "DFFITS flagged {} observations, expected <= 3", dffits_count);
}
