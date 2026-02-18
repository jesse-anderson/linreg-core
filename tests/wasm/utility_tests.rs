// ============================================================================
// Utility Function Tests
// ============================================================================
//
// Tests for statistical utility functions.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use linreg_core::wasm::*;

#[wasm_bindgen_test]
fn test_wasm_get_t_cdf() {
    // t = 0 should give CDF = 0.5
    let cdf = get_t_cdf(0.0, 10.0);
    assert!((cdf - 0.5).abs() < 0.01, "t=0 should give CDF ≈ 0.5");

    // Large positive t should give CDF close to 1
    let cdf_high = get_t_cdf(10.0, 10.0);
    assert!(cdf_high > 0.99, "Large t should give CDF close to 1");
}

#[wasm_bindgen_test]
fn test_wasm_get_t_critical() {
    let t_crit = get_t_critical(0.05, 10.0);

    // For df=10, alpha=0.05, t-critical ≈ 2.23
    assert!(
        t_crit > 2.0 && t_crit < 2.5,
        "t-critical for df=10, alpha=0.05 should be ~2.23, got {}",
        t_crit
    );
}

#[wasm_bindgen_test]
fn test_wasm_get_normal_inverse() {
    // p = 0.5 should give z = 0
    let z = get_normal_inverse(0.5);
    assert!((z - 0.0).abs() < 0.01, "p=0.5 should give z ≈ 0");

    // p = 0.975 should give z ≈ 1.96
    let z_975 = get_normal_inverse(0.975);
    assert!(
        z_975 > 1.9 && z_975 < 2.0,
        "p=0.975 should give z ≈ 1.96, got {}",
        z_975
    );
}

#[wasm_bindgen_test]
fn test_wasm_get_version() {
    let version = get_version();

    assert!(!version.is_empty(), "Version should not be empty");
    assert!(version.contains('.'), "Version should contain dots");
}

// ============================================================================
// Stats Utility Functions
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_stats_mean() {
    let data = serde_json::to_string(&vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let mean: f64 = serde_json::from_str(&stats_mean(data)).unwrap();
    assert!((mean - 3.0).abs() < 1e-10, "mean of [1..5] should be 3.0, got {}", mean);
}

#[wasm_bindgen_test]
fn test_wasm_stats_variance() {
    let data = serde_json::to_string(&vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
    let var: f64 = serde_json::from_str(&stats_variance(data)).unwrap();
    assert!(var > 0.0, "variance should be positive");
}

#[wasm_bindgen_test]
fn test_wasm_stats_stddev() {
    let data = serde_json::to_string(&vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
    let var: f64 = serde_json::from_str(&stats_variance(data.clone())).unwrap();
    let std: f64 = serde_json::from_str(&stats_stddev(data)).unwrap();
    assert!((std - var.sqrt()).abs() < 1e-10, "stddev should equal sqrt(variance)");
}

#[wasm_bindgen_test]
fn test_wasm_stats_median_odd() {
    let data = serde_json::to_string(&vec![3.0, 1.0, 5.0, 2.0, 4.0]).unwrap();
    let median: f64 = serde_json::from_str(&stats_median(data)).unwrap();
    assert!((median - 3.0).abs() < 1e-10, "median of [1,2,3,4,5] should be 3.0, got {}", median);
}

#[wasm_bindgen_test]
fn test_wasm_stats_median_even() {
    let data = serde_json::to_string(&vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let median: f64 = serde_json::from_str(&stats_median(data)).unwrap();
    assert!((median - 2.5).abs() < 1e-10, "median of [1,2,3,4] should be 2.5, got {}", median);
}

#[wasm_bindgen_test]
fn test_wasm_stats_quantile() {
    let data = serde_json::to_string(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
    let min: f64 = serde_json::from_str(&stats_quantile(data.clone(), 0.0)).unwrap();
    let max: f64 = serde_json::from_str(&stats_quantile(data, 1.0)).unwrap();
    assert!((min - 1.0).abs() < 1e-10, "0th quantile should be min (1.0), got {}", min);
    assert!((max - 10.0).abs() < 1e-10, "1st quantile should be max (10.0), got {}", max);
}

#[wasm_bindgen_test]
fn test_wasm_stats_correlation() {
    let x = serde_json::to_string(&vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = serde_json::to_string(&vec![2.0, 4.0, 6.0, 8.0, 10.0]).unwrap();
    let corr: f64 = serde_json::from_str(&stats_correlation(x, y)).unwrap();
    assert!((corr - 1.0).abs() < 1e-10, "Perfect positive correlation should be 1.0, got {}", corr);
}

#[wasm_bindgen_test]
fn test_wasm_stats_correlation_negative() {
    let x = serde_json::to_string(&vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = serde_json::to_string(&vec![10.0, 8.0, 6.0, 4.0, 2.0]).unwrap();
    let corr: f64 = serde_json::from_str(&stats_correlation(x, y)).unwrap();
    assert!((corr - (-1.0)).abs() < 1e-10, "Perfect negative correlation should be -1.0, got {}", corr);
}
