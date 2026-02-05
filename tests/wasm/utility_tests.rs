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
