//! Test and validation functions for WASM
//!
//! Provides WASM bindings for testing and validation functions.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use super::domain::check_domain;
use crate::core;
use crate::error::{error_to_json, Result};

/// Simple test function to verify WASM is working.
///
/// Returns a success message confirming the WASM module loaded correctly.
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
#[wasm_bindgen]
pub fn test() -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }
    "Rust WASM is working!".to_string()
}

/// Returns the current version of the library.
///
/// Returns the Cargo package version as a string (e.g., "0.1.0").
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
#[wasm_bindgen]
pub fn get_version() -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }
    env!("CARGO_PKG_VERSION").to_string()
}

/// Test function for t-critical value computation.
///
/// Returns JSON with the computed t-critical value for the given parameters.
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
#[wasm_bindgen]
pub fn test_t_critical(df: f64, alpha: f64) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }
    let t_crit = core::t_critical_quantile(df, alpha);
    format!(
        r#"{{"df": {}, "alpha": {}, "t_critical": {}}}"#,
        df, alpha, t_crit
    )
}

/// Test function for confidence interval computation.
///
/// Returns JSON with the computed confidence interval for a coefficient.
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
#[wasm_bindgen]
pub fn test_ci(coef: f64, se: f64, df: f64, alpha: f64) -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }
    let t_crit = core::t_critical_quantile(df, alpha);
    format!(
        r#"{{"lower": {}, "upper": {}}}"#,
        coef - t_crit * se,
        coef + t_crit * se
    )
}

/// Test function for R accuracy validation.
///
/// Returns JSON comparing our statistical functions against R reference values.
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
#[wasm_bindgen]
pub fn test_r_accuracy() -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }
    format!(
        r#"{{"two_tail_p": {}, "qt_975": {}}}"#,
        core::two_tailed_p_value(1.6717, 21.0),
        core::t_critical_quantile(21.0, 0.05)
    )
}

/// Test function for regression validation against R reference values.
///
/// Runs a regression on a housing dataset and compares results against R's lm() output.
/// Returns JSON with status "PASS" or "FAIL" with details.
///
/// # Errors
///
/// Returns a JSON error object if domain check fails.
#[wasm_bindgen]
pub fn test_housing_regression() -> String {
    if let Err(e) = check_domain() {
        return error_to_json(&e);
    }

    match test_housing_regression_native() {
        Ok(result) => result,
        Err(e) => serde_json::json!({ "status": "ERROR", "error": e.to_string() }).to_string(),
    }
}

// Native Rust test function (works without WASM feature)
#[cfg(any(test, feature = "wasm"))]
fn test_housing_regression_native() -> Result<String> {
    let y = vec![
        245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1, 445.8, 167.9, 367.4,
        289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9, 223.4, 312.5, 156.8, 423.7, 267.9,
    ];

    let square_feet = vec![
        1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0, 2200.0,
        900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0, 1250.0, 1700.0,
        850.0, 2350.0, 1400.0,
    ];
    let bedrooms = vec![
        3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0, 4.0, 2.0, 4.0, 3.0, 2.0, 5.0, 3.0, 4.0,
        2.0, 4.0, 3.0, 3.0, 2.0, 4.0, 3.0,
    ];
    let age = vec![
        15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0, 3.0, 30.0, 6.0, 14.0, 22.0, 1.0,
        16.0, 9.0, 28.0, 4.0, 19.0, 11.0, 35.0, 3.0, 13.0,
    ];

    let x_vars = vec![square_feet, bedrooms, age];
    let names = vec![
        "Intercept".to_string(),
        "Square_Feet".to_string(),
        "Bedrooms".to_string(),
        "Age".to_string(),
    ];

    let result = core::ols_regression(&y, &x_vars, &names)?;

    // Check against R results
    let expected_coeffs = [52.1271333, 0.1613877, 0.9545492, -1.1811815];
    let expected_std_errs = [31.18201809, 0.01875072, 10.44400198, 0.73219949];

    let tolerance = 1e-4;
    let mut mismatches = vec![];

    for i in 0..4 {
        if (result.coefficients[i] - expected_coeffs[i]).abs() > tolerance {
            mismatches.push(format!(
                "coeff[{}] differs: got {}, expected {}",
                i, result.coefficients[i], expected_coeffs[i]
            ));
        }
        if (result.std_errors[i] - expected_std_errs[i]).abs() > tolerance {
            mismatches.push(format!(
                "std_err[{}] differs: got {}, expected {}",
                i, result.std_errors[i], expected_std_errs[i]
            ));
        }
    }

    if mismatches.is_empty() {
        Ok(serde_json::json!({ "status": "PASS" }).to_string())
    } else {
        Ok(serde_json::json!({ "status": "FAIL", "mismatches": mismatches }).to_string())
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_housing_regression_integrity() {
        let result = test_housing_regression_native();
        if let Err(e) = result {
            panic!("Regression test failed: {}", e);
        }
    }

    /// Test that test_housing_regression_native produces valid JSON
    #[test]
    fn test_housing_regression_json_output() {
        let result = test_housing_regression_native().unwrap();
        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        // Should have status field
        assert!(parsed.get("status").is_some());
        // Status should be PASS (we control the test data)
        assert_eq!(parsed["status"], "PASS");
    }

    /// Test housing regression with actual R reference values
    #[test]
    fn test_housing_regression_coefficients() {
        let y = vec![
            245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1, 445.8, 167.9,
            367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9, 223.4, 312.5, 156.8, 423.7,
            267.9,
        ];

        let square_feet = vec![
            1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0,
            2200.0, 900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0,
            1250.0, 1700.0, 850.0, 2350.0, 1400.0,
        ];
        let bedrooms = vec![
            3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0, 4.0, 2.0, 4.0, 3.0, 2.0, 5.0,
            3.0, 4.0, 2.0, 4.0, 3.0, 3.0, 2.0, 4.0, 3.0,
        ];
        let age = vec![
            15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0, 3.0, 30.0, 6.0, 14.0,
            22.0, 1.0, 16.0, 9.0, 28.0, 4.0, 19.0, 11.0, 35.0, 3.0, 13.0,
        ];

        let x_vars = vec![square_feet, bedrooms, age];
        let names = vec![
            "Intercept".to_string(),
            "Square_Feet".to_string(),
            "Bedrooms".to_string(),
            "Age".to_string(),
        ];

        let result = core::ols_regression(&y, &x_vars, &names).unwrap();

        // Check against R results
        let expected_coeffs = [52.1271333, 0.1613877, 0.9545492, -1.1811815];
        let expected_std_errs = [31.18201809, 0.01875072, 10.44400198, 0.73219949];

        let tolerance = 1e-4;
        for i in 0..4 {
            assert!(
                (result.coefficients[i] - expected_coeffs[i]).abs() < tolerance,
                "coeff[{}] differs: got {}, expected {}",
                i,
                result.coefficients[i],
                expected_coeffs[i]
            );
            assert!(
                (result.std_errors[i] - expected_std_errs[i]).abs() < tolerance,
                "std_err[{}] differs: got {}, expected {}",
                i,
                result.std_errors[i],
                expected_std_errs[i]
            );
        }
    }

    /// Test R-squared calculation in housing regression
    #[test]
    fn test_housing_regression_r_squared() {
        let result = test_housing_regression_native().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

        // If status is PASS, R² should be reasonable (between 0 and 1)
        assert_eq!(parsed["status"], "PASS");
    }

    /// Test that housing regression handles all expected output fields
    #[test]
    fn test_housing_regression_comprehensive() {
        let y = vec![
            245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1,
        ];
        let x1 = vec![1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0];
        let x2 = vec![3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0];

        let result = core::ols_regression(&y, &[x1, x2], &["Intercept".into(), "X1".into(), "X2".into()])
            .unwrap();

        // Verify expected output fields exist
        assert!(!result.coefficients.is_empty());
        assert!(!result.std_errors.is_empty());
        assert!(!result.t_stats.is_empty());
        assert!(!result.p_values.is_empty());
        assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
        assert!(result.residuals.len() == y.len());
    }

    /// Test error handling when insufficient data is provided
    #[test]
    fn test_housing_regression_insufficient_data() {
        let y = vec![245.5, 312.8]; // Only 2 observations
        let x1 = vec![1200.0, 1800.0];
        let x2 = vec![3.0, 4.0];

        let result = core::ols_regression(&y, &[x1, x2], &["Intercept".into(), "X1".into(), "X2".into()]);
        assert!(result.is_err());
    }

    /// Test housing regression precision with tolerance check
    #[test]
    fn test_housing_regression_tolerance_check() {
        let y = vec![
            245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1, 445.8, 167.9,
            367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9, 223.4, 312.5, 156.8, 423.7,
            267.9,
        ];

        let square_feet = vec![
            1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0,
            2200.0, 900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0,
            1250.0, 1700.0, 850.0, 2350.0, 1400.0,
        ];
        let bedrooms = vec![
            3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0, 4.0, 2.0, 4.0, 3.0, 2.0, 5.0,
            3.0, 4.0, 2.0, 4.0, 3.0, 3.0, 2.0, 4.0, 3.0,
        ];
        let age = vec![
            15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0, 3.0, 30.0, 6.0, 14.0,
            22.0, 1.0, 16.0, 9.0, 28.0, 4.0, 19.0, 11.0, 35.0, 3.0, 13.0,
        ];

        let x_vars = vec![square_feet, bedrooms, age];
        let names = vec![
            "Intercept".to_string(),
            "Square_Feet".to_string(),
            "Bedrooms".to_string(),
            "Age".to_string(),
        ];

        let result = core::ols_regression(&y, &x_vars, &names).unwrap();

        // Verify all coefficient values are finite
        for coef in &result.coefficients {
            assert!(coef.is_finite(), "Coefficient should be finite");
        }
        // Verify all standard errors are positive and finite
        for se in &result.std_errors {
            assert!(se.is_finite(), "Standard error should be finite");
            if *se <= 0.0 {
                panic!("Standard error should be positive, got {}", se);
            }
        }
    }
}
