// ============================================================================
// Core Regression Validation Tests
// ============================================================================
//
// These tests validate the Rust OLS implementation against reference values from
// R and Python statistical libraries using the housing regression dataset.

use crate::common::{
    assert_close_to, get_housing_data, load_validation_results, print_comparison_python,
    print_comparison_r, STAT_TOLERANCE, TIGHT_TOLERANCE,
};

use linreg_core::core;
use linreg_core::diagnostics::{self, RainbowMethod};

// ============================================================================
// R Validation Tests
// ============================================================================

#[test]
fn validate_against_r_reference() {
    println!("\n========== R VALIDATION ==========\n");

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let r_results_file = current_dir.join("verification/results/r/R_results.json");

    if !r_results_file.exists() {
        panic!("R validation file not found at {:?}. Run verification/scripts/runners/run_all_diagnostics_r.R to generate it.", r_results_file);
    }

    let expected = load_validation_results(&r_results_file);
    let (y, x_vars) = get_housing_data();
    let names = vec![
        "Intercept".to_string(),
        "Square_Feet".to_string(),
        "Bedrooms".to_string(),
        "Age".to_string(),
    ];

    // Run OLS regression
    let result = core::ols_regression(&y, &x_vars, &names).expect("OLS regression should succeed");

    println!("  ──────────────────────────────────────────────────────────");
    println!("  COEFFICIENTS");
    println!("  ──────────────────────────────────────────────────────────");
    println!();

    for i in 0..4 {
        println!("  [{}] {}", i, names[i]);
        println!();

        print_comparison_r(
            "Coefficient",
            result.coefficients[i],
            expected.coefficients[i],
            "  ",
        );
        assert_close_to(
            result.coefficients[i],
            expected.coefficients[i],
            TIGHT_TOLERANCE,
            &format!("coeff[{}]", i),
        );

        print_comparison_r(
            "Std Error",
            result.std_errors[i],
            expected.std_errors[i],
            "  ",
        );
        assert_close_to(
            result.std_errors[i],
            expected.std_errors[i],
            TIGHT_TOLERANCE,
            &format!("std_err[{}]", i),
        );

        print_comparison_r("t-statistic", result.t_stats[i], expected.t_stats[i], "  ");
        assert_close_to(
            result.t_stats[i],
            expected.t_stats[i],
            TIGHT_TOLERANCE,
            &format!("t_stat[{}]", i),
        );

        print_comparison_r("p-value", result.p_values[i], expected.p_values[i], "  ");
        assert_close_to(
            result.p_values[i],
            expected.p_values[i],
            1e-8,
            &format!("p_value[{}]", i),
        );
    }

    println!("  ──────────────────────────────────────────────────────────");
    println!("  MODEL FIT STATISTICS");
    println!("  ──────────────────────────────────────────────────────────");
    println!();

    print_comparison_r("R-squared", result.r_squared, expected.r_squared, "  ");
    assert_close_to(
        result.r_squared,
        expected.r_squared,
        TIGHT_TOLERANCE,
        "R-squared",
    );

    print_comparison_r(
        "Adjusted R-squared",
        result.adj_r_squared,
        expected.adj_r_squared,
        "  ",
    );
    assert_close_to(
        result.adj_r_squared,
        expected.adj_r_squared,
        TIGHT_TOLERANCE,
        "Adj R-squared",
    );

    print_comparison_r(
        "F-statistic",
        result.f_statistic,
        expected.f_statistic,
        "  ",
    );
    assert_close_to(
        result.f_statistic,
        expected.f_statistic,
        TIGHT_TOLERANCE,
        "F statistic",
    );

    println!();
    print_comparison_r("Log-Likelihood", result.log_likelihood, expected.log_likelihood, "  ");
    assert_close_to(
        result.log_likelihood,
        expected.log_likelihood,
        TIGHT_TOLERANCE,
        "Log-Likelihood",
    );

    print_comparison_r("AIC", result.aic, expected.aic, "  ");
    assert_close_to(result.aic, expected.aic, TIGHT_TOLERANCE, "AIC");

    print_comparison_r("BIC", result.bic, expected.bic, "  ");
    assert_close_to(result.bic, expected.bic, TIGHT_TOLERANCE, "BIC");

    println!("  ──────────────────────────────────────────────────────────");
    println!("  VARIANCE INFLATION FACTORS (VIF)");
    println!("  ──────────────────────────────────────────────────────────");
    println!();

    for expected_vif in &expected.vif {
        let rust_vif = result
            .vif
            .iter()
            .find(|v| v.variable == expected_vif.variable)
            .unwrap_or_else(|| {
                panic!("VIF for {} not found in Rust output", expected_vif.variable)
            });

        print_comparison_r(&rust_vif.variable, rust_vif.vif, expected_vif.vif, "  ");
        assert_close_to(
            rust_vif.vif,
            expected_vif.vif,
            TIGHT_TOLERANCE,
            &format!("VIF for {}", expected_vif.variable),
        );
    }

    println!("  ──────────────────────────────────────────────────────────");
    println!("  DIAGNOSTIC TESTS");
    println!("  ──────────────────────────────────────────────────────────");
    println!();

    // Diagnostic tests
    if let Some(ref expected_rainbow) = expected.rainbow {
        let rainbow_result = diagnostics::rainbow_test(&y, &x_vars, 0.5, RainbowMethod::R)
            .expect("Rainbow test should succeed");
        let r_result = rainbow_result
            .r_result
            .as_ref()
            .expect("R result should exist");
        assert_close_to(
            r_result.statistic,
            expected_rainbow.statistic,
            STAT_TOLERANCE,
            "Rainbow statistic",
        );
        assert_close_to(
            r_result.p_value,
            expected_rainbow.p_value,
            STAT_TOLERANCE,
            "Rainbow p-value",
        );

        println!("  Rainbow Test (Linearity)");
        print_comparison_r(
            "F-statistic",
            r_result.statistic,
            expected_rainbow.statistic,
            "  ",
        );
        print_comparison_r("p-value", r_result.p_value, expected_rainbow.p_value, "  ");
    }

    // Breusch-Pagan test
    if let Some(ref expected_bp) = expected.breusch_pagan {
        let bp_result = diagnostics::breusch_pagan_test(&y, &x_vars)
            .expect("Breusch-Pagan test should succeed");
        assert_close_to(
            bp_result.statistic,
            expected_bp.statistic,
            STAT_TOLERANCE,
            "BP statistic",
        );
        assert_close_to(
            bp_result.p_value,
            expected_bp.p_value,
            STAT_TOLERANCE,
            "BP p-value",
        );

        println!("  Breusch-Pagan Test (Heteroscedasticity)");
        print_comparison_r(
            "LM-statistic",
            bp_result.statistic,
            expected_bp.statistic,
            "  ",
        );
        print_comparison_r("p-value", bp_result.p_value, expected_bp.p_value, "  ");
    }

    // White test
    if let Some(ref expected_white) = expected.white {
        let white_result = diagnostics::white_test(&y, &x_vars, diagnostics::WhiteMethod::R)
            .expect("White test should succeed");
        let white_r = white_result
            .r_result
            .as_ref()
            .expect("R result should be present");
        assert_close_to(
            white_r.statistic,
            expected_white.statistic,
            STAT_TOLERANCE,
            "White statistic",
        );
        assert_close_to(
            white_r.p_value,
            expected_white.p_value,
            STAT_TOLERANCE,
            "White p-value",
        );

        println!("  White Test (Heteroscedasticity)");
        print_comparison_r(
            "LM-statistic",
            white_r.statistic,
            expected_white.statistic,
            "  ",
        );
        print_comparison_r("p-value", white_r.p_value, expected_white.p_value, "  ");
    }

    // Anderson-Darling test
    const AD_TOLERANCE: f64 = 0.001;
    if let Some(ref expected_ad) = expected.anderson_darling {
        let ad_result = diagnostics::anderson_darling_test(&y, &x_vars)
            .expect("Anderson-Darling test should succeed");
        assert_close_to(
            ad_result.statistic,
            expected_ad.statistic,
            AD_TOLERANCE,
            "AD statistic",
        );
        assert_close_to(
            ad_result.p_value,
            expected_ad.p_value,
            AD_TOLERANCE,
            "AD p-value",
        );

        println!("  Anderson-Darling Test (Normality)");
        print_comparison_r(
            "A-squared",
            ad_result.statistic,
            expected_ad.statistic,
            "  ",
        );
        print_comparison_r("p-value", ad_result.p_value, expected_ad.p_value, "  ");
    }

    // Shapiro-Wilk test
    const SW_TOLERANCE: f64 = 0.001;
    if let Some(ref expected_sw) = expected.shapiro_wilk {
        let sw_result =
            diagnostics::shapiro_wilk_test(&y, &x_vars).expect("Shapiro-Wilk test should succeed");
        assert_close_to(
            sw_result.statistic,
            expected_sw.statistic,
            SW_TOLERANCE,
            "SW statistic",
        );
        assert_close_to(
            sw_result.p_value,
            expected_sw.p_value,
            SW_TOLERANCE,
            "SW p-value",
        );

        println!("  Shapiro-Wilk Test (Normality)");
        print_comparison_r(
            "W statistic",
            sw_result.statistic,
            expected_sw.statistic,
            "  ",
        );
        print_comparison_r("p-value", sw_result.p_value, expected_sw.p_value, "  ");
    }

    println!("\n✓ All R validation checks passed!");
}

// ============================================================================
// Python Validation Tests
// ============================================================================

#[test]
fn validate_against_python_reference() {
    println!("\n========== PYTHON VALIDATION ==========\n");

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let py_results_file = current_dir.join("verification/results/python/Python_results.json");

    if !py_results_file.exists() {
        panic!("Python validation file not found at {:?}. Run verification/scripts/runners/run_all_diagnostics_python.py to generate it.", py_results_file);
    }

    let expected = load_validation_results(&py_results_file);
    let (y, x_vars) = get_housing_data();
    let names = vec![
        "Intercept".to_string(),
        "Square_Feet".to_string(),
        "Bedrooms".to_string(),
        "Age".to_string(),
    ];

    // Run OLS regression
    let result = core::ols_regression(&y, &x_vars, &names).expect("OLS regression should succeed");

    println!("  ──────────────────────────────────────────────────────────");
    println!("  COEFFICIENTS");
    println!("  ──────────────────────────────────────────────────────────");
    println!();

    for i in 0..4 {
        println!("  [{}] {}", i, names[i]);
        println!();

        print_comparison_python(
            "Coefficient",
            result.coefficients[i],
            expected.coefficients[i],
            "  ",
        );
        assert_close_to(
            result.coefficients[i],
            expected.coefficients[i],
            TIGHT_TOLERANCE,
            &format!("coeff[{}]", i),
        );

        print_comparison_python(
            "Std Error",
            result.std_errors[i],
            expected.std_errors[i],
            "  ",
        );
        assert_close_to(
            result.std_errors[i],
            expected.std_errors[i],
            TIGHT_TOLERANCE,
            &format!("std_err[{}]", i),
        );

        print_comparison_python("t-statistic", result.t_stats[i], expected.t_stats[i], "  ");
        assert_close_to(
            result.t_stats[i],
            expected.t_stats[i],
            TIGHT_TOLERANCE,
            &format!("t_stat[{}]", i),
        );

        print_comparison_python("p-value", result.p_values[i], expected.p_values[i], "  ");
        assert_close_to(
            result.p_values[i],
            expected.p_values[i],
            1e-8,
            &format!("p_value[{}]", i),
        );
    }

    println!("  ──────────────────────────────────────────────────────────");
    println!("  MODEL FIT STATISTICS");
    println!("  ──────────────────────────────────────────────────────────");
    println!();

    print_comparison_python("R-squared", result.r_squared, expected.r_squared, "  ");
    assert_close_to(
        result.r_squared,
        expected.r_squared,
        TIGHT_TOLERANCE,
        "R-squared",
    );

    print_comparison_python(
        "Adjusted R-squared",
        result.adj_r_squared,
        expected.adj_r_squared,
        "  ",
    );
    assert_close_to(
        result.adj_r_squared,
        expected.adj_r_squared,
        TIGHT_TOLERANCE,
        "Adj R-squared",
    );

    print_comparison_python(
        "F-statistic",
        result.f_statistic,
        expected.f_statistic,
        "  ",
    );
    assert_close_to(
        result.f_statistic,
        expected.f_statistic,
        TIGHT_TOLERANCE,
        "F statistic",
    );

    println!("  ──────────────────────────────────────────────────────────");
    println!("  VARIANCE INFLATION FACTORS (VIF)");
    println!("  ──────────────────────────────────────────────────────────");
    println!();

    for expected_vif in &expected.vif {
        let rust_vif = result
            .vif
            .iter()
            .find(|v| v.variable == expected_vif.variable)
            .unwrap_or_else(|| {
                panic!("VIF for {} not found in Rust output", expected_vif.variable)
            });

        print_comparison_python(&rust_vif.variable, rust_vif.vif, expected_vif.vif, "  ");
        assert_close_to(
            rust_vif.vif,
            expected_vif.vif,
            TIGHT_TOLERANCE,
            &format!("VIF for {}", expected_vif.variable),
        );
    }

    println!("  ──────────────────────────────────────────────────────────");
    println!("  DIAGNOSTIC TESTS");
    println!("  ──────────────────────────────────────────────────────────");
    println!();

    // Diagnostic tests - Python method for Rainbow and White
    if let Some(ref expected_rainbow) = expected.rainbow {
        let rainbow_result = diagnostics::rainbow_test(&y, &x_vars, 0.5, RainbowMethod::Python)
            .expect("Rainbow test should succeed");
        let py_result = rainbow_result
            .python_result
            .as_ref()
            .expect("Python result should exist");
        assert_close_to(
            py_result.statistic,
            expected_rainbow.statistic,
            STAT_TOLERANCE,
            "Rainbow statistic",
        );
        assert_close_to(
            py_result.p_value,
            expected_rainbow.p_value,
            STAT_TOLERANCE,
            "Rainbow p-value",
        );

        println!("  Rainbow Test (Linearity)");
        print_comparison_python(
            "F-statistic",
            py_result.statistic,
            expected_rainbow.statistic,
            "  ",
        );
        print_comparison_python("p-value", py_result.p_value, expected_rainbow.p_value, "  ");
    }

    // White test (Python method for Python validation)
    if let Some(ref expected_white) = expected.white {
        let white_result = diagnostics::white_test(&y, &x_vars, diagnostics::WhiteMethod::Python)
            .expect("White test should succeed");
        let white_py = white_result
            .python_result
            .as_ref()
            .expect("Python result should be present");
        assert_close_to(
            white_py.statistic,
            expected_white.statistic,
            STAT_TOLERANCE,
            "White statistic",
        );
        assert_close_to(
            white_py.p_value,
            expected_white.p_value,
            STAT_TOLERANCE,
            "White p-value",
        );

        println!("  White Test (Heteroscedasticity)");
        print_comparison_python(
            "LM-statistic",
            white_py.statistic,
            expected_white.statistic,
            "  ",
        );
        print_comparison_python("p-value", white_py.p_value, expected_white.p_value, "  ");
    }

    // Breusch-Pagan test
    if let Some(ref expected_bp) = expected.breusch_pagan {
        let bp_result = diagnostics::breusch_pagan_test(&y, &x_vars)
            .expect("Breusch-Pagan test should succeed");
        assert_close_to(
            bp_result.statistic,
            expected_bp.statistic,
            STAT_TOLERANCE,
            "BP statistic",
        );
        assert_close_to(
            bp_result.p_value,
            expected_bp.p_value,
            STAT_TOLERANCE,
            "BP p-value",
        );

        println!("  Breusch-Pagan Test (Heteroscedasticity)");
        print_comparison_python(
            "LM-statistic",
            bp_result.statistic,
            expected_bp.statistic,
            "  ",
        );
        print_comparison_python("p-value", bp_result.p_value, expected_bp.p_value, "  ");
    }

    // Anderson-Darling test
    const AD_TOLERANCE_PY: f64 = 0.001;
    if let Some(ref expected_ad) = expected.anderson_darling {
        let ad_result = diagnostics::anderson_darling_test(&y, &x_vars)
            .expect("Anderson-Darling test should succeed");
        assert_close_to(
            ad_result.statistic,
            expected_ad.statistic,
            AD_TOLERANCE_PY,
            "AD statistic",
        );
        assert_close_to(
            ad_result.p_value,
            expected_ad.p_value,
            AD_TOLERANCE_PY,
            "AD p-value",
        );

        println!("  Anderson-Darling Test (Normality)");
        print_comparison_python(
            "A-squared",
            ad_result.statistic,
            expected_ad.statistic,
            "  ",
        );
        print_comparison_python("p-value", ad_result.p_value, expected_ad.p_value, "  ");
    }

    // Shapiro-Wilk test
    const SW_TOLERANCE_PY: f64 = 0.001;
    if let Some(ref expected_sw) = expected.shapiro_wilk {
        let sw_result =
            diagnostics::shapiro_wilk_test(&y, &x_vars).expect("Shapiro-Wilk test should succeed");
        assert_close_to(
            sw_result.statistic,
            expected_sw.statistic,
            SW_TOLERANCE_PY,
            "SW statistic",
        );
        assert_close_to(
            sw_result.p_value,
            expected_sw.p_value,
            SW_TOLERANCE_PY,
            "SW p-value",
        );

        println!("  Shapiro-Wilk Test (Normality)");
        print_comparison_python(
            "W statistic",
            sw_result.statistic,
            expected_sw.statistic,
            "  ",
        );
        print_comparison_python("p-value", sw_result.p_value, expected_sw.p_value, "  ");
    }

    println!("\n✓ All Python validation checks passed!");
}

// ============================================================================
// Method-Specific Tests
// ============================================================================

/// Test that R method for Rainbow test matches R reference
#[test]
fn test_rainbow_r_method() {
    let (y, x_vars) = get_housing_data();
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let r_results_file = current_dir.join("verification/results/r/R_results.json");

    let expected = load_validation_results(&r_results_file);
    let result = diagnostics::rainbow_test(&y, &x_vars, 0.5, RainbowMethod::R)
        .expect("Rainbow R method should succeed");

    assert!(result.r_result.is_some(), "R result should be present");
    assert!(
        result.python_result.is_none(),
        "Python result should NOT be present"
    );

    let r_result = result.r_result.as_ref().unwrap();
    let expected_rainbow = expected
        .rainbow
        .as_ref()
        .expect("Rainbow data should exist in JSON");

    assert_close_to(
        r_result.statistic,
        expected_rainbow.statistic,
        STAT_TOLERANCE,
        "Rainbow R statistic",
    );
    assert_close_to(
        r_result.p_value,
        expected_rainbow.p_value,
        STAT_TOLERANCE,
        "Rainbow R p-value",
    );

    println!(
        "✓ Rainbow R method: F = {:.4}, p = {:.4} (expected: F = {:.4}, p = {:.4})",
        r_result.statistic, r_result.p_value, expected_rainbow.statistic, expected_rainbow.p_value
    );
}

/// Test that Python method for Rainbow test matches Python reference
#[test]
fn test_rainbow_python_method() {
    let (y, x_vars) = get_housing_data();
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let py_results_file = current_dir.join("verification/results/python/Python_results.json");

    let expected = load_validation_results(&py_results_file);
    let result = diagnostics::rainbow_test(&y, &x_vars, 0.5, RainbowMethod::Python)
        .expect("Rainbow Python method should succeed");

    assert!(result.r_result.is_none(), "R result should NOT be present");
    assert!(
        result.python_result.is_some(),
        "Python result should be present"
    );

    let py_result = result.python_result.as_ref().unwrap();
    let expected_rainbow = expected
        .rainbow
        .as_ref()
        .expect("Rainbow data should exist in JSON");

    assert_close_to(
        py_result.statistic,
        expected_rainbow.statistic,
        STAT_TOLERANCE,
        "Rainbow Python statistic",
    );
    assert_close_to(
        py_result.p_value,
        expected_rainbow.p_value,
        STAT_TOLERANCE,
        "Rainbow Python p-value",
    );

    println!(
        "✓ Rainbow Python method: F = {:.4}, p = {:.4} (expected: F = {:.4}, p = {:.4})",
        py_result.statistic,
        py_result.p_value,
        expected_rainbow.statistic,
        expected_rainbow.p_value
    );
}

/// Test that "both" method for Rainbow returns both R and Python results
#[test]
fn test_rainbow_both_methods() {
    let (y, x_vars) = get_housing_data();
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let r_results_file = current_dir.join("verification/results/r/R_results.json");
    let py_results_file = current_dir.join("verification/results/python/Python_results.json");

    let expected_r = load_validation_results(&r_results_file);
    let expected_py = load_validation_results(&py_results_file);

    let result = diagnostics::rainbow_test(&y, &x_vars, 0.5, RainbowMethod::Both)
        .expect("Rainbow Both method should succeed");

    assert!(result.r_result.is_some(), "R result should be present");
    assert!(
        result.python_result.is_some(),
        "Python result should be present"
    );

    let r_result = result.r_result.as_ref().unwrap();
    let py_result = result.python_result.as_ref().unwrap();

    let expected_rainbow_r = expected_r
        .rainbow
        .as_ref()
        .expect("Rainbow data should exist in R JSON");
    let expected_rainbow_py = expected_py
        .rainbow
        .as_ref()
        .expect("Rainbow data should exist in Python JSON");

    assert_close_to(
        r_result.statistic,
        expected_rainbow_r.statistic,
        STAT_TOLERANCE,
        "Rainbow Both R statistic",
    );
    assert_close_to(
        r_result.p_value,
        expected_rainbow_r.p_value,
        STAT_TOLERANCE,
        "Rainbow Both R p-value",
    );
    assert_close_to(
        py_result.statistic,
        expected_rainbow_py.statistic,
        STAT_TOLERANCE,
        "Rainbow Both Python statistic",
    );
    assert_close_to(
        py_result.p_value,
        expected_rainbow_py.p_value,
        STAT_TOLERANCE,
        "Rainbow Both Python p-value",
    );

    println!("✓ Rainbow Both method:");
    println!(
        "  R:     F = {:.4}, p = {:.4}",
        r_result.statistic, r_result.p_value
    );
    println!(
        "  Python: F = {:.4}, p = {:.4}",
        py_result.statistic, py_result.p_value
    );
}

/// Test basic regression integrity (smoke test)
#[test]
fn verify_housing_regression_integrity() {
    let (y, x_vars) = get_housing_data();
    let names = vec![
        "Intercept".to_string(),
        "Square_Feet".to_string(),
        "Bedrooms".to_string(),
        "Age".to_string(),
    ];

    let result = core::ols_regression(&y, &x_vars, &names).expect("OLS regression should succeed");

    // Basic sanity checks
    assert_eq!(
        result.coefficients.len(),
        4,
        "Should have 4 coefficients (including intercept)"
    );
    assert!(
        result.r_squared > 0.0 && result.r_squared <= 1.0,
        "R-squared should be between 0 and 1"
    );
    assert!(
        result.r_squared > 0.9,
        "Housing data should have high R-squared"
    );
    assert!(result.f_statistic > 0.0, "F statistic should be positive");

    println!("✓ Regression integrity check passed:");
    println!("  R-squared = {:.4}", result.r_squared);
    println!("  F = {:.4}", result.f_statistic);
}
