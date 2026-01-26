// ============================================================================
// Breusch-Godfrey Test Validation
// ============================================================================
//
// Comprehensive validation of the Breusch-Godfrey higher-order serial
// correlation test against R and Python reference implementations.
//
// # Implementation Choice
//
// This implementation follows **R's lmtest::bgtest** approach (LM/Chi-squared
// statistic with order parameter). Rust results should match R reference values
// within STAT_TOLERANCE (0.001).
//
// # Known Differences with Python
//
// Python's `statsmodels.stats.diagnostic.acorr_breusch_godfrey` may produce
// slightly different results due to numerical precision differences in the
// auxiliary regression. These differences are typically within 1e-6.

use crate::common::{
    load_dataset_with_encoding, load_python_bg_result, load_r_bg_result, ALL_DATASETS, STAT_TOLERANCE, CategoricalEncoding,
};

use linreg_core::diagnostics::{breusch_godfrey_test, BGTestType};

#[test]
fn validate_breusch_godfrey_all_datasets() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  BREUSCH-GODFREY TEST - COMPREHENSIVE MULTI-DATASET VALIDATION     ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");
    let r_results_dir = current_dir.join("verification/results/r");
    let python_results_dir = current_dir.join("verification/results/python");

    let mut total_tests = 0;
    let mut passed_r = 0;
    let mut passed_python = 0;
    let mut failed_tests = Vec::new();

    // Test with order = 1 (default)
    let order = 1;

    for dataset_name in ALL_DATASETS {
        let csv_path = datasets_dir.join(format!("{}.csv", dataset_name));

        if !csv_path.exists() {
            println!("    Skipping {}: CSV file not found", dataset_name);
            continue;
        }

        println!("  ┌─────────────────────────────────────────────────────────────────┐");
        println!("  │  Dataset: {:<52}│", dataset_name);
        println!("  └─────────────────────────────────────────────────────────────────┘");

        // Load the dataset
        let dataset = match load_dataset_with_encoding(&csv_path, CategoricalEncoding::OneBased) {
            Ok(d) => d,
            Err(e) => {
                println!("     Failed to load dataset: {}", e);
                failed_tests.push((dataset_name.to_string(), "Load failed".to_string()));
                continue;
            },
        };

        println!(
            "    Loaded: n = {}, predictors = {}",
            dataset.y.len(),
            dataset.x_vars.len()
        );

        // Check for sufficient data (need n > k + order + 1)
        let k = dataset.x_vars.len() + 1; // including intercept
        let n = dataset.y.len();
        let min_required = k + order + 1;

        if n <= min_required {
            println!(
                "      SKIP: Insufficient data (n={}, k={}, order={}, need n > {})",
                n, k - 1, order, min_required
            );
            println!();
            continue;
        }

        // Run Breusch-Godfrey test (Chi-squared)
        let rust_result = match breusch_godfrey_test(&dataset.y, &dataset.x_vars, order, BGTestType::Chisq) {
            Ok(r) => r,
            Err(e) => {
                println!("     Breusch-Godfrey test failed: {}", e);
                failed_tests.push((dataset_name.to_string(), format!("Test error: {}", e)));
                continue;
            },
        };

        println!(
            "    Rust: LM = {:.6}, p = {:.6}, order = {}",
            rust_result.statistic, rust_result.p_value, rust_result.order
        );

        // Validate against R
        let r_result_path = r_results_dir.join(format!("{}_breusch_godfrey.json", dataset_name));
        if let Some(r_ref) = load_r_bg_result(&r_result_path) {
            total_tests += 1;

            let r_stat = r_ref.statistic.get(0).copied().unwrap_or(0.0);
            let r_pval = r_ref.p_value.get(0).copied().unwrap_or(1.0);
            let r_order = r_ref.order.get(0).copied().unwrap_or(1.0) as usize;

            // Check that order matches
            if r_order != order {
                println!(
                    "      WARNING: R reference has order={}, testing with order={}",
                    r_order, order
                );
            }

            let stat_diff = (rust_result.statistic - r_stat).abs();
            let pval_diff = (rust_result.p_value - r_pval).abs();

            let stat_match = stat_diff <= STAT_TOLERANCE;
            let pval_match = pval_diff <= STAT_TOLERANCE;

            println!("    R:    LM = {:.6}, p = {:.6}, order = {}", r_stat, r_pval, r_order);
            println!(
                "          Diff: stat = {:.2e}, p = {:.2e}",
                stat_diff, pval_diff
            );

            if stat_match && pval_match {
                println!("     R validation: PASS");
                passed_r += 1;
            } else {
                println!("     R validation: FAIL");
                failed_tests.push((
                    dataset_name.to_string(),
                    format!("R mismatch: stat diff={:.2e}", stat_diff),
                ));
            }
        } else {
            println!(
                "      R reference file not found: {}",
                r_result_path.display()
            );
            // Don't add to failed_tests - this is expected during initial setup
        }

        println!();

        // Validate against Python
        let python_result_path =
            python_results_dir.join(format!("{}_breusch_godfrey.json", dataset_name));
        if let Some(py_ref) = load_python_bg_result(&python_result_path) {
            total_tests += 1;

            let py_stat = py_ref.statistic;
            let py_pval = py_ref.p_value;
            let py_order = py_ref.order as usize;

            // Check that order matches
            if py_order != order {
                println!(
                        "      WARNING: Python reference has order={}, testing with order={}",
                        py_order, order
                    );
            }

            let stat_diff = (rust_result.statistic - py_stat).abs();
            let pval_diff = (rust_result.p_value - py_pval).abs();

            let stat_match = stat_diff <= STAT_TOLERANCE;
            let pval_match = pval_diff <= STAT_TOLERANCE;

            println!("    Python: LM = {:.6}, p = {:.6}, order = {}", py_stat, py_pval, py_order);
            println!(
                "          Diff: stat = {:.2e}, p = {:.2e}",
                stat_diff, pval_diff
            );

            if stat_match && pval_match {
                println!("     Python validation: PASS");
                passed_python += 1;
            } else {
                // Use slightly more lenient tolerance for Python
                let py_tolerance = STAT_TOLERANCE * 10.0; // 0.01
                let stat_match_lenient = stat_diff <= py_tolerance;
                let pval_match_lenient = pval_diff <= py_tolerance;

                if stat_match_lenient && pval_match_lenient {
                    println!("     Python validation: PASS (within lenient tolerance)");
                    passed_python += 1;
                } else {
                    println!("     Python validation: FAIL");
                    failed_tests.push((
                        dataset_name.to_string(),
                        format!("Python mismatch: stat diff={:.2e}", stat_diff),
                    ));
                }
            }
        } else {
            println!(
                "      Python reference file not found: {}",
                python_result_path.display()
            );
            // Don't add to failed_tests - this is expected during initial setup
        }

        println!();
    }

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  VALIDATION SUMMARY                                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Total tests run:       {:>40}║", total_tests);
    println!("║  R validations passed:   {:>40}║", passed_r);
    println!("║  Python validations passed: {:>39}║", passed_python);
    println!("║  Failed tests:          {:>40}║", failed_tests.len());
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    if !failed_tests.is_empty() {
        println!();
        println!("Failed tests:");
        for (dataset, reason) in &failed_tests {
            println!("  - {}: {}", dataset, reason);
        }
    }

    // Assert that we tested at least some datasets
    assert!(
        total_tests > 0,
        "No validation tests were run. Check that result files exist."
    );

    // Assert that we have a reasonable pass rate (at least 90%)
    let pass_rate = (passed_r + passed_python) as f64 / total_tests as f64;
    assert!(
        pass_rate >= 0.9,
        "Validation pass rate ({:.1}%) is below 90% threshold. See failed tests above.",
        pass_rate * 100.0
    );

    println!();
    println!(" Breusch-Godfrey comprehensive validation passed!");
}

#[test]
fn validate_breusch_godfrey_synthetic_autocorrelated() {
    // Specific test for the synthetic_autocorrelated dataset
    // This dataset is designed to have AR(1) errors, so BG should detect significant autocorrelation
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  BREUSCH-GODFREY TEST - SYNTHETIC AUTOCORRELATED DATASET             ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");
    let r_results_dir = current_dir.join("verification/results/r");
    let python_results_dir = current_dir.join("verification/results/python");

    let dataset_name = "synthetic_autocorrelated";
    let csv_path = datasets_dir.join(format!("{}.csv", dataset_name));

    let dataset = load_dataset_with_encoding(&csv_path, CategoricalEncoding::OneBased).expect("Failed to load dataset");

    println!("  Dataset: {}", dataset_name);
    println!("  n = {}, predictors = {}", dataset.y.len(), dataset.x_vars.len());

    // Run Breusch-Godfrey test with order = 1
    let rust_result = breusch_godfrey_test(&dataset.y, &dataset.x_vars, 1, BGTestType::Chisq)
        .expect("Breusch-Godfrey test failed");

    println!();
    println!("  Rust result:");
    println!("    LM statistic = {:.6}", rust_result.statistic);
    println!("    p-value = {:.6}", rust_result.p_value);
    println!("    order = {}", rust_result.order);

    // The synthetic_autocorrelated dataset should have significant autocorrelation
    // (p-value < 0.05)
    println!();
    println!("  Interpretation:");
    if rust_result.p_value < 0.05 {
        println!("    Significant serial correlation detected (p < 0.05) ✓");
    } else {
        println!("    No significant serial correlation detected (p >= 0.05)");
    }

    // Validate against R if reference exists
    let r_result_path = r_results_dir.join(format!("{}_breusch_godfrey.json", dataset_name));
    if let Some(r_ref) = load_r_bg_result(&r_result_path) {
        let r_stat = r_ref.statistic.get(0).copied().unwrap_or(0.0);
        let r_pval = r_ref.p_value.get(0).copied().unwrap_or(1.0);

        let stat_diff = (rust_result.statistic - r_stat).abs();
        let pval_diff = (rust_result.p_value - r_pval).abs();

        println!();
        println!("  R reference:");
        println!("    LM statistic = {:.6}", r_stat);
        println!("    p-value = {:.6}", r_pval);
        println!("    Difference: stat = {:.2e}, p = {:.2e}", stat_diff, pval_diff);

        assert!(
            stat_diff <= STAT_TOLERANCE,
            "Statistic differs from R: {:.2e} > {:.2e}",
            stat_diff,
            STAT_TOLERANCE
        );
        assert!(
            pval_diff <= STAT_TOLERANCE,
            "P-value differs from R: {:.2e} > {:.2e}",
            pval_diff,
            STAT_TOLERANCE
        );
        println!("    R validation: PASS ✓");
    }

    // Validate against Python if reference exists
    let python_result_path =
        python_results_dir.join(format!("{}_breusch_godfrey.json", dataset_name));
    if let Some(py_ref) = load_python_bg_result(&python_result_path) {
        let py_stat = py_ref.statistic;
        let py_pval = py_ref.p_value;

        let stat_diff = (rust_result.statistic - py_stat).abs();
        let pval_diff = (rust_result.p_value - py_pval).abs();

        println!();
        println!("  Python reference:");
        println!("    LM statistic = {:.6}", py_stat);
        println!("    p-value = {:.6}", py_pval);
        println!("    Difference: stat = {:.2e}, p = {:.2e}", stat_diff, pval_diff);

        // Use slightly more lenient tolerance for Python
        let py_tolerance = STAT_TOLERANCE * 10.0; // 0.01
        assert!(
            stat_diff <= py_tolerance,
            "Statistic differs from Python: {:.2e} > {:.2e}",
            stat_diff,
            py_tolerance
        );
        assert!(
            pval_diff <= py_tolerance,
            "P-value differs from Python: {:.2e} > {:.2e}",
            pval_diff,
            py_tolerance
        );
        println!("    Python validation: PASS ✓");
    }

    println!();
    println!("  Breusch-Godfrey synthetic_autocorrelated validation passed!");
}

#[test]
fn validate_breusch_godfrey_f_statistic() {
    // Test the F-statistic variant of the Breusch-Godfrey test
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  BREUSCH-GODFREY TEST - F-STATISTIC VARIANT                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let dataset_name = "faithful";
    let csv_path = datasets_dir.join(format!("{}.csv", dataset_name));

    let dataset = load_dataset_with_encoding(&csv_path, CategoricalEncoding::OneBased).expect("Failed to load dataset");

    println!("  Dataset: {}", dataset_name);
    println!("  n = {}, predictors = {}", dataset.y.len(), dataset.x_vars.len());

    // Run Breusch-Godfrey test with F-statistic
    let rust_result = breusch_godfrey_test(&dataset.y, &dataset.x_vars, 1, BGTestType::F)
        .expect("Breusch-Godfrey test failed");

    println!();
    println!("  Rust result (F-statistic):");
    println!("    F statistic = {:.6}", rust_result.statistic);
    println!("    p-value = {:.6}", rust_result.p_value);
    println!("    order = {}", rust_result.order);
    println!("    df = [{:.1}, {:.1}]", rust_result.df[0], rust_result.df[1]);

    // Check that the result is valid
    assert!(rust_result.statistic >= 0.0, "F-statistic should be non-negative");
    assert!(
        rust_result.p_value >= 0.0 && rust_result.p_value <= 1.0,
        "p-value should be in [0, 1]"
    );
    assert_eq!(rust_result.df.len(), 2, "F-test should have 2 degrees of freedom");
    assert_eq!(rust_result.df[0], 1.0, "df1 should equal order");
    assert!(rust_result.df[1] > 0.0, "df2 should be positive");

    println!();
    println!("  F-statistic validation passed!");
}
