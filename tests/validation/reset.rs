// ============================================================================
// RESET Test Validation
// ============================================================================
//
// Comprehensive validation of the RESET (Regression Specification Error Test)
// against R and Python reference implementations.
//
// The RESET test checks for functional form misspecification by testing
// whether powers of fitted values, regressors, or the first principal
// component significantly improve the model fit.

use crate::common::{
    load_dataset, load_python_diagnostic_result, load_r_diagnostic_result, ALL_DATASETS,
    RESET_TOLERANCE,
};

use linreg_core::diagnostics::{reset_test, ResetType};

#[test]
fn validate_reset_all_datasets() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  RESET TEST - COMPREHENSIVE MULTI-DATASET VALIDATION              ║");
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

    // Use default powers [2, 3] for validation (matches R's default)
    let powers = vec![2, 3];

    for dataset_name in ALL_DATASETS {
        let csv_path = datasets_dir.join(format!("{}.csv", dataset_name));
        if !csv_path.exists() {
            println!("    Skipping {}: CSV file not found", dataset_name);
            continue;
        }

        println!("  ┌─────────────────────────────────────────────────────────────────┐");
        println!("  │  Dataset: {:<52}│", dataset_name);
        println!("  └─────────────────────────────────────────────────────────────────┘");

        let dataset = match load_dataset(&csv_path) {
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

        // Run RESET test (Fitted type - R's default)
        let rust_result = match reset_test(&dataset.y, &dataset.x_vars, &powers, ResetType::Fitted) {
            Ok(r) => r,
            Err(e) => {
                println!("     RESET test failed: {}", e);
                failed_tests.push((
                    dataset_name.to_string(),
                    format!("RESET test error: {}", e),
                ));
                continue;
            },
        };

        println!(
            "    Rust: F = {:.6}, p = {:.6}",
            rust_result.statistic, rust_result.p_value
        );

        // Validate against R
        let r_result_path = r_results_dir.join(format!("{}_reset.json", dataset_name));
        if let Some(r_ref) = load_r_diagnostic_result(&r_result_path) {
            total_tests += 1;

            // R JSON uses arrays for values
            let r_stat = r_ref.statistic.get(0).copied().unwrap_or(0.0);
            let r_pval = r_ref.p_value.get(0).copied().unwrap_or(1.0);

            let stat_diff = (rust_result.statistic - r_stat).abs();
            let pval_diff = (rust_result.p_value - r_pval).abs();

            println!("    R:        F = {:.6}, p = {:.6}", r_stat, r_pval);
            println!(
                "              Diff: stat = {:.2e}, p = {:.2e}",
                stat_diff, pval_diff
            );

            if stat_diff <= RESET_TOLERANCE && pval_diff <= RESET_TOLERANCE {
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
            println!("     R validation: FAIL (missing reference)");
            failed_tests.push((
                dataset_name.to_string(),
                format!("R reference file missing: {}", r_result_path.display()),
            ));
        }

        // Validate against Python
        let python_result_path = python_results_dir.join(format!("{}_reset.json", dataset_name));
        if let Some(py_ref) = load_python_diagnostic_result(&python_result_path) {
            total_tests += 1;

            let py_stat = py_ref.statistic;
            let py_pval = py_ref.p_value;

            let stat_diff = (rust_result.statistic - py_stat).abs();
            let pval_diff = (rust_result.p_value - py_pval).abs();

            println!("    Python:   F = {:.6}, p = {:.6}", py_stat, py_pval);
            println!(
                "              Diff: stat = {:.2e}, p = {:.2e}",
                stat_diff, pval_diff
            );

            if stat_diff <= RESET_TOLERANCE && pval_diff <= RESET_TOLERANCE {
                println!("     Python validation: PASS");
                passed_python += 1;
            } else {
                println!("     Python validation: FAIL");
                failed_tests.push((
                    dataset_name.to_string(),
                    format!("Python mismatch: stat diff={:.2e}", stat_diff),
                ));
            }
        } else {
            println!(
                "      Python reference file not found: {}",
                python_result_path.display()
            );
            println!("     Python validation: FAIL (missing reference)");
            failed_tests.push((
                dataset_name.to_string(),
                format!("Python reference file missing: {}", python_result_path.display()),
            ));
        }

        println!();
    }

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  RESET VALIDATION SUMMARY                                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Total tests run:       {:>40}║", total_tests);
    println!("║  R validations passed:   {:>40}║", passed_r);
    println!("║  Python validations passed: {:>39}║", passed_python);
    println!("║  Failed tests:          {:>40}║", failed_tests.len());
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    assert!(total_tests > 0, "No RESET validation tests were run.");
    let pass_rate = (passed_r + passed_python) as f64 / total_tests as f64;
    assert!(
        pass_rate >= 0.9,
        "RESET validation pass rate ({:.1}%) is below 90%.",
        pass_rate * 100.0
    );

    println!();
    println!(" RESET comprehensive validation passed!");
}
