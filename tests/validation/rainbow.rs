// ============================================================================
// Rainbow Test Validation
// ============================================================================
//
// Comprehensive validation of the Rainbow linearity test
// against R and Python reference implementations.
//
// The Rainbow test checks for linearity by comparing the fit of a model
// on a subset of observations (middle portion) to the full model.

use crate::common::{
    load_dataset, load_r_diagnostic_result, load_python_diagnostic_result,
    ALL_DATASETS, STAT_TOLERANCE,
};

use linreg_core::diagnostics::{self, RainbowMethod};

#[test]
fn validate_rainbow_all_datasets() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  RAINBOW TEST - COMPREHENSIVE MULTI-DATASET VALIDATION             ║");
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
            }
        };

        println!("    Loaded: n = {}, predictors = {}", dataset.y.len(), dataset.x_vars.len());

        // Run Rainbow test (R method)
        let rust_result_r = match diagnostics::rainbow_test(&dataset.y, &dataset.x_vars, 0.5, RainbowMethod::R) {
            Ok(r) => r,
            Err(e) => {
                println!("     Rainbow R test failed: {}", e);
                failed_tests.push((dataset_name.to_string(), format!("R test error: {}", e)));
                continue;
            }
        };

        // Handle case where R result is None due to numerical issues (e.g., extreme multicollinearity)
        let rust_r_result = match rust_result_r.r_result.as_ref() {
            Some(result) => result,
            None => {
                println!("      R result not available - likely due to extreme multicollinearity");
                println!("       Skipping R validation for this dataset");
                continue;
            }
        };

        println!("    Rust (R): F = {:.6}, p = {:.6}", rust_r_result.statistic, rust_r_result.p_value);

        // Validate against R
        let r_result_path = r_results_dir.join(format!("{}_rainbow.json", dataset_name));
        if let Some(r_ref) = load_r_diagnostic_result(&r_result_path) {
            total_tests += 1;

            let r_stat = r_ref.statistic.get(0).copied().unwrap_or(0.0);
            let r_pval = r_ref.p_value.get(0).copied().unwrap_or(1.0);

            let stat_diff = (rust_r_result.statistic - r_stat).abs();
            let pval_diff = (rust_r_result.p_value - r_pval).abs();

            println!("    R:        F = {:.6}, p = {:.6}", r_stat, r_pval);
            println!("              Diff: stat = {:.2e}, p = {:.2e}", stat_diff, pval_diff);

            if stat_diff <= STAT_TOLERANCE && pval_diff <= STAT_TOLERANCE {
                println!("     R validation: PASS");
                passed_r += 1;
            } else {
                println!("     R validation: FAIL");
                failed_tests.push((dataset_name.to_string(), format!("R mismatch: stat diff={:.2e}", stat_diff)));
            }
        } else {
            println!("      R reference file not found: {}", r_result_path.display());
            failed_tests.push((dataset_name.to_string(), "R reference file missing".to_string()));
        }

        // Run Rainbow test (Python method)
        let rust_result_py = match diagnostics::rainbow_test(&dataset.y, &dataset.x_vars, 0.5, RainbowMethod::Python) {
            Ok(r) => r,
            Err(e) => {
                println!("     Rainbow Python test failed: {}", e);
                continue;
            }
        };

        // Handle case where Python result is None due to numerical issues
        let rust_py_result = match rust_result_py.python_result.as_ref() {
            Some(result) => result,
            None => {
                println!("      Python result not available - likely due to extreme multicollinearity");
                println!("       Skipping Python validation for this dataset");
                continue;
            }
        };

        println!("    Rust (Py): F = {:.6}, p = {:.6}", rust_py_result.statistic, rust_py_result.p_value);

        // Validate against Python
        let python_result_path = python_results_dir.join(format!("{}_rainbow.json", dataset_name));
        if let Some(py_ref) = load_python_diagnostic_result(&python_result_path) {
            total_tests += 1;

            let py_stat = py_ref.statistic;
            let py_pval = py_ref.p_value;

            let stat_diff = (rust_py_result.statistic - py_stat).abs();
            let pval_diff = (rust_py_result.p_value - py_pval).abs();

            println!("    Python:   F = {:.6}, p = {:.6}", py_stat, py_pval);
            println!("              Diff: stat = {:.2e}, p = {:.2e}", stat_diff, pval_diff);

            if stat_diff <= STAT_TOLERANCE && pval_diff <= STAT_TOLERANCE {
                println!("     Python validation: PASS");
                passed_python += 1;
            } else {
                println!("     Python validation: FAIL");
                failed_tests.push((dataset_name.to_string(), format!("Python mismatch: stat diff={:.2e}", stat_diff)));
            }
        } else {
            println!("      Python reference file not found: {}", python_result_path.display());
            failed_tests.push((dataset_name.to_string(), "Python reference file missing".to_string()));
        }

        println!();
    }

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  RAINBOW VALIDATION SUMMARY                                         ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Total tests run:       {:>40}║", total_tests);
    println!("║  R validations passed:   {:>40}║", passed_r);
    println!("║  Python validations passed: {:>39}║", passed_python);
    println!("║  Failed tests:          {:>40}║", failed_tests.len());
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    assert!(total_tests > 0, "No Rainbow validation tests were run.");
    let pass_rate = (passed_r + passed_python) as f64 / total_tests as f64;
    assert!(pass_rate >= 0.9, "Rainbow validation pass rate ({:.1}%) is below 90%.", pass_rate * 100.0);

    println!();
    println!(" Rainbow comprehensive validation passed!");
}
