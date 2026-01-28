// ============================================================================
// Durbin-Watson Test Validation
// ============================================================================
//
// Comprehensive validation of the Durbin-Watson autocorrelation test
// against R and Python reference implementations.
//
// The Durbin-Watson test detects first-order autocorrelation in residuals
// by analyzing the ratio of the sum of squared differences to residual variance.

use crate::common::{
    load_dataset, load_python_diagnostic_result, load_r_diagnostic_result, ALL_DATASETS,
    DURBIN_WATSON_TOLERANCE,
};

use linreg_core::diagnostics::durbin_watson_test;

#[test]
fn validate_durbin_watson_all_datasets() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  DURBIN-WATSON TEST - COMPREHENSIVE MULTI-DATASET VALIDATION       ║");
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
            },
        };

        println!(
            "    Loaded: n = {}, predictors = {}",
            dataset.y.len(),
            dataset.x_vars.len()
        );

        // Run Durbin-Watson test
        let rust_result = match durbin_watson_test(&dataset.y, &dataset.x_vars) {
            Ok(r) => r,
            Err(e) => {
                println!("     Durbin-Watson test failed: {}", e);
                failed_tests.push((
                    dataset_name.to_string(),
                    format!("DW test error: {}", e),
                ));
                continue;
            },
        };

        println!(
            "    Rust: DW = {:.6}",
            rust_result.statistic
        );

        // Validate against R
        let r_result_path = r_results_dir.join(format!("{}_durbin_watson.json", dataset_name));
        if let Some(r_ref) = load_r_diagnostic_result(&r_result_path) {
            total_tests += 1;

            let r_stat = r_ref.statistic.get(0).copied().unwrap_or(2.0);

            let stat_diff = (rust_result.statistic - r_stat).abs();

            println!("    R:        DW = {:.6}", r_stat);
            println!("              Diff: stat = {:.2e}", stat_diff);

            if stat_diff <= DURBIN_WATSON_TOLERANCE {
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
            failed_tests.push((
                dataset_name.to_string(),
                "R reference file missing".to_string(),
            ));
        }

        // Validate against Python
        let python_result_path =
            python_results_dir.join(format!("{}_durbin_watson.json", dataset_name));
        if let Some(py_ref) = load_python_diagnostic_result(&python_result_path) {
            total_tests += 1;

            let py_stat = py_ref.statistic;

            let stat_diff = (rust_result.statistic - py_stat).abs();

            println!("    Python:   DW = {:.6}", py_stat);
            println!("              Diff: stat = {:.2e}", stat_diff);

            if stat_diff <= DURBIN_WATSON_TOLERANCE {
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
            failed_tests.push((
                dataset_name.to_string(),
                "Python reference file missing".to_string(),
            ));
        }

        println!();
    }

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  DURBIN-WATSON VALIDATION SUMMARY                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Total tests run:       {:>40}║", total_tests);
    println!("║  R validations passed:   {:>40}║", passed_r);
    println!("║  Python validations passed: {:>39}║", passed_python);
    println!("║  Failed tests:          {:>40}║", failed_tests.len());
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    if !failed_tests.is_empty() {
        println!("\n  Failed tests:");
        for (dataset, reason) in &failed_tests {
            println!("    - {}: {}", dataset, reason);
        }
    }

    assert!(total_tests > 0, "No Durbin-Watson validation tests were run.");
    let pass_rate = (passed_r + passed_python) as f64 / total_tests as f64;
    assert!(
        pass_rate >= 0.9,
        "Durbin-Watson validation pass rate ({:.1}%) is below 90%.",
        pass_rate * 100.0
    );

    println!();
    println!(" Durbin-Watson comprehensive validation passed!");
}
