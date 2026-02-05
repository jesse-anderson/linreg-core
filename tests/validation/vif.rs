// ============================================================================
// VIF (Variance Inflation Factor) Test Validation
// ============================================================================
//
// Comprehensive validation of the VIF multicollinearity test
// against R and Python reference implementations.
//
// The VIF test measures how much the variance of a regression coefficient
// is inflated due to multicollinearity among predictor variables.

use crate::common::{
    load_dataset, load_python_vif_result, load_r_vif_result, ALL_DATASETS, VIF_TOLERANCE,
};

use linreg_core::diagnostics;

#[test]
fn validate_vif_all_datasets() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  VIF TEST - COMPREHENSIVE MULTI-DATASET VALIDATION                   ║");
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

        println!(
            "    Loaded: n = {}, predictors = {}",
            dataset.y.len(),
            dataset.x_vars.len()
        );

        // VIF requires at least 2 predictors
        if dataset.x_vars.len() < 2 {
            println!("     Skipping: VIF requires at least 2 predictors");
            continue;
        }

        // Run VIF test
        let rust_result = match diagnostics::vif_test(&dataset.y, &dataset.x_vars) {
            Ok(r) => r,
            Err(e) => {
                println!("     VIF test failed: {}", e);
                failed_tests.push((
                    dataset_name.to_string(),
                    format!("VIF test error: {}", e),
                ));
                continue;
            }
        };

        println!("    Rust: max_vif = {:.6}", rust_result.max_vif);
        println!("    Variables:");
        for detail in &rust_result.vif_results {
            println!(
                "      {}: VIF = {:.6}, R² = {:.6}",
                detail.variable, detail.vif, detail.rsquared
            );
        }

        // Validate against R
        let r_result_path = r_results_dir.join(format!("{}_vif.json", dataset_name));
        if let Some(r_ref) = load_r_vif_result(&r_result_path) {
            total_tests += 1;

            let r_max_vif = r_ref.max_vif.get(0).copied().unwrap_or(0.0);

            // Compare max_vif
            let max_vif_diff = (rust_result.max_vif - r_max_vif).abs();

            // Compare individual VIF values
            let mut all_vif_match = true;
            for (rust_detail, r_entry) in rust_result
                .vif_results
                .iter()
                .zip(r_ref.vif_results.iter())
            {
                let vif_diff = (rust_detail.vif - r_entry.vif_result).abs();
                if vif_diff > VIF_TOLERANCE {
                    all_vif_match = false;
                    println!(
                        "      {}: Rust VIF = {:.10}, R VIF = {:.10}, diff = {:.2e}",
                        rust_detail.variable, rust_detail.vif, r_entry.vif_result, vif_diff
                    );
                }
            }

            println!("    R:        max_vif = {:.6}", r_max_vif);
            println!("              Diff: max_vif = {:.2e}", max_vif_diff);

            if max_vif_diff <= VIF_TOLERANCE && all_vif_match {
                println!("     R validation: PASS");
                passed_r += 1;
            } else {
                println!("     R validation: FAIL");
                failed_tests.push((
                    dataset_name.to_string(),
                    format!("R mismatch: max_vif diff={:.2e}", max_vif_diff),
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
        let python_result_path = python_results_dir.join(format!("{}_vif.json", dataset_name));
        if let Some(py_ref) = load_python_vif_result(&python_result_path) {
            total_tests += 1;

            let py_max_vif = py_ref.max_vif;

            // Compare max_vif
            let max_vif_diff = (rust_result.max_vif - py_max_vif).abs();

            // Compare individual VIF values
            let mut all_vif_match = true;
            for (rust_detail, &py_vif) in rust_result
                .vif_results
                .iter()
                .zip(py_ref.vif_values.iter())
            {
                let vif_diff = (rust_detail.vif - py_vif).abs();
                if vif_diff > VIF_TOLERANCE {
                    all_vif_match = false;
                    println!(
                        "      {}: Rust VIF = {:.10}, Python VIF = {:.10}, diff = {:.2e}",
                        rust_detail.variable, rust_detail.vif, py_vif, vif_diff
                    );
                }
            }

            println!("    Python:   max_vif = {:.6}", py_max_vif);
            println!("              Diff: max_vif = {:.2e}", max_vif_diff);

            if max_vif_diff <= VIF_TOLERANCE && all_vif_match {
                println!("     Python validation: PASS");
                passed_python += 1;
            } else {
                println!("     Python validation: FAIL");
                failed_tests.push((
                    dataset_name.to_string(),
                    format!("Python mismatch: max_vif diff={:.2e}", max_vif_diff),
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
    println!("║  VIF VALIDATION SUMMARY                                               ║");
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

    assert!(total_tests > 0, "No VIF validation tests were run.");
    let pass_rate = (passed_r + passed_python) as f64 / total_tests as f64;
    assert!(
        pass_rate >= 0.9,
        "VIF validation pass rate ({:.1}%) is below 90%.",
        pass_rate * 100.0
    );

    println!();
    println!(" VIF comprehensive validation passed!");
}
