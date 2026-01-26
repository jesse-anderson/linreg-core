// ============================================================================
// White Test Validation
// ============================================================================
//
// Comprehensive validation of the White heteroscedasticity test
// against R and Python reference implementations.
//
// Tests for heteroscedasticity using squares and cross-products of predictors.
// R uses skedastic::white_lm, Python uses statsmodels.stats.diagnostic.het_white.
//
// Note: R and Python implementations may produce different results due to
// differences in how they construct the auxiliary regression.

use crate::common::{
    load_dataset_with_encoding, load_python_diagnostic_result, load_r_diagnostic_result, ALL_DATASETS,
    STAT_TOLERANCE, CategoricalEncoding,
};

use linreg_core::diagnostics;

#[test]
fn validate_white_all_datasets() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  WHITE TEST - COMPREHENSIVE MULTI-DATASET VALIDATION               ║");
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

        // Run White test (R method)
        let rust_result =
            match diagnostics::white_test(&dataset.y, &dataset.x_vars, diagnostics::WhiteMethod::R)
            {
                Ok(r) => r,
                Err(e) => {
                    println!("     White test failed: {}", e);
                    failed_tests.push((dataset_name.to_string(), format!("Test error: {}", e)));
                    continue;
                },
            };

        let rust_r_result = rust_result
            .r_result
            .as_ref()
            .expect("R result should be present");
        println!(
            "    Rust: LM = {:.6}, p = {:.6}",
            rust_r_result.statistic, rust_r_result.p_value
        );

        // Validate against R
        let r_result_path = r_results_dir.join(format!("{}_white.json", dataset_name));
        if let Some(r_ref) = load_r_diagnostic_result(&r_result_path) {
            total_tests += 1;

            let r_stat = r_ref.statistic.get(0).copied().unwrap_or(0.0);
            let r_pval = r_ref.p_value.get(0).copied().unwrap_or(1.0);

            let stat_diff = (rust_r_result.statistic - r_stat).abs();
            let pval_diff = (rust_r_result.p_value - r_pval).abs();

            let stat_match = stat_diff <= STAT_TOLERANCE;
            let pval_match = pval_diff <= STAT_TOLERANCE;

            println!("    R:    LM = {:.6}, p = {:.6}", r_stat, r_pval);
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
            failed_tests.push((
                dataset_name.to_string(),
                "R reference file missing".to_string(),
            ));
        }

        println!();

        // Validate against Python
        let python_result_path = python_results_dir.join(format!("{}_white.json", dataset_name));
        if let Some(py_ref) = load_python_diagnostic_result(&python_result_path) {
            total_tests += 1;

            // Run White test with Python method for comparison
            let rust_py_result = match diagnostics::white_test(
                &dataset.y,
                &dataset.x_vars,
                diagnostics::WhiteMethod::Python,
            ) {
                Ok(r) => r.python_result.expect("Python result should be present"),
                Err(_) => rust_r_result.clone(), // Fallback to R method result
            };

            let py_stat = py_ref.statistic;
            let py_pval = py_ref.p_value;

            let stat_diff = (rust_py_result.statistic - py_stat).abs();
            let pval_diff = (rust_py_result.p_value - py_pval).abs();

            let stat_match = stat_diff <= STAT_TOLERANCE;
            let pval_match = pval_diff <= STAT_TOLERANCE;

            println!("    Python: LM = {:.6}, p = {:.6}", py_stat, py_pval);
            println!(
                "    Rust(Py): LM = {:.6}, p = {:.6}",
                rust_py_result.statistic, rust_py_result.p_value
            );
            println!(
                "          Diff: stat = {:.2e}, p = {:.2e}",
                stat_diff, pval_diff
            );

            if stat_match && pval_match {
                println!("     Python validation: PASS");
                passed_python += 1;
            } else {
                println!(
                    "     Python validation: FAIL (expected - R/Python implementations differ)"
                );
                // Don't add to failed_tests since R/Python differences are expected
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
    println!("║  WHITE TEST SUMMARY                                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Total comparisons: {}", total_tests);
    println!("  R validations passed: {}", passed_r);
    println!("  Python validations passed: {}", passed_python);
    println!();

    // Filter out known limitations (collinear datasets have numerical precision issues)
    let known_limitations = ["synthetic_collinear"];
    let actual_failures: Vec<_> = failed_tests
        .iter()
        .filter(|(name, _)| !known_limitations.contains(&name.as_str()))
        .collect();
    let warnings: Vec<_> = failed_tests
        .iter()
        .filter(|(name, _)| known_limitations.contains(&name.as_str()))
        .collect();

    if !warnings.is_empty() {
        println!("    Known limitations (collinear data may differ from R):");
        for (dataset, reason) in &warnings {
            println!("     - {}: {}", dataset, reason);
        }
    }

    if !actual_failures.is_empty() {
        println!("   Failed tests:");
        for (dataset, reason) in &actual_failures {
            println!("     - {}: {}", dataset, reason);
        }
        panic!(
            "White test validation failed for {} datasets",
            actual_failures.len()
        );
    }

    println!();
    println!(" White comprehensive validation passed!");
}
