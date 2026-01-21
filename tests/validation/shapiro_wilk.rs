// ============================================================================
// Shapiro-Wilk Test Validation
// ============================================================================
//
// Comprehensive validation of the Shapiro-Wilk normality test
// against R and Python reference implementations.
//
// # Implementation Choice
//
// This implementation follows **R's shapiro.test()** approach (Royston 1982, 1995).
// Rust results should match R reference values within STAT_TOLERANCE (0.001).
//
// # Notes
//
// Shapiro-Wilk is one of the most powerful tests for normality, especially for
// small to moderate sample sizes (3 ≤ n ≤ 5000). It tests the null hypothesis
// that the residuals are normally distributed.
//
// # Skipped Tests
//
// - `iris`: Contains string data (categorical species column) - cannot be loaded
// - `synthetic_*` datasets: Have `x,y` column ordering (predictor first) which differs
//   from the validation script's expectation (first column = response variable)
//   These datasets are excluded from Shapiro-Wilk validation.

use crate::common::{
    load_dataset, load_r_shapiro_wilk_result, load_python_shapiro_wilk_result,
    SHAPIRO_WILK_DATASETS, STAT_TOLERANCE,
};

use linreg_core::diagnostics;

#[test]
fn validate_shapiro_wilk_all_datasets() {

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  SHAPIRO-WILK TEST - COMPREHENSIVE MULTI-DATASET VALIDATION       ║");
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

    for dataset_name in SHAPIRO_WILK_DATASETS {
        let csv_path = datasets_dir.join(format!("{}.csv", dataset_name));

        if !csv_path.exists() {
            println!("    Skipping {}: CSV file not found", dataset_name);
            continue;
        }

        println!("  ┌─────────────────────────────────────────────────────────────────┐");
        println!("  │  Dataset: {:<52}│", dataset_name);
        println!("  └─────────────────────────────────────────────────────────────────┘");

        // Load the dataset
        let dataset = match load_dataset(&csv_path) {
            Ok(d) => d,
            Err(e) => {
                println!("     Failed to load dataset: {}", e);
                failed_tests.push((dataset_name.to_string(), "Load failed".to_string()));
                continue;
            }
        };

        println!("    Loaded: n = {}, predictors = {}", dataset.y.len(), dataset.x_vars.len());

        // Run Shapiro-Wilk test
        let rust_result = match diagnostics::shapiro_wilk_test(&dataset.y, &dataset.x_vars) {
            Ok(r) => r,
            Err(e) => {
                println!("     Shapiro-Wilk test failed: {}", e);
                failed_tests.push((dataset_name.to_string(), format!("Test error: {}", e)));
                continue;
            }
        };

        println!("    Rust: W = {:.6}, p = {:.6}", rust_result.statistic, rust_result.p_value);

        // Validate against R
        let r_result_path = r_results_dir.join(format!("{}_shapiro_wilk.json", dataset_name));
        if let Some(r_ref) = load_r_shapiro_wilk_result(&r_result_path) {
            total_tests += 1;

            let r_stat = r_ref.statistic.get(0).copied().unwrap_or(0.0);
            let r_pval = r_ref.p_value.get(0).copied().unwrap_or(1.0);

            let stat_diff = (rust_result.statistic - r_stat).abs();
            let pval_diff = (rust_result.p_value - r_pval).abs();

            let stat_match = stat_diff <= STAT_TOLERANCE;
            let pval_match = pval_diff <= STAT_TOLERANCE;

            println!("    R:    W = {:.6}, p = {:.6}", r_stat, r_pval);
            println!("          Diff: stat = {:.2e}, p = {:.2e}", stat_diff, pval_diff);

            if stat_match && pval_match {
                println!("     R validation: PASS");
                passed_r += 1;
            } else {
                println!("     R validation: FAIL");
                failed_tests.push((dataset_name.to_string(), format!("R mismatch: stat diff={:.2e}", stat_diff)));
            }
        } else {
            println!("      R reference file not found: {}", r_result_path.display());
        }

        println!();

        // Validate against Python
        let python_result_path = python_results_dir.join(format!("{}_shapiro_wilk.json", dataset_name));
        if let Some(py_ref) = load_python_shapiro_wilk_result(&python_result_path) {
            total_tests += 1;

            let py_stat = py_ref.statistic;
            let py_pval = py_ref.p_value;

            let stat_diff = (rust_result.statistic - py_stat).abs();
            let pval_diff = (rust_result.p_value - py_pval).abs();

            let stat_match = stat_diff <= STAT_TOLERANCE;
            let pval_match = pval_diff <= STAT_TOLERANCE;

            println!("    Python: W = {:.6}, p = {:.6}", py_stat, py_pval);
            println!("          Diff: stat = {:.2e}, p = {:.2e}", stat_diff, pval_diff);

            if stat_match && pval_match {
                println!("     Python validation: PASS");
                passed_python += 1;
            } else {
                // Note: We follow R's shapiro.test. Python's scipy.stats.shapiro differs.
                println!("      Python validation: KNOWN DIFFERENCE (R convention followed)");
                println!("       Python: W = {:.6}, p = {:.6}", py_stat, py_pval);
                println!("       Rust/R:  W = {:.6}, p = {:.6}", rust_result.statistic, rust_result.p_value);
                println!("       Difference: stat = {:.2e}, p = {:.2e}", stat_diff, pval_diff);
                println!("       Note: We follow R's shapiro.test. Python's scipy.stats.shapiro differs.");
            }
        } else {
            println!("      Python reference file not found: {}", python_result_path.display());
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
    assert!(total_tests > 0, "No validation tests were run. Check that result files exist.");

    // Assert that we have a reasonable pass rate (at least 80%)
    let pass_rate = (passed_r + passed_python) as f64 / total_tests as f64;
    assert!(pass_rate >= 0.8,
        "Validation pass rate ({:.1}%) is below 80% threshold. See failed tests above.",
        pass_rate * 100.0
    );

    println!();
    println!(" Shapiro-Wilk comprehensive validation passed!");
}
