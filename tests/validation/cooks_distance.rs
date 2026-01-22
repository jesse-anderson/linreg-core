// ============================================================================
// Cook's Distance Validation
// ============================================================================
//
// Comprehensive validation of Cook's Distance computation
// against R and Python reference implementations.
//
// Cook's Distance identifies influential observations that have
// both high residuals and high leverage.

use crate::common::{
    load_dataset, load_r_cooks_result, load_python_cooks_result,
    ALL_DATASETS, COOKS_TOLERANCE,
};

use linreg_core::diagnostics;

#[test]
fn validate_cooks_distance_all_datasets() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  COOK'S DISTANCE - COMPREHENSIVE MULTI-DATASET VALIDATION          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");
    let r_results_dir = current_dir.join("verification/results/r");
    let python_results_dir = current_dir.join("verification/results/python");

    // Cook's distance validation against R (all datasets) and Python (mtcars only)
    // R reference files exist for all datasets, Python only for mtcars
    let datasets = ALL_DATASETS;

    let mut total_tests = 0;
    let mut passed_r = 0;
    let mut passed_python = 0;
    let mut failed_tests = Vec::new();

    for dataset_name in datasets {
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

        // Run Cook's distance test
        let rust_cooks = match diagnostics::cooks_distance_test(&dataset.y, &dataset.x_vars) {
            Ok(c) => c,
            Err(e) => {
                println!("     Cook's distance test failed: {}", e);
                failed_tests.push((dataset_name.to_string(), format!("Cook's error: {}", e)));
                continue;
            }
        };

        // Find max distance and its index
        let (rust_max_dist, rust_max_idx) = rust_cooks.distances
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, v)| (*v, i))
            .unwrap_or((0.0, 0));

        println!("    Rust: max_dist = {:.6}, max_idx = {}", rust_max_dist, rust_max_idx);

        // Validate against R
        let r_result_path = r_results_dir.join(format!("{}_cooks_distance.json", dataset_name));
        if let Some(r_ref) = load_r_cooks_result(&r_result_path) {
            total_tests += 1;

            let r_distances = &r_ref.distances;
            let r_max_dist = r_ref.max_distance.get(0).copied().unwrap_or(0.0);
            let r_max_idx = r_ref.max_index.get(0).copied().unwrap_or(0);

            // Compare all distances
            let mut all_match = true;
            for (i, (rust_d, r_d)) in rust_cooks.distances.iter().zip(r_distances.iter()).enumerate() {
                if (rust_d - r_d).abs() > COOKS_TOLERANCE {
                    all_match = false;
                    println!("      Distance[{}] mismatch: Rust = {:.2e}, R = {:.2e}, diff = {:.2e}",
                        i, rust_d, r_d, (rust_d - r_d).abs());
                }
            }

            let max_dist_diff = (rust_max_dist - r_max_dist).abs();
            // R uses 1-based indexing, Rust uses 0-based
            let max_idx_match = rust_max_idx + 1 == r_max_idx;

            println!("    R:    max_dist = {:.6}, max_idx = {}", r_max_dist, r_max_idx);
            println!("          Diff: max_dist = {:.2e}, max_idx_match = {}",
                max_dist_diff, max_idx_match);

            if all_match && max_idx_match && max_dist_diff < COOKS_TOLERANCE {
                println!("     R validation: PASS");
                passed_r += 1;
            } else {
                println!("     R validation: FAIL");
                failed_tests.push((dataset_name.to_string(), "R Cook's mismatch".to_string()));
            }
        } else {
            println!("      R reference file not found: {}", r_result_path.display());
            failed_tests.push((dataset_name.to_string(), "R reference file missing".to_string()));
        }

        // Validate against Python
        let python_result_path = python_results_dir.join(format!("{}_cooks_distance.json", dataset_name));
        if let Some(py_ref) = load_python_cooks_result(&python_result_path) {
            total_tests += 1;

            let py_max_dist = py_ref.max_distance;
            let py_max_idx = py_ref.max_index;

            // Compare all distances
            let mut all_match = true;
            for (i, (rust_d, py_d)) in rust_cooks.distances.iter().zip(py_ref.distances.iter()).enumerate() {
                if (rust_d - py_d).abs() > COOKS_TOLERANCE {
                    all_match = false;
                    println!("      Distance[{}] mismatch: Rust = {:.2e}, Python = {:.2e}, diff = {:.2e}",
                        i, rust_d, py_d, (rust_d - py_d).abs());
                }
            }

            let max_dist_diff = (rust_max_dist - py_max_dist).abs();
            // Python uses 1-based indexing, Rust uses 0-based
            let max_idx_match = rust_max_idx + 1 == py_max_idx;

            println!("    Python: max_dist = {:.6}, max_idx = {}", py_max_dist, py_max_idx);
            println!("          Diff: max_dist = {:.2e}, max_idx_match = {}",
                max_dist_diff, max_idx_match);

            if all_match && max_idx_match && max_dist_diff < COOKS_TOLERANCE {
                println!("     Python validation: PASS");
                passed_python += 1;
            } else {
                println!("     Python validation: FAIL");
                failed_tests.push((dataset_name.to_string(), "Python Cook's mismatch".to_string()));
            }
        } else {
            println!("      Python reference file not found: {}", python_result_path.display());
            failed_tests.push((dataset_name.to_string(), "Python reference file missing".to_string()));
        }

        println!();
    }

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  COOK'S DISTANCE VALIDATION SUMMARY                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Total tests run:       {:>40}║", total_tests);
    println!("║  R validations passed:   {:>40}║", passed_r);
    println!("║  Python validations passed: {:>39}║", passed_python);
    println!("║  Failed tests:          {:>40}║", failed_tests.len());
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    assert!(total_tests > 0, "No Cook's Distance validation tests were run.");
    let pass_rate = (passed_r + passed_python) as f64 / total_tests as f64;
    assert!(pass_rate >= 0.9, "Cook's Distance validation pass rate ({:.1}%) is below 90%.", pass_rate * 100.0);

    println!();
    println!(" Cook's Distance comprehensive validation passed!");
}
