// ============================================================================
// DFFITS Validation
// ============================================================================
//
// Comprehensive validation of DFFITS computation
// against R and Python reference implementations.
//
// DFFITS measures the influence of each observation on its own fitted value.

use crate::common::{
    load_dataset, load_python_dffits_result, load_r_dffits_result, ALL_DATASETS, DFFITS_TOLERANCE,
};

use linreg_core::diagnostics;

#[test]
fn validate_dffits_all_datasets() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  DFFITS - COMPREHENSIVE MULTI-DATASET VALIDATION                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");
    let r_results_dir = current_dir.join("verification/results/r");
    let python_results_dir = current_dir.join("verification/results/python");

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
            },
        };

        println!(
            "    Loaded: n = {}, predictors = {}",
            dataset.y.len(),
            dataset.x_vars.len()
        );

        // Run DFFITS test
        let rust_dffits = match diagnostics::dffits_test(&dataset.y, &dataset.x_vars) {
            Ok(d) => d,
            Err(e) => {
                println!("     DFFITS test failed: {}", e);
                failed_tests.push((dataset_name.to_string(), format!("DFFITS error: {}", e)));
                continue;
            },
        };

        // Find max absolute DFFITS value and its index
        let (max_dffits, max_idx) = rust_dffits
            .dffits
            .iter()
            .enumerate()
            .map(|(i, v)| (v.abs(), i))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0.0, 0));

        println!(
            "    Rust: max_dffits = {:.6}, max_idx = {}",
            max_dffits, max_idx + 1
        );

        // Validate against R
        let r_result_path = r_results_dir.join(format!("{}_dffits.json", dataset_name));
        if let Some(r_ref) = load_r_dffits_result(&r_result_path) {
            total_tests += 1;

            // Compare all DFFITS values using relative tolerance
            let mut all_match = true;
            let mut max_diff = 0.0;
            let mut max_rel_error = 0.0;
            const REL_TOLERANCE: f64 = 0.01; // 1% relative tolerance
            const ABS_TOLERANCE: f64 = 1e-6; // Absolute tolerance for near-zero values

            for (i, (rust_d, r_d)) in rust_dffits.dffits.iter().zip(r_ref.dffits.iter()).enumerate() {
                let diff = (rust_d - r_d).abs();
                if diff > max_diff {
                    max_diff = diff;
                }

                // Use relative tolerance for non-zero values, absolute for near-zero
                let rel_error = if r_d.abs() < ABS_TOLERANCE {
                    diff
                } else {
                    diff / r_d.abs()
                };

                if rel_error > max_rel_error {
                    max_rel_error = rel_error;
                }

                if rel_error > REL_TOLERANCE {
                    all_match = false;
                    if rel_error > 1e-3 {
                        println!(
                            "      DFFITS[{}] mismatch: Rust = {:.2e}, R = {:.2e}, diff = {:.2e}, rel_error = {:.2e}",
                            i + 1, rust_d, r_d, diff, rel_error
                        );
                    }
                }
            }

            // Compare influential observations
            let r_influential = &r_ref.influential_observations;
            let rust_influential_set: std::collections::HashSet<_> =
                rust_dffits.influential_observations.iter().cloned().collect();

            let influential_match = rust_influential_set
                == r_influential.iter().cloned().collect::<std::collections::HashSet<_>>();

            println!(
                "    R:    n = {}, p = {}, threshold = {:.6}",
                r_ref.n.get(0).copied().unwrap_or(0),
                r_ref.p.get(0).copied().unwrap_or(0),
                r_ref.threshold.get(0).copied().unwrap_or(0.0)
            );
            println!(
                "          Diff: max_diff = {:.2e}, influential_match = {}",
                max_diff, influential_match
            );

            if all_match && influential_match && max_rel_error <= REL_TOLERANCE {
                println!("     R validation: PASS");
                passed_r += 1;
            } else {
                println!("     R validation: FAIL");
                failed_tests.push((dataset_name.to_string(), "R DFFITS mismatch".to_string()));
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
            python_results_dir.join(format!("{}_dffits.json", dataset_name));
        if let Some(py_ref) = load_python_dffits_result(&python_result_path) {
            total_tests += 1;

            // Compare all DFFITS values using relative tolerance
            let mut all_match = true;
            let mut max_diff = 0.0;
            let mut max_rel_error = 0.0;
            const REL_TOLERANCE: f64 = 0.01; // 1% relative tolerance
            const ABS_TOLERANCE: f64 = 1e-6; // Absolute tolerance for near-zero values

            for (i, (rust_d, py_d)) in rust_dffits.dffits.iter().zip(py_ref.dffits.iter()).enumerate() {
                let diff = (rust_d - py_d).abs();
                if diff > max_diff {
                    max_diff = diff;
                }

                // Use relative tolerance for non-zero values, absolute for near-zero
                let rel_error = if py_d.abs() < ABS_TOLERANCE {
                    diff
                } else {
                    diff / py_d.abs()
                };

                if rel_error > max_rel_error {
                    max_rel_error = rel_error;
                }

                if rel_error > REL_TOLERANCE {
                    all_match = false;
                    if rel_error > 1e-3 {
                        println!(
                            "      DFFITS[{}] mismatch: Rust = {:.2e}, Python = {:.2e}, diff = {:.2e}, rel_error = {:.2e}",
                            i + 1, rust_d, py_d, diff, rel_error
                        );
                    }
                }
            }

            // Compare influential observations
            let py_influential_set: std::collections::HashSet<_> =
                py_ref.influential_observations.iter().cloned().collect();
            let rust_influential_set: std::collections::HashSet<_> =
                rust_dffits.influential_observations.iter().cloned().collect();

            let influential_match = rust_influential_set == py_influential_set;

            println!(
                "    Python: n = {}, p = {}, threshold = {:.6}",
                py_ref.n, py_ref.p, py_ref.threshold
            );
            println!(
                "          Diff: max_diff = {:.2e}, influential_match = {}",
                max_diff, influential_match
            );

            if all_match && influential_match && max_rel_error <= REL_TOLERANCE {
                println!("     Python validation: PASS");
                passed_python += 1;
            } else {
                println!("     Python validation: FAIL");
                failed_tests.push((
                    dataset_name.to_string(),
                    "Python DFFITS mismatch".to_string(),
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
    println!("║  DFFITS VALIDATION SUMMARY                                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Total tests run:       {:>40}║", total_tests);
    println!("║  R validations passed:   {:>40}║", passed_r);
    println!("║  Python validations passed: {:>39}║", passed_python);
    println!("║  Failed tests:          {:>40}║", failed_tests.len());
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    assert!(
        total_tests > 0,
        "No DFFITS validation tests were run."
    );
    let pass_rate = (passed_r + passed_python) as f64 / total_tests as f64;
    assert!(
        pass_rate >= 0.9,
        "DFFITS validation pass rate ({:.1}%) is below 90%.",
        pass_rate * 100.0
    );

    println!();
    println!(" DFFITS comprehensive validation passed!");
}
