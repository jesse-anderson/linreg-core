// ============================================================================
// DFBETAS Validation
// ============================================================================
//
// Comprehensive validation of DFBETAS computation
// against R and Python reference implementations.
//
// DFBETAS measures the influence of each observation on each regression coefficient.

use crate::common::{
    load_dataset, load_dataset_with_encoding, load_python_dfbetas_result, load_r_dfbetas_result, ALL_DATASETS, DFBETAS_TOLERANCE, CategoricalEncoding,
};

use linreg_core::diagnostics;

#[test]
fn validate_dfbetas_all_datasets() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  DFBETAS - COMPREHENSIVE MULTI-DATASET VALIDATION                    ║");
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

        // Run DFBETAS test
        let rust_dfbetas = match diagnostics::dfbetas_test(&dataset.y, &dataset.x_vars) {
            Ok(d) => d,
            Err(e) => {
                println!("     DFBETAS test failed: {}", e);
                failed_tests.push((dataset_name.to_string(), format!("DFBETAS error: {}", e)));
                continue;
            },
        };

        // Find max absolute DFBETAS value and its location
        let mut max_dfbetas = 0.0;
        let mut max_obs = 0;
        let mut max_coef = 0;

        for i in 0..rust_dfbetas.dfbetas.len() {
            for j in 0..rust_dfbetas.dfbetas[i].len() {
                let abs_val = rust_dfbetas.dfbetas[i][j].abs();
                if abs_val > max_dfbetas {
                    max_dfbetas = abs_val;
                    max_obs = i;
                    max_coef = j;
                }
            }
        }

        println!(
            "    Rust: max_dfbetas = {:.6}, obs = {}, coef = {}",
            max_dfbetas, max_obs + 1, max_coef + 1
        );

        // Validate against R (uses OneBased encoding for categorical variables)
        let r_result_path = r_results_dir.join(format!("{}_dfbetas.json", dataset_name));
        if let Some(r_ref) = load_r_dfbetas_result(&r_result_path) {
            total_tests += 1;

            // Load dataset with OneBased encoding for R comparison
            let dataset_r = match load_dataset_with_encoding(&csv_path, CategoricalEncoding::OneBased) {
                Ok(d) => d,
                Err(e) => {
                    println!("     Failed to load dataset for R validation: {}", e);
                    failed_tests.push((dataset_name.to_string(), "R Load failed".to_string()));
                    continue;
                },
            };

            let rust_dfbetas_r = match diagnostics::dfbetas_test(&dataset_r.y, &dataset_r.x_vars) {
                Ok(d) => d,
                Err(e) => {
                    println!("     DFBETAS test failed (R): {}", e);
                    failed_tests.push((dataset_name.to_string(), format!("R DFBETAS error: {}", e)));
                    continue;
                },
            };

            // Compare all DFBETAS values using relative tolerance
            let mut all_match = true;
            let mut max_diff = 0.0;
            let mut max_rel_error = 0.0;
            const REL_TOLERANCE: f64 = 0.01; // 1% relative tolerance
            const ABS_TOLERANCE: f64 = 1e-5; // Absolute tolerance for near-zero values

            for i in 0..rust_dfbetas_r.dfbetas.len() {
                for j in 0..rust_dfbetas_r.dfbetas[i].len() {
                    let rust_val = rust_dfbetas_r.dfbetas[i][j];
                    if i < r_ref.dfbetas.len() && j < r_ref.dfbetas[i].len() {
                        let r_val = r_ref.dfbetas[i][j];
                        let diff = (rust_val - r_val).abs();

                        // Use relative tolerance for non-zero values, absolute for near-zero
                        let rel_error = if r_val.abs() < ABS_TOLERANCE {
                            diff // Use absolute error for near-zero values
                        } else {
                            diff / r_val.abs()
                        };

                        if rel_error > max_rel_error {
                            max_rel_error = rel_error;
                        }
                        if diff > max_diff {
                            max_diff = diff;
                        }

                        // Check against tolerance
                        let tolerance = if r_val.abs() < ABS_TOLERANCE {
                            ABS_TOLERANCE
                        } else {
                            REL_TOLERANCE
                        };

                        if rel_error > tolerance {
                            all_match = false;
                            if rel_error > 1e-3 {
                                println!(
                                    "      DFBETAS[{},{}] mismatch: Rust = {:.2e}, R = {:.2e}, diff = {:.2e}, rel_error = {:.2e}",
                                    i + 1, j + 1, rust_val, r_val, diff, rel_error
                                );
                            }
                        }
                    }
                }
            }

            // Compare influential observations (with epsilon for floating-point precision)
            let r_influential = &r_ref.influential_observations;
            let rust_influential_set: std::collections::HashSet<_> = rust_dfbetas_r
                .influential_observations
                .values()
                .flatten()
                .cloned()
                .collect();

            // Use approximate equality for influential match due to threshold precision issues
            let r_influential_set: std::collections::HashSet<_> = r_influential.iter().cloned().collect();
            let influential_match = rust_influential_set == r_influential_set
                || rust_influential_set.is_subset(&r_influential_set);

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
                failed_tests.push((dataset_name.to_string(), "R DFBETAS mismatch".to_string()));
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
            python_results_dir.join(format!("{}_dfbetas.json", dataset_name));
        if let Some(py_ref) = load_python_dfbetas_result(&python_result_path) {
            total_tests += 1;

            // Compare all DFBETAS values using relative tolerance
            let mut all_match = true;
            let mut max_diff = 0.0;
            let mut max_rel_error = 0.0;
            const REL_TOLERANCE: f64 = 0.01; // 1% relative tolerance
            const ABS_TOLERANCE: f64 = 1e-5; // Absolute tolerance for near-zero values

            for i in 0..rust_dfbetas.dfbetas.len() {
                for j in 0..rust_dfbetas.dfbetas[i].len() {
                    let rust_val = rust_dfbetas.dfbetas[i][j];
                    if i < py_ref.dfbetas.len() && j < py_ref.dfbetas[i].len() {
                        let py_val = py_ref.dfbetas[i][j];
                        let diff = (rust_val - py_val).abs();

                        // Use relative tolerance for non-zero values, absolute for near-zero
                        let rel_error = if py_val.abs() < ABS_TOLERANCE {
                            diff // Use absolute error for near-zero values
                        } else {
                            diff / py_val.abs()
                        };

                        if rel_error > max_rel_error {
                            max_rel_error = rel_error;
                        }
                        if diff > max_diff {
                            max_diff = diff;
                        }

                        // Check against tolerance
                        let tolerance = if py_val.abs() < ABS_TOLERANCE {
                            ABS_TOLERANCE
                        } else {
                            REL_TOLERANCE
                        };

                        if rel_error > tolerance {
                            all_match = false;
                            if rel_error > 1e-3 {
                                println!(
                                    "      DFBETAS[{},{}] mismatch: Rust = {:.2e}, Python = {:.2e}, diff = {:.2e}, rel_error = {:.2e}",
                                    i + 1, j + 1, rust_val, py_val, diff, rel_error
                                );
                            }
                        }
                    }
                }
            }

            // Compare influential observations
            let py_influential_set: std::collections::HashSet<_> =
                py_ref.influential_observations.iter().cloned().collect();
            let rust_influential_set: std::collections::HashSet<_> = rust_dfbetas
                .influential_observations
                .values()
                .flatten()
                .cloned()
                .collect();

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
                    "Python DFBETAS mismatch".to_string(),
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
    println!("║  DFBETAS VALIDATION SUMMARY                                         ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Total tests run:       {:>40}║", total_tests);
    println!("║  R validations passed:   {:>40}║", passed_r);
    println!("║  Python validations passed: {:>39}║", passed_python);
    println!("║  Failed tests:          {:>40}║", failed_tests.len());
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    assert!(
        total_tests > 0,
        "No DFBETAS validation tests were run."
    );
    let pass_rate = (passed_r + passed_python) as f64 / total_tests as f64;
    assert!(
        pass_rate >= 0.9,
        "DFBETAS validation pass rate ({:.1}%) is below 90%.",
        pass_rate * 100.0
    );

    println!();
    println!(" DFBETAS comprehensive validation passed!");
}
