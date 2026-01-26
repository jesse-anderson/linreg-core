// ============================================================================
// Per-Dataset OLS Validation Tests
// ============================================================================
//
// These tests validate the Rust OLS implementation against reference values
// from R (lm) and Python (statsmodels) for multiple datasets.
//
// Note: Excludes synthetic_collinear and synthetic_high_vif which are
// designed to test multicollinearity edge cases (numerically unstable for OLS).

use crate::common::{load_dataset_with_encoding, load_ols_by_dataset_result, TIGHT_TOLERANCE, CategoricalEncoding};
use linreg_core::core;

const TEST_DATASETS: &[&str] = &[
    "bodyfat",
    "cars_stopping",
    "faithful",
    "lh",
    "longley",
    "mtcars",
    "prostate",
    // "synthetic_collinear",   // Excluded: perfect collinearity causes numerical instability
    // "synthetic_high_vif",     // Excluded: high VIF (>5) causes numerical instability
    "synthetic_interaction",
    "synthetic_multiple",
    "synthetic_autocorrelated",
    "synthetic_heteroscedastic",
    "synthetic_nonlinear",
    "synthetic_nonnormal",
    "synthetic_outliers",
    "synthetic_simple_linear",
    "synthetic_small",
    "ToothGrowth",
];

/// Validate OLS against R reference for a specific dataset
fn validate_ols_r_dataset(dataset_name: &str) {
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let r_results_dir = current_dir.join("verification/results/r");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let csv_path = datasets_dir.join(format!("{}.csv", dataset_name));
    let r_result_path = r_results_dir.join(format!("{}_ols.json", dataset_name));

    // Load dataset with R-compatible 1-based categorical encoding
    let dataset =
        load_dataset_with_encoding(&csv_path, CategoricalEncoding::OneBased)
            .expect(&format!("Failed to load {} dataset", dataset_name));

    // Load R reference - panic loudly if not found
    let r_ref = load_ols_by_dataset_result(&r_result_path).unwrap_or_else(|| {
        panic!(
            "R OLS result file not found: {}\n \
             Run: Rscript verification/scripts/runners/run_all_diagnostics_r.R",
            r_result_path.display()
        )
    });

    // Build variable names
    let mut names = vec!["Intercept".to_string()];
    names.extend(dataset.variable_names);

    // Run OLS regression
    let result = match core::ols_regression(&dataset.y, &dataset.x_vars, &names) {
        Ok(r) => r,
        Err(e) => {
            println!("   OLS regression failed: {}", e);
            return;
        },
    };

    // Validate coefficients
    let mut all_passed = true;
    for i in 0..r_ref.coefficients.len() {
        let diff = (result.coefficients[i] - r_ref.coefficients[i]).abs();
        if diff > TIGHT_TOLERANCE {
            println!(
                "   coef[{}]: Rust = {:.8}, R = {:.8}, diff = {:.2e}",
                i, result.coefficients[i], r_ref.coefficients[i], diff
            );
            all_passed = false;
        }
    }

    // Validate R-squared
    let rsq_diff = (result.r_squared - r_ref.r_squared).abs();
    if rsq_diff > TIGHT_TOLERANCE {
        println!(
            "   R²: Rust = {:.8}, R = {:.8}, diff = {:.2e}",
            result.r_squared, r_ref.r_squared, rsq_diff
        );
        all_passed = false;
    }

    // Validate F-statistic
    let f_diff = (result.f_statistic - r_ref.f_statistic).abs();
    if f_diff > TIGHT_TOLERANCE {
        println!(
            "   F: Rust = {:.8}, R = {:.8}, diff = {:.2e}",
            result.f_statistic, r_ref.f_statistic, f_diff
        );
        all_passed = false;
    }

    if all_passed {
        println!("   {} OLS validation: PASS", dataset_name);
    } else {
        panic!("{} OLS validation: FAILED", dataset_name);
    }
}

#[test]
fn validate_ols_r_all_datasets() {
    println!("\n========== PER-DATASET OLS VALIDATION (R) ==========\n");

    for dataset in TEST_DATASETS {
        println!("--- Dataset: {} ---", dataset);
        validate_ols_r_dataset(dataset);
    }
}
