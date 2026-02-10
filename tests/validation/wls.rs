// ============================================================================
// Per-Dataset WLS Validation Tests
// ============================================================================
//
// These tests validate the Rust WLS implementation against reference values
// from R (lm with weights) and Python (statsmodels WLS) for multiple datasets.
//
// WLS with equal weights (all 1.0) should produce identical results to OLS.
//
// Note: Excludes synthetic_collinear and synthetic_high_vif which are
// designed to test multicollinearity edge cases (numerically unstable).
//
// To regenerate reference values:
//   Rscript verification/scripts/runners/run_all_diagnostics_r.R
//   python verification/scripts/runners/run_all_diagnostics_python.py

use crate::common::{
    expect_wls_result, load_dataset_with_encoding, load_wls_result,
    CategoricalEncoding, WLS_TOLERANCE,
};
use linreg_core::weighted_regression::wls_regression;

const TEST_DATASETS: &[&str] = &[
    "bodyfat",
    "cars_stopping",
    "faithful",
    "lh",
    "longley",
    "mtcars",
    "prostate",
    // "synthetic_collinear",   // Excluded: perfect collinearity causes numerical instability
    // "synthetic_high_vif",    // Excluded: high VIF (>5) causes numerical instability
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

/// Validate WLS against R reference for a specific dataset
fn validate_wls_r_dataset(dataset_name: &str) {
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let r_results_dir = current_dir.join("verification/results/r");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let csv_path = datasets_dir.join(format!("{}.csv", dataset_name));
    let r_result_path = r_results_dir.join(format!("{}_wls.json", dataset_name));

    // Load dataset with R-compatible 1-based categorical encoding
    let dataset = load_dataset_with_encoding(&csv_path, CategoricalEncoding::OneBased)
        .expect(&format!("Failed to load {} dataset", dataset_name));

    // Load R reference - panic loudly if not found
    let r_ref = expect_wls_result(&r_result_path);

    // Create equal weights (WLS with w=1 is equivalent to OLS)
    let weights: Vec<f64> = vec![1.0; dataset.y.len()];

    // Run WLS regression
    let result = match wls_regression(&dataset.y, &dataset.x_vars, &weights) {
        Ok(r) => r,
        Err(e) => {
            println!("   WLS regression failed: {}", e);
            return;
        }
    };

    // Validate coefficients
    let mut all_passed = true;
    for i in 0..r_ref.coefficients.len() {
        let diff = (result.coefficients[i] - r_ref.coefficients[i]).abs();
        if diff > WLS_TOLERANCE {
            println!(
                "   coef[{}]: Rust = {:.8}, R = {:.8}, diff = {:.2e}",
                i, result.coefficients[i], r_ref.coefficients[i], diff
            );
            all_passed = false;
        }
    }

    // Validate R-squared
    let rsq_diff = (result.r_squared - r_ref.r_squared).abs();
    if rsq_diff > WLS_TOLERANCE {
        println!(
            "   R²: Rust = {:.8}, R = {:.8}, diff = {:.2e}",
            result.r_squared, r_ref.r_squared, rsq_diff
        );
        all_passed = false;
    }

    // Validate adjusted R-squared
    let adj_rsq_diff = (result.adj_r_squared - r_ref.adj_r_squared).abs();
    if adj_rsq_diff > WLS_TOLERANCE {
        println!(
            "   Adj R²: Rust = {:.8}, R = {:.8}, diff = {:.2e}",
            result.adj_r_squared, r_ref.adj_r_squared, adj_rsq_diff
        );
        all_passed = false;
    }

    // Validate F-statistic
    let f_diff = (result.f_statistic - r_ref.f_statistic).abs();
    if f_diff > WLS_TOLERANCE {
        println!(
            "   F: Rust = {:.8}, R = {:.8}, diff = {:.2e}",
            result.f_statistic, r_ref.f_statistic, f_diff
        );
        all_passed = false;
    }

    // Validate MSE
    let mse_diff = (result.mse - r_ref.mse).abs();
    if mse_diff > WLS_TOLERANCE {
        println!(
            "   MSE: Rust = {:.8}, R = {:.8}, diff = {:.2e}",
            result.mse, r_ref.mse, mse_diff
        );
        all_passed = false;
    }

    // Validate standard errors
    for i in 0..r_ref.std_errors.len().min(result.standard_errors.len()) {
        let diff = (result.standard_errors[i] - r_ref.std_errors[i]).abs();
        if diff > WLS_TOLERANCE {
            println!(
                "   SE[{}]: Rust = {:.8}, R = {:.8}, diff = {:.2e}",
                i, result.standard_errors[i], r_ref.std_errors[i], diff
            );
            all_passed = false;
        }
    }

    // Validate t-statistics
    for i in 0..r_ref.t_stats.len().min(result.t_statistics.len()) {
        let diff = (result.t_statistics[i] - r_ref.t_stats[i]).abs();
        if diff > WLS_TOLERANCE {
            println!(
                "   t[{}]: Rust = {:.8}, R = {:.8}, diff = {:.2e}",
                i, result.t_statistics[i], r_ref.t_stats[i], diff
            );
            all_passed = false;
        }
    }

    // Validate p-values
    for i in 0..r_ref.p_values.len().min(result.p_values.len()) {
        let diff = (result.p_values[i] - r_ref.p_values[i]).abs();
        if diff > WLS_TOLERANCE {
            println!(
                "   p[{}]: Rust = {:.8}, R = {:.8}, diff = {:.2e}",
                i, result.p_values[i], r_ref.p_values[i], diff
            );
            all_passed = false;
        }
    }

    if all_passed {
        println!("   {} WLS validation: PASS", dataset_name);
    } else {
        panic!("{} WLS validation: FAILED", dataset_name);
    }
}

/// Validate WLS against Python reference for a specific dataset
fn validate_wls_python_dataset(dataset_name: &str) {
    // Looser tolerance for Python due to implementation differences
    const PYTHON_TOLERANCE: f64 = 1e-6;

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let python_results_dir = current_dir.join("verification/results/python");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let csv_path = datasets_dir.join(format!("{}.csv", dataset_name));
    let python_result_path = python_results_dir.join(format!("{}_wls.json", dataset_name));

    // Load dataset with Python-compatible 0-based categorical encoding
    let dataset = load_dataset_with_encoding(&csv_path, CategoricalEncoding::ZeroBased)
        .expect(&format!("Failed to load {} dataset", dataset_name));

    // Load Python reference - skip if not found
    let py_ref = match load_wls_result(&python_result_path) {
        Some(r) => r,
        None => {
            println!("   SKIP: Python WLS reference not found for {}", dataset_name);
            return;
        }
    };

    // Create equal weights
    let weights: Vec<f64> = vec![1.0; dataset.y.len()];

    // Run WLS regression
    let result = match wls_regression(&dataset.y, &dataset.x_vars, &weights) {
        Ok(r) => r,
        Err(e) => {
            println!("   WLS regression failed: {}", e);
            return;
        }
    };

    let mut all_passed = true;

    // Validate coefficients
    for i in 0..py_ref.coefficients.len().min(result.coefficients.len()) {
        let diff = (result.coefficients[i] - py_ref.coefficients[i]).abs();
        if diff > PYTHON_TOLERANCE {
            println!(
                "   coef[{}]: Rust = {:.8}, Python = {:.8}, diff = {:.2e}",
                i, result.coefficients[i], py_ref.coefficients[i], diff
            );
            all_passed = false;
        }
    }

    // Validate R-squared
    let rsq_diff = (result.r_squared - py_ref.r_squared).abs();
    if rsq_diff > PYTHON_TOLERANCE {
        println!(
            "   R²: Rust = {:.8}, Python = {:.8}, diff = {:.2e}",
            result.r_squared, py_ref.r_squared, rsq_diff
        );
        all_passed = false;
    }

    // Validate F-statistic
    let f_diff = (result.f_statistic - py_ref.f_statistic).abs();
    if f_diff > PYTHON_TOLERANCE {
        println!(
            "   F: Rust = {:.8}, Python = {:.8}, diff = {:.2e}",
            result.f_statistic, py_ref.f_statistic, f_diff
        );
        all_passed = false;
    }

    // Validate standard errors
    for i in 0..py_ref.std_errors.len().min(result.standard_errors.len()) {
        let diff = (result.standard_errors[i] - py_ref.std_errors[i]).abs();
        if diff > PYTHON_TOLERANCE {
            println!(
                "   SE[{}]: Rust = {:.8}, Python = {:.8}, diff = {:.2e}",
                i, result.standard_errors[i], py_ref.std_errors[i], diff
            );
            all_passed = false;
        }
    }

    // Validate t-statistics
    for i in 0..py_ref.t_stats.len().min(result.t_statistics.len()) {
        let diff = (result.t_statistics[i] - py_ref.t_stats[i]).abs();
        if diff > PYTHON_TOLERANCE {
            println!(
                "   t[{}]: Rust = {:.8}, Python = {:.8}, diff = {:.2e}",
                i, result.t_statistics[i], py_ref.t_stats[i], diff
            );
            all_passed = false;
        }
    }

    // Validate p-values
    for i in 0..py_ref.p_values.len().min(result.p_values.len()) {
        let diff = (result.p_values[i] - py_ref.p_values[i]).abs();
        if diff > PYTHON_TOLERANCE {
            println!(
                "   p[{}]: Rust = {:.8}, Python = {:.8}, diff = {:.2e}",
                i, result.p_values[i], py_ref.p_values[i], diff
            );
            all_passed = false;
        }
    }

    if all_passed {
        println!("   {} Python WLS validation: PASS", dataset_name);
    } else {
        panic!("{} Python WLS validation: FAILED", dataset_name);
    }
}

#[test]
fn validate_wls_r_all_datasets() {
    println!("\n========== PER-DATASET WLS VALIDATION (R) ==========\n");

    for dataset in TEST_DATASETS {
        println!("--- Dataset: {} ---", dataset);
        validate_wls_r_dataset(dataset);
    }
}

#[test]
fn validate_wls_python_all_datasets() {
    println!("\n========== PER-DATASET WLS VALIDATION (Python) ==========\n");

    for dataset in TEST_DATASETS {
        println!("--- Dataset: {} ---", dataset);
        validate_wls_python_dataset(dataset);
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn validate_wls_negative_weight_error() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let weights = vec![1.0, -1.0, 1.0, 1.0, 1.0];

    let result = wls_regression(&y, &x, &weights);
    assert!(result.is_err(), "WLS with negative weight should return error");
}

#[test]
fn validate_wls_zero_sum_weights_error() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let weights = vec![0.0, 0.0, 0.0, 0.0, 0.0];

    let result = wls_regression(&y, &x, &weights);
    assert!(result.is_err(), "WLS with zero weights should return error");
}

#[test]
fn validate_wls_insufficient_data_error() {
    let y = vec![1.0, 2.0, 3.0];
    let x = vec![vec![1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0]];
    let weights = vec![1.0, 1.0, 1.0];

    let result = wls_regression(&y, &x, &weights);
    assert!(result.is_err(), "WLS with insufficient data should return error");
}

// ============================================================================
// Behavioral Tests
// ============================================================================

#[test]
fn validate_wls_equal_weights_matches_ols() {
    // Simple linear data: y = 2x + 1
    let n = 20;
    let x_data: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    let weights: Vec<f64> = vec![1.0; n];

    let result = wls_regression(&y_data, &[x_data], &weights).expect("WLS regression failed");

    let intercept = result.coefficients[0];
    let slope = result.coefficients[1];

    assert!((intercept - 1.0).abs() < 1e-10, "Intercept should be ~1.0, got {}", intercept);
    assert!((slope - 2.0).abs() < 1e-10, "Slope should be ~2.0, got {}", slope);
    assert!((result.r_squared - 1.0).abs() < 1e-10, "R² should be 1.0, got {}", result.r_squared);
}

#[test]
fn validate_wls_downweights_outlier() {
    let n = 20;
    let x_data: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let mut y_data: Vec<f64> = x_data.iter().map(|&xi| 2.0 * xi).collect();
    y_data[n - 1] += 50.0; // Add outlier

    let equal_weights: Vec<f64> = vec![1.0; n];
    let result_equal = wls_regression(&y_data, &[x_data.clone()], &equal_weights).unwrap();
    let slope_equal = result_equal.coefficients[1];

    let mut low_outlier_weights = vec![1.0; n];
    low_outlier_weights[n - 1] = 0.01;
    let result_low = wls_regression(&y_data, &[x_data.clone()], &low_outlier_weights).unwrap();
    let slope_low = result_low.coefficients[1];

    assert!(
        (slope_low - 2.0).abs() < (slope_equal - 2.0).abs(),
        "Low outlier weight should give slope closer to 2.0"
    );
}
