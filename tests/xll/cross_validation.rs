//! Cross Validation XLL UDF tests.
//!
//! Tests the LINREG.KFOLDOLS, LINREG.KFOLDRIDGE, LINREG.KFOLDLASSO,
//! and LINREG.KFOLDELASTICNET UDFs.

use crate::xll::common::*;
use linreg_core::xll::{
    xl_linreg_kfoldols, xl_linreg_kfoldridge, xl_linreg_kfoldlasso,
    xl_linreg_kfoldelasticnet, xlAutoFree12,
};

// ============================================================================
// Test Data
// ============================================================================

/// Dataset large enough for 5-fold CV (20 observations, 2 predictors)
fn cv_data() -> (Vec<f64>, Vec<Vec<f64>>) {
    let (y, x_vars) = mtcars_subset(); // 20 obs, 4 predictors
    (y, x_vars)
}

/// Build y, X, and optional n_folds ranges for CV tests.
struct CvRanges {
    y_oper: XLOPER12,
    _y_cells: Vec<XLOPER12>,
    x_oper: XLOPER12,
    _x_cells: Vec<XLOPER12>,
}

fn build_cv_ranges() -> CvRanges {
    let (y_data, x_data) = cv_data();
    let (y_oper, y_cells) = build_column_range(&y_data);
    let (x_oper, x_cells) = build_matrix_range(&x_data);
    CvRanges {
        y_oper, _y_cells: y_cells,
        x_oper, _x_cells: x_cells,
    }
}

// ============================================================================
// KFOLDOLS Tests
// ============================================================================

#[test]
fn test_kfoldols_returns_cv_table() {
    let ranges = build_cv_ranges();
    let result = XlResultGuard::new(xl_linreg_kfoldols(
        &ranges.y_oper, &ranges.x_oper, std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 2, "CV output should be 2 columns");
    assert_eq!(rows, 13, "CV output should be 13 rows (header + 8 metrics + blank + 3 summary)");

    // Check header labels
    assert_eq!(result.cell_string(0, 0), "Metric");
    assert_eq!(result.cell_string(0, 1), "Value");

    // Check metric labels
    assert_eq!(result.cell_string(1, 0), "Mean MSE");
    assert_eq!(result.cell_string(2, 0), "Std MSE");
    assert_eq!(result.cell_string(3, 0), "Mean RMSE");
    assert_eq!(result.cell_string(4, 0), "Std RMSE");
    assert_eq!(result.cell_string(5, 0), "Mean MAE");
    assert_eq!(result.cell_string(6, 0), "Std MAE");
    assert_eq!(result.cell_string(7, 0), "Mean R²");
    assert_eq!(result.cell_string(8, 0), "Std R²");

    // Check summary labels
    assert_eq!(result.cell_string(10, 0), "n Folds");
    assert_eq!(result.cell_string(11, 0), "n Samples");
    assert_eq!(result.cell_string(12, 0), "Train R²");
}

#[test]
fn test_kfoldols_values_are_reasonable() {
    let ranges = build_cv_ranges();
    let result = XlResultGuard::new(xl_linreg_kfoldols(
        &ranges.y_oper, &ranges.x_oper, std::ptr::null(),
    ));

    // Mean MSE should be positive
    let mean_mse = result.cell_f64(1, 1);
    assert!(mean_mse > 0.0, "Mean MSE should be positive, got {}", mean_mse);

    // Mean RMSE should be sqrt-ish of MSE
    let mean_rmse = result.cell_f64(3, 1);
    assert!(mean_rmse > 0.0, "Mean RMSE should be positive");

    // Mean R² — can be negative for CV on test folds, but typically positive for good models
    let mean_r2 = result.cell_f64(7, 1);
    assert!(mean_r2 < 1.0, "Mean R² should be < 1.0");

    // n_folds defaults to 5
    let n_folds = result.cell_f64(10, 1);
    assert_eq!(n_folds, 5.0);

    // n_samples = 20 (mtcars subset)
    let n_samples = result.cell_f64(11, 1);
    assert_eq!(n_samples, 20.0);
}

#[test]
fn test_kfoldols_custom_folds() {
    let ranges = build_cv_ranges();
    let nfolds_oper = XLOPER12::from_f64(4.0);

    let result = XlResultGuard::new(xl_linreg_kfoldols(
        &ranges.y_oper, &ranges.x_oper, &nfolds_oper,
    ));
    assert!(result.is_multi());
    let n_folds = result.cell_f64(10, 1);
    assert_eq!(n_folds, 4.0);
}

#[test]
fn test_kfoldols_null_inputs() {
    let ranges = build_cv_ranges();

    // null y
    let result = XlResultGuard::new(xl_linreg_kfoldols(
        std::ptr::null(), &ranges.x_oper, std::ptr::null(),
    ));
    assert!(result.is_error());

    // null x
    let result = XlResultGuard::new(xl_linreg_kfoldols(
        &ranges.y_oper, std::ptr::null(), std::ptr::null(),
    ));
    assert!(result.is_error());
}

// ============================================================================
// KFOLDRIDGE Tests
// ============================================================================

#[test]
fn test_kfoldridge_returns_cv_table() {
    let ranges = build_cv_ranges();
    let lambda = XLOPER12::from_f64(1.0);

    let result = XlResultGuard::new(xl_linreg_kfoldridge(
        &ranges.y_oper, &ranges.x_oper, &lambda, std::ptr::null(), std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 2);
    assert_eq!(rows, 13);

    // Verify header
    assert_eq!(result.cell_string(0, 0), "Metric");

    // Mean MSE should be positive
    let mean_mse = result.cell_f64(1, 1);
    assert!(mean_mse > 0.0);

    // Default 5 folds
    assert_eq!(result.cell_f64(10, 1), 5.0);
}

#[test]
fn test_kfoldridge_custom_folds_and_standardize() {
    let ranges = build_cv_ranges();
    let lambda = XLOPER12::from_f64(0.5);
    let nfolds = XLOPER12::from_f64(3.0);
    let standardize = XLOPER12::from_f64(0.0); // false

    let result = XlResultGuard::new(xl_linreg_kfoldridge(
        &ranges.y_oper, &ranges.x_oper, &lambda, &nfolds, &standardize,
    ));
    assert!(result.is_multi());
    assert_eq!(result.cell_f64(10, 1), 3.0);
}

// ============================================================================
// KFOLDLASSO Tests
// ============================================================================

#[test]
fn test_kfoldlasso_returns_cv_table() {
    let ranges = build_cv_ranges();
    let lambda = XLOPER12::from_f64(0.1);

    let result = XlResultGuard::new(xl_linreg_kfoldlasso(
        &ranges.y_oper, &ranges.x_oper, &lambda, std::ptr::null(), std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 2);
    assert_eq!(rows, 13);

    let mean_mse = result.cell_f64(1, 1);
    assert!(mean_mse > 0.0);
}

// ============================================================================
// KFOLDELASTICNET Tests
// ============================================================================

#[test]
fn test_kfoldelasticnet_returns_cv_table() {
    let ranges = build_cv_ranges();
    let lambda = XLOPER12::from_f64(0.1);
    let alpha = XLOPER12::from_f64(0.5);

    let result = XlResultGuard::new(xl_linreg_kfoldelasticnet(
        &ranges.y_oper, &ranges.x_oper, &lambda, &alpha,
        std::ptr::null(), std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 2);
    assert_eq!(rows, 13);

    let mean_mse = result.cell_f64(1, 1);
    assert!(mean_mse > 0.0);
    assert_eq!(result.cell_f64(10, 1), 5.0); // default folds
}

#[test]
fn test_kfoldelasticnet_custom_params() {
    let ranges = build_cv_ranges();
    let lambda = XLOPER12::from_f64(0.05);
    let alpha = XLOPER12::from_f64(0.7);
    let nfolds = XLOPER12::from_f64(4.0);
    let standardize = XLOPER12::from_f64(1.0);

    let result = XlResultGuard::new(xl_linreg_kfoldelasticnet(
        &ranges.y_oper, &ranges.x_oper, &lambda, &alpha, &nfolds, &standardize,
    ));
    assert!(result.is_multi());
    assert_eq!(result.cell_f64(10, 1), 4.0);
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_stress_cv_udfs() {
    let ranges = build_cv_ranges();
    let lambda = XLOPER12::from_f64(0.5);
    let alpha = XLOPER12::from_f64(0.5);
    let baseline = MemSnapshot::now();
    let n_iters = 500;
    let log_interval = n_iters / 10;

    for i in 0..n_iters {
        let result = match i % 4 {
            0 => xl_linreg_kfoldols(&ranges.y_oper, &ranges.x_oper, std::ptr::null()),
            1 => xl_linreg_kfoldridge(&ranges.y_oper, &ranges.x_oper, &lambda, std::ptr::null(), std::ptr::null()),
            2 => xl_linreg_kfoldlasso(&ranges.y_oper, &ranges.x_oper, &lambda, std::ptr::null(), std::ptr::null()),
            _ => xl_linreg_kfoldelasticnet(&ranges.y_oper, &ranges.x_oper, &lambda, &alpha, std::ptr::null(), std::ptr::null()),
        };
        assert!(!result.is_null());
        xlAutoFree12(result);

        if (i + 1) % log_interval == 0 {
            log_mem("CV stress", i + 1, &baseline);
        }
    }
}
