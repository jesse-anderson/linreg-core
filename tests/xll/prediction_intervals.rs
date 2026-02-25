//! Prediction Intervals XLL UDF tests.
//!
//! Tests the LINREG.PREDICTIONINTERVALS UDF.

use crate::xll::common::*;
use linreg_core::xll::{xl_linreg_predictionintervals, xlAutoFree12};

// ============================================================================
// Test Data
// ============================================================================

/// Simple linear data with noise for PI testing (10 obs, 1 predictor)
fn pi_data() -> (Vec<f64>, Vec<f64>) {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = vec![3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0, 17.2, 18.8, 21.1];
    (y, x)
}

/// Multi-predictor data for PI testing
fn pi_multi_data() -> (Vec<f64>, Vec<Vec<f64>>) {
    mtcars_subset() // 20 obs, 4 predictors
}

// ============================================================================
// Output Format Tests
// ============================================================================

#[test]
fn test_pi_returns_6col_table() {
    let (y_data, x_data) = pi_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);

    // Predict at 2 new points
    let new_x = vec![11.0, 12.0];
    let (new_x_oper, _new_x_cells) = build_column_range(&new_x);

    let result = XlResultGuard::new(xl_linreg_predictionintervals(
        &y_oper, &x_oper, &new_x_oper, std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 6, "PI should have 6 columns");
    // 1 header + 2 data + 2 summary = 5
    assert_eq!(rows, 5, "PI should have 5 rows for 2 new observations");

    // Check header labels
    assert_eq!(result.cell_string(0, 0), "Obs");
    assert_eq!(result.cell_string(0, 1), "Predicted");
    assert_eq!(result.cell_string(0, 2), "Lower");
    assert_eq!(result.cell_string(0, 3), "Upper");
    assert_eq!(result.cell_string(0, 4), "SE");
    assert_eq!(result.cell_string(0, 5), "Leverage");

    // Check summary labels
    assert_eq!(result.cell_string(3, 0), "Alpha");
    assert_eq!(result.cell_string(4, 0), "df Residuals");
}

#[test]
fn test_pi_values_are_reasonable() {
    let (y_data, x_data) = pi_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);

    let new_x = vec![5.5]; // near center of training data
    let (new_x_oper, _new_x_cells) = build_column_range(&new_x);

    let result = XlResultGuard::new(xl_linreg_predictionintervals(
        &y_oper, &x_oper, &new_x_oper, std::ptr::null(),
    ));
    assert!(result.is_multi());

    // Obs index = 1
    assert_eq!(result.cell_f64(1, 0), 1.0);

    // Predicted value should be roughly 2*5.5 + 1 ≈ 12
    let predicted = result.cell_f64(1, 1);
    assert!(predicted > 8.0 && predicted < 16.0, "Predicted {} out of range", predicted);

    // Lower < Predicted < Upper
    let lower = result.cell_f64(1, 2);
    let upper = result.cell_f64(1, 3);
    assert!(lower < predicted, "Lower {} should be < predicted {}", lower, predicted);
    assert!(upper > predicted, "Upper {} should be > predicted {}", upper, predicted);

    // SE > 0
    let se = result.cell_f64(1, 4);
    assert!(se > 0.0, "SE should be positive");

    // Leverage > 0
    let leverage = result.cell_f64(1, 5);
    assert!(leverage > 0.0, "Leverage should be positive");

    // Alpha defaults to 0.05
    let alpha = result.cell_f64(2, 1);
    assert!((alpha - 0.05).abs() < 1e-10);

    // df residuals = n - k - 1 = 10 - 1 - 1 = 8
    let df = result.cell_f64(3, 1);
    assert_eq!(df, 8.0);
}

#[test]
fn test_pi_custom_alpha() {
    let (y_data, x_data) = pi_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);

    let new_x = vec![6.0];
    let (new_x_oper, _new_x_cells) = build_column_range(&new_x);
    let alpha_oper = XLOPER12::from_f64(0.01); // 99% PI

    let result_99 = XlResultGuard::new(xl_linreg_predictionintervals(
        &y_oper, &x_oper, &new_x_oper, &alpha_oper,
    ));
    let alpha_oper_95 = XLOPER12::from_f64(0.05);
    let result_95 = XlResultGuard::new(xl_linreg_predictionintervals(
        &y_oper, &x_oper, &new_x_oper, &alpha_oper_95,
    ));

    // 99% PI should be wider than 95% PI
    let width_99 = result_99.cell_f64(1, 3) - result_99.cell_f64(1, 2);
    let width_95 = result_95.cell_f64(1, 3) - result_95.cell_f64(1, 2);
    assert!(width_99 > width_95, "99% PI ({}) should be wider than 95% PI ({})", width_99, width_95);

    // Alpha in summary should be 0.01
    assert!((result_99.cell_f64(2, 1) - 0.01).abs() < 1e-10);
}

#[test]
fn test_pi_multiple_new_points() {
    let (y_data, x_data) = pi_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);

    let new_x = vec![3.0, 5.0, 8.0, 12.0, 15.0];
    let (new_x_oper, _new_x_cells) = build_column_range(&new_x);

    let result = XlResultGuard::new(xl_linreg_predictionintervals(
        &y_oper, &x_oper, &new_x_oper, std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, _cols) = result.dimensions();
    // 1 header + 5 data + 2 summary = 8
    assert_eq!(rows, 8);

    // Obs indices should be 1..5
    for i in 0..5 {
        assert_eq!(result.cell_f64(i + 1, 0), (i + 1) as f64);
    }

    // Extrapolation points (12, 15) should have wider PI than center (5)
    let width_center = result.cell_f64(2, 3) - result.cell_f64(2, 2); // obs 2 = x=5
    let width_extrap = result.cell_f64(5, 3) - result.cell_f64(5, 2); // obs 5 = x=15
    assert!(width_extrap > width_center, "Extrapolation should have wider PI");
}

#[test]
fn test_pi_multi_predictor() {
    let (y_data, x_data) = pi_multi_data(); // 20 obs, 4 predictors
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_matrix_range(&x_data);

    // Predict at 2 new points with 4 predictors each
    let new_x_cols: Vec<Vec<f64>> = vec![
        vec![6.0, 4.0],    // cyl
        vec![200.0, 100.0], // disp
        vec![150.0, 80.0],  // hp
        vec![3.0, 2.5],     // wt
    ];
    let (new_x_oper, _new_x_cells) = build_matrix_range(&new_x_cols);

    let result = XlResultGuard::new(xl_linreg_predictionintervals(
        &y_oper, &x_oper, &new_x_oper, std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 6);
    assert_eq!(rows, 5); // 1 header + 2 data + 2 summary
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_pi_null_inputs() {
    let (y_data, x_data) = pi_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);
    let new_x = vec![6.0];
    let (new_x_oper, _new_x_cells) = build_column_range(&new_x);

    // null y
    let result = XlResultGuard::new(xl_linreg_predictionintervals(
        std::ptr::null(), &x_oper, &new_x_oper, std::ptr::null(),
    ));
    assert!(result.is_error());

    // null x
    let result = XlResultGuard::new(xl_linreg_predictionintervals(
        &y_oper, std::ptr::null(), &new_x_oper, std::ptr::null(),
    ));
    assert!(result.is_error());

    // null new_x
    let result = XlResultGuard::new(xl_linreg_predictionintervals(
        &y_oper, &x_oper, std::ptr::null(), std::ptr::null(),
    ));
    assert!(result.is_error());
}

#[test]
fn test_pi_dimension_mismatch_returns_error() {
    let (y_data, x_data) = pi_data(); // 1 predictor
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);

    // new_x has 2 predictor columns but training has 1
    let new_x_cols: Vec<Vec<f64>> = vec![
        vec![6.0],
        vec![7.0],
    ];
    let (new_x_oper, _new_x_cells) = build_matrix_range(&new_x_cols);

    let result = XlResultGuard::new(xl_linreg_predictionintervals(
        &y_oper, &x_oper, &new_x_oper, std::ptr::null(),
    ));
    // Should return #VALUE! because dimension mismatch is caught before calling prediction_intervals
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_VALUE);
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_stress_prediction_intervals() {
    let (y_data, x_data) = pi_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);
    let new_x = vec![5.5, 11.0, 15.0];
    let (new_x_oper, _new_x_cells) = build_column_range(&new_x);

    let baseline = MemSnapshot::now();
    let n_iters = 500;
    let log_interval = n_iters / 10;

    for i in 0..n_iters {
        let result = xl_linreg_predictionintervals(
            &y_oper, &x_oper, &new_x_oper, std::ptr::null(),
        );
        assert!(!result.is_null());
        xlAutoFree12(result);

        if (i + 1) % log_interval == 0 {
            log_mem("PI stress", i + 1, &baseline);
        }
    }
}
