// XLL tests for LINREG.WLS() UDF.

use super::common::*;

pub use linreg_core::xll::xl_linreg_wls as xll_wls;

// ============================================================================
// Basic Functionality
// ============================================================================

#[test]
fn test_wls_returns_multi_array() {
    let (y_data, x_data) = simple_linear_data();
    let weights: Vec<f64> = vec![1.0; y_data.len()];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);
    let (w_range, _w_cells) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));
    assert!(result.is_multi(), "WLS should return a multi-cell array");
}

#[test]
fn test_wls_output_dimensions() {
    let (y_data, x_data) = simple_linear_data();
    let weights: Vec<f64> = vec![1.0; y_data.len()];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);
    let (w_range, _w_cells) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));
    let (rows, cols) = result.dimensions();

    // 1 header + 2 coefficients (Intercept, X1) + 7 summary = 10 rows
    assert_eq!(rows, 10, "Should have 10 rows (1 header + 2 coefs + 7 summary)");
    assert_eq!(cols, 5, "Should have 5 columns");
}

#[test]
fn test_wls_header_row() {
    let (y_data, x_data) = simple_linear_data();
    let weights: Vec<f64> = vec![1.0; y_data.len()];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);
    let (w_range, _w_cells) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));

    assert_eq!(result.cell_string(0, 0), "Term");
    assert_eq!(result.cell_string(0, 1), "Coefficient");
    assert_eq!(result.cell_string(0, 2), "Std Error");
    assert_eq!(result.cell_string(0, 3), "t Stat");
    assert_eq!(result.cell_string(0, 4), "p-Value");
}

#[test]
fn test_wls_coefficient_labels() {
    let (y_data, x_data) = simple_linear_data();
    let weights: Vec<f64> = vec![1.0; y_data.len()];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);
    let (w_range, _w_cells) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));

    assert_eq!(result.cell_string(1, 0), "Intercept");
    assert_eq!(result.cell_string(2, 0), "X1");
}

#[test]
fn test_wls_summary_labels() {
    let (y_data, x_data) = simple_linear_data();
    let weights: Vec<f64> = vec![1.0; y_data.len()];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);
    let (w_range, _w_cells) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));

    // Summary rows start after header (1) + coefficients (2) = row 3
    assert_eq!(result.cell_string(3, 0), "R-squared");
    assert_eq!(result.cell_string(4, 0), "Adj R-squared");
    assert_eq!(result.cell_string(5, 0), "F-statistic");
    assert_eq!(result.cell_string(6, 0), "F p-value");
    assert_eq!(result.cell_string(7, 0), "Resid Std Err");
    assert_eq!(result.cell_string(8, 0), "MSE");
    assert_eq!(result.cell_string(9, 0), "RMSE");
}

// ============================================================================
// Uniform Weights = OLS
// ============================================================================

#[test]
fn test_wls_uniform_weights_matches_ols() {
    // WLS with all weights = 1 should produce identical results to OLS
    let (y_data, x_data) = simple_linear_data();
    let weights: Vec<f64> = vec![1.0; y_data.len()];

    let (y_range_ols, _y1) = build_column_range(&y_data);
    let (x_range_ols, _x1) = build_matrix_range(&[x_data.clone()]);
    let ols_result = XlResultGuard::new(xll_ols(&y_range_ols, &x_range_ols));

    let (y_range_wls, _y2) = build_column_range(&y_data);
    let (x_range_wls, _x2) = build_matrix_range(&[x_data]);
    let (w_range, _w) = build_column_range(&weights);
    let wls_result = XlResultGuard::new(xll_wls(&y_range_wls, &x_range_wls, &w_range));

    // Compare coefficients
    let ols_intercept = ols_result.cell_f64(1, 1);
    let wls_intercept = wls_result.cell_f64(1, 1);
    assert!(
        (ols_intercept - wls_intercept).abs() < 1e-9,
        "Intercept should match: OLS={}, WLS={}",
        ols_intercept, wls_intercept
    );

    let ols_slope = ols_result.cell_f64(2, 1);
    let wls_slope = wls_result.cell_f64(2, 1);
    assert!(
        (ols_slope - wls_slope).abs() < 1e-9,
        "Slope should match: OLS={}, WLS={}",
        ols_slope, wls_slope
    );

    // Compare R²
    let ols_r2 = ols_result.cell_f64(3, 1);
    let wls_r2 = wls_result.cell_f64(3, 1);
    assert!(
        (ols_r2 - wls_r2).abs() < 1e-9,
        "R² should match: OLS={}, WLS={}",
        ols_r2, wls_r2
    );
}

// ============================================================================
// Non-Uniform Weights
// ============================================================================

#[test]
fn test_wls_nonuniform_weights() {
    let (y_data, x_data) = simple_linear_data();
    // Give higher weight to later observations
    let weights: Vec<f64> = (1..=y_data.len()).map(|i| i as f64).collect();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let (w_range, _w) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));
    assert!(result.is_multi(), "WLS with non-uniform weights should succeed");

    let r2 = result.cell_f64(3, 1);
    assert!(r2 > 0.99, "R² should be high for linear data, got {}", r2);
}

#[test]
fn test_wls_multiple_regression() {
    let (y_data, x_cols) = mtcars_subset();
    let weights: Vec<f64> = vec![1.0; y_data.len()];
    let n_predictors = x_cols.len();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&x_cols);
    let (w_range, _w) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));
    let (rows, cols) = result.dimensions();

    // 1 header + (n_predictors + 1) coefs + 7 summary
    let expected_rows = 1 + (n_predictors + 1) + 7;
    assert_eq!(rows, expected_rows);
    assert_eq!(cols, 5);

    // Variable names
    assert_eq!(result.cell_string(1, 0), "Intercept");
    assert_eq!(result.cell_string(2, 0), "X1");
    assert_eq!(result.cell_string(3, 0), "X2");
    assert_eq!(result.cell_string(4, 0), "X3");
    assert_eq!(result.cell_string(5, 0), "X4");
}

#[test]
fn test_wls_t_stat_equals_coef_over_se() {
    let (y_data, x_data) = simple_linear_data();
    let weights: Vec<f64> = (1..=y_data.len()).map(|i| i as f64).collect();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let (w_range, _w) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));

    for row in 1..=2 {
        let coef = result.cell_f64(row, 1);
        let se = result.cell_f64(row, 2);
        let t_stat = result.cell_f64(row, 3);

        if se.abs() > 1e-15 {
            let computed_t = coef / se;
            assert!(
                (computed_t - t_stat).abs() < 1e-6,
                "t-stat should equal coef/se for row {} (coef={}, se={}, t={})",
                row, coef, se, t_stat
            );
        }
    }
}

#[test]
fn test_wls_p_values_in_range() {
    let (y_data, x_cols) = mtcars_subset();
    let weights: Vec<f64> = vec![1.0; y_data.len()];
    let n_coefs = x_cols.len() + 1;
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&x_cols);
    let (w_range, _w) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));

    for i in 0..n_coefs {
        let p = result.cell_f64(i + 1, 4);
        assert!(p >= 0.0 && p <= 1.0, "p-value at row {} should be in [0, 1], got {}", i + 1, p);
    }
}

#[test]
fn test_wls_summary_values_reasonable() {
    let (y_data, x_cols) = mtcars_subset();
    let weights: Vec<f64> = vec![1.0; y_data.len()];
    let n_coefs = x_cols.len() + 1;
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&x_cols);
    let (w_range, _w) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));

    let summary_start = 1 + n_coefs;
    let r2 = result.cell_f64(summary_start, 1);
    let adj_r2 = result.cell_f64(summary_start + 1, 1);
    let f_stat = result.cell_f64(summary_start + 2, 1);
    let f_pval = result.cell_f64(summary_start + 3, 1);
    let resid_se = result.cell_f64(summary_start + 4, 1);
    let mse = result.cell_f64(summary_start + 5, 1);
    let rmse = result.cell_f64(summary_start + 6, 1);

    assert!(r2 > 0.5 && r2 <= 1.0, "R² should be reasonable, got {}", r2);
    assert!(adj_r2 <= r2 + 1e-9, "Adj R² should be <= R²");
    assert!(f_stat > 0.0, "F-statistic should be positive");
    assert!(f_pval >= 0.0 && f_pval <= 1.0, "F p-value should be in [0, 1]");
    assert!(resid_se > 0.0, "Residual std error should be positive");
    assert!(mse > 0.0, "MSE should be positive");
    assert!(rmse > 0.0, "RMSE should be positive");
    assert!((rmse - mse.sqrt()).abs() < 1e-9, "RMSE should equal sqrt(MSE)");
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_wls_null_y_returns_error() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weights = vec![1.0; 5];
    let (x_range, _x) = build_column_range(&x_data);
    let (w_range, _w) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(std::ptr::null(), &x_range, &w_range));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_VALUE);
}

#[test]
fn test_wls_null_x_returns_error() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weights = vec![1.0; 5];
    let (y_range, _y) = build_column_range(&y_data);
    let (w_range, _w) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, std::ptr::null(), &w_range));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_VALUE);
}

#[test]
fn test_wls_null_weights_returns_error() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, std::ptr::null()));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_VALUE);
}

#[test]
fn test_wls_missing_weights_returns_error() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let w_missing = XLOPER12::missing();

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_missing));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_VALUE);
}

#[test]
fn test_wls_error_in_weights_propagates() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let (w_range, _w) = build_column_with_error(&weights, 2, XLERR_NUM);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_NUM);
}

#[test]
fn test_wls_nil_cell_in_weights_returns_error() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let (w_range, _w) = build_column_with_nil(&weights, 1);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_VALUE);
}

// ============================================================================
// Summary Row Blank Cells
// ============================================================================

#[test]
fn test_wls_summary_blank_cells_are_empty_strings() {
    let (y_data, x_data) = simple_linear_data();
    let weights: Vec<f64> = vec![1.0; y_data.len()];
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let (w_range, _w) = build_column_range(&weights);

    let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));

    // Summary rows: cols 2, 3, 4 should be empty strings
    // Summary starts at row 3 (after header + 2 coefs), goes to row 9
    for summary_row in 3..10 {
        for col in 2..5 {
            let cell = result.cell(summary_row, col);
            assert_eq!(
                cell.base_type(), XLTYPE_STR,
                "Summary row {} col {} should be a string", summary_row, col
            );
            let s = cell.as_string().unwrap();
            assert!(s.is_empty(), "Summary row {} col {} should be empty", summary_row, col);
        }
    }
}

// ============================================================================
// Stress Test
// ============================================================================

#[test]
fn test_wls_stress_with_mem_tracking() {
    let (y_data, x_data) = simple_linear_data();
    let weights: Vec<f64> = (1..=y_data.len()).map(|i| i as f64).collect();
    let baseline = MemSnapshot::now();
    eprintln!();
    log_mem("wls", 0, &baseline);

    for i in 1..=500 {
        let (y_range, _y) = build_column_range(&y_data);
        let (x_range, _x) = build_matrix_range(&[x_data.clone()]);
        let (w_range, _w) = build_column_range(&weights);

        let result = XlResultGuard::new(xll_wls(&y_range, &x_range, &w_range));
        assert!(result.is_multi(), "Iteration {} should succeed", i);

        if i % 50 == 0 {
            log_mem("wls", i, &baseline);
        }
    }
}
