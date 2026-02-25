// XLL tests for LINREG.OLS() UDF.

use super::common::*;

// ============================================================================
// Basic Functionality
// ============================================================================

#[test]
fn test_ols_returns_multi_array() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
    assert!(result.is_multi(), "OLS should return a multi-cell array");
}

#[test]
fn test_ols_output_dimensions() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
    let (rows, cols) = result.dimensions();

    // 1 header + 2 coefficients (Intercept, X1) + 6 summary = 9 rows
    assert_eq!(rows, 9, "Should have 9 rows (1 header + 2 coefs + 6 summary)");
    assert_eq!(cols, 5, "Should have 5 columns (Term, Coefficient, Std Error, t Stat, p-Value)");
}

#[test]
fn test_ols_header_row() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    assert_eq!(result.cell_string(0, 0), "Term");
    assert_eq!(result.cell_string(0, 1), "Coefficient");
    assert_eq!(result.cell_string(0, 2), "Std Error");
    assert_eq!(result.cell_string(0, 3), "t Stat");
    assert_eq!(result.cell_string(0, 4), "p-Value");
}

#[test]
fn test_ols_coefficient_labels() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    assert_eq!(result.cell_string(1, 0), "Intercept");
    assert_eq!(result.cell_string(2, 0), "X1");
}

#[test]
fn test_ols_summary_labels() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    // Summary rows start after header (1) + coefficients (2) = row 3
    assert_eq!(result.cell_string(3, 0), "R-squared");
    assert_eq!(result.cell_string(4, 0), "Adj R-squared");
    assert_eq!(result.cell_string(5, 0), "F-statistic");
    assert_eq!(result.cell_string(6, 0), "F p-value");
    assert_eq!(result.cell_string(7, 0), "MSE");
    assert_eq!(result.cell_string(8, 0), "RMSE");
}

// ============================================================================
// Coefficient Values
// ============================================================================

#[test]
fn test_ols_simple_linear_coefficients() {
    // y = 2 + 3x -> intercept ≈ 2.0, slope ≈ 3.0
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    let intercept = result.cell_f64(1, 1);
    let slope = result.cell_f64(2, 1);

    assert!((intercept - 2.0).abs() < 1e-9, "Intercept should be ~2.0, got {}", intercept);
    assert!((slope - 3.0).abs() < 1e-9, "Slope should be ~3.0, got {}", slope);
}

#[test]
fn test_ols_r_squared_perfect_fit() {
    // Perfect linear data -> R² = 1.0
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    let r2 = result.cell_f64(3, 1); // R-squared row
    assert!((r2 - 1.0).abs() < 1e-9, "R² should be 1.0 for perfect linear data, got {}", r2);
}

#[test]
fn test_ols_std_errors_positive() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    let se_intercept = result.cell_f64(1, 2);
    let se_slope = result.cell_f64(2, 2);

    assert!(se_intercept >= 0.0, "Intercept SE should be non-negative, got {}", se_intercept);
    assert!(se_slope >= 0.0, "Slope SE should be non-negative, got {}", se_slope);
}

#[test]
fn test_ols_t_stat_equals_coef_over_se() {
    let (y_data, x_cols) = mtcars_subset();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&x_cols);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    // Check for each coefficient: t = coef / se
    let n_coefs = x_cols.len() + 1; // +1 for intercept
    for i in 0..n_coefs {
        let row = i + 1; // skip header
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
fn test_ols_p_values_in_range() {
    let (y_data, x_cols) = mtcars_subset();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&x_cols);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    let n_coefs = x_cols.len() + 1;
    for i in 0..n_coefs {
        let p = result.cell_f64(i + 1, 4);
        assert!(p >= 0.0 && p <= 1.0, "p-value at row {} should be in [0, 1], got {}", i + 1, p);
    }
}

// ============================================================================
// Multiple Regression
// ============================================================================

#[test]
fn test_ols_multiple_regression_dimensions() {
    let (y_data, x_cols) = mtcars_subset();
    let n_predictors = x_cols.len();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&x_cols);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
    let (rows, cols) = result.dimensions();

    // 1 header + (n_predictors + 1) coefficients + 6 summary
    let expected_rows = 1 + (n_predictors + 1) + 6;
    assert_eq!(rows, expected_rows, "Multiple regression should have {} rows", expected_rows);
    assert_eq!(cols, 5);
}

#[test]
fn test_ols_multiple_regression_variable_names() {
    let (y_data, x_cols) = mtcars_subset();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&x_cols);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    assert_eq!(result.cell_string(1, 0), "Intercept");
    assert_eq!(result.cell_string(2, 0), "X1");
    assert_eq!(result.cell_string(3, 0), "X2");
    assert_eq!(result.cell_string(4, 0), "X3");
    assert_eq!(result.cell_string(5, 0), "X4");
}

#[test]
fn test_ols_multiple_regression_r_squared_reasonable() {
    let (y_data, x_cols) = mtcars_subset();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&x_cols);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    let r2 = result.cell_f64(6, 1); // row 6 = R-squared (after 1 header + 5 coefs)
    let adj_r2 = result.cell_f64(7, 1);

    assert!(r2 > 0.5 && r2 <= 1.0, "R² should be reasonable, got {}", r2);
    assert!(adj_r2 <= r2 + 1e-9, "Adj R² should be <= R², got {} vs {}", adj_r2, r2);
}

#[test]
fn test_ols_summary_values_reasonable() {
    let (y_data, x_cols) = mtcars_subset();
    let n_coefs = x_cols.len() + 1;
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&x_cols);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    let summary_start = 1 + n_coefs; // after header + coefficient rows
    let f_stat = result.cell_f64(summary_start + 2, 1);
    let f_pval = result.cell_f64(summary_start + 3, 1);
    let mse = result.cell_f64(summary_start + 4, 1);
    let rmse = result.cell_f64(summary_start + 5, 1);

    assert!(f_stat > 0.0, "F-statistic should be positive, got {}", f_stat);
    assert!(f_pval >= 0.0 && f_pval <= 1.0, "F p-value should be in [0, 1], got {}", f_pval);
    assert!(mse > 0.0, "MSE should be positive, got {}", mse);
    assert!(rmse > 0.0, "RMSE should be positive, got {}", rmse);
    assert!((rmse - mse.sqrt()).abs() < 1e-9, "RMSE should equal sqrt(MSE)");
}

// ============================================================================
// Single-Cell Inputs
// ============================================================================

#[test]
fn test_ols_single_value_inputs() {
    // A single numeric cell arrives as xltypeNum, not a 1x1 multi
    let y_scalar = XLOPER12::from_f64(5.0);
    let x_scalar = XLOPER12::from_f64(2.0);

    let result = XlResultGuard::new(xll_ols(&y_scalar, &x_scalar));

    // Single observation with 1 predictor — too few observations for OLS
    // (n=1, p=1 means df=0), so this should either error or produce degenerate output
    // The important thing is it doesn't crash
    assert!(
        result.is_error() || result.is_multi(),
        "Single value input should return error or array, got type {}",
        result.base_type()
    );
}

// ============================================================================
// Summary Row Blank Cells
// ============================================================================

#[test]
fn test_ols_summary_blank_cells_are_empty_strings() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    // Summary rows: cols 2, 3, 4 should be empty strings (not nil/0)
    // R-squared row is at index 3 (after header + 2 coefs)
    for summary_row in 3..9 {
        for col in 2..5 {
            let cell = result.cell(summary_row, col);
            let base = cell.base_type();
            assert_eq!(
                base,
                linreg_core::xll::types::XLTYPE_STR,
                "Summary row {} col {} should be a string (empty), got type {}",
                summary_row, col, base
            );
            let s = cell.as_string().unwrap();
            assert!(s.is_empty(), "Summary row {} col {} should be empty string, got '{}'", summary_row, col, s);
        }
    }
}
