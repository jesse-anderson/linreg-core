// XLL tests for LINREG.RIDGE(), LINREG.LASSO(), LINREG.ELASTICNET(), and
// LINREG.LAMBDAPATH() UDFs.

use super::common::*;

pub use linreg_core::xll::{xl_linreg_ridge as xll_ridge, xl_linreg_lasso as xll_lasso, xl_linreg_elasticnet as xll_elasticnet};
use linreg_core::xll::{xl_linreg_lambdapath, xlAutoFree12};

// ============================================================================
// Ridge — Basic Functionality
// ============================================================================

#[test]
fn test_ridge_returns_multi_array() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(1.0);

    let result = XlResultGuard::new(xll_ridge(&y_range, &x_range, &lambda, std::ptr::null()));
    assert!(result.is_multi(), "Ridge should return a multi-cell array");
}

#[test]
fn test_ridge_output_dimensions() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(1.0);

    let result = XlResultGuard::new(xll_ridge(&y_range, &x_range, &lambda, std::ptr::null()));
    let (rows, cols) = result.dimensions();

    // 1 header + 1 intercept + 1 slope + 1 blank + 5 summary = 9
    assert_eq!(rows, 9);
    assert_eq!(cols, 2, "Ridge uses 2-column layout (Term, Coefficient)");
}

#[test]
fn test_ridge_header_and_labels() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(0.5);

    let result = XlResultGuard::new(xll_ridge(&y_range, &x_range, &lambda, std::ptr::null()));

    assert_eq!(result.cell_string(0, 0), "Term");
    assert_eq!(result.cell_string(0, 1), "Coefficient");
    assert_eq!(result.cell_string(1, 0), "Intercept");
    assert_eq!(result.cell_string(2, 0), "X1");
    // Blank separator
    assert_eq!(result.cell_string(3, 0), "");
    // Summary
    assert_eq!(result.cell_string(4, 0), "R-squared");
    assert_eq!(result.cell_string(5, 0), "Adj R-squared");
    assert_eq!(result.cell_string(6, 0), "MSE");
    assert_eq!(result.cell_string(7, 0), "Lambda");
    assert_eq!(result.cell_string(8, 0), "Eff. df");
}

#[test]
fn test_ridge_lambda_value_in_output() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(2.5);

    let result = XlResultGuard::new(xll_ridge(&y_range, &x_range, &lambda, std::ptr::null()));

    // Lambda row is at index 7 (0-based)
    let lambda_out = result.cell_f64(7, 1);
    assert!((lambda_out - 2.5).abs() < 1e-9, "Lambda should be 2.5, got {}", lambda_out);
}

#[test]
fn test_ridge_summary_values_reasonable() {
    let (y_data, x_cols) = mtcars_subset();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&x_cols);
    let lambda = XLOPER12::from_f64(1.0);

    let result = XlResultGuard::new(xll_ridge(&y_range, &x_range, &lambda, std::ptr::null()));

    let n_coefs = x_cols.len(); // slopes only
    let summary_start = 1 + 1 + n_coefs + 1; // header + intercept + slopes + blank
    let r2 = result.cell_f64(summary_start, 1);
    let mse = result.cell_f64(summary_start + 2, 1);
    let eff_df = result.cell_f64(summary_start + 4, 1);

    assert!(r2 > 0.0 && r2 <= 1.0, "R² should be in (0, 1], got {}", r2);
    assert!(mse > 0.0, "MSE should be positive");
    assert!(eff_df > 0.0, "Effective df should be positive");
}

#[test]
fn test_ridge_standardize_default_true() {
    // Passing null for standardize should default to true
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(1.0);

    let result = XlResultGuard::new(xll_ridge(&y_range, &x_range, &lambda, std::ptr::null()));
    assert!(result.is_multi(), "Should succeed with default standardize");
}

#[test]
fn test_ridge_standardize_false() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(1.0);
    let std_false = XLOPER12::from_f64(0.0); // FALSE

    let result = XlResultGuard::new(xll_ridge(&y_range, &x_range, &lambda, &std_false));
    assert!(result.is_multi(), "Should succeed with standardize=false");
}

#[test]
fn test_ridge_multiple_predictors() {
    let (y_data, x_cols) = mtcars_subset();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&x_cols);
    let lambda = XLOPER12::from_f64(0.5);

    let result = XlResultGuard::new(xll_ridge(&y_range, &x_range, &lambda, std::ptr::null()));
    let (rows, cols) = result.dimensions();

    // 1 header + 1 intercept + 4 slopes + 1 blank + 5 summary = 12
    assert_eq!(rows, 12);
    assert_eq!(cols, 2);

    // Check variable names
    assert_eq!(result.cell_string(2, 0), "X1");
    assert_eq!(result.cell_string(3, 0), "X2");
    assert_eq!(result.cell_string(4, 0), "X3");
    assert_eq!(result.cell_string(5, 0), "X4");
}

// ============================================================================
// Lasso — Basic Functionality
// ============================================================================

#[test]
fn test_lasso_returns_multi_array() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(0.1);

    let result = XlResultGuard::new(xll_lasso(&y_range, &x_range, &lambda, std::ptr::null()));
    assert!(result.is_multi(), "Lasso should return a multi-cell array");
}

#[test]
fn test_lasso_output_dimensions() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(0.1);

    let result = XlResultGuard::new(xll_lasso(&y_range, &x_range, &lambda, std::ptr::null()));
    let (rows, cols) = result.dimensions();

    // 1 header + 1 intercept + 1 slope + 1 blank + 6 summary = 10
    assert_eq!(rows, 10);
    assert_eq!(cols, 2);
}

#[test]
fn test_lasso_summary_labels() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(0.1);

    let result = XlResultGuard::new(xll_lasso(&y_range, &x_range, &lambda, std::ptr::null()));

    // Summary starts after header(1) + intercept(1) + slopes(1) + blank(1) = row 4
    assert_eq!(result.cell_string(4, 0), "R-squared");
    assert_eq!(result.cell_string(5, 0), "Adj R-squared");
    assert_eq!(result.cell_string(6, 0), "MSE");
    assert_eq!(result.cell_string(7, 0), "Lambda");
    assert_eq!(result.cell_string(8, 0), "Non-zero");
    assert_eq!(result.cell_string(9, 0), "Converged");
}

#[test]
fn test_lasso_converged_is_string() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(0.1);

    let result = XlResultGuard::new(xll_lasso(&y_range, &x_range, &lambda, std::ptr::null()));

    let converged = result.cell_string(9, 1);
    assert!(
        converged == "Yes" || converged == "No",
        "Converged should be 'Yes' or 'No', got '{}'", converged
    );
}

#[test]
fn test_lasso_nonzero_count() {
    let (y_data, x_cols) = mtcars_subset();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&x_cols);
    let lambda = XLOPER12::from_f64(0.1);

    let result = XlResultGuard::new(xll_lasso(&y_range, &x_range, &lambda, std::ptr::null()));

    let n_coefs = x_cols.len();
    let nonzero_row = 1 + 1 + n_coefs + 1 + 4; // header + intercept + slopes + blank + 4 stats
    let nonzero = result.cell_f64(nonzero_row, 1);
    assert!(nonzero >= 0.0, "Non-zero count should be non-negative");
    assert!(nonzero <= x_cols.len() as f64, "Non-zero should be <= number of predictors");
}

// ============================================================================
// Elastic Net — Basic Functionality
// ============================================================================

#[test]
fn test_elasticnet_returns_multi_array() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(0.1);
    let alpha = XLOPER12::from_f64(0.5);

    let result = XlResultGuard::new(xll_elasticnet(&y_range, &x_range, &lambda, &alpha, std::ptr::null()));
    assert!(result.is_multi(), "Elastic Net should return a multi-cell array");
}

#[test]
fn test_elasticnet_output_dimensions() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(0.1);
    let alpha = XLOPER12::from_f64(0.5);

    let result = XlResultGuard::new(xll_elasticnet(&y_range, &x_range, &lambda, &alpha, std::ptr::null()));
    let (rows, cols) = result.dimensions();

    // 1 header + 1 intercept + 1 slope + 1 blank + 7 summary = 11
    assert_eq!(rows, 11);
    assert_eq!(cols, 2);
}

#[test]
fn test_elasticnet_summary_labels() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(0.1);
    let alpha = XLOPER12::from_f64(0.5);

    let result = XlResultGuard::new(xll_elasticnet(&y_range, &x_range, &lambda, &alpha, std::ptr::null()));

    assert_eq!(result.cell_string(4, 0), "R-squared");
    assert_eq!(result.cell_string(5, 0), "Adj R-squared");
    assert_eq!(result.cell_string(6, 0), "MSE");
    assert_eq!(result.cell_string(7, 0), "Lambda");
    assert_eq!(result.cell_string(8, 0), "Alpha");
    assert_eq!(result.cell_string(9, 0), "Non-zero");
    assert_eq!(result.cell_string(10, 0), "Converged");
}

#[test]
fn test_elasticnet_alpha_in_output() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(0.1);
    let alpha = XLOPER12::from_f64(0.7);

    let result = XlResultGuard::new(xll_elasticnet(&y_range, &x_range, &lambda, &alpha, std::ptr::null()));

    let alpha_out = result.cell_f64(8, 1);
    assert!((alpha_out - 0.7).abs() < 1e-9, "Alpha should be 0.7, got {}", alpha_out);
}

#[test]
fn test_elasticnet_alpha_zero_is_ridge() {
    // alpha=0 should behave like Ridge
    let (y_data, x_data) = simple_linear_data();

    let (y1, _y1c) = build_column_range(&y_data);
    let (x1, _x1c) = build_matrix_range(&[x_data.clone()]);
    let lambda = XLOPER12::from_f64(1.0);
    let alpha_zero = XLOPER12::from_f64(0.0);
    let en_result = XlResultGuard::new(xll_elasticnet(&y1, &x1, &lambda, &alpha_zero, std::ptr::null()));

    let (y2, _y2c) = build_column_range(&y_data);
    let (x2, _x2c) = build_matrix_range(&[x_data]);
    let ridge_result = XlResultGuard::new(xll_ridge(&y2, &x2, &lambda, std::ptr::null()));

    // Intercepts should match
    let en_intercept = en_result.cell_f64(1, 1);
    let ridge_intercept = ridge_result.cell_f64(1, 1);
    assert!(
        (en_intercept - ridge_intercept).abs() < 1e-6,
        "EN(alpha=0) intercept should match Ridge: {} vs {}",
        en_intercept, ridge_intercept
    );
}

// ============================================================================
// Error Cases (shared across all three)
// ============================================================================

#[test]
fn test_ridge_null_y_returns_error() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x) = build_column_range(&x_data);
    let lambda = XLOPER12::from_f64(1.0);

    let result = XlResultGuard::new(xll_ridge(std::ptr::null(), &x_range, &lambda, std::ptr::null()));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_VALUE);
}

#[test]
fn test_ridge_missing_lambda_returns_error() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda_missing = XLOPER12::missing();

    let result = XlResultGuard::new(xll_ridge(&y_range, &x_range, &lambda_missing, std::ptr::null()));
    assert!(result.is_error(), "Missing lambda should return error");
    assert_eq!(result.error_code(), XLERR_VALUE);
}

#[test]
fn test_lasso_null_x_returns_error() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y) = build_column_range(&y_data);
    let lambda = XLOPER12::from_f64(0.1);

    let result = XlResultGuard::new(xll_lasso(&y_range, std::ptr::null(), &lambda, std::ptr::null()));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_VALUE);
}

#[test]
fn test_elasticnet_missing_alpha_returns_error() {
    let (y_data, x_data) = simple_linear_data();
    let (y_range, _y) = build_column_range(&y_data);
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(0.1);
    let alpha_missing = XLOPER12::missing();

    let result = XlResultGuard::new(xll_elasticnet(&y_range, &x_range, &lambda, &alpha_missing, std::ptr::null()));
    assert!(result.is_error(), "Missing alpha should return error");
    assert_eq!(result.error_code(), XLERR_VALUE);
}

#[test]
fn test_ridge_error_in_y_propagates() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y) = build_column_with_error(&y_data, 0, XLERR_NUM);
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x) = build_matrix_range(&[x_data]);
    let lambda = XLOPER12::from_f64(1.0);

    let result = XlResultGuard::new(xll_ridge(&y_range, &x_range, &lambda, std::ptr::null()));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_NUM);
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_ridge_stress_with_mem_tracking() {
    let (y_data, x_data) = simple_linear_data();
    let baseline = MemSnapshot::now();
    eprintln!();
    log_mem("ridge", 0, &baseline);

    for i in 1..=500 {
        let (y_range, _y) = build_column_range(&y_data);
        let (x_range, _x) = build_matrix_range(&[x_data.clone()]);
        let lambda = XLOPER12::from_f64(1.0);

        let result = XlResultGuard::new(xll_ridge(&y_range, &x_range, &lambda, std::ptr::null()));
        assert!(result.is_multi(), "Iteration {} should succeed", i);

        if i % 50 == 0 {
            log_mem("ridge", i, &baseline);
        }
    }
}

#[test]
fn test_lasso_stress_with_mem_tracking() {
    let (y_data, x_data) = simple_linear_data();
    let baseline = MemSnapshot::now();
    eprintln!();
    log_mem("lasso", 0, &baseline);

    for i in 1..=500 {
        let (y_range, _y) = build_column_range(&y_data);
        let (x_range, _x) = build_matrix_range(&[x_data.clone()]);
        let lambda = XLOPER12::from_f64(0.1);

        let result = XlResultGuard::new(xll_lasso(&y_range, &x_range, &lambda, std::ptr::null()));
        assert!(result.is_multi(), "Iteration {} should succeed", i);

        if i % 50 == 0 {
            log_mem("lasso", i, &baseline);
        }
    }
}

#[test]
fn test_elasticnet_stress_with_mem_tracking() {
    let (y_data, x_data) = simple_linear_data();
    let baseline = MemSnapshot::now();
    eprintln!();
    log_mem("enet", 0, &baseline);

    for i in 1..=500 {
        let (y_range, _y) = build_column_range(&y_data);
        let (x_range, _x) = build_matrix_range(&[x_data.clone()]);
        let lambda = XLOPER12::from_f64(0.1);
        let alpha = XLOPER12::from_f64(0.5);

        let result = XlResultGuard::new(xll_elasticnet(&y_range, &x_range, &lambda, &alpha, std::ptr::null()));
        assert!(result.is_multi(), "Iteration {} should succeed", i);

        if i % 50 == 0 {
            log_mem("enet", i, &baseline);
        }
    }
}

// ============================================================================
// Lambda Path — Basic Functionality
// ============================================================================

#[test]
fn test_lambdapath_returns_column() {
    let (y_data, x_data) = mtcars_subset();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_matrix_range(&x_data);

    let result = XlResultGuard::new(xl_linreg_lambdapath(
        &y_oper, &x_oper, std::ptr::null(), std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 1, "Lambda path should be single column");
    // 1 header + 100 lambdas = 101
    assert_eq!(rows, 101);

    // Header
    assert_eq!(result.cell_string(0, 0), "Lambda");
}

#[test]
fn test_lambdapath_values_are_decreasing() {
    let (y_data, x_data) = mtcars_subset();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_matrix_range(&x_data);

    let result = XlResultGuard::new(xl_linreg_lambdapath(
        &y_oper, &x_oper, std::ptr::null(), std::ptr::null(),
    ));

    // Lambda values should be decreasing (largest to smallest)
    let first = result.cell_f64(1, 0);
    let last = result.cell_f64(100, 0);
    assert!(first > last, "Lambdas should be decreasing: first={}, last={}", first, last);
    assert!(first > 0.0, "First lambda should be positive");
    assert!(last > 0.0, "Last lambda should be positive");
}

#[test]
fn test_lambdapath_custom_count() {
    let (y_data, x_data) = mtcars_subset();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_matrix_range(&x_data);
    let nlambda = XLOPER12::from_f64(20.0);

    let result = XlResultGuard::new(xl_linreg_lambdapath(
        &y_oper, &x_oper, &nlambda, std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, _cols) = result.dimensions();
    // 1 header + 20 lambdas = 21
    assert_eq!(rows, 21);
}

#[test]
fn test_lambdapath_custom_alpha() {
    let (y_data, x_data) = mtcars_subset();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_matrix_range(&x_data);
    let nlambda = XLOPER12::from_f64(10.0);
    let alpha = XLOPER12::from_f64(0.5); // elastic net

    let result = XlResultGuard::new(xl_linreg_lambdapath(
        &y_oper, &x_oper, &nlambda, &alpha,
    ));
    assert!(result.is_multi());
    let (rows, _cols) = result.dimensions();
    assert_eq!(rows, 11); // 1 header + 10 lambdas

    // All values should be positive
    for i in 1..=10 {
        let v = result.cell_f64(i, 0);
        assert!(v > 0.0, "Lambda at row {} should be positive, got {}", i, v);
    }
}

#[test]
fn test_lambdapath_null_inputs() {
    let (y_data, x_data) = mtcars_subset();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_matrix_range(&x_data);

    let result = XlResultGuard::new(xl_linreg_lambdapath(
        std::ptr::null(), &x_oper, std::ptr::null(), std::ptr::null(),
    ));
    assert!(result.is_error());

    let result = XlResultGuard::new(xl_linreg_lambdapath(
        &y_oper, std::ptr::null(), std::ptr::null(), std::ptr::null(),
    ));
    assert!(result.is_error());
}

#[test]
fn test_stress_lambdapath() {
    let (y_data, x_data) = mtcars_subset();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_matrix_range(&x_data);
    let baseline = MemSnapshot::now();
    let n_iters = 500;
    let log_interval = n_iters / 10;

    for i in 0..n_iters {
        let result = xl_linreg_lambdapath(
            &y_oper, &x_oper, std::ptr::null(), std::ptr::null(),
        );
        assert!(!result.is_null());
        xlAutoFree12(result);

        if (i + 1) % log_interval == 0 {
            log_mem("LambdaPath stress", i + 1, &baseline)
        }
    }
}
