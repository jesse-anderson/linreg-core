//! Polynomial regression XLL UDF tests.
//!
//! Tests LINREG.POLYNOMIAL UDF. Lambda path tests are in regularized.rs.

use crate::xll::common::*;
use linreg_core::xll::{
    xl_linreg_polynomial, xlAutoFree12,
};

// ============================================================================
// Test Data
// ============================================================================

/// Quadratic data: y = 1 + 2x + x^2
fn quadratic_data() -> (Vec<f64>, Vec<f64>) {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();
    (y, x)
}

/// Noisy polynomial data for realistic tests
fn noisy_poly_data() -> (Vec<f64>, Vec<f64>) {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = vec![4.1, 9.3, 16.8, 25.2, 36.1, 49.5, 63.8, 81.2, 100.9, 121.0];
    (y, x)
}

// ============================================================================
// POLYNOMIAL Output Format Tests
// ============================================================================

#[test]
fn test_polynomial_returns_ols_table() {
    let (y_data, x_data) = quadratic_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xl_linreg_polynomial(
        &y_oper, &x_oper, std::ptr::null(), std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 5, "Polynomial output should have 5 columns (OLS format)");

    // Header row
    assert_eq!(result.cell_string(0, 0), "Term");
    assert_eq!(result.cell_string(0, 1), "Coefficient");
    assert_eq!(result.cell_string(0, 2), "Std Error");
    assert_eq!(result.cell_string(0, 3), "t Stat");
    assert_eq!(result.cell_string(0, 4), "p-Value");

    // Should have: header + 3 coefs (Intercept, x, x^2) + 6 summary = 10
    assert_eq!(rows, 10);
}

#[test]
fn test_polynomial_term_names() {
    let (y_data, x_data) = quadratic_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xl_linreg_polynomial(
        &y_oper, &x_oper, std::ptr::null(), std::ptr::null(),
    ));

    // Polynomial term names from the core
    assert_eq!(result.cell_string(1, 0), "Intercept");
    // The polynomial module names features as "x", "x^2", etc.
    let x_name = result.cell_string(2, 0);
    let x2_name = result.cell_string(3, 0);
    assert!(x_name.contains("x") || x_name.contains("X"), "x term should contain 'x', got '{}'", x_name);
    assert!(x2_name.contains("2"), "x^2 term should contain '2', got '{}'", x2_name);
}

#[test]
fn test_polynomial_perfect_fit_quadratic() {
    let (y_data, x_data) = quadratic_data(); // y = 1 + 2x + x^2 (exact)
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xl_linreg_polynomial(
        &y_oper, &x_oper, std::ptr::null(), std::ptr::null(),
    ));

    // R² should be ~1.0 for perfect fit
    // Summary starts at row 4 (after 3 coefficient rows + header)
    let r2_label = result.cell_string(4, 0);
    assert_eq!(r2_label, "R-squared");
    let r2 = result.cell_f64(4, 1);
    assert!((r2 - 1.0).abs() < 1e-6, "R² should be ~1.0 for perfect quadratic fit, got {}", r2);
}

#[test]
fn test_polynomial_custom_degree() {
    let (y_data, x_data) = noisy_poly_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);
    let degree = XLOPER12::from_f64(3.0);

    let result = XlResultGuard::new(xl_linreg_polynomial(
        &y_oper, &x_oper, &degree, std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 5);
    // header + 4 coefs (Intercept, x, x^2, x^3) + 6 summary = 11
    assert_eq!(rows, 11);
}

#[test]
fn test_polynomial_with_centering() {
    let (y_data, x_data) = noisy_poly_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);
    let center = XLOPER12::from_f64(1.0); // true

    let result = XlResultGuard::new(xl_linreg_polynomial(
        &y_oper, &x_oper, std::ptr::null(), &center,
    ));
    assert!(result.is_multi());

    // R² should still be valid
    let r2 = result.cell_f64(4, 1);
    assert!(r2 > 0.9, "R² should be high for polynomial data, got {}", r2);
}

#[test]
fn test_polynomial_degree_1_is_simple_linear() {
    let (y_data, x_data) = noisy_poly_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);
    let degree = XLOPER12::from_f64(1.0);

    let result = XlResultGuard::new(xl_linreg_polynomial(
        &y_oper, &x_oper, &degree, std::ptr::null(),
    ));
    assert!(result.is_multi());
    let (rows, _cols) = result.dimensions();
    // header + 2 coefs (Intercept, x) + 6 summary = 9
    assert_eq!(rows, 9);
}

// ============================================================================
// POLYNOMIAL Error Cases
// ============================================================================

#[test]
fn test_polynomial_null_inputs() {
    let (y_data, x_data) = quadratic_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xl_linreg_polynomial(
        std::ptr::null(), &x_oper, std::ptr::null(), std::ptr::null(),
    ));
    assert!(result.is_error());

    let result = XlResultGuard::new(xl_linreg_polynomial(
        &y_oper, std::ptr::null(), std::ptr::null(), std::ptr::null(),
    ));
    assert!(result.is_error());
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_stress_polynomial() {
    let (y_data, x_data) = noisy_poly_data();
    let (y_oper, _y_cells) = build_column_range(&y_data);
    let (x_oper, _x_cells) = build_column_range(&x_data);
    let baseline = MemSnapshot::now();
    let n_iters = 500;
    let log_interval = n_iters / 10;

    for i in 0..n_iters {
        let result = xl_linreg_polynomial(
            &y_oper, &x_oper, std::ptr::null(), std::ptr::null(),
        );
        assert!(!result.is_null());
        xlAutoFree12(result);

        if (i + 1) % log_interval == 0 {
            log_mem("Polynomial stress", i + 1, &baseline);
        }
    }
}
