// XLL tests for diagnostic UDFs.

use super::common::*;

use linreg_core::xll::{
    xl_linreg_breuschpagan, xl_linreg_white, xl_linreg_jarquebera,
    xl_linreg_shapirowilk, xl_linreg_andersondarling, xl_linreg_harveycollier,
    xl_linreg_rainbow, xl_linreg_reset, xl_linreg_durbinwatson,
    xl_linreg_breuschgodfrey, xl_linreg_vif, xl_linreg_cooksdistance,
    xl_linreg_dffits, xl_linreg_dfbetas,
};

// ============================================================================
// Test Data
// ============================================================================

/// Heteroscedastic data for heteroscedasticity tests.
fn heteroscedastic_data() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().enumerate().map(|(i, &xi)| {
        // Variance increases with x
        let noise = if i % 2 == 0 { xi * 0.3 } else { -xi * 0.3 };
        2.0 + 1.5 * xi + noise
    }).collect();
    (y, x)
}

/// Non-normal residuals data.
fn nonnormal_data() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| {
        // Add nonlinear component to create non-normal residuals
        1.0 + 2.0 * xi + 0.1 * xi * xi
    }).collect();
    (y, x)
}

/// Multi-predictor data for VIF/multicollinearity tests.
fn multi_predictor_data() -> (Vec<f64>, Vec<Vec<f64>>) {
    mtcars_subset()
}

// ============================================================================
// Helper: build y+X ranges from data
// ============================================================================

struct TestRanges {
    y_range: XLOPER12,
    x_range: XLOPER12,
    _y_cells: Vec<XLOPER12>,
    _x_cells: Vec<XLOPER12>,
}

fn build_test_ranges(y: &[f64], x_cols: &[Vec<f64>]) -> TestRanges {
    let (y_range, _y_cells) = build_column_range(y);
    let (x_range, _x_cells) = build_matrix_range(x_cols);
    TestRanges { y_range, x_range, _y_cells, _x_cells }
}

fn build_single_predictor_ranges(y: &[f64], x: &[f64]) -> TestRanges {
    build_test_ranges(y, &[x.to_vec()])
}

// ============================================================================
// Breusch-Pagan
// ============================================================================

#[test]
fn test_breuschpagan_returns_2x2() {
    let (y, x) = heteroscedastic_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_breuschpagan(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (2, 2));
}

#[test]
fn test_breuschpagan_labels() {
    let (y, x) = heteroscedastic_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_breuschpagan(&r.y_range, &r.x_range));
    assert_eq!(result.cell_string(0, 0), "Statistic");
    assert_eq!(result.cell_string(1, 0), "p-Value");
}

#[test]
fn test_breuschpagan_values_are_finite() {
    let (y, x) = heteroscedastic_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_breuschpagan(&r.y_range, &r.x_range));
    let stat = result.cell_f64(0, 1);
    let pval = result.cell_f64(1, 1);
    assert!(stat.is_finite() && stat >= 0.0, "Statistic should be non-negative finite");
    assert!(pval.is_finite() && pval >= 0.0 && pval <= 1.0, "p-value should be in [0,1]");
}

#[test]
fn test_breuschpagan_multi_predictor() {
    let (y, x_vars) = multi_predictor_data();
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_breuschpagan(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (2, 2));
    let stat = result.cell_f64(0, 1);
    assert!(stat.is_finite() && stat >= 0.0);
}

// ============================================================================
// White Test
// ============================================================================

#[test]
fn test_white_returns_2x2() {
    let (y, x) = heteroscedastic_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_white(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (2, 2));
    assert_eq!(result.cell_string(0, 0), "Statistic");
    assert_eq!(result.cell_string(1, 0), "p-Value");
}

#[test]
fn test_white_values_are_finite() {
    let (y, x) = heteroscedastic_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_white(&r.y_range, &r.x_range));
    let stat = result.cell_f64(0, 1);
    let pval = result.cell_f64(1, 1);
    assert!(stat.is_finite() && stat >= 0.0);
    assert!(pval.is_finite() && pval >= 0.0 && pval <= 1.0);
}

// ============================================================================
// Jarque-Bera
// ============================================================================

#[test]
fn test_jarquebera_returns_2x2() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_jarquebera(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (2, 2));
    assert_eq!(result.cell_string(0, 0), "Statistic");
    assert_eq!(result.cell_string(1, 0), "p-Value");
}

#[test]
fn test_jarquebera_values_are_finite() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_jarquebera(&r.y_range, &r.x_range));
    let stat = result.cell_f64(0, 1);
    let pval = result.cell_f64(1, 1);
    assert!(stat.is_finite() && stat >= 0.0);
    assert!(pval.is_finite() && pval >= 0.0 && pval <= 1.0);
}

// ============================================================================
// Shapiro-Wilk
// ============================================================================

#[test]
fn test_shapirowilk_returns_2x2() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_shapirowilk(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (2, 2));
    assert_eq!(result.cell_string(0, 0), "Statistic");
    assert_eq!(result.cell_string(1, 0), "p-Value");
}

#[test]
fn test_shapirowilk_statistic_in_0_1() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_shapirowilk(&r.y_range, &r.x_range));
    let stat = result.cell_f64(0, 1);
    assert!(stat > 0.0 && stat <= 1.0, "Shapiro-Wilk statistic should be in (0, 1], got {}", stat);
}

// ============================================================================
// Anderson-Darling
// ============================================================================

#[test]
fn test_andersondarling_returns_2x2() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_andersondarling(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (2, 2));
    assert_eq!(result.cell_string(0, 0), "Statistic");
    assert_eq!(result.cell_string(1, 0), "p-Value");
}

#[test]
fn test_andersondarling_values_are_finite() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_andersondarling(&r.y_range, &r.x_range));
    let stat = result.cell_f64(0, 1);
    let pval = result.cell_f64(1, 1);
    assert!(stat.is_finite() && stat >= 0.0);
    assert!(pval.is_finite() && pval >= 0.0 && pval <= 1.0);
}

// ============================================================================
// Harvey-Collier
// ============================================================================

#[test]
fn test_harveycollier_returns_2x2() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_harveycollier(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (2, 2));
    assert_eq!(result.cell_string(0, 0), "Statistic");
    assert_eq!(result.cell_string(1, 0), "p-Value");
}

#[test]
fn test_harveycollier_values_are_finite() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_harveycollier(&r.y_range, &r.x_range));
    let stat = result.cell_f64(0, 1);
    let pval = result.cell_f64(1, 1);
    assert!(stat.is_finite());
    assert!(pval.is_finite() && pval >= 0.0 && pval <= 1.0);
}

// ============================================================================
// Rainbow
// ============================================================================

#[test]
fn test_rainbow_default_fraction() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    // Missing fraction arg -> default 0.5
    let missing = XLOPER12 { val: XLOPER12Val { w: 0 }, xltype: XLTYPE_MISSING };
    let result = XlResultGuard::new(xl_linreg_rainbow(&r.y_range, &r.x_range, &missing));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (2, 2));
    assert_eq!(result.cell_string(0, 0), "Statistic");
}

#[test]
fn test_rainbow_explicit_fraction() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    let frac = XLOPER12::from_f64(0.6);
    let result = XlResultGuard::new(xl_linreg_rainbow(&r.y_range, &r.x_range, &frac));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (2, 2));
    let stat = result.cell_f64(0, 1);
    let pval = result.cell_f64(1, 1);
    assert!(stat.is_finite() && stat >= 0.0);
    assert!(pval.is_finite() && pval >= 0.0 && pval <= 1.0);
}

// ============================================================================
// RESET
// ============================================================================

#[test]
fn test_reset_returns_2x2() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_reset(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (2, 2));
    assert_eq!(result.cell_string(0, 0), "Statistic");
    assert_eq!(result.cell_string(1, 0), "p-Value");
}

#[test]
fn test_reset_values_are_finite() {
    let (y, x) = nonnormal_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_reset(&r.y_range, &r.x_range));
    let stat = result.cell_f64(0, 1);
    let pval = result.cell_f64(1, 1);
    assert!(stat.is_finite() && stat >= 0.0);
    assert!(pval.is_finite() && pval >= 0.0 && pval <= 1.0);
}

// ============================================================================
// Durbin-Watson
// ============================================================================

#[test]
fn test_durbinwatson_returns_2x2() {
    let (y, x) = heteroscedastic_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_durbinwatson(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (2, 2));
    assert_eq!(result.cell_string(0, 0), "DW Statistic");
    assert_eq!(result.cell_string(1, 0), "Autocorrelation");
}

#[test]
fn test_durbinwatson_statistic_range() {
    let (y, x) = heteroscedastic_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_durbinwatson(&r.y_range, &r.x_range));
    let dw = result.cell_f64(0, 1);
    let rho = result.cell_f64(1, 1);
    assert!(dw >= 0.0 && dw <= 4.0, "DW statistic should be in [0, 4], got {}", dw);
    assert!(rho >= -1.0 && rho <= 1.0, "Autocorrelation should be in [-1, 1], got {}", rho);
}

// ============================================================================
// Breusch-Godfrey
// ============================================================================

#[test]
fn test_breuschgodfrey_default_lag() {
    let (y, x) = heteroscedastic_data();
    let r = build_single_predictor_ranges(&y, &x);
    let missing = XLOPER12 { val: XLOPER12Val { w: 0 }, xltype: XLTYPE_MISSING };
    let result = XlResultGuard::new(xl_linreg_breuschgodfrey(&r.y_range, &r.x_range, &missing));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (3, 2));
    assert_eq!(result.cell_string(0, 0), "Statistic");
    assert_eq!(result.cell_string(1, 0), "p-Value");
    assert_eq!(result.cell_string(2, 0), "df");
}

#[test]
fn test_breuschgodfrey_explicit_lag() {
    let (y, x) = heteroscedastic_data();
    let r = build_single_predictor_ranges(&y, &x);
    let lag = XLOPER12::from_f64(2.0);
    let result = XlResultGuard::new(xl_linreg_breuschgodfrey(&r.y_range, &r.x_range, &lag));
    assert!(result.is_multi());
    assert_eq!(result.dimensions(), (3, 2));
    let stat = result.cell_f64(0, 1);
    let pval = result.cell_f64(1, 1);
    let df = result.cell_f64(2, 1);
    assert!(stat.is_finite() && stat >= 0.0);
    assert!(pval.is_finite() && pval >= 0.0 && pval <= 1.0);
    assert!(df >= 1.0, "df should be >= 1 for lag order 2");
}

// ============================================================================
// VIF
// ============================================================================

#[test]
fn test_vif_returns_labeled_array() {
    let (y, x_vars) = multi_predictor_data();
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_vif(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 2, "VIF should be 2 columns (Variable, VIF)");
    assert_eq!(rows, 1 + x_vars.len(), "VIF should have header + 1 row per predictor");
    assert_eq!(result.cell_string(0, 0), "Variable");
    assert_eq!(result.cell_string(0, 1), "VIF");
}

#[test]
fn test_vif_labels_are_x1_x2_etc() {
    let (y, x_vars) = multi_predictor_data();
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_vif(&r.y_range, &r.x_range));
    for i in 0..x_vars.len() {
        assert_eq!(result.cell_string(i + 1, 0), format!("X{}", i + 1));
    }
}

#[test]
fn test_vif_values_are_positive() {
    let (y, x_vars) = multi_predictor_data();
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_vif(&r.y_range, &r.x_range));
    for i in 0..x_vars.len() {
        let vif = result.cell_f64(i + 1, 1);
        assert!(vif >= 1.0, "VIF should be >= 1.0, got {} for X{}", vif, i + 1);
    }
}

// ============================================================================
// Cook's Distance
// ============================================================================

#[test]
fn test_cooksdistance_returns_column() {
    let (y, x_vars) = multi_predictor_data();
    let n = y.len();
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_cooksdistance(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 1, "Cook's distance should be 1 column");
    assert_eq!(rows, 1 + n, "Should have header + n values");
    assert_eq!(result.cell_string(0, 0), "Cook's D");
}

#[test]
fn test_cooksdistance_values_nonnegative() {
    let (y, x_vars) = multi_predictor_data();
    let n = y.len();
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_cooksdistance(&r.y_range, &r.x_range));
    for i in 0..n {
        let d = result.cell_f64(i + 1, 0);
        assert!(d.is_finite() && d >= 0.0, "Cook's D should be non-negative, got {} for obs {}", d, i + 1);
    }
}

// ============================================================================
// DFFITS
// ============================================================================

#[test]
fn test_dffits_returns_column() {
    let (y, x_vars) = multi_predictor_data();
    let n = y.len();
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_dffits(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(cols, 1, "DFFITS should be 1 column");
    assert_eq!(rows, 1 + n, "Should have header + n values");
    assert_eq!(result.cell_string(0, 0), "DFFITS");
}

#[test]
fn test_dffits_values_are_finite() {
    let (y, x_vars) = multi_predictor_data();
    let n = y.len();
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_dffits(&r.y_range, &r.x_range));
    for i in 0..n {
        let d = result.cell_f64(i + 1, 0);
        assert!(d.is_finite(), "DFFITS should be finite, got {} for obs {}", d, i + 1);
    }
}

// ============================================================================
// DFBETAS
// ============================================================================

#[test]
fn test_dfbetas_returns_matrix() {
    let (y, x_vars) = multi_predictor_data();
    let n = y.len();
    let p = x_vars.len() + 1; // intercept + predictors
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_dfbetas(&r.y_range, &r.x_range));
    assert!(result.is_multi());
    let (rows, cols) = result.dimensions();
    assert_eq!(rows, 1 + n, "Should have header + n observation rows");
    assert_eq!(cols, 1 + p, "Should have Obs col + p coefficient columns");
}

#[test]
fn test_dfbetas_header_row() {
    let (y, x_vars) = multi_predictor_data();
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_dfbetas(&r.y_range, &r.x_range));
    assert_eq!(result.cell_string(0, 0), "Obs");
    assert_eq!(result.cell_string(0, 1), "Intercept");
    for j in 1..x_vars.len() + 1 {
        assert_eq!(result.cell_string(0, j + 1), format!("X{}", j));
    }
}

#[test]
fn test_dfbetas_obs_column_is_1_based() {
    let (y, x_vars) = multi_predictor_data();
    let n = y.len();
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_dfbetas(&r.y_range, &r.x_range));
    for i in 0..n {
        let obs = result.cell_f64(i + 1, 0);
        assert_eq!(obs, (i + 1) as f64, "Obs column should be 1-based");
    }
}

#[test]
fn test_dfbetas_values_are_finite() {
    let (y, x_vars) = multi_predictor_data();
    let n = y.len();
    let p = x_vars.len() + 1;
    let r = build_test_ranges(&y, &x_vars);
    let result = XlResultGuard::new(xl_linreg_dfbetas(&r.y_range, &r.x_range));
    for i in 0..n {
        for j in 0..p {
            let v = result.cell_f64(i + 1, j + 1);
            assert!(v.is_finite(), "DFBETAS[{},{}] should be finite, got {}", i, j, v);
        }
    }
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_diagnostic_null_y_returns_error() {
    let (_, x) = simple_linear_data();
    let (x_range, _x_cells) = build_matrix_range(&[x]);
    let result = XlResultGuard::new(xl_linreg_breuschpagan(std::ptr::null(), &x_range));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_VALUE);
}

#[test]
fn test_diagnostic_null_x_returns_error() {
    let (y, _) = simple_linear_data();
    let (y_range, _y_cells) = build_column_range(&y);
    let result = XlResultGuard::new(xl_linreg_breuschpagan(&y_range, std::ptr::null()));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_VALUE);
}

#[test]
fn test_vif_single_predictor_returns_error_array() {
    // VIF requires >= 2 predictors — returns error message in array
    let (y, x) = simple_linear_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_vif(&r.y_range, &r.x_range));
    assert!(result.is_multi(), "VIF error should return a multi array with error message");
    let stat_val = result.cell_string(0, 1);
    assert!(stat_val.starts_with("#ERR:"), "Error message should start with #ERR:, got: {}", stat_val);
}

#[test]
fn test_diagnostic_nil_in_y_returns_error() {
    // Nil cell in input -> input parsing error -> still returns XLOPER12 error
    // (parse_yx! uses return_xl_error for input errors, not build_diagnostic_error)
    let (y, x) = simple_linear_data();
    let (y_range, _y_cells) = build_column_with_nil(&y, 3);
    let (x_range, _x_cells) = build_matrix_range(&[x]);
    let result = XlResultGuard::new(xl_linreg_jarquebera(&y_range, &x_range));
    assert!(result.is_error());
}

#[test]
fn test_diagnostic_error_cell_propagates() {
    // Error cell in input -> input parsing error -> still returns XLOPER12 error
    let (y, x) = simple_linear_data();
    let (y_range, _y_cells) = build_column_with_error(&y, 2, XLERR_NUM);
    let (x_range, _x_cells) = build_matrix_range(&[x]);
    let result = XlResultGuard::new(xl_linreg_shapirowilk(&y_range, &x_range));
    assert!(result.is_error());
    assert_eq!(result.error_code(), XLERR_NUM);
}

#[test]
fn test_durbinwatson_perfect_fit_returns_error_array() {
    // Perfect fit data -> zero residuals -> DW returns error message array
    let (y, x) = simple_linear_data();
    let r = build_single_predictor_ranges(&y, &x);
    let result = XlResultGuard::new(xl_linreg_durbinwatson(&r.y_range, &r.x_range));
    assert!(result.is_multi(), "DW error should return a multi array");
    let stat_val = result.cell_string(0, 1);
    assert!(stat_val.starts_with("#ERR:"), "Error message should start with #ERR:, got: {}", stat_val);
}

// ============================================================================
// Stress Tests with Memory Tracking
// ============================================================================

const STRESS_ITERATIONS: usize = 500;
const LOG_INTERVAL: usize = STRESS_ITERATIONS / 10;

#[test]
fn test_stress_simple_diagnostics() {
    let (y, x_vars) = multi_predictor_data();
    let r = build_test_ranges(&y, &x_vars);
    let baseline = MemSnapshot::now();

    for i in 0..STRESS_ITERATIONS {
        // Cycle through 6 simple diagnostics
        let result = match i % 6 {
            0 => XlResultGuard::new(xl_linreg_breuschpagan(&r.y_range, &r.x_range)),
            1 => XlResultGuard::new(xl_linreg_jarquebera(&r.y_range, &r.x_range)),
            2 => XlResultGuard::new(xl_linreg_shapirowilk(&r.y_range, &r.x_range)),
            3 => XlResultGuard::new(xl_linreg_andersondarling(&r.y_range, &r.x_range)),
            4 => XlResultGuard::new(xl_linreg_harveycollier(&r.y_range, &r.x_range)),
            _ => XlResultGuard::new(xl_linreg_reset(&r.y_range, &r.x_range)),
        };
        assert!(result.is_multi());
        if (i + 1) % LOG_INTERVAL == 0 {
            log_mem("diag-simple", i + 1, &baseline);
        }
    }
}

#[test]
fn test_stress_influence_diagnostics() {
    let (y, x_vars) = multi_predictor_data();
    let r = build_test_ranges(&y, &x_vars);
    let baseline = MemSnapshot::now();

    for i in 0..STRESS_ITERATIONS {
        let result = match i % 4 {
            0 => XlResultGuard::new(xl_linreg_cooksdistance(&r.y_range, &r.x_range)),
            1 => XlResultGuard::new(xl_linreg_dffits(&r.y_range, &r.x_range)),
            2 => XlResultGuard::new(xl_linreg_vif(&r.y_range, &r.x_range)),
            _ => XlResultGuard::new(xl_linreg_dfbetas(&r.y_range, &r.x_range)),
        };
        assert!(result.is_multi());
        if (i + 1) % LOG_INTERVAL == 0 {
            log_mem("diag-influence", i + 1, &baseline);
        }
    }
}

#[test]
fn test_stress_all_diagnostics() {
    let (y, x_vars) = multi_predictor_data();
    let r = build_test_ranges(&y, &x_vars);
    let missing = XLOPER12 { val: XLOPER12Val { w: 0 }, xltype: XLTYPE_MISSING };
    let baseline = MemSnapshot::now();

    for i in 0..STRESS_ITERATIONS {
        // Cycle through all 14 diagnostic UDFs
        let result = match i % 14 {
            0 => XlResultGuard::new(xl_linreg_breuschpagan(&r.y_range, &r.x_range)),
            1 => XlResultGuard::new(xl_linreg_white(&r.y_range, &r.x_range)),
            2 => XlResultGuard::new(xl_linreg_jarquebera(&r.y_range, &r.x_range)),
            3 => XlResultGuard::new(xl_linreg_shapirowilk(&r.y_range, &r.x_range)),
            4 => XlResultGuard::new(xl_linreg_andersondarling(&r.y_range, &r.x_range)),
            5 => XlResultGuard::new(xl_linreg_harveycollier(&r.y_range, &r.x_range)),
            6 => XlResultGuard::new(xl_linreg_rainbow(&r.y_range, &r.x_range, &missing)),
            7 => XlResultGuard::new(xl_linreg_reset(&r.y_range, &r.x_range)),
            8 => XlResultGuard::new(xl_linreg_durbinwatson(&r.y_range, &r.x_range)),
            9 => XlResultGuard::new(xl_linreg_breuschgodfrey(&r.y_range, &r.x_range, &missing)),
            10 => XlResultGuard::new(xl_linreg_vif(&r.y_range, &r.x_range)),
            11 => XlResultGuard::new(xl_linreg_cooksdistance(&r.y_range, &r.x_range)),
            12 => XlResultGuard::new(xl_linreg_dffits(&r.y_range, &r.x_range)),
            _ => XlResultGuard::new(xl_linreg_dfbetas(&r.y_range, &r.x_range)),
        };
        assert!(result.is_multi());
        if (i + 1) % LOG_INTERVAL == 0 {
            log_mem("diag-all", i + 1, &baseline);
        }
    }
}
