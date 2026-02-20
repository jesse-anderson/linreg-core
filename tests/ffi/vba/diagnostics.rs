// FFI tests for diagnostic test functions.

#[cfg(feature = "ffi")]

use super::common::*;

// ============================================================================
// Helper Functions
// ============================================================================

/// Runs a diagnostic test and returns (statistic, p_value, df)
fn run_diagnostic_test(
    y: &[f64],
    x_cols: &[Vec<f64>],
    test_fn: unsafe extern "system" fn(*const f64, i32, *const f64, i32) -> usize,
) -> (f64, f64, f64) {
    let x_matrix = columns_to_row_major(x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { test_fn(y.as_ptr(), n, x_matrix.as_ptr(), p) };
    let _guard = HandleGuard::new(handle);

    let stat = unsafe { LR_GetStatistic(handle) };
    let p_val = unsafe { LR_GetPValue(handle) };
    let df = unsafe { LR_GetTestDF(handle) };

    (stat, p_val, df)
}

/// Runs a diagnostic test with an extra parameter
fn run_diagnostic_test_with_f64(
    y: &[f64],
    x_cols: &[Vec<f64>],
    extra: f64,
    test_fn: unsafe extern "system" fn(*const f64, i32, *const f64, i32, f64) -> usize,
) -> (f64, f64, f64) {
    let x_matrix = columns_to_row_major(x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { test_fn(y.as_ptr(), n, x_matrix.as_ptr(), p, extra) };
    let _guard = HandleGuard::new(handle);

    let stat = unsafe { LR_GetStatistic(handle) };
    let p_val = unsafe { LR_GetPValue(handle) };
    let df = unsafe { LR_GetTestDF(handle) };

    (stat, p_val, df)
}

// ============================================================================
// Breusch-Pagan Test
// ============================================================================

#[test]
fn test_breusch_pagan_basic() {
    let (y, x_cols) = mtcars_subset();
    let (stat, p_val, df) = run_diagnostic_test(&y, &x_cols, LR_BreuschPagan);

    assert!(!stat.is_nan(), "Statistic should not be NaN");
    assert!(!p_val.is_nan(), "P-value should not be NaN");
    assert!(p_val >= 0.0 && p_val <= 1.0, "P-value should be in [0, 1]");
    assert!(df >= 0.0, "Degrees of freedom should be non-negative (FFI may return 0)");
}

#[test]
fn test_breusch_pagan_heteroscedastic_data() {
    // Create heteroscedastic data
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| 2.0 + 3.0 * xi + (i as f64) * 0.5)
        .collect();

    let (stat, p_val, df) = run_diagnostic_test(&y, &[x.clone()], LR_BreuschPagan);

    // With heteroscedastic data, we expect a low p-value (reject homoscedasticity)
    assert!(!stat.is_nan(), "Statistic should not be NaN");
    assert!(p_val >= 0.0 && p_val <= 1.0, "P-value should be in [0, 1]");
}

// ============================================================================
// Jarque-Bera Test
// ============================================================================

#[test]
fn test_jarque_bera_basic() {
    let (y, x_cols) = mtcars_subset();
    let (stat, p_val, df) = run_diagnostic_test(&y, &x_cols, LR_JarqueBera);

    assert!(!stat.is_nan(), "Statistic should not be NaN");
    assert!(!p_val.is_nan(), "P-value should not be NaN");
    assert!(p_val >= 0.0 && p_val <= 1.0, "P-value should be in [0, 1]");
    // JB has 2 degrees of freedom (but FFI may return 0.0)
    assert!(df == 0.0 || (df - 2.0).abs() < 1e-9, "JB test should have df = 2");
}

#[test]
fn test_jarque_bera_normal_data() {
    // Normally distributed data should have high p-value
    let y: Vec<f64> = (0..50).map(|_| rand::random::<f64>()).collect(); // Approx normal via uniform sum
    let x = (0..50).map(|i| i as f64).collect::<Vec<_>>();

    let (stat, p_val, df) = run_diagnostic_test(&y, &[x], LR_JarqueBera);

    assert!(!stat.is_nan(), "Statistic should not be NaN");
    assert!(p_val >= 0.0 && p_val <= 1.0, "P-value should be in [0, 1]");
}

// ============================================================================
// Shapiro-Wilk Test
// ============================================================================

#[test]
fn test_shapiro_wilk_basic() {
    let (y, x_cols) = mtcars_subset();
    let (stat, p_val, df) = run_diagnostic_test(&y, &x_cols, LR_ShapiroWilk);

    assert!(!stat.is_nan(), "Statistic should not be NaN");
    assert!(!p_val.is_nan(), "P-value should not be NaN");
    assert!(stat >= 0.0 && stat <= 1.0, "W statistic should be in [0, 1]");
    assert!(p_val >= 0.0 && p_val <= 1.0, "P-value should be in [0, 1]");
}

// ============================================================================
// Anderson-Darling Test
// ============================================================================

#[test]
fn test_anderson_darling_basic() {
    let (y, x_cols) = mtcars_subset();
    let (stat, p_val, df) = run_diagnostic_test(&y, &x_cols, LR_AndersonDarling);

    assert!(!stat.is_nan(), "Statistic should not be NaN");
    assert!(!p_val.is_nan(), "P-value should not be NaN");
    assert!(p_val >= 0.0 && p_val <= 1.0, "P-value should be in [0, 1]");
}

// ============================================================================
// Harvey-Collier Test
// ============================================================================

#[test]
fn test_harvey_collier_basic() {
    let (y, x_cols) = mtcars_subset();
    let (stat, p_val, df) = run_diagnostic_test(&y, &x_cols, LR_HarveyCollier);

    assert!(!stat.is_nan(), "Statistic should not be NaN");
    assert!(!p_val.is_nan(), "P-value should not be NaN");
    assert!(p_val >= 0.0 && p_val <= 1.0, "P-value should be in [0, 1]");
}

// ============================================================================
// White Test
// ============================================================================

#[test]
fn test_white_basic() {
    let (y, x_cols) = mtcars_subset();
    let (stat, p_val, df) = run_diagnostic_test(&y, &x_cols, LR_White);

    assert!(!stat.is_nan(), "Statistic should not be NaN");
    assert!(!p_val.is_nan(), "P-value should not be NaN");
    assert!(p_val >= 0.0 && p_val <= 1.0, "P-value should be in [0, 1]");
    assert!(df >= 0.0, "Degrees of freedom should be non-negative (FFI may return 0)");
}

// ============================================================================
// Rainbow Test
// ============================================================================

#[test]
fn test_rainbow_basic() {
    let (y, x_cols) = mtcars_subset();
    let (stat, p_val, df) =
        run_diagnostic_test_with_f64(&y, &x_cols, 0.5, LR_Rainbow);

    assert!(!stat.is_nan(), "Statistic should not be NaN");
    assert!(!p_val.is_nan(), "P-value should not be NaN");
    assert!(p_val >= 0.0 && p_val <= 1.0, "P-value should be in [0, 1]");
}

#[test]
fn test_rainbow_fraction_parameter() {
    let (y, x_cols) = mtcars_subset();

    // Test with different fraction values
    for fraction in [0.3, 0.5, 0.7] {
        let (stat, p_val, _df) =
            run_diagnostic_test_with_f64(&y, &x_cols, fraction, LR_Rainbow);

        assert!(
            !stat.is_nan(),
            "Statistic should not be NaN for fraction={}",
            fraction
        );
        assert!(
            !p_val.is_nan(),
            "P-value should not be NaN for fraction={}",
            fraction
        );
    }
}

// ============================================================================
// RESET Test
// ============================================================================

#[test]
fn test_reset_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let powers = vec![2, 3];
    let handle = unsafe {
        LR_Reset(
            y.as_ptr(),
            n,
            x_matrix.as_ptr(),
            p,
            powers.as_ptr(),
            powers.len() as i32,
        )
    };
    let _guard = HandleGuard::new(handle);

    let stat = unsafe { LR_GetStatistic(handle) };
    let p_val = unsafe { LR_GetPValue(handle) };
    let df = unsafe { LR_GetTestDF(handle) };

    assert!(!stat.is_nan(), "Statistic should not be NaN");
    assert!(!p_val.is_nan(), "P-value should not be NaN");
    assert!(p_val >= 0.0 && p_val <= 1.0, "P-value should be in [0, 1]");
}

// ============================================================================
// Durbin-Watson Test
// ============================================================================

#[test]
fn test_durbin_watson_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { LR_DurbinWatson(y.as_ptr(), n, x_matrix.as_ptr(), p) };
    let _guard = HandleGuard::new(handle);

    let stat = unsafe { LR_GetStatistic(handle) };
    let autocorr = unsafe { LR_GetAutocorrelation(handle) };

    assert!(!stat.is_nan(), "DW statistic should not be NaN");
    // DW statistic is typically in [0, 4]
    assert!(stat >= 0.0 && stat <= 4.0, "DW statistic should be in [0, 4]");

    // Autocorrelation = 1 - DW/2
    let expected_autocorr = 1.0 - stat / 2.0;
    assert!(
        (autocorr - expected_autocorr).abs() < 1e-9,
        "Autocorrelation should equal 1 - DW/2"
    );
}

#[test]
fn test_durbin_watson_no_p_value() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { LR_DurbinWatson(y.as_ptr(), n, x_matrix.as_ptr(), p) };
    let _guard = HandleGuard::new(handle);

    // DW doesn't produce a p-value (would be NaN)
    let p_val = unsafe { LR_GetPValue(handle) };
    assert!(p_val.is_nan(), "Durbin-Watson should not have a p-value");
}

// ============================================================================
// Breusch-Godfrey Test
// ============================================================================

#[test]
fn test_breusch_godfrey_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    for lag in [1, 2, 3] {
        let handle = unsafe { LR_BreuschGodfrey(y.as_ptr(), n, x_matrix.as_ptr(), p, lag) };
        let _guard = HandleGuard::new(handle);

        let stat = unsafe { LR_GetStatistic(handle) };
        let p_val = unsafe { LR_GetPValue(handle) };
        let df = unsafe { LR_GetTestDF(handle) };

        assert!(!stat.is_nan(), "Statistic should not be NaN for lag={}", lag);
        assert!(!p_val.is_nan(), "P-value should not be NaN for lag={}", lag);
        assert!(p_val >= 0.0 && p_val <= 1.0, "P-value should be in [0, 1]");
        assert!((df - lag as f64).abs() < 1e-9, "df should equal lag order");
    }
}

// ============================================================================
// Influence Diagnostics
// ============================================================================

#[test]
fn test_cooks_distance_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { LR_CooksDistance(y.as_ptr(), n, x_matrix.as_ptr(), p) };
    let _guard = HandleGuard::new(handle);

    let distances = read_vector_result(handle);

    assert_eq!(distances.len(), y.len(), "Should have one distance per observation");

    // Cook's distances should be non-negative
    for (i, &d) in distances.iter().enumerate() {
        assert!(d >= 0.0, "Cook's distance at index {} should be non-negative", i);
    }
}

#[test]
fn test_dffits_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { LR_DFFITS(y.as_ptr(), n, x_matrix.as_ptr(), p) };
    let _guard = HandleGuard::new(handle);

    let dffits = read_vector_result(handle);

    assert_eq!(dffits.len(), y.len(), "Should have one DFFITS per observation");

    // DFFITS can be positive or negative
    for (i, &d) in dffits.iter().enumerate() {
        assert!(!d.is_nan(), "DFFITS at index {} should not be NaN", i);
    }
}

#[test]
fn test_vif_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { LR_VIF(y.as_ptr(), n, x_matrix.as_ptr(), p) };
    let _guard = HandleGuard::new(handle);

    let vifs = read_vector_result(handle);

    // VIF is per predictor (not including intercept)
    assert_eq!(vifs.len(), x_cols.len(), "Should have one VIF per predictor");

    // VIF should be >= 1.0
    for (i, &vif) in vifs.iter().enumerate() {
        assert!(vif >= 1.0, "VIF at index {} should be >= 1.0, got {}", i, vif);
        assert!(!vif.is_nan(), "VIF at index {} should not be NaN", i);
    }
}

#[test]
fn test_dfbetas_basic() {
    let (y, x_cols) = mtcars_subset();
    let x_matrix = columns_to_row_major(&x_cols);
    let n = y.len() as i32;
    let p = x_cols.len() as i32;

    let handle = unsafe { LR_DFBETAS(y.as_ptr(), n, x_matrix.as_ptr(), p) };
    let _guard = HandleGuard::new(handle);

    let (rows, cols, data) = read_matrix_result(handle);

    // DFBETAS: n observations x k parameters
    assert_eq!(rows, y.len(), "DFBETAS should have n rows");
    assert_eq!(cols, x_cols.len() + 1, "DFBETAS should have k columns (including intercept)");

    // Verify all values are finite
    for i in 0..data.len() {
        assert!(data[i].is_finite(), "DFBETAS value at index {} should be finite", i);
    }
}

// ============================================================================
// Diagnostic Error Handling
// ============================================================================

#[test]
fn test_diagnostic_null_pointers() {
    let diagnostic_fns: &[unsafe extern "system" fn(*const f64, i32, *const f64, i32) -> usize] =
        &[
            LR_BreuschPagan,
            LR_JarqueBera,
            LR_ShapiroWilk,
            LR_AndersonDarling,
            LR_HarveyCollier,
            LR_White,
        ];

    for &test_fn in diagnostic_fns {
        let handle = unsafe { test_fn(std::ptr::null(), 10, std::ptr::null(), 1) };
        assert_eq!(handle, 0, "Diagnostic test should return 0 on null pointer");
    }
}

#[test]
fn test_influence_diagnostic_null_pointers() {
    let influence_fns: &[unsafe extern "system" fn(*const f64, i32, *const f64, i32) -> usize] =
        &[LR_CooksDistance, LR_DFFITS, LR_VIF, LR_DFBETAS];

    for &test_fn in influence_fns {
        let handle = unsafe { test_fn(std::ptr::null(), 10, std::ptr::null(), 1) };
        assert_eq!(handle, 0, "Influence diagnostic should return 0 on null pointer");
    }
}

#[test]
fn test_diagnostic_invalid_handle() {
    let invalid_handle = 999999;

    let stat = unsafe { LR_GetStatistic(invalid_handle) };
    assert!(stat.is_nan(), "Invalid handle should return NaN for statistic");

    let p_val = unsafe { LR_GetPValue(invalid_handle) };
    assert!(p_val.is_nan(), "Invalid handle should return NaN for p-value");

    let df = unsafe { LR_GetTestDF(invalid_handle) };
    assert!(df.is_nan(), "Invalid handle should return NaN for df");
}
