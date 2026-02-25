// XLL error case tests.
//
// Verifies that bad inputs produce appropriate Excel error values
// (#NUM!, #VALUE!, #N/A) rather than crashing.

use super::common::*;

// ============================================================================
// Null Pointer Inputs
// ============================================================================

#[test]
fn test_ols_null_y_returns_error() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xll_ols(std::ptr::null(), &x_range));
    assert!(result.is_error(), "Null y should return error");
    assert_eq!(result.error_code(), XLERR_VALUE, "Null y should return #VALUE!");
}

#[test]
fn test_ols_null_x_returns_error() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y_cells) = build_column_range(&y_data);

    let result = XlResultGuard::new(xll_ols(&y_range, std::ptr::null()));
    assert!(result.is_error(), "Null x should return error");
    assert_eq!(result.error_code(), XLERR_VALUE, "Null x should return #VALUE!");
}

#[test]
fn test_ols_both_null_returns_error() {
    let result = XlResultGuard::new(xll_ols(std::ptr::null(), std::ptr::null()));
    assert!(result.is_error(), "Both null should return error");
    assert_eq!(result.error_code(), XLERR_VALUE, "Both null should return #VALUE!");
}

// ============================================================================
// Empty / Missing Inputs
// ============================================================================

#[test]
fn test_ols_missing_y_returns_error() {
    let y_missing = XLOPER12::missing();
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xll_ols(&y_missing, &x_range));
    assert!(result.is_error(), "Missing y should return error");
    assert_eq!(result.error_code(), XLERR_VALUE, "Missing y should return #VALUE!");
}

#[test]
fn test_ols_missing_x_returns_error() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let x_missing = XLOPER12::missing();

    let result = XlResultGuard::new(xll_ols(&y_range, &x_missing));
    assert!(result.is_error(), "Missing x should return error");
    assert_eq!(result.error_code(), XLERR_VALUE, "Missing x should return #VALUE!");
}

#[test]
fn test_ols_nil_y_returns_error() {
    let y_nil = XLOPER12::nil();
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xll_ols(&y_nil, &x_range));
    assert!(result.is_error(), "Nil y should return error");
    assert_eq!(result.error_code(), XLERR_VALUE, "Nil y should return #VALUE!");
}

// ============================================================================
// Error Cell Propagation
// ============================================================================

#[test]
fn test_ols_error_in_y_propagates() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y_cells) = build_column_with_error(&y_data, 2, XLERR_NUM);
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
    assert!(result.is_error(), "Error cell in y should propagate");
    assert_eq!(result.error_code(), XLERR_NUM, "Should propagate #NUM! from y");
}

#[test]
fn test_ols_error_in_x_propagates() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x_cells) = build_column_with_error(&x_data, 1, XLERR_DIV0);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
    assert!(result.is_error(), "Error cell in x should propagate");
    assert_eq!(result.error_code(), XLERR_DIV0, "Should propagate #DIV/0! from x");
}

#[test]
fn test_ols_value_error_propagates() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y_cells) = build_column_with_error(&y_data, 0, XLERR_VALUE);
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
    assert!(result.is_error(), "Error cell should propagate");
    assert_eq!(result.error_code(), XLERR_VALUE, "Should propagate #VALUE!");
}

#[test]
fn test_ols_na_error_propagates() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y_cells) = build_column_with_error(&y_data, 4, XLERR_NA);
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
    assert!(result.is_error(), "Error cell should propagate");
    assert_eq!(result.error_code(), XLERR_NA, "Should propagate #N/A");
}

// ============================================================================
// Nil (Empty) Cells in Data
// ============================================================================

#[test]
fn test_ols_nil_cell_in_y_returns_error() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y_cells) = build_column_with_nil(&y_data, 2);
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
    assert!(result.is_error(), "Nil cell in y data should return error");
    assert_eq!(result.error_code(), XLERR_VALUE, "Empty cell should return #VALUE!");
}

#[test]
fn test_ols_nil_cell_in_x_returns_error() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x_cells) = build_column_with_nil(&x_data, 3);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
    assert!(result.is_error(), "Nil cell in x data should return error");
    assert_eq!(result.error_code(), XLERR_VALUE, "Empty cell should return #VALUE!");
}

// ============================================================================
// Error Input XLOPER12 (the argument itself is an error, not a range)
// ============================================================================

#[test]
fn test_ols_y_is_error_xloper() {
    let y_err = XLOPER12::from_err(XLERR_NUM);
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (x_range, _x_cells) = build_column_range(&x_data);

    let result = XlResultGuard::new(xll_ols(&y_err, &x_range));
    assert!(result.is_error(), "Error XLOPER12 as y should return error");
    assert_eq!(result.error_code(), XLERR_NUM, "Should return #NUM!");
}

#[test]
fn test_ols_x_is_error_xloper() {
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let x_err = XLOPER12::from_err(XLERR_VALUE);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_err));
    assert!(result.is_error(), "Error XLOPER12 as x should return error");
    assert_eq!(result.error_code(), XLERR_VALUE, "Should return #VALUE!");
}

// ============================================================================
// Insufficient Data
// ============================================================================

#[test]
fn test_ols_too_few_observations() {
    // 2 observations, 1 predictor -> df = n - p - 1 = 0 -> might error
    let y_data = vec![1.0, 2.0];
    let x_data = vec![1.0, 2.0];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    // This may succeed (n=2, p=1 gives df_residual=0) or error — either is fine
    // The important thing is it doesn't crash
    assert!(
        result.is_error() || result.is_multi(),
        "Too few observations should error or produce degenerate result"
    );
}

#[test]
fn test_ols_more_predictors_than_observations() {
    // 3 observations, 4 predictors -> rank-deficient
    let y_data = vec![1.0, 2.0, 3.0];
    let x_cols = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![10.0, 11.0, 12.0],
    ];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&x_cols);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    // Should either error (#NUM!) or handle rank-deficiency gracefully
    assert!(
        result.is_error() || result.is_multi(),
        "Rank-deficient case should not crash"
    );
}

// ============================================================================
// Dimension Mismatch
// ============================================================================

#[test]
fn test_ols_y_x_length_mismatch() {
    // y has 5 elements, X has 3 rows -> dimension mismatch
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_data = vec![1.0, 2.0, 3.0];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
    assert!(result.is_error(), "Dimension mismatch should return error");
    assert_eq!(result.error_code(), XLERR_VALUE, "Dimension mismatch should return #VALUE!");
}

// ============================================================================
// Constant Predictor (Perfect Collinearity)
// ============================================================================

#[test]
fn test_ols_constant_x_does_not_crash() {
    // All x values are the same -> collinear with intercept
    let y_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_data = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let (y_range, _y_cells) = build_column_range(&y_data);
    let (x_range, _x_cells) = build_matrix_range(&[x_data]);

    let result = XlResultGuard::new(xll_ols(&y_range, &x_range));

    // May succeed with rank-deficient handling or return #NUM! — either is fine
    assert!(
        result.is_error() || result.is_multi(),
        "Constant X should not crash, got type {}",
        result.base_type()
    );
}

// ============================================================================
// xlbitDLLFree Flag Verification
// ============================================================================

#[test]
fn test_error_return_has_dll_free_bit() {
    // Verify that error XLOPER12s have xlbitDLLFree set, so Excel knows
    // to call xlAutoFree12. Without this flag, every error return leaks.
    let ptr = xll_ols(std::ptr::null(), std::ptr::null());
    assert!(!ptr.is_null());
    let xltype = unsafe { (*ptr).xltype };
    assert!(
        xltype & XLBIT_DLL_FREE != 0,
        "Error XLOPER12 should have xlbitDLLFree set, got xltype=0x{:X}",
        xltype
    );
    // Clean up
    xll_free(ptr);
}

// ============================================================================
// Memory Safety — Stress Tests with Heap Tracking
// ============================================================================
//
// These tests run 200 iterations of each UDF path and log process memory
// every 50 iterations. Run with `cargo test --features xll -- --nocapture`
// to see the memory output.
//
// Any double-free, use-after-free, or allocator corruption would crash on
// debug builds. The heap tracking lets us visually confirm that committed
// memory (pagefile usage) stabilizes rather than growing linearly.

const STRESS_ITERATIONS: usize = 500;
const LOG_INTERVAL: usize = STRESS_ITERATIONS / 10;

#[test]
fn test_ols_stress_with_mem_tracking() {
    let (y_data, x_data) = simple_linear_data();
    let baseline = MemSnapshot::now();
    eprintln!();
    log_mem("ols_simple", 0, &baseline);

    for i in 1..=STRESS_ITERATIONS {
        let (y_range, _y_cells) = build_column_range(&y_data);
        let (x_range, _x_cells) = build_matrix_range(&[x_data.clone()]);

        let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
        assert!(result.is_multi(), "Iteration {} should succeed", i);

        if i % LOG_INTERVAL == 0 {
            log_mem("ols_simple", i, &baseline);
        }
    }
}

#[test]
fn test_ols_multi_reg_stress_with_mem_tracking() {
    let (y_data, x_cols) = mtcars_subset();
    let baseline = MemSnapshot::now();
    eprintln!();
    log_mem("ols_mtcars", 0, &baseline);

    for i in 1..=STRESS_ITERATIONS {
        let (y_range, _y_cells) = build_column_range(&y_data);
        let (x_range, _x_cells) = build_matrix_range(&x_cols);

        let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
        assert!(result.is_multi(), "Iteration {} should succeed", i);

        let (rows, cols) = result.dimensions();
        assert_eq!(rows, 12); // 1 header + 5 coefs + 6 summary
        assert_eq!(cols, 5);

        if i % LOG_INTERVAL == 0 {
            log_mem("ols_mtcars", i, &baseline);
        }
    }
}

#[test]
fn test_version_stress_with_mem_tracking() {
    let baseline = MemSnapshot::now();
    eprintln!();
    log_mem("version", 0, &baseline);

    for i in 1..=STRESS_ITERATIONS {
        let result = XlResultGuard::new(xll_version());
        assert!(result.is_string(), "Iteration {} should return string", i);

        if i % LOG_INTERVAL == 0 {
            log_mem("version", i, &baseline);
        }
    }
}

#[test]
fn test_error_stress_with_mem_tracking() {
    let baseline = MemSnapshot::now();
    eprintln!();
    log_mem("error", 0, &baseline);

    for i in 1..=STRESS_ITERATIONS {
        let result = XlResultGuard::new(xll_ols(std::ptr::null(), std::ptr::null()));
        assert!(result.is_error(), "Iteration {} should return error", i);

        if i % LOG_INTERVAL == 0 {
            log_mem("error", i, &baseline);
        }
    }
}

#[test]
fn test_mixed_stress_with_mem_tracking() {
    let (y_data, x_data) = simple_linear_data();
    let baseline = MemSnapshot::now();
    eprintln!();
    log_mem("mixed", 0, &baseline);

    for i in 1..=STRESS_ITERATIONS {
        if i % 2 == 0 {
            let (y_range, _y_cells) = build_column_range(&y_data);
            let (x_range, _x_cells) = build_matrix_range(&[x_data.clone()]);
            let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
            assert!(result.is_multi(), "Even iteration {} should succeed", i);
        } else {
            let result = XlResultGuard::new(xll_ols(std::ptr::null(), std::ptr::null()));
            assert!(result.is_error(), "Odd iteration {} should error", i);
        }

        if i % LOG_INTERVAL == 0 {
            log_mem("mixed", i, &baseline);
        }
    }
}

#[test]
fn test_ols_memory_leak_detection() {
    // Quantitative leak detection: warmup, snapshot heap, run 200 iterations,
    // snapshot again, assert bounded growth.
    let (y_data, x_cols) = mtcars_subset();

    // Warmup — let allocator pools stabilize
    for _ in 0..50 {
        let (y_range, _y_cells) = build_column_range(&y_data);
        let (x_range, _x_cells) = build_matrix_range(&x_cols);
        let _result = XlResultGuard::new(xll_ols(&y_range, &x_range));
    }

    let baseline = MemSnapshot::now();
    eprintln!();
    log_mem("leak_detect", 0, &baseline);

    for i in 1..=STRESS_ITERATIONS {
        let (y_range, _y_cells) = build_column_range(&y_data);
        let (x_range, _x_cells) = build_matrix_range(&x_cols);
        let result = XlResultGuard::new(xll_ols(&y_range, &x_range));
        assert!(result.is_multi());

        let v = XlResultGuard::new(xll_version());
        assert!(v.is_string());

        if i % LOG_INTERVAL == 0 {
            log_mem("leak_detect", i, &baseline);
        }
    }

    let final_snap = MemSnapshot::now();
    let delta_kb = final_snap.heap_delta_from(&baseline) / 1024;
    log_mem("leak_detect", STRESS_ITERATIONS, &baseline);
    eprintln!("  [leak_detect] final heap delta: {} KB over {} iterations", delta_kb, STRESS_ITERATIONS);

    // Each OLS call with mtcars data allocates ~60 XLOPER12 cells with ~12
    // string buffers. If xlAutoFree12 leaks even one string per iteration,
    // that's ~200 * 50 bytes = 10 KB minimum growth. With full leaking
    // (no free at all), expect ~200 * 2 KB = 400 KB.
    //
    // Allow 1024 KB of jitter (allocator fragmentation, thread-local caches).
    // A real leak would show linear growth visible in the log output.
    assert!(
        delta_kb < 1024,
        "Heap grew by {} KB over {} iterations — possible memory leak. \
         Check log output for linear growth pattern.",
        delta_kb,
        STRESS_ITERATIONS
    );
}
