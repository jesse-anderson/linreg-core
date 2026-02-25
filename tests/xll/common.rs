//! Common utilities for XLL integration tests.
//!
//! Provides helpers to build XLOPER12 inputs (simulating Excel ranges),
//! inspect returned XLOPER12 results, and manage memory cleanup.
//!
//! # Memory Rules
//!
//! - **Input XLOPER12s** are owned by the test (simulating Excel ownership).
//!   They are stack/Vec-allocated and dropped normally. Do NOT call xlAutoFree12 on them.
//! - **Returned XLOPER12s** are DLL-owned (have `xlbitDLLFree` set).
//!   Must be freed via `xlAutoFree12` after inspection.

pub use linreg_core::xll::types::*;
use linreg_core::xll::xlAutoFree12;
use std::mem::ManuallyDrop;

// ============================================================================
// Process Memory Tracking (Windows)
// ============================================================================

#[repr(C)]
#[allow(non_snake_case)]
struct PROCESS_MEMORY_COUNTERS {
    cb: u32,
    PageFaultCount: u32,
    PeakWorkingSetSize: usize,
    WorkingSetSize: usize,
    QuotaPeakPagedPoolUsage: usize,
    QuotaPagedPoolUsage: usize,
    QuotaPeakNonPagedPoolUsage: usize,
    QuotaNonPagedPoolUsage: usize,
    PagefileUsage: usize,
    PeakPagefileUsage: usize,
}

extern "system" {
    fn GetCurrentProcess() -> *mut std::ffi::c_void;
    fn K32GetProcessMemoryInfo(
        process: *mut std::ffi::c_void,
        ppsmemCounters: *mut PROCESS_MEMORY_COUNTERS,
        cb: u32,
    ) -> i32;
}

/// Snapshot of process memory usage.
#[derive(Clone, Copy)]
pub struct MemSnapshot {
    /// Current physical memory (working set) in bytes.
    pub working_set: usize,
    /// Committed virtual memory (pagefile usage) in bytes — closest to "heap size".
    pub pagefile: usize,
    /// Peak working set in bytes.
    pub peak_working_set: usize,
}

impl MemSnapshot {
    /// Take a snapshot of the current process memory usage.
    pub fn now() -> Self {
        let mut counters = PROCESS_MEMORY_COUNTERS {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            PageFaultCount: 0,
            PeakWorkingSetSize: 0,
            WorkingSetSize: 0,
            QuotaPeakPagedPoolUsage: 0,
            QuotaPagedPoolUsage: 0,
            QuotaPeakNonPagedPoolUsage: 0,
            QuotaNonPagedPoolUsage: 0,
            PagefileUsage: 0,
            PeakPagefileUsage: 0,
        };
        unsafe {
            K32GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut counters,
                counters.cb,
            );
        }
        Self {
            working_set: counters.WorkingSetSize,
            pagefile: counters.PagefileUsage,
            peak_working_set: counters.PeakWorkingSetSize,
        }
    }

    /// Format as human-readable string (KB).
    pub fn display(&self) -> String {
        format!(
            "heap: {} KB, working_set: {} KB, peak: {} KB",
            self.pagefile / 1024,
            self.working_set / 1024,
            self.peak_working_set / 1024,
        )
    }

    /// Pagefile delta from a baseline, in bytes.
    pub fn heap_delta_from(&self, baseline: &MemSnapshot) -> i64 {
        self.pagefile as i64 - baseline.pagefile as i64
    }
}

/// Print a memory status line for stress tests.
/// Only visible when running with `cargo test -- --nocapture`.
pub fn log_mem(label: &str, iter: usize, baseline: &MemSnapshot) {
    let now = MemSnapshot::now();
    let delta = now.heap_delta_from(baseline);
    let sign = if delta >= 0 { "+" } else { "" };
    eprintln!(
        "  [{}] iter {:>4} | {} | delta: {}{} KB",
        label,
        iter,
        now.display(),
        sign,
        delta / 1024,
    );
}

// Re-export XLL functions for convenient use in tests
pub use linreg_core::xll::xlAutoFree12 as xll_free;
pub use linreg_core::xll::xl_linreg_version as xll_version;
pub use linreg_core::xll::xl_linreg_ols as xll_ols;

// ============================================================================
// Test Data Fixtures
// ============================================================================

/// Simple linear regression test data: y = 2 + 3x (perfect fit)
pub fn simple_linear_data() -> (Vec<f64>, Vec<f64>) {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
    (y, x)
}

/// Multiple regression test data (mtcars subset - 20 observations, 4 predictors)
pub fn mtcars_subset() -> (Vec<f64>, Vec<Vec<f64>>) {
    let y = vec![
        21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2,
        17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9,
    ];
    let cyl = vec![6.0, 6.0, 4.0, 6.0, 8.0, 6.0, 8.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0];
    let disp = vec![160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472.0, 460.0, 440.0, 78.7, 75.7, 71.1];
    let hp = vec![110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0, 123.0, 180.0, 180.0, 180.0, 205.0, 215.0, 230.0, 66.0, 52.0, 65.0];
    let wt = vec![2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440, 3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835];
    (y, vec![cyl, disp, hp, wt])
}

// ============================================================================
// XLOPER12 Input Builders (simulate Excel ranges)
// ============================================================================

/// Build a column-vector XLOPER12 (n_rows x 1) from f64 values.
///
/// Returns (oper, cells) — caller must keep `cells` alive while `oper` is in use.
pub fn build_column_range(values: &[f64]) -> (XLOPER12, Vec<XLOPER12>) {
    let mut cells: Vec<XLOPER12> = values.iter().map(|&v| XLOPER12::from_f64(v)).collect();
    let oper = XLOPER12 {
        val: XLOPER12Val {
            array: ManuallyDrop::new(XLOPER12Array {
                lparray: cells.as_mut_ptr(),
                rows: values.len() as i32,
                columns: 1,
            }),
        },
        xltype: XLTYPE_MULTI,
    };
    (oper, cells)
}

/// Build a matrix XLOPER12 (n_rows x n_cols) from column vectors.
///
/// Input is column-major (each inner Vec is one predictor column).
/// Output is row-major (matching Excel's xltypeMulti layout).
///
/// Returns (oper, cells) — caller must keep `cells` alive while `oper` is in use.
pub fn build_matrix_range(columns: &[Vec<f64>]) -> (XLOPER12, Vec<XLOPER12>) {
    let n_rows = columns[0].len();
    let n_cols = columns.len();
    let mut cells = Vec::with_capacity(n_rows * n_cols);
    // Row-major order (how Excel stores xltypeMulti)
    for row in 0..n_rows {
        for col in 0..n_cols {
            cells.push(XLOPER12::from_f64(columns[col][row]));
        }
    }
    let oper = XLOPER12 {
        val: XLOPER12Val {
            array: ManuallyDrop::new(XLOPER12Array {
                lparray: cells.as_mut_ptr(),
                rows: n_rows as i32,
                columns: n_cols as i32,
            }),
        },
        xltype: XLTYPE_MULTI,
    };
    (oper, cells)
}

/// Build a column range with a nil (empty) cell at the given index.
pub fn build_column_with_nil(values: &[f64], nil_index: usize) -> (XLOPER12, Vec<XLOPER12>) {
    let mut cells: Vec<XLOPER12> = values.iter().map(|&v| XLOPER12::from_f64(v)).collect();
    cells[nil_index] = XLOPER12::nil();
    let oper = XLOPER12 {
        val: XLOPER12Val {
            array: ManuallyDrop::new(XLOPER12Array {
                lparray: cells.as_mut_ptr(),
                rows: values.len() as i32,
                columns: 1,
            }),
        },
        xltype: XLTYPE_MULTI,
    };
    (oper, cells)
}

/// Build a column range with an error cell at the given index.
pub fn build_column_with_error(values: &[f64], err_index: usize, err_code: i32) -> (XLOPER12, Vec<XLOPER12>) {
    let mut cells: Vec<XLOPER12> = values.iter().map(|&v| XLOPER12::from_f64(v)).collect();
    cells[err_index] = XLOPER12::from_err(err_code);
    let oper = XLOPER12 {
        val: XLOPER12Val {
            array: ManuallyDrop::new(XLOPER12Array {
                lparray: cells.as_mut_ptr(),
                rows: values.len() as i32,
                columns: 1,
            }),
        },
        xltype: XLTYPE_MULTI,
    };
    (oper, cells)
}

// ============================================================================
// Result Inspection Helpers
// ============================================================================

/// RAII guard for a returned XLOPER12 pointer. Calls xlAutoFree12 on drop.
pub struct XlResultGuard {
    pub ptr: *mut XLOPER12,
}

impl XlResultGuard {
    pub fn new(ptr: *mut XLOPER12) -> Self {
        assert!(!ptr.is_null(), "XLL function returned null pointer");
        Self { ptr }
    }

    /// Get the base type of the returned XLOPER12.
    pub fn base_type(&self) -> u32 {
        unsafe { (*self.ptr).base_type() }
    }

    /// Check if the result is an error.
    pub fn is_error(&self) -> bool {
        self.base_type() == XLTYPE_ERR
    }

    /// Get the error code (panics if not an error type).
    pub fn error_code(&self) -> i32 {
        assert!(self.is_error(), "Expected error XLOPER12, got type {}", self.base_type());
        unsafe { (*self.ptr).val.err }
    }

    /// Check if the result is a string.
    pub fn is_string(&self) -> bool {
        self.base_type() == XLTYPE_STR
    }

    /// Get the string value (panics if not a string type).
    pub fn as_string(&self) -> String {
        assert!(self.is_string(), "Expected string XLOPER12, got type {}", self.base_type());
        unsafe { (*self.ptr).as_string().unwrap() }
    }

    /// Check if the result is a multi-cell array.
    pub fn is_multi(&self) -> bool {
        self.base_type() == XLTYPE_MULTI
    }

    /// Get array dimensions (rows, cols). Panics if not a multi type.
    pub fn dimensions(&self) -> (usize, usize) {
        assert!(self.is_multi(), "Expected multi XLOPER12, got type {}", self.base_type());
        unsafe {
            let arr = &*std::ptr::addr_of!((*self.ptr).val.array);
            (arr.rows as usize, arr.columns as usize)
        }
    }

    /// Get a cell from the array at (row, col). Panics if out of bounds or not multi.
    pub fn cell(&self, row: usize, col: usize) -> &XLOPER12 {
        let (rows, cols) = self.dimensions();
        assert!(row < rows && col < cols, "Cell ({}, {}) out of bounds ({}, {})", row, col, rows, cols);
        unsafe {
            let arr = &*std::ptr::addr_of!((*self.ptr).val.array);
            &*arr.lparray.add(row * cols + col)
        }
    }

    /// Get a string from cell (row, col).
    pub fn cell_string(&self, row: usize, col: usize) -> String {
        let cell = self.cell(row, col);
        cell.as_string().unwrap_or_else(|| panic!("Cell ({}, {}) is not a string, type={}", row, col, cell.base_type()))
    }

    /// Get an f64 from cell (row, col).
    pub fn cell_f64(&self, row: usize, col: usize) -> f64 {
        let cell = self.cell(row, col);
        cell.as_f64().unwrap_or_else(|| panic!("Cell ({}, {}) is not numeric, type={}", row, col, cell.base_type()))
    }
}

impl Drop for XlResultGuard {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            xlAutoFree12(self.ptr);
        }
    }
}
