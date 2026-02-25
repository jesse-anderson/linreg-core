//! Conversion helpers between Excel XLOPER12 ranges and Rust types.
//!
//! # Gotchas discovered during development
//!
//! - When arguments are registered as type `Q`, Excel coerces cell references
//!   into `xltypeMulti` arrays automatically.  We never see `xltypeSRef` or
//!   `xltypeRef` — those only appear with type `U` registration.
//! - A single cell passed as an argument arrives as `xltypeNum`, `xltypeStr`,
//!   etc. — NOT as a 1x1 `xltypeMulti`.  Must handle both cases.
//! - Empty cells inside a multi-cell range arrive as `xltypeNil`.
//! - Cells containing errors arrive as `xltypeErr` — we propagate them.

use super::types::*;

/// Extract a column vector of f64 from an XLOPER12 (single value or range).
///
/// For a 2D range, reads all values in column-major order flattened into one
/// vector (used for the Y variable — must be a single column).
pub fn xloper_to_f64_vec(p: *const XLOPER12) -> Result<Vec<f64>, i32> {
    if p.is_null() {
        return Err(XLERR_VALUE);
    }
    let oper = unsafe { &*p };
    match oper.base_type() {
        XLTYPE_NUM => Ok(vec![unsafe { oper.val.num }]),
        XLTYPE_INT => Ok(vec![unsafe { oper.val.w } as f64]),
        XLTYPE_MISSING | XLTYPE_NIL => Err(XLERR_VALUE),
        XLTYPE_ERR => Err(unsafe { oper.val.err }),
        XLTYPE_MULTI => {
            let arr = unsafe { &*std::ptr::addr_of!(oper.val.array) };
            let total = (arr.rows * arr.columns) as usize;
            let mut result = Vec::with_capacity(total);
            for i in 0..total {
                let elem = unsafe { &*arr.lparray.add(i) };
                match elem.base_type() {
                    XLTYPE_NUM => result.push(unsafe { elem.val.num }),
                    XLTYPE_INT => result.push(unsafe { elem.val.w } as f64),
                    XLTYPE_NIL => return Err(XLERR_VALUE), // empty cell in data
                    XLTYPE_ERR => return Err(unsafe { elem.val.err }),
                    _ => return Err(XLERR_VALUE),
                }
            }
            Ok(result)
        }
        _ => Err(XLERR_VALUE),
    }
}

/// Extract a 2D range into column-major `Vec<Vec<f64>>` — the format
/// `ols_regression` expects for `x_vars`.
///
/// A range of shape (n_rows, n_cols) becomes `n_cols` vectors each of length
/// `n_rows`.  Also returns `(n_rows, n_cols)`.
pub fn xloper_to_columns(p: *const XLOPER12) -> Result<(Vec<Vec<f64>>, usize, usize), i32> {
    if p.is_null() {
        return Err(XLERR_VALUE);
    }
    let oper = unsafe { &*p };
    match oper.base_type() {
        // Single numeric value → 1 column, 1 row
        XLTYPE_NUM => Ok((vec![vec![unsafe { oper.val.num }]], 1, 1)),
        XLTYPE_INT => Ok((vec![vec![unsafe { oper.val.w } as f64]], 1, 1)),
        XLTYPE_MISSING | XLTYPE_NIL => Err(XLERR_VALUE),
        XLTYPE_ERR => Err(unsafe { oper.val.err }),
        XLTYPE_MULTI => {
            let arr = unsafe { &*std::ptr::addr_of!(oper.val.array) };
            let n_rows = arr.rows as usize;
            let n_cols = arr.columns as usize;
            if n_rows == 0 || n_cols == 0 {
                return Err(XLERR_VALUE);
            }

            // Build column-major: columns[col][row]
            let mut columns = vec![Vec::with_capacity(n_rows); n_cols];
            for row in 0..n_rows {
                for col in 0..n_cols {
                    let idx = row * n_cols + col; // row-major in XLOPER12
                    let elem = unsafe { &*arr.lparray.add(idx) };
                    match elem.base_type() {
                        XLTYPE_NUM => columns[col].push(unsafe { elem.val.num }),
                        XLTYPE_INT => columns[col].push(unsafe { elem.val.w } as f64),
                        XLTYPE_NIL => return Err(XLERR_VALUE),
                        XLTYPE_ERR => return Err(unsafe { elem.val.err }),
                        _ => return Err(XLERR_VALUE),
                    }
                }
            }
            Ok((columns, n_rows, n_cols))
        }
        _ => Err(XLERR_VALUE),
    }
}

/// Build an xltypeMulti XLOPER12 from a grid of cells.
///
/// `cells` is a flat row-major array of XLOPER12 values.  The returned
/// XLOPER12 is heap-allocated with `xlbitDLLFree` set — caller returns it
/// directly to Excel, which will call `xlAutoFree12` after copying.
///
/// # Memory contract
///
/// - The `cells` Vec is leaked (forgotten) and reclaimed in `xlAutoFree12`.
/// - Any string XLOPER12s inside `cells` must have been created with
///   `XLOPER12::from_str()` (which allocates its own buffer).  They are
///   freed recursively in `xlAutoFree12`.
/// - Numeric / error / nil cells have no allocations.
pub fn build_multi(mut cells: Vec<XLOPER12>, rows: usize, cols: usize) -> *mut XLOPER12 {
    debug_assert_eq!(cells.len(), rows * cols);

    let lparray = cells.as_mut_ptr();
    std::mem::forget(cells);

    let result = Box::new(XLOPER12 {
        val: XLOPER12Val {
            array: std::mem::ManuallyDrop::new(XLOPER12Array {
                lparray,
                rows: rows as i32,
                columns: cols as i32,
            }),
        },
        xltype: XLTYPE_MULTI | XLBIT_DLL_FREE,
    });
    Box::into_raw(result)
}

/// Return a heap-allocated error XLOPER12 suitable as a UDF return value.
pub fn return_xl_error(code: i32) -> *mut XLOPER12 {
    Box::into_raw(Box::new(XLOPER12 {
        val: XLOPER12Val { err: code },
        xltype: XLTYPE_ERR | XLBIT_DLL_FREE,
    }))
}

/// Build a design matrix (with intercept column) from column-major predictor vectors.
///
/// Takes `n_cols` predictor columns each of length `n_rows`, prepends a column of 1.0s,
/// and returns a row-major `Matrix` of shape `(n_rows, n_cols + 1)`.
pub fn columns_to_design_matrix(
    columns: &[Vec<f64>],
    n_rows: usize,
) -> crate::linalg::Matrix {
    let n_cols = columns.len();
    let mut data = Vec::with_capacity(n_rows * (n_cols + 1));
    for row in 0..n_rows {
        data.push(1.0); // intercept
        for col in 0..n_cols {
            data.push(columns[col][row]);
        }
    }
    crate::linalg::Matrix::new(n_rows, n_cols + 1, data)
}
