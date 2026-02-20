//! Diagnostic test FFI bindings.
//!
//! Each function fits an OLS model internally, runs the diagnostic, stores a
//! lightweight `DiagnosticResult` (statistic + p-value), and returns a handle.
//! Call `LR_GetStatistic`, `LR_GetPValue`, and `LR_GetDF` to retrieve results.
//! Call `LR_Free` when done.
//!
//! Durbin-Watson does not produce a p-value; `LR_GetPValue` returns NAN for
//! that handle.  Use `LR_GetAutocorrelation` to read ρ ≈ 1 − DW/2.

use std::slice;

use super::store::{insert, set_last_error, with};
use super::types::{DiagnosticResult, FitResult};
use crate::diagnostics;

// ── Helper: unpack raw pointers into owned vecs ───────────────────────────────

unsafe fn unpack(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    if y_ptr.is_null() || x_ptr.is_null() || n <= 0 || p <= 0 {
        return None;
    }
    let n = n as usize;
    let p = p as usize;
    let y = slice::from_raw_parts(y_ptr, n).to_vec();
    let x_flat = slice::from_raw_parts(x_ptr, n * p);
    let mut x_vars: Vec<Vec<f64>> = vec![Vec::with_capacity(n); p];
    for row in 0..n {
        for col in 0..p {
            x_vars[col].push(x_flat[row * p + col]);
        }
    }
    Some((y, x_vars))
}

/// Store a DiagnosticResult and return its handle.
fn store_diag(statistic: f64, p_value: f64, df: f64, autocorrelation: f64) -> usize {
    insert(FitResult::Diagnostic(DiagnosticResult {
        statistic,
        p_value,
        df,
        autocorrelation,
    }))
}

// ── Scalar getters shared by all diagnostic handles ───────────────────────────

/// Return the test statistic for a diagnostic handle.
#[no_mangle]
pub extern "system" fn LR_GetStatistic(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Diagnostic(d) => d.statistic,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Return the p-value for a diagnostic handle (NAN for Durbin-Watson).
#[no_mangle]
pub extern "system" fn LR_GetPValue(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Diagnostic(d) => d.p_value,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Return degrees of freedom for a diagnostic handle (0.0 if not applicable).
#[no_mangle]
pub extern "system" fn LR_GetTestDF(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Diagnostic(d) => d.df,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

/// Return the estimated autocorrelation ρ for a Durbin-Watson handle (0.0 otherwise).
#[no_mangle]
pub extern "system" fn LR_GetAutocorrelation(handle: usize) -> f64 {
    with(handle, |r| match r {
        FitResult::Diagnostic(d) => d.autocorrelation,
        _ => f64::NAN,
    })
    .unwrap_or(f64::NAN)
}

// ── Diagnostic fit functions ──────────────────────────────────────────────────

macro_rules! diag_fn {
    ($fn_name:ident, $call:expr, $label:literal) => {
        #[no_mangle]
        pub extern "system" fn $fn_name(
            y_ptr: *const f64,
            n: i32,
            x_ptr: *const f64,
            p: i32,
        ) -> usize {
            let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
                Some(v) => v,
                None => {
                    set_last_error(concat!($label, ": null pointer or non-positive dimensions"));
                    return 0;
                }
            };
            match $call(&y, &x_vars) {
                Ok(r) => store_diag(r.statistic, r.p_value, 0.0, 0.0),
                Err(e) => {
                    set_last_error(&e.to_string());
                    0
                }
            }
        }
    };
}

diag_fn!(LR_BreuschPagan,    diagnostics::breusch_pagan_test,   "LR_BreuschPagan");
diag_fn!(LR_JarqueBera,      diagnostics::jarque_bera_test,      "LR_JarqueBera");
diag_fn!(LR_ShapiroWilk,     diagnostics::shapiro_wilk_test,     "LR_ShapiroWilk");
diag_fn!(LR_AndersonDarling, diagnostics::anderson_darling_test, "LR_AndersonDarling");

/// Harvey-Collier test for linearity (R method).
#[no_mangle]
pub extern "system" fn LR_HarveyCollier(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_HarveyCollier: null pointer or non-positive dimensions");
            return 0;
        }
    };
    match diagnostics::harvey_collier_test(&y, &x_vars, diagnostics::HarveyCollierMethod::R) {
        Ok(r) => store_diag(r.statistic, r.p_value, 0.0, 0.0),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

/// Breusch-Pagan White test (R method).
#[no_mangle]
pub extern "system" fn LR_White(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_White: null pointer or non-positive dimensions");
            return 0;
        }
    };
    match diagnostics::r_white_method(&y, &x_vars) {
        Ok(r) => store_diag(r.statistic, r.p_value, 0.0, 0.0),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

/// Rainbow test for linearity.
///
/// - `fraction` – fraction of central data used (typically 0.5)
#[no_mangle]
pub extern "system" fn LR_Rainbow(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    fraction: f64,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_Rainbow: null pointer or non-positive dimensions");
            return 0;
        }
    };
    match diagnostics::rainbow_test(&y, &x_vars, fraction, diagnostics::RainbowMethod::R) {
        Ok(output) => {
            if let Some(r) = output.r_result {
                store_diag(r.statistic, r.p_value, 0.0, 0.0)
            } else {
                set_last_error("LR_Rainbow: no R result returned");
                0
            }
        }
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

/// RESET test for model specification error.
///
/// - `powers_ptr` – pointer to array of `powers_len` power values (e.g. {2, 3})
/// - `powers_len` – number of powers
#[no_mangle]
pub extern "system" fn LR_Reset(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    powers_ptr: *const i32,
    powers_len: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_Reset: null pointer or non-positive dimensions");
            return 0;
        }
    };
    if powers_ptr.is_null() || powers_len <= 0 {
        set_last_error("LR_Reset: null powers pointer or non-positive powers_len");
        return 0;
    }
    let powers: Vec<usize> = unsafe { slice::from_raw_parts(powers_ptr, powers_len as usize) }
        .iter()
        .map(|&v| v as usize)
        .collect();

    match diagnostics::reset_test(&y, &x_vars, &powers, diagnostics::ResetType::Fitted) {
        Ok(r) => store_diag(r.statistic, r.p_value, 0.0, 0.0),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

/// Durbin-Watson test for first-order autocorrelation.
///
/// `LR_GetPValue` returns NAN for this handle.
/// Use `LR_GetAutocorrelation` to read ρ ≈ 1 − DW/2.
#[no_mangle]
pub extern "system" fn LR_DurbinWatson(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_DurbinWatson: null pointer or non-positive dimensions");
            return 0;
        }
    };
    match diagnostics::durbin_watson_test(&y, &x_vars) {
        Ok(r) => store_diag(r.statistic, f64::NAN, 0.0, r.autocorrelation),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

// ── Influence diagnostic fit functions ───────────────────────────────────────
//
// These return Vector / Matrix handles.  Use the generic getters defined in
// ols.rs (LR_GetVectorLength, LR_GetVector, LR_GetMatrixRows, LR_GetMatrixCols,
// LR_GetMatrix) to retrieve the results.

/// Cook's distance for each observation.
///
/// Returns a handle to a `Vector` of n Cook's distance values.
/// Use `LR_GetVectorLength` and `LR_GetVector` to retrieve values.
#[no_mangle]
pub extern "system" fn LR_CooksDistance(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_CooksDistance: null pointer or non-positive dimensions");
            return 0;
        }
    };
    match diagnostics::cooks_distance_test(&y, &x_vars) {
        Ok(r) => insert(FitResult::Vector(r.distances)),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

/// DFFITS for each observation (influence on its own fitted value).
///
/// Returns a handle to a `Vector` of n DFFITS values.
/// Use `LR_GetVectorLength` and `LR_GetVector` to retrieve values.
#[no_mangle]
pub extern "system" fn LR_DFFITS(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_DFFITS: null pointer or non-positive dimensions");
            return 0;
        }
    };
    match diagnostics::dffits_test(&y, &x_vars) {
        Ok(r) => insert(FitResult::Vector(r.dffits)),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

/// Variance Inflation Factor for each predictor (multicollinearity check).
///
/// Requires p >= 2 predictors.  Returns a handle to a `Vector` of p VIF values
/// (one per predictor, in the same order as the input columns, excluding intercept).
/// Use `LR_GetVectorLength` and `LR_GetVector` to retrieve values.
#[no_mangle]
pub extern "system" fn LR_VIF(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_VIF: null pointer or non-positive dimensions");
            return 0;
        }
    };
    match diagnostics::vif_test(&y, &x_vars) {
        Ok(r) => {
            let values: Vec<f64> = r.vif_results.iter().map(|v| v.vif).collect();
            insert(FitResult::Vector(values))
        }
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

/// DFBETAS matrix: influence of each observation on each coefficient.
///
/// Returns a handle to a `Matrix` with shape (n × p_total) where
/// p_total = k + 1 (intercept + k predictors).
/// Row i, column j = standardised change in coefficient j when obs i is omitted.
/// Use `LR_GetMatrixRows`, `LR_GetMatrixCols`, and `LR_GetMatrix` to retrieve.
#[no_mangle]
pub extern "system" fn LR_DFBETAS(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_DFBETAS: null pointer or non-positive dimensions");
            return 0;
        }
    };
    match diagnostics::dfbetas_test(&y, &x_vars) {
        Ok(r) => {
            let rows = r.n;
            let cols = r.p; // includes intercept
            // Flatten Vec<Vec<f64>> to row-major Vec<f64>
            let data: Vec<f64> = r.dfbetas.into_iter().flatten().collect();
            insert(FitResult::Matrix { data, rows, cols })
        }
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}

/// Breusch-Godfrey test for higher-order serial correlation.
///
/// - `lag_order` – maximum lag order to test (>= 1)
#[no_mangle]
pub extern "system" fn LR_BreuschGodfrey(
    y_ptr: *const f64,
    n: i32,
    x_ptr: *const f64,
    p: i32,
    lag_order: i32,
) -> usize {
    let (y, x_vars) = match unsafe { unpack(y_ptr, n, x_ptr, p) } {
        Some(v) => v,
        None => {
            set_last_error("LR_BreuschGodfrey: null pointer or non-positive dimensions");
            return 0;
        }
    };
    let order = lag_order.max(1) as usize;
    match diagnostics::breusch_godfrey_test(&y, &x_vars, order, diagnostics::BGTestType::Chisq) {
        Ok(r) => store_diag(r.statistic, r.p_value, r.df.first().copied().unwrap_or(0.0), 0.0),
        Err(e) => {
            set_last_error(&e.to_string());
            0
        }
    }
}
