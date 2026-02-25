//! XLL add-in bindings for Excel.
//!
//! Exposes linreg-core functions as Excel worksheet UDFs (User-Defined Functions).
//! Users can type `=LINREG.VERSION()` or `=LINREG.OLS(A1:A100, B1:E100)`
//! directly in cells.
//!
//! Compiled only when the `xll` feature is enabled:
//!
//! ```bash
//! cargo build --release --features xll --no-default-features --target x86_64-pc-windows-msvc
//! ```
//!
//! The resulting DLL must be renamed from `.dll` to `.xll` for Excel to
//! recognise it as an add-in.

pub mod convert;
pub mod entrypoint;
pub mod register;
pub mod types;

use convert::{build_multi, columns_to_design_matrix, return_xl_error, xloper_to_columns, xloper_to_f64_vec};
use register::Reg;
use types::*;

use crate::core::{ols_regression, RegressionOutput};
use crate::weighted_regression::{wls_regression, WlsFit};
use crate::regularized::{
    ridge::{ridge_fit, RidgeFit, RidgeFitOptions},
    lasso::{lasso_fit, LassoFit, LassoFitOptions},
    elastic_net::{elastic_net_fit, ElasticNetFit, ElasticNetOptions},
};
use crate::diagnostics;
use crate::cross_validation::{kfold_cv_ols, kfold_cv_ridge, kfold_cv_lasso, kfold_cv_elastic_net, KFoldOptions, CVResult};
use crate::prediction_intervals;
use crate::polynomial::{polynomial_regression, PolynomialOptions, PolynomialFit};
use crate::regularized::path::{make_lambda_path, LambdaPathOptions};

// ── xlAutoOpen — called by Excel when the XLL is loaded ─────────────────────

/// Register all UDFs with Excel.  Returns 1 on success.
#[no_mangle]
pub extern "system" fn xlAutoOpen() -> i32 {
    let reg = Reg::new();

    // VERSION — no args, returns string
    reg.add(
        "xl_linreg_version",
        "Q$",
        "LINREG.VERSION",
        "",
        "LinReg",
        "Returns the linreg-core library version",
        &[],
    );

    // OLS — two args (y_range, x_range), returns array
    // Type string "QQQ$": returns XLOPER12, takes 2 XLOPER12 args, thread-safe
    reg.add(
        "xl_linreg_ols",
        "QQQ$",
        "LINREG.OLS",
        "y_range, x_range",
        "LinReg",
        "OLS regression — returns coefficient table and fit statistics",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
        ],
    );

    // WLS — three args (y_range, x_range, weights_range), returns array
    reg.add(
        "xl_linreg_wls",
        "QQQQ$",
        "LINREG.WLS",
        "y_range, x_range, weights_range",
        "LinReg",
        "Weighted Least Squares — returns coefficient table and fit statistics",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
            "Observation weights (positive values, same length as y)",
        ],
    );

    // RIDGE — args: y_range, x_range, lambda, [standardize]
    reg.add(
        "xl_linreg_ridge",
        "QQQQQ$",
        "LINREG.RIDGE",
        "y_range, x_range, lambda, [standardize]",
        "LinReg",
        "Ridge regression (L2) — returns coefficient table and fit statistics",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
            "Regularization penalty (lambda >= 0)",
            "Standardize predictors before fitting? (default TRUE)",
        ],
    );

    // LASSO — args: y_range, x_range, lambda, [standardize]
    reg.add(
        "xl_linreg_lasso",
        "QQQQQ$",
        "LINREG.LASSO",
        "y_range, x_range, lambda, [standardize]",
        "LinReg",
        "Lasso regression (L1) — returns coefficient table and fit statistics",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
            "Regularization penalty (lambda >= 0)",
            "Standardize predictors before fitting? (default TRUE)",
        ],
    );

    // ELASTICNET — args: y_range, x_range, lambda, alpha, [standardize]
    reg.add(
        "xl_linreg_elasticnet",
        "QQQQQQ$",
        "LINREG.ELASTICNET",
        "y_range, x_range, lambda, alpha, [standardize]",
        "LinReg",
        "Elastic Net regression (L1+L2) — returns coefficient table and fit statistics",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
            "Regularization penalty (lambda >= 0)",
            "L1/L2 mixing (0=Ridge, 1=Lasso, between=Elastic Net)",
            "Standardize predictors before fitting? (default TRUE)",
        ],
    );

    // ── Diagnostic tests ──────────────────────────────────────────────────

    // Simple diagnostics: (y, X) -> 2x2 labeled array
    let yx_help: &[&str] = &[
        "Response variable (single column)",
        "Predictor variables (one or more columns)",
    ];
    reg.add(
        "xl_linreg_breuschpagan",
        "QQQ$",
        "LINREG.BREUSCHPAGAN",
        "y_range, x_range",
        "LinReg",
        "Breusch-Pagan heteroscedasticity test — tests if error variance depends on X",
        yx_help,
    );
    reg.add(
        "xl_linreg_white",
        "QQQ$",
        "LINREG.WHITE",
        "y_range, x_range",
        "LinReg",
        "White heteroscedasticity test (R method) — general test for non-constant variance",
        yx_help,
    );
    reg.add(
        "xl_linreg_jarquebera",
        "QQQ$",
        "LINREG.JARQUEBERA",
        "y_range, x_range",
        "LinReg",
        "Jarque-Bera normality test — tests if residuals are normally distributed",
        yx_help,
    );
    reg.add(
        "xl_linreg_shapirowilk",
        "QQQ$",
        "LINREG.SHAPIROWILK",
        "y_range, x_range",
        "LinReg",
        "Shapiro-Wilk normality test — tests if residuals are normally distributed (n<=5000)",
        yx_help,
    );
    reg.add(
        "xl_linreg_andersondarling",
        "QQQ$",
        "LINREG.ANDERSONDARLING",
        "y_range, x_range",
        "LinReg",
        "Anderson-Darling normality test — tail-sensitive normality test on residuals",
        yx_help,
    );
    reg.add(
        "xl_linreg_harveycollier",
        "QQQ$",
        "LINREG.HARVEYCOLLIER",
        "y_range, x_range",
        "LinReg",
        "Harvey-Collier linearity test — tests for functional form misspecification",
        yx_help,
    );

    // Parameterized diagnostics
    reg.add(
        "xl_linreg_rainbow",
        "QQQQ$",
        "LINREG.RAINBOW",
        "y_range, x_range, [fraction]",
        "LinReg",
        "Rainbow linearity test — compares fit on a subset vs full data",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
            "Fraction of data for the center subset (default 0.5)",
        ],
    );
    reg.add(
        "xl_linreg_reset",
        "QQQ$",
        "LINREG.RESET",
        "y_range, x_range",
        "LinReg",
        "RESET specification error test — tests for omitted nonlinear terms (powers 2,3)",
        yx_help,
    );
    reg.add(
        "xl_linreg_durbinwatson",
        "QQQ$",
        "LINREG.DURBINWATSON",
        "y_range, x_range",
        "LinReg",
        "Durbin-Watson test — detects first-order residual autocorrelation",
        yx_help,
    );
    reg.add(
        "xl_linreg_breuschgodfrey",
        "QQQQ$",
        "LINREG.BREUSCHGODFREY",
        "y_range, x_range, [lag_order]",
        "LinReg",
        "Breusch-Godfrey LM test — detects higher-order serial correlation",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
            "Number of lags to test (default 1)",
        ],
    );

    // Influence & multicollinearity diagnostics
    reg.add(
        "xl_linreg_vif",
        "QQQ$",
        "LINREG.VIF",
        "y_range, x_range",
        "LinReg",
        "Variance Inflation Factor — measures multicollinearity per predictor (>10 is high)",
        yx_help,
    );
    reg.add(
        "xl_linreg_cooksdistance",
        "QQQ$",
        "LINREG.COOKSDISTANCE",
        "y_range, x_range",
        "LinReg",
        "Cook's distance — identifies influential observations affecting all coefficients",
        yx_help,
    );
    reg.add(
        "xl_linreg_dffits",
        "QQQ$",
        "LINREG.DFFITS",
        "y_range, x_range",
        "LinReg",
        "DFFITS — measures influence of each observation on its own fitted value",
        yx_help,
    );
    reg.add(
        "xl_linreg_dfbetas",
        "QQQ$",
        "LINREG.DFBETAS",
        "y_range, x_range",
        "LinReg",
        "DFBETAS — measures influence of each observation on each coefficient",
        yx_help,
    );

    // ── Cross Validation ──────────────────────────────────────────────────

    reg.add(
        "xl_linreg_kfoldols",
        "QQQQ$",
        "LINREG.KFOLDOLS",
        "y_range, x_range, [n_folds]",
        "LinReg",
        "K-Fold Cross Validation for OLS — returns mean/std of MSE, RMSE, MAE, R²",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
            "Number of folds (default 5, minimum 2)",
        ],
    );
    reg.add(
        "xl_linreg_kfoldridge",
        "QQQQQQ$",
        "LINREG.KFOLDRIDGE",
        "y_range, x_range, lambda, [n_folds], [standardize]",
        "LinReg",
        "K-Fold Cross Validation for Ridge regression",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
            "Regularization penalty (lambda >= 0)",
            "Number of folds (default 5, minimum 2)",
            "Standardize predictors before fitting? (default TRUE)",
        ],
    );
    reg.add(
        "xl_linreg_kfoldlasso",
        "QQQQQQ$",
        "LINREG.KFOLDLASSO",
        "y_range, x_range, lambda, [n_folds], [standardize]",
        "LinReg",
        "K-Fold Cross Validation for Lasso regression",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
            "Regularization penalty (lambda >= 0)",
            "Number of folds (default 5, minimum 2)",
            "Standardize predictors before fitting? (default TRUE)",
        ],
    );
    reg.add(
        "xl_linreg_kfoldelasticnet",
        "QQQQQQQ$",
        "LINREG.KFOLDELASTICNET",
        "y_range, x_range, lambda, alpha, [n_folds], [standardize]",
        "LinReg",
        "K-Fold Cross Validation for Elastic Net regression",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
            "Regularization penalty (lambda >= 0)",
            "L1/L2 mixing (0=Ridge, 1=Lasso, between=Elastic Net)",
            "Number of folds (default 5, minimum 2)",
            "Standardize predictors before fitting? (default TRUE)",
        ],
    );

    // ── Prediction Intervals ──────────────────────────────────────────────

    reg.add(
        "xl_linreg_predictionintervals",
        "QQQQQ$",
        "LINREG.PREDICTIONINTERVALS",
        "y_range, x_range, new_x_range, [alpha]",
        "LinReg",
        "Prediction intervals — predicted values with lower/upper bounds, SE, and leverage",
        &[
            "Training response variable (single column)",
            "Training predictor variables (one or more columns)",
            "New predictor values to predict at (same number of columns as x_range)",
            "Significance level: 0.05 = 95% PI, 0.01 = 99% PI (default 0.05)",
        ],
    );

    // ── Polynomial & Lambda Path ──────────────────────────────────────────

    reg.add(
        "xl_linreg_polynomial",
        "QQQQQ$",
        "LINREG.POLYNOMIAL",
        "y_range, x_range, [degree], [center]",
        "LinReg",
        "Polynomial regression — fits y = b0 + b1*x + b2*x² + ... + bd*x^d",
        &[
            "Response variable (single column)",
            "Single predictor variable (single column)",
            "Polynomial degree (default 2; degree=1 is simple linear)",
            "Center x before fitting? Reduces multicollinearity (default FALSE)",
        ],
    );
    reg.add(
        "xl_linreg_lambdapath",
        "QQQQQ$",
        "LINREG.LAMBDAPATH",
        "y_range, x_range, [n_lambda], [alpha]",
        "LinReg",
        "Lambda path — generates log-spaced lambda values for Ridge/Lasso/Elastic Net",
        &[
            "Response variable (single column)",
            "Predictor variables (one or more columns)",
            "Number of lambda values to generate (default 100)",
            "L1/L2 mixing for lambda_max calculation (0=Ridge, 1=Lasso; default 1.0)",
        ],
    );

    1
}

/// Called by Excel when the XLL is unloaded.
#[no_mangle]
pub extern "system" fn xlAutoClose() -> i32 {
    1
}

/// Called by Excel's Add-in Manager to get the display name.
#[no_mangle]
pub extern "system" fn xlAddInManagerInfo12(action: *const XLOPER12) -> *mut XLOPER12 {
    if !action.is_null() {
        let oper = unsafe { &*action };
        let is_one = match oper.base_type() {
            XLTYPE_NUM => (unsafe { oper.val.num }) == 1.0,
            XLTYPE_INT => (unsafe { oper.val.w }) == 1,
            _ => false,
        };
        if is_one {
            return Box::into_raw(Box::new(XLOPER12::from_str("LinReg Core")));
        }
    }
    return_xl_error(XLERR_VALUE)
}

// ── xlAutoFree12 — memory management ────────────────────────────────────────

/// Called by Excel after it copies a returned XLOPER12 whose `xltype` has
/// `xlbitDLLFree` set.  We must free all DLL-allocated memory here.
#[no_mangle]
pub extern "system" fn xlAutoFree12(p: *mut XLOPER12) {
    if p.is_null() {
        return;
    }
    unsafe {
        let base = (*p).xltype & 0x0FFF;
        match base {
            XLTYPE_STR => {
                let ptr = (*p).val.str_;
                if !ptr.is_null() {
                    let len = *ptr as usize + 1; // length prefix + chars
                    let _ = Vec::from_raw_parts(ptr, len, len);
                }
            }
            XLTYPE_MULTI => {
                let arr = &*std::ptr::addr_of!((*p).val.array);
                let total = (arr.rows * arr.columns) as usize;
                // Free strings inside array elements
                for i in 0..total {
                    let elem = &*arr.lparray.add(i);
                    if (elem.xltype & 0x0FFF) == XLTYPE_STR {
                        let ptr = elem.val.str_;
                        if !ptr.is_null() {
                            let len = *ptr as usize + 1;
                            let _ = Vec::from_raw_parts(ptr, len, len);
                        }
                    }
                }
                // Free the array of XLOPER12s itself
                let _ = Vec::from_raw_parts(arr.lparray, total, total);
            }
            _ => {}
        }
        // Free the XLOPER12 struct (was Box::into_raw)
        let _ = Box::from_raw(p);
    }
}

// ── UDF implementations ────────────────────────────────────────────────────

/// `=LINREG.VERSION()` — returns the library version string.
#[no_mangle]
pub extern "system" fn xl_linreg_version() -> *mut XLOPER12 {
    let result = XLOPER12::from_str(env!("CARGO_PKG_VERSION"));
    Box::into_raw(Box::new(result))
}

/// `=LINREG.OLS(y_range, x_range)` — runs OLS regression and returns a
/// coefficient table with fit statistics.
///
/// Output format (matches VBA wrapper):
/// ```text
/// | Term          | Coefficient | Std Error | t Stat  | p-Value |
/// | Intercept     | ...         | ...       | ...     | ...     |
/// | X1            | ...         | ...       | ...     | ...     |
/// | ...           |             |           |         |         |
/// | R-squared     | ...         |           |         |         |
/// | Adj R-squared | ...         |           |         |         |
/// | F-statistic   | ...         |           |         |         |
/// | F p-value     | ...         |           |         |         |
/// | MSE           | ...         |           |         |         |
/// | RMSE          | ...         |           |         |         |
/// ```
#[no_mangle]
pub extern "system" fn xl_linreg_ols(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    // 1. Parse y vector
    let y = match xloper_to_f64_vec(y_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    // 2. Parse X matrix into column-major vectors
    let (x_vars, _n_rows, n_cols) = match xloper_to_columns(x_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    // 3. Build variable names: Intercept, X1, X2, ...
    let mut names = vec!["Intercept".to_string()];
    for i in 1..=n_cols {
        names.push(format!("X{}", i));
    }

    // 4. Run OLS
    let result = match ols_regression(&y, &x_vars, &names) {
        Ok(r) => r,
        Err(e) => return return_xl_error(linreg_err_to_xl(&e)),
    };

    // 5. Build output array
    build_ols_output(&result)
}

/// `=LINREG.WLS(y_range, x_range, weights_range)` — runs Weighted Least Squares
/// regression and returns a coefficient table with fit statistics.
///
/// Output format matches OLS but includes residual std error instead of RMSE:
/// ```text
/// | Term          | Coefficient | Std Error | t Stat  | p-Value |
/// | Intercept     | ...         | ...       | ...     | ...     |
/// | X1            | ...         | ...       | ...     | ...     |
/// | ...           |             |           |         |         |
/// | R-squared     | ...         |           |         |         |
/// | Adj R-squared | ...         |           |         |         |
/// | F-statistic   | ...         |           |         |         |
/// | F p-value     | ...         |           |         |         |
/// | Resid Std Err | ...         |           |         |         |
/// | MSE           | ...         |           |         |         |
/// | RMSE          | ...         |           |         |         |
/// ```
#[no_mangle]
pub extern "system" fn xl_linreg_wls(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    weights_range: *const XLOPER12,
) -> *mut XLOPER12 {
    // 1. Parse y vector
    let y = match xloper_to_f64_vec(y_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    // 2. Parse X matrix into column-major vectors
    let (x_vars, _n_rows, _n_cols) = match xloper_to_columns(x_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    // 3. Parse weights vector
    let weights = match xloper_to_f64_vec(weights_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    // 4. Run WLS
    let result = match wls_regression(&y, &x_vars, &weights) {
        Ok(r) => r,
        Err(e) => return return_xl_error(linreg_err_to_xl(&e)),
    };

    // 5. Build output array
    build_wls_output(&result)
}

/// Helper: parse an optional boolean XLOPER12 argument (default = true).
/// Missing/nil -> true.  Numeric: 0 -> false, nonzero -> true.
fn parse_optional_bool(p: *const XLOPER12, default: bool) -> Result<bool, i32> {
    if p.is_null() {
        return Ok(default);
    }
    let oper = unsafe { &*p };
    match oper.base_type() {
        XLTYPE_MISSING | XLTYPE_NIL => Ok(default),
        XLTYPE_NUM => Ok(unsafe { oper.val.num } != 0.0),
        XLTYPE_INT => Ok(unsafe { oper.val.w } != 0),
        XLTYPE_BOOL => Ok(unsafe { oper.val.xbool } != 0),
        XLTYPE_ERR => Err(unsafe { oper.val.err }),
        _ => Err(XLERR_VALUE),
    }
}

/// Helper: parse a required f64 scalar XLOPER12 argument.
fn parse_required_f64(p: *const XLOPER12) -> Result<f64, i32> {
    if p.is_null() {
        return Err(XLERR_VALUE);
    }
    let oper = unsafe { &*p };
    match oper.base_type() {
        XLTYPE_NUM => Ok(unsafe { oper.val.num }),
        XLTYPE_INT => Ok(unsafe { oper.val.w } as f64),
        XLTYPE_MISSING | XLTYPE_NIL => Err(XLERR_VALUE),
        XLTYPE_ERR => Err(unsafe { oper.val.err }),
        _ => Err(XLERR_VALUE),
    }
}

/// Helper: parse an optional f64 scalar XLOPER12 argument with a default.
/// Missing/nil -> default value.
fn parse_optional_f64(p: *const XLOPER12, default: f64) -> Result<f64, i32> {
    if p.is_null() {
        return Ok(default);
    }
    let oper = unsafe { &*p };
    match oper.base_type() {
        XLTYPE_MISSING | XLTYPE_NIL => Ok(default),
        XLTYPE_NUM => Ok(unsafe { oper.val.num }),
        XLTYPE_INT => Ok(unsafe { oper.val.w } as f64),
        XLTYPE_ERR => Err(unsafe { oper.val.err }),
        _ => Err(XLERR_VALUE),
    }
}

/// `=LINREG.RIDGE(y_range, x_range, lambda, [standardize])` — Ridge regression (L2).
///
/// Output format (matches VBA wrapper — 2 columns):
/// ```text
/// | Term          | Coefficient |
/// | Intercept     | ...         |
/// | X1            | ...         |
/// | ...           |             |
/// |               |             |   <- blank separator row
/// | R-squared     | ...         |
/// | Adj R-squared | ...         |
/// | MSE           | ...         |
/// | Lambda        | ...         |
/// | Eff. df       | ...         |
/// ```
#[no_mangle]
pub extern "system" fn xl_linreg_ridge(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    lambda_arg: *const XLOPER12,
    standardize_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let y = match xloper_to_f64_vec(y_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let (x_vars, n_rows, _n_cols) = match xloper_to_columns(x_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let lambda = match parse_required_f64(lambda_arg) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let standardize = match parse_optional_bool(standardize_arg, true) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    let x = columns_to_design_matrix(&x_vars, n_rows);
    let options = RidgeFitOptions {
        lambda,
        standardize,
        intercept: true,
        ..Default::default()
    };

    let result = match ridge_fit(&x, &y, &options) {
        Ok(r) => r,
        Err(e) => return return_xl_error(linreg_err_to_xl(&e)),
    };

    build_ridge_output(&result)
}

/// `=LINREG.LASSO(y_range, x_range, lambda, [standardize])` — Lasso regression (L1).
///
/// Output format (matches VBA wrapper — 2 columns):
/// ```text
/// | Term          | Coefficient |
/// | Intercept     | ...         |
/// | X1            | ...         |
/// | ...           |             |
/// |               |             |   <- blank separator row
/// | R-squared     | ...         |
/// | Adj R-squared | ...         |
/// | MSE           | ...         |
/// | Lambda        | ...         |
/// | Non-zero      | ...         |
/// | Converged     | Yes/No      |
/// ```
#[no_mangle]
pub extern "system" fn xl_linreg_lasso(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    lambda_arg: *const XLOPER12,
    standardize_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let y = match xloper_to_f64_vec(y_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let (x_vars, n_rows, _n_cols) = match xloper_to_columns(x_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let lambda = match parse_required_f64(lambda_arg) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let standardize = match parse_optional_bool(standardize_arg, true) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    let x = columns_to_design_matrix(&x_vars, n_rows);
    let options = LassoFitOptions {
        lambda,
        standardize,
        intercept: true,
        ..Default::default()
    };

    let result = match lasso_fit(&x, &y, &options) {
        Ok(r) => r,
        Err(e) => return return_xl_error(linreg_err_to_xl(&e)),
    };

    build_lasso_output(&result)
}

/// `=LINREG.ELASTICNET(y_range, x_range, lambda, alpha, [standardize])` — Elastic Net (L1+L2).
///
/// Output format (matches VBA wrapper — 2 columns):
/// ```text
/// | Term          | Coefficient |
/// | Intercept     | ...         |
/// | X1            | ...         |
/// | ...           |             |
/// |               |             |   <- blank separator row
/// | R-squared     | ...         |
/// | Adj R-squared | ...         |
/// | MSE           | ...         |
/// | Lambda        | ...         |
/// | Alpha         | ...         |
/// | Non-zero      | ...         |
/// | Converged     | Yes/No      |
/// ```
#[no_mangle]
pub extern "system" fn xl_linreg_elasticnet(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    lambda_arg: *const XLOPER12,
    alpha_arg: *const XLOPER12,
    standardize_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let y = match xloper_to_f64_vec(y_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let (x_vars, n_rows, _n_cols) = match xloper_to_columns(x_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let lambda = match parse_required_f64(lambda_arg) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let alpha = match parse_required_f64(alpha_arg) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let standardize = match parse_optional_bool(standardize_arg, true) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    let x = columns_to_design_matrix(&x_vars, n_rows);
    let options = ElasticNetOptions {
        lambda,
        alpha,
        standardize,
        intercept: true,
        ..Default::default()
    };

    let result = match elastic_net_fit(&x, &y, &options) {
        Ok(r) => r,
        Err(e) => return return_xl_error(linreg_err_to_xl(&e)),
    };

    build_elastic_net_output(&result)
}

// ── Diagnostic UDFs ──────────────────────────────────────────────────────

/// Helper: parse y and X ranges into vectors, returning an error XLOPER12 on failure.
macro_rules! parse_yx {
    ($y_range:expr, $x_range:expr) => {{
        let y = match xloper_to_f64_vec($y_range) {
            Ok(v) => v,
            Err(code) => return return_xl_error(code),
        };
        let (x_vars, _n_rows, _n_cols) = match xloper_to_columns($x_range) {
            Ok(v) => v,
            Err(code) => return return_xl_error(code),
        };
        (y, x_vars)
    }};
}

/// `=LINREG.BREUSCHPAGAN(y_range, x_range)` — Breusch-Pagan heteroscedasticity test.
#[no_mangle]
pub extern "system" fn xl_linreg_breuschpagan(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::breusch_pagan_test(&y, &x_vars) {
        Ok(r) => build_simple_diagnostic(&r),
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.WHITE(y_range, x_range)` — White heteroscedasticity test (R method).
#[no_mangle]
pub extern "system" fn xl_linreg_white(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::r_white_method(&y, &x_vars) {
        Ok(r) => build_diagnostic_2x2("Statistic", r.statistic, "p-Value", r.p_value),
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.JARQUEBERA(y_range, x_range)` — Jarque-Bera normality test.
#[no_mangle]
pub extern "system" fn xl_linreg_jarquebera(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::jarque_bera_test(&y, &x_vars) {
        Ok(r) => build_simple_diagnostic(&r),
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.SHAPIROWILK(y_range, x_range)` — Shapiro-Wilk normality test.
#[no_mangle]
pub extern "system" fn xl_linreg_shapirowilk(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::shapiro_wilk_test(&y, &x_vars) {
        Ok(r) => build_simple_diagnostic(&r),
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.ANDERSONDARLING(y_range, x_range)` — Anderson-Darling normality test.
#[no_mangle]
pub extern "system" fn xl_linreg_andersondarling(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::anderson_darling_test(&y, &x_vars) {
        Ok(r) => build_simple_diagnostic(&r),
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.HARVEYCOLLIER(y_range, x_range)` — Harvey-Collier linearity test.
#[no_mangle]
pub extern "system" fn xl_linreg_harveycollier(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::harvey_collier_test(&y, &x_vars, diagnostics::HarveyCollierMethod::R) {
        Ok(r) => build_simple_diagnostic(&r),
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.RAINBOW(y_range, x_range, [fraction])` — Rainbow linearity test.
///
/// `fraction` defaults to 0.5 if omitted.
#[no_mangle]
pub extern "system" fn xl_linreg_rainbow(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    fraction_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    let fraction = match parse_optional_f64(fraction_arg, 0.5) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    match diagnostics::rainbow_test(&y, &x_vars, fraction, diagnostics::RainbowMethod::R) {
        Ok(output) => {
            if let Some(r) = output.r_result {
                build_diagnostic_2x2("Statistic", r.statistic, "p-Value", r.p_value)
            } else {
                return_xl_error(XLERR_NUM)
            }
        }
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.RESET(y_range, x_range)` — RESET specification error test (powers 2,3).
#[no_mangle]
pub extern "system" fn xl_linreg_reset(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::reset_test(&y, &x_vars, &[2, 3], diagnostics::ResetType::Fitted) {
        Ok(r) => build_simple_diagnostic(&r),
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.DURBINWATSON(y_range, x_range)` — Durbin-Watson autocorrelation test.
///
/// Returns DW statistic and estimated autocorrelation (no p-value).
#[no_mangle]
pub extern "system" fn xl_linreg_durbinwatson(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::durbin_watson_test(&y, &x_vars) {
        Ok(r) => build_diagnostic_2x2(
            "DW Statistic",
            r.statistic,
            "Autocorrelation",
            r.autocorrelation,
        ),
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.BREUSCHGODFREY(y_range, x_range, [lag_order])` — Breusch-Godfrey
/// serial correlation test.  `lag_order` defaults to 1 if omitted.
#[no_mangle]
pub extern "system" fn xl_linreg_breuschgodfrey(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    lag_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    let lag_order = match parse_optional_f64(lag_arg, 1.0) {
        Ok(v) => v as usize,
        Err(code) => return return_xl_error(code),
    };
    let order = lag_order.max(1);
    match diagnostics::breusch_godfrey_test(&y, &x_vars, order, diagnostics::BGTestType::Chisq) {
        Ok(r) => {
            let df = r.df.first().copied().unwrap_or(0.0);
            let mut cells = Vec::with_capacity(6);
            cells.push(XLOPER12::from_str("Statistic"));
            cells.push(XLOPER12::from_f64(r.statistic));
            cells.push(XLOPER12::from_str("p-Value"));
            cells.push(XLOPER12::from_f64(r.p_value));
            cells.push(XLOPER12::from_str("df"));
            cells.push(XLOPER12::from_f64(df));
            build_multi(cells, 3, 2)
        }
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.VIF(y_range, x_range)` — VIF per predictor.
///
/// Output: (p+1) x 2 array: header row + one row per predictor.
#[no_mangle]
pub extern "system" fn xl_linreg_vif(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::vif_test(&y, &x_vars) {
        Ok(r) => {
            let n_vars = r.vif_results.len();
            let n_rows = 1 + n_vars; // header + data
            let mut cells = Vec::with_capacity(n_rows * 2);
            cells.push(XLOPER12::from_str("Variable"));
            cells.push(XLOPER12::from_str("VIF"));
            for (i, v) in r.vif_results.iter().enumerate() {
                cells.push(XLOPER12::from_str(&format!("X{}", i + 1)));
                cells.push(XLOPER12::from_f64(v.vif));
            }
            build_multi(cells, n_rows, 2)
        }
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.COOKSDISTANCE(y_range, x_range)` — Cook's distance per observation.
///
/// Output: (n+1) x 1 array: header + one value per observation.
#[no_mangle]
pub extern "system" fn xl_linreg_cooksdistance(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::cooks_distance_test(&y, &x_vars) {
        Ok(r) => build_vector_column("Cook's D", &r.distances),
        Err(e) => build_vector_error("Cook's D", &e),
    }
}

/// `=LINREG.DFFITS(y_range, x_range)` — DFFITS per observation.
///
/// Output: (n+1) x 1 array: header + one value per observation.
#[no_mangle]
pub extern "system" fn xl_linreg_dffits(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::dffits_test(&y, &x_vars) {
        Ok(r) => build_vector_column("DFFITS", &r.dffits),
        Err(e) => build_vector_error("DFFITS", &e),
    }
}

/// `=LINREG.DFBETAS(y_range, x_range)` — DFBETAS influence matrix.
///
/// Output: (n+1) x (p+1) array: header row + observation rows.
/// Columns: Obs | Intercept | X1 | X2 | ...
#[no_mangle]
pub extern "system" fn xl_linreg_dfbetas(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    match diagnostics::dfbetas_test(&y, &x_vars) {
        Ok(r) => {
            let n = r.n;
            let p = r.p; // includes intercept
            let n_rows = 1 + n; // header + observations
            let n_cols = 1 + p; // Obs column + coefficient columns
            let mut cells = Vec::with_capacity(n_rows * n_cols);

            // Header row
            cells.push(XLOPER12::from_str("Obs"));
            cells.push(XLOPER12::from_str("Intercept"));
            for j in 1..p {
                cells.push(XLOPER12::from_str(&format!("X{}", j)));
            }

            // Data rows
            for i in 0..n {
                cells.push(XLOPER12::from_f64((i + 1) as f64)); // 1-based obs index
                for j in 0..p {
                    cells.push(XLOPER12::from_f64(r.dfbetas[i][j]));
                }
            }

            build_multi(cells, n_rows, n_cols)
        }
        Err(e) => build_matrix_error(&e),
    }
}

// ── Cross Validation UDFs ──────────────────────────────────────────────────

/// `=LINREG.KFOLDOLS(y_range, x_range, [n_folds])` — K-Fold CV for OLS.
///
/// `n_folds` defaults to 5 if omitted.
///
/// Output: 2-column table with Mean/Std for MSE, RMSE, MAE, R², plus summary.
#[no_mangle]
pub extern "system" fn xl_linreg_kfoldols(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    nfolds_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    let n_folds = match parse_optional_f64(nfolds_arg, 5.0) {
        Ok(v) => (v as usize).max(2),
        Err(code) => return return_xl_error(code),
    };

    let mut names = vec!["Intercept".to_string()];
    for i in 0..x_vars.len() {
        names.push(format!("X{}", i + 1));
    }
    let options = KFoldOptions { n_folds, shuffle: false, seed: None };

    match kfold_cv_ols(&y, &x_vars, &names, &options) {
        Ok(r) => build_cv_output(&r),
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.KFOLDRIDGE(y_range, x_range, lambda, [n_folds], [standardize])`
#[no_mangle]
pub extern "system" fn xl_linreg_kfoldridge(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    lambda_arg: *const XLOPER12,
    nfolds_arg: *const XLOPER12,
    standardize_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    let lambda = match parse_required_f64(lambda_arg) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let n_folds = match parse_optional_f64(nfolds_arg, 5.0) {
        Ok(v) => (v as usize).max(2),
        Err(code) => return return_xl_error(code),
    };
    let standardize = match parse_optional_bool(standardize_arg, true) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    let options = KFoldOptions { n_folds, shuffle: false, seed: None };

    match kfold_cv_ridge(&x_vars, &y, lambda, standardize, &options) {
        Ok(r) => build_cv_output(&r),
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.KFOLDLASSO(y_range, x_range, lambda, [n_folds], [standardize])`
#[no_mangle]
pub extern "system" fn xl_linreg_kfoldlasso(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    lambda_arg: *const XLOPER12,
    nfolds_arg: *const XLOPER12,
    standardize_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    let lambda = match parse_required_f64(lambda_arg) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let n_folds = match parse_optional_f64(nfolds_arg, 5.0) {
        Ok(v) => (v as usize).max(2),
        Err(code) => return return_xl_error(code),
    };
    let standardize = match parse_optional_bool(standardize_arg, true) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    let options = KFoldOptions { n_folds, shuffle: false, seed: None };

    match kfold_cv_lasso(&x_vars, &y, lambda, standardize, &options) {
        Ok(r) => build_cv_output(&r),
        Err(e) => build_diagnostic_error(&e),
    }
}

/// `=LINREG.KFOLDELASTICNET(y_range, x_range, lambda, alpha, [n_folds], [standardize])`
#[no_mangle]
pub extern "system" fn xl_linreg_kfoldelasticnet(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    lambda_arg: *const XLOPER12,
    alpha_arg: *const XLOPER12,
    nfolds_arg: *const XLOPER12,
    standardize_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);
    let lambda = match parse_required_f64(lambda_arg) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let alpha = match parse_required_f64(alpha_arg) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let n_folds = match parse_optional_f64(nfolds_arg, 5.0) {
        Ok(v) => (v as usize).max(2),
        Err(code) => return return_xl_error(code),
    };
    let standardize = match parse_optional_bool(standardize_arg, true) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    let options = KFoldOptions { n_folds, shuffle: false, seed: None };

    match kfold_cv_elastic_net(&x_vars, &y, lambda, alpha, standardize, &options) {
        Ok(r) => build_cv_output(&r),
        Err(e) => build_diagnostic_error(&e),
    }
}

// ── Prediction Intervals UDF ──────────────────────────────────────────────

/// `=LINREG.PREDICTIONINTERVALS(y_range, x_range, new_x_range, [alpha])`
///
/// Computes prediction intervals for new observations.
/// `alpha` defaults to 0.05 (95% PI) if omitted.
///
/// Output: (n_new + 3) x 6 array:
/// ```text
/// | Obs | Predicted | Lower    | Upper    | SE      | Leverage |
/// | 1   | ...       | ...      | ...      | ...     | ...      |
/// | ... |           |          |          |         |          |
/// | Alpha        | 0.05  |          |          |         |          |
/// | df Residuals | ...   |          |          |         |          |
/// ```
#[no_mangle]
pub extern "system" fn xl_linreg_predictionintervals(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    new_x_range: *const XLOPER12,
    alpha_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let (y, x_vars) = parse_yx!(y_range, x_range);

    // Parse new X data
    let (new_x_cols, _new_n_rows, new_n_cols) = match xloper_to_columns(new_x_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    // Validate: new_x must have the same number of predictor columns as x
    if new_n_cols != x_vars.len() {
        return return_xl_error(XLERR_VALUE);
    }

    let alpha = match parse_optional_f64(alpha_arg, 0.05) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    // Convert new_x columns to &[&[f64]] as required by prediction_intervals()
    let new_x_refs: Vec<&[f64]> = new_x_cols.iter().map(|c| c.as_slice()).collect();

    match prediction_intervals::prediction_intervals(&y, &x_vars, &new_x_refs, alpha) {
        Ok(r) => build_pi_output(&r),
        Err(e) => build_diagnostic_error(&e),
    }
}

// ── Polynomial & Lambda Path UDFs ─────────────────────────────────────────

/// `=LINREG.POLYNOMIAL(y_range, x_range, [degree], [center])` — Polynomial regression.
///
/// `degree` defaults to 2, `center` defaults to FALSE (0).
/// x_range must be a single column.
///
/// Output: same 5-column OLS table format with polynomial term names
/// (Intercept, x, x^2, x^3, ...).
#[no_mangle]
pub extern "system" fn xl_linreg_polynomial(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    degree_arg: *const XLOPER12,
    center_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let y = match xloper_to_f64_vec(y_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let x = match xloper_to_f64_vec(x_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let degree = match parse_optional_f64(degree_arg, 2.0) {
        Ok(v) => (v as usize).max(1),
        Err(code) => return return_xl_error(code),
    };
    let center = match parse_optional_bool(center_arg, false) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    let options = PolynomialOptions {
        degree,
        center,
        standardize: false,
        intercept: true,
    };

    match polynomial_regression(&y, &x, &options) {
        Ok(fit) => build_polynomial_output(&fit),
        Err(e) => return_xl_error(linreg_err_to_xl(&e)),
    }
}

/// `=LINREG.LAMBDAPATH(y_range, x_range, [n_lambda], [alpha])` — Lambda path.
///
/// Generates a logarithmically-spaced sequence of lambda values for regularization.
/// `n_lambda` defaults to 100, `alpha` defaults to 1.0 (Lasso).
///
/// Output: single column with lambda values (largest to smallest).
#[no_mangle]
pub extern "system" fn xl_linreg_lambdapath(
    y_range: *const XLOPER12,
    x_range: *const XLOPER12,
    nlambda_arg: *const XLOPER12,
    alpha_arg: *const XLOPER12,
) -> *mut XLOPER12 {
    let y = match xloper_to_f64_vec(y_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let (x_vars, n_rows, _n_cols) = match xloper_to_columns(x_range) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };
    let nlambda = match parse_optional_f64(nlambda_arg, 100.0) {
        Ok(v) => (v as usize).max(2),
        Err(code) => return return_xl_error(code),
    };
    let alpha = match parse_optional_f64(alpha_arg, 1.0) {
        Ok(v) => v,
        Err(code) => return return_xl_error(code),
    };

    let x = columns_to_design_matrix(&x_vars, n_rows);
    let options = LambdaPathOptions {
        nlambda,
        alpha,
        ..Default::default()
    };

    let lambdas = make_lambda_path(&x, &y, &options, None, Some(0));
    build_vector_column("Lambda", &lambdas)
}

// ── Output builders ────────────────────────────────────────────────────────

/// Build the OLS result as a multi-cell XLOPER12 array.
fn build_ols_output(r: &RegressionOutput) -> *mut XLOPER12 {
    let n_coefs = r.coefficients.len();
    let n_summary = 6; // R², Adj R², F-stat, F p-value, MSE, RMSE
    let n_rows = 1 + n_coefs + n_summary; // header + coefs + summary
    let n_cols = 5; // Term | Coefficient | Std Error | t Stat | p-Value

    let mut cells: Vec<XLOPER12> = Vec::with_capacity(n_rows * n_cols);

    // Header row
    cells.push(XLOPER12::from_str("Term"));
    cells.push(XLOPER12::from_str("Coefficient"));
    cells.push(XLOPER12::from_str("Std Error"));
    cells.push(XLOPER12::from_str("t Stat"));
    cells.push(XLOPER12::from_str("p-Value"));

    // Coefficient rows
    for i in 0..n_coefs {
        cells.push(XLOPER12::from_str(&r.variable_names[i]));
        cells.push(XLOPER12::from_f64(r.coefficients[i]));
        cells.push(XLOPER12::from_f64(r.std_errors[i]));
        cells.push(XLOPER12::from_f64(r.t_stats[i]));
        cells.push(XLOPER12::from_f64(r.p_values[i]));
    }

    // Summary rows (value in col 1, rest empty)
    let summary_rows: [(&str, f64); 6] = [
        ("R-squared", r.r_squared),
        ("Adj R-squared", r.adj_r_squared),
        ("F-statistic", r.f_statistic),
        ("F p-value", r.f_p_value),
        ("MSE", r.mse),
        ("RMSE", r.rmse),
    ];

    for (label, value) in &summary_rows {
        cells.push(XLOPER12::from_str(label));
        cells.push(XLOPER12::from_f64(*value));
        cells.push(XLOPER12::from_str(""));
        cells.push(XLOPER12::from_str(""));
        cells.push(XLOPER12::from_str(""));
    }

    build_multi(cells, n_rows, n_cols)
}

/// Build the WLS result as a multi-cell XLOPER12 array.
fn build_wls_output(r: &WlsFit) -> *mut XLOPER12 {
    let n_coefs = r.coefficients.len();
    let n_summary = 7; // R², Adj R², F-stat, F p-value, Resid Std Err, MSE, RMSE
    let n_rows = 1 + n_coefs + n_summary; // header + coefs + summary
    let n_cols = 5; // Term | Coefficient | Std Error | t Stat | p-Value

    let mut cells: Vec<XLOPER12> = Vec::with_capacity(n_rows * n_cols);

    // Header row
    cells.push(XLOPER12::from_str("Term"));
    cells.push(XLOPER12::from_str("Coefficient"));
    cells.push(XLOPER12::from_str("Std Error"));
    cells.push(XLOPER12::from_str("t Stat"));
    cells.push(XLOPER12::from_str("p-Value"));

    // Coefficient rows — WLS uses "Intercept", "X1", "X2", ...
    for i in 0..n_coefs {
        let name = if i == 0 {
            "Intercept".to_string()
        } else {
            format!("X{}", i)
        };
        cells.push(XLOPER12::from_str(&name));
        cells.push(XLOPER12::from_f64(r.coefficients[i]));
        cells.push(XLOPER12::from_f64(r.standard_errors[i]));
        cells.push(XLOPER12::from_f64(r.t_statistics[i]));
        cells.push(XLOPER12::from_f64(r.p_values[i]));
    }

    // Summary rows (value in col 1, rest empty)
    let summary_rows: [(&str, f64); 7] = [
        ("R-squared", r.r_squared),
        ("Adj R-squared", r.adj_r_squared),
        ("F-statistic", r.f_statistic),
        ("F p-value", r.f_p_value),
        ("Resid Std Err", r.residual_std_error),
        ("MSE", r.mse),
        ("RMSE", r.rmse),
    ];

    for (label, value) in &summary_rows {
        cells.push(XLOPER12::from_str(label));
        cells.push(XLOPER12::from_f64(*value));
        cells.push(XLOPER12::from_str(""));
        cells.push(XLOPER12::from_str(""));
        cells.push(XLOPER12::from_str(""));
    }

    build_multi(cells, n_rows, n_cols)
}

/// Build a 2-column regularized regression output.
///
/// Shared layout for Ridge/Lasso/Elastic Net (matches VBA wrapper):
///   Row 0:        header ["Term", "Coefficient"]
///   Row 1:        ["Intercept", intercept]
///   Rows 2..p+1:  ["X1"..."Xp", coef]
///   Row p+2:      blank separator
///   Rows p+3+:    model-specific summary stats
fn build_regularized_header(
    intercept: f64,
    coefficients: &[f64],
) -> Vec<XLOPER12> {
    let mut cells = Vec::new();

    // Header row
    cells.push(XLOPER12::from_str("Term"));
    cells.push(XLOPER12::from_str("Coefficient"));

    // Intercept row
    cells.push(XLOPER12::from_str("Intercept"));
    cells.push(XLOPER12::from_f64(intercept));

    // Slope coefficient rows
    for (i, &coef) in coefficients.iter().enumerate() {
        cells.push(XLOPER12::from_str(&format!("X{}", i + 1)));
        cells.push(XLOPER12::from_f64(coef));
    }

    // Blank separator row
    cells.push(XLOPER12::from_str(""));
    cells.push(XLOPER12::from_str(""));

    cells
}

/// Build the Ridge result as a 2-column XLOPER12 array.
fn build_ridge_output(r: &RidgeFit) -> *mut XLOPER12 {
    let n_coefs = r.coefficients.len(); // slopes only
    let mut cells = build_regularized_header(r.intercept, &r.coefficients);

    // Summary stats (5 rows)
    let summary: [(&str, f64); 5] = [
        ("R-squared", r.r_squared),
        ("Adj R-squared", r.adj_r_squared),
        ("MSE", r.mse),
        ("Lambda", r.lambda),
        ("Eff. df", r.df),
    ];
    for (label, value) in &summary {
        cells.push(XLOPER12::from_str(label));
        cells.push(XLOPER12::from_f64(*value));
    }

    // 1 header + 1 intercept + n_coefs slopes + 1 blank + 5 summary
    let n_rows = 1 + 1 + n_coefs + 1 + 5;
    build_multi(cells, n_rows, 2)
}

/// Build the Lasso result as a 2-column XLOPER12 array.
fn build_lasso_output(r: &LassoFit) -> *mut XLOPER12 {
    let n_coefs = r.coefficients.len();
    let mut cells = build_regularized_header(r.intercept, &r.coefficients);

    // Summary stats (6 rows)
    let summary: [(&str, f64); 5] = [
        ("R-squared", r.r_squared),
        ("Adj R-squared", r.adj_r_squared),
        ("MSE", r.mse),
        ("Lambda", r.lambda),
        ("Non-zero", r.n_nonzero as f64),
    ];
    for (label, value) in &summary {
        cells.push(XLOPER12::from_str(label));
        cells.push(XLOPER12::from_f64(*value));
    }
    // Converged as string
    cells.push(XLOPER12::from_str("Converged"));
    cells.push(XLOPER12::from_str(if r.converged { "Yes" } else { "No" }));

    let n_rows = 1 + 1 + n_coefs + 1 + 6;
    build_multi(cells, n_rows, 2)
}

/// Build the Elastic Net result as a 2-column XLOPER12 array.
fn build_elastic_net_output(r: &ElasticNetFit) -> *mut XLOPER12 {
    let n_coefs = r.coefficients.len();
    let mut cells = build_regularized_header(r.intercept, &r.coefficients);

    // Summary stats (7 rows)
    let summary: [(&str, f64); 6] = [
        ("R-squared", r.r_squared),
        ("Adj R-squared", r.adj_r_squared),
        ("MSE", r.mse),
        ("Lambda", r.lambda),
        ("Alpha", r.alpha),
        ("Non-zero", r.n_nonzero as f64),
    ];
    for (label, value) in &summary {
        cells.push(XLOPER12::from_str(label));
        cells.push(XLOPER12::from_f64(*value));
    }
    // Converged as string
    cells.push(XLOPER12::from_str("Converged"));
    cells.push(XLOPER12::from_str(if r.converged { "Yes" } else { "No" }));

    let n_rows = 1 + 1 + n_coefs + 1 + 7;
    build_multi(cells, n_rows, 2)
}

// ── Polynomial output builder ─────────────────────────────────────────────

/// Build the polynomial regression result as a 5-column OLS table.
///
/// Uses the same format as `build_ols_output` but the PolynomialFit
/// already has polynomial term names (Intercept, x, x^2, ...) in its
/// OLS output, so we just forward to the standard OLS builder.
fn build_polynomial_output(fit: &PolynomialFit) -> *mut XLOPER12 {
    build_ols_output(&fit.ols_output)
}

// ── CV output builder ─────────────────────────────────────────────────────

/// Build the CV result as a 2-column XLOPER12 array.
///
/// Layout:
/// ```text
/// | Metric     | Value   |
/// | Mean MSE   | ...     |
/// | Std MSE    | ...     |
/// | Mean RMSE  | ...     |
/// | Std RMSE   | ...     |
/// | Mean MAE   | ...     |
/// | Std MAE    | ...     |
/// | Mean R²    | ...     |
/// | Std R²     | ...     |
/// |            |         |   <- blank separator
/// | n Folds    | ...     |
/// | n Samples  | ...     |
/// | Train R²   | ...     |
/// ```
fn build_cv_output(r: &CVResult) -> *mut XLOPER12 {
    let n_rows = 1 + 8 + 1 + 3; // header + 8 metrics + blank + 3 summary = 13
    let mut cells = Vec::with_capacity(n_rows * 2);

    // Header
    cells.push(XLOPER12::from_str("Metric"));
    cells.push(XLOPER12::from_str("Value"));

    // Metric rows
    let metrics: [(&str, f64); 8] = [
        ("Mean MSE", r.mean_mse),
        ("Std MSE", r.std_mse),
        ("Mean RMSE", r.mean_rmse),
        ("Std RMSE", r.std_rmse),
        ("Mean MAE", r.mean_mae),
        ("Std MAE", r.std_mae),
        ("Mean R²", r.mean_r_squared),
        ("Std R²", r.std_r_squared),
    ];
    for (label, value) in &metrics {
        cells.push(XLOPER12::from_str(label));
        cells.push(XLOPER12::from_f64(*value));
    }

    // Blank separator
    cells.push(XLOPER12::from_str(""));
    cells.push(XLOPER12::from_str(""));

    // Summary
    cells.push(XLOPER12::from_str("n Folds"));
    cells.push(XLOPER12::from_f64(r.n_folds as f64));
    cells.push(XLOPER12::from_str("n Samples"));
    cells.push(XLOPER12::from_f64(r.n_samples as f64));
    cells.push(XLOPER12::from_str("Train R²"));
    cells.push(XLOPER12::from_f64(r.mean_train_r_squared));

    build_multi(cells, n_rows, 2)
}

// ── PI output builder ─────────────────────────────────────────────────────

/// Build the prediction intervals result as a 6-column XLOPER12 array.
///
/// Layout:
/// ```text
/// | Obs | Predicted | Lower | Upper | SE   | Leverage |
/// | 1   | ...       | ...   | ...   | ...  | ...      |
/// | ... |           |       |       |      |          |
/// | Alpha        | 0.05 |       |       |      |          |
/// | df Residuals | ...  |       |       |      |          |
/// ```
fn build_pi_output(r: &prediction_intervals::PredictionIntervalOutput) -> *mut XLOPER12 {
    let n_new = r.predicted.len();
    let n_rows = 1 + n_new + 2; // header + data + 2 summary
    let n_cols = 6;

    let mut cells = Vec::with_capacity(n_rows * n_cols);

    // Header
    cells.push(XLOPER12::from_str("Obs"));
    cells.push(XLOPER12::from_str("Predicted"));
    cells.push(XLOPER12::from_str("Lower"));
    cells.push(XLOPER12::from_str("Upper"));
    cells.push(XLOPER12::from_str("SE"));
    cells.push(XLOPER12::from_str("Leverage"));

    // Data rows
    for i in 0..n_new {
        cells.push(XLOPER12::from_f64((i + 1) as f64));
        cells.push(XLOPER12::from_f64(r.predicted[i]));
        cells.push(XLOPER12::from_f64(r.lower_bound[i]));
        cells.push(XLOPER12::from_f64(r.upper_bound[i]));
        cells.push(XLOPER12::from_f64(r.se_pred[i]));
        cells.push(XLOPER12::from_f64(r.leverage[i]));
    }

    // Summary rows
    cells.push(XLOPER12::from_str("Alpha"));
    cells.push(XLOPER12::from_f64(r.alpha));
    for _ in 2..n_cols { cells.push(XLOPER12::from_str("")); }

    cells.push(XLOPER12::from_str("df Residuals"));
    cells.push(XLOPER12::from_f64(r.df_residuals));
    for _ in 2..n_cols { cells.push(XLOPER12::from_str("")); }

    build_multi(cells, n_rows, n_cols)
}

// ── Diagnostic output builders ────────────────────────────────────────────

/// Build a 2x2 diagnostic output: two labeled rows.
fn build_diagnostic_2x2(
    label1: &str,
    value1: f64,
    label2: &str,
    value2: f64,
) -> *mut XLOPER12 {
    let mut cells = Vec::with_capacity(4);
    cells.push(XLOPER12::from_str(label1));
    cells.push(XLOPER12::from_f64(value1));
    cells.push(XLOPER12::from_str(label2));
    cells.push(XLOPER12::from_f64(value2));
    build_multi(cells, 2, 2)
}

/// Build a 2x2 diagnostic from a `DiagnosticTestResult` (Statistic + p-Value).
fn build_simple_diagnostic(r: &diagnostics::DiagnosticTestResult) -> *mut XLOPER12 {
    build_diagnostic_2x2("Statistic", r.statistic, "p-Value", r.p_value)
}

/// Build a single-column output with a header string and n numeric values.
fn build_vector_column(header: &str, values: &[f64]) -> *mut XLOPER12 {
    let n_rows = 1 + values.len();
    let mut cells = Vec::with_capacity(n_rows);
    cells.push(XLOPER12::from_str(header));
    for &v in values {
        cells.push(XLOPER12::from_f64(v));
    }
    build_multi(cells, n_rows, 1)
}

/// Build a 2x2 diagnostic error array with an error message in the value cell.
///
/// Instead of returning a bare `#VALUE!` error, diagnostic UDFs return the
/// normal labeled layout with the error message as a string:
/// ```text
/// | Statistic | #ERR: residuals near zero |
/// | p-Value   |                           |
/// ```
fn build_diagnostic_error(e: &crate::Error) -> *mut XLOPER12 {
    let msg = format!("#ERR: {}", e);
    let mut cells = Vec::with_capacity(4);
    cells.push(XLOPER12::from_str("Statistic"));
    cells.push(XLOPER12::from_str(&msg));
    cells.push(XLOPER12::from_str("p-Value"));
    cells.push(XLOPER12::from_str(""));
    build_multi(cells, 2, 2)
}

/// Build a vector-column diagnostic error with an error message in the header.
fn build_vector_error(header: &str, e: &crate::Error) -> *mut XLOPER12 {
    let msg = format!("#ERR: {}", e);
    let mut cells = Vec::with_capacity(2);
    cells.push(XLOPER12::from_str(header));
    cells.push(XLOPER12::from_str(&msg));
    build_multi(cells, 2, 1)
}

/// Build a matrix diagnostic error (DFBETAS) with an error message.
fn build_matrix_error(e: &crate::Error) -> *mut XLOPER12 {
    let msg = format!("#ERR: {}", e);
    let mut cells = Vec::with_capacity(2);
    cells.push(XLOPER12::from_str("DFBETAS"));
    cells.push(XLOPER12::from_str(&msg));
    build_multi(cells, 1, 2)
}

// ── Error mapping ──────────────────────────────────────────────────────────

/// Map linreg-core errors to Excel error codes.
fn linreg_err_to_xl(e: &crate::Error) -> i32 {
    use crate::Error;
    match e {
        Error::SingularMatrix => XLERR_NUM,
        Error::InsufficientData { .. } => XLERR_NUM,
        Error::InvalidInput(_) => XLERR_VALUE,
        Error::DimensionMismatch { .. } => XLERR_VALUE,
        Error::ComputationFailed(_) => XLERR_NUM,
        _ => XLERR_NA,
    }
}
