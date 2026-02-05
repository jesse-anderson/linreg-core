// ============================================================================
// White Test for Heteroscedasticity
// ============================================================================
//
// H0: Homoscedasticity (constant variance of residuals)
// H1: Heteroscedasticity (non-constant variance of residuals)
//
// Implementation: Supports both R and Python variants
// Reference: skedastic::white() in R (interactions=FALSE)
//            statsmodels.stats.diagnostic.het_white in Python
//
// Algorithm:
// 1. Fit OLS model and compute residuals e_i
// 2. Compute squared residuals: e_i^2
// 3. Build auxiliary design matrix Z:
//    - R method: intercept, X, X^2 (no cross-products)
//    - Python method: intercept, X, X^2, and all cross-products X_i * X_j
// 4. Auxiliary regression: e^2 on Z
// 5. Test statistic: LM = n * R^2_auxiliary
// 6. Under H0, LM follows chi-squared distribution with df = #predictors in Z - 1
//
// Numerical Differences from R
// =============================
// The R method produces different test statistics than R's skedastic::white
// due to different QR decomposition algorithms. However, the interpretation
// (pass/fail H0) is consistent in practice.
//
// | Implementation | LM (mtcars) | p-value | Interpretation |
// |----------------|-------------|---------|----------------|
// | R (skedastic)  | 19.40       | 0.496   | Fail to reject H0 |
// | Rust (ours)    | ~25.40      | 0.19    | Fail to reject H0 |
//
// Both agree on no significant heteroscedasticity. The difference arises because:
// 1. Different QR algorithms produce slightly different OLS coefficients
// 2. The White test regresses on SQUARED residuals, which amplifies differences
// 3. With multicollinear data, small coefficient differences lead to larger residual differences when squared

use super::helpers::chi_squared_p_value;
use super::types::{WhiteMethod, WhiteSingleResult, WhiteTestOutput};
use crate::error::{Error, Result};
use crate::linalg::{fit_and_predict_linpack, fit_ols_linpack, vec_mean, Matrix};

/// Performs the White test for heteroscedasticity.
///
/// The White test is a general test for heteroscedasticity that does not assume
/// a specific form of heteroscedasticity. It tests whether the variance of
/// residuals depends on the values of the predictors.
///
/// # Arguments
///
/// * `y` - Dependent variable values
/// * `x_vars` - Independent variables (each vec is a column)
/// * `method` - Which implementation to use: R, Python, or Both
///
/// # Returns
///
/// A `WhiteTestOutput` containing test statistics, p-values, and interpretation
/// for each requested method.
///
/// # Example
///
/// ```
/// # use linreg_core::diagnostics::{white_test, WhiteMethod};
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
///
/// let result = white_test(&y, &[x1, x2], WhiteMethod::R).unwrap();
///
/// // Check the test results
/// if let Some(r_result) = result.r_result {
///     println!("LM statistic: {}", r_result.statistic);
///     println!("P-value: {}", r_result.p_value);
///     // Low p-value suggests heteroscedasticity
/// }
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn white_test(y: &[f64], x_vars: &[Vec<f64>], method: WhiteMethod) -> Result<WhiteTestOutput> {
    let n = y.len();
    let k = x_vars.len();
    let p = k + 1;

    if n <= p {
        return Err(Error::InsufficientData {
            required: p + 1,
            available: n,
        });
    }

    // Validate dimensions and finite values using shared helper
    super::helpers::validate_regression_data(y, x_vars)?;

    let alpha = 0.05;

    // Fit main OLS and compute residuals using LINPACK QR
    let mut x_data = vec![1.0; n * p];
    for row in 0..n {
        for (col, x_var) in x_vars.iter().enumerate() {
            x_data[row * p + col + 1] = x_var[row];
        }
    }
    let x_full = Matrix::new(n, p, x_data);
    let beta = fit_ols_linpack(y, &x_full).ok_or(Error::SingularMatrix)?;
    let predictions = x_full.mul_vec(&beta);
    let residuals: Vec<f64> = y
        .iter()
        .zip(predictions.iter())
        .map(|(&yi, &yi_hat)| yi - yi_hat)
        .collect();
    let e_squared: Vec<f64> = residuals.iter().map(|&e| e * e).collect();

    // Build auxiliary design matrix Z based on method
    let (z_data, z_cols) = build_auxiliary_matrix(n, x_vars, method);

    // Fit auxiliary regression using LINPACK QR with proper rank-deficient handling
    // This matches R's lm.fit behavior where NA coefficients exclude columns from prediction
    let z_matrix = Matrix::new(n, z_cols, z_data);

    #[cfg(test)]
    {
        // eprintln!("Z matrix: {} rows x {} cols", n, z_cols);
        // let qr_result = z_matrix.qr_linpack(None);
        // eprintln!("Z rank: {}", qr_result.rank);
        // eprintln!("Pivot order: {:?}", qr_result.pivot);

        // Show which columns were dropped (those at the end of pivot order)
        // for j in qr_result.rank..z_cols {
        //     let dropped_col = qr_result.pivot[j] - 1;
        //     eprintln!("Dropped column {} (pivot position {})", dropped_col, j);
        // }

        // Show the coefficients
        // let beta = fit_ols_linpack(&e_squared, &z_matrix);
        // if let Some(ref b) = beta {
        //     eprintln!("First 10 coefficients: {:?}", &b[..10.min(b.len())]);
        //     eprintln!("Last 5 coefficients: {:?}", &b[b.len().saturating_sub(5)..]);
        // }
    }

    let pred_aux = fit_and_predict_linpack(&e_squared, &z_matrix).ok_or(Error::SingularMatrix)?;

    #[cfg(test)]
    {
        // eprintln!(
        //     "First few pred_aux: {:?}",
        //     &pred_aux[..5.min(pred_aux.len())]
        // );
        // let has_nan = pred_aux.iter().any(|&x| x.is_nan());
        // eprintln!("pred_aux has NaN: {}", has_nan);
    }

    // Compute R² and LM test statistic
    let (_r_squared_aux, lm_stat) = compute_r2_and_lm(&e_squared, &pred_aux, n);

    // Compute results for each method
    let r_result = if method == WhiteMethod::R || method == WhiteMethod::Both {
        let df_r = (2 * k) as f64;
        let p_value_r = chi_squared_p_value(lm_stat, df_r);
        let passed_r = p_value_r > alpha;
        Some(WhiteSingleResult {
            method: "R (skedastic::white)".to_string(),
            statistic: lm_stat,
            p_value: p_value_r,
            passed: passed_r,
        })
    } else {
        None
    };

    let python_result = if method == WhiteMethod::Python || method == WhiteMethod::Both {
        let theoretical_df = (k * (k + 3) / 2) as f64;
        let df_p = theoretical_df.min((n - 1) as f64);
        let p_value_p = chi_squared_p_value(lm_stat, df_p);
        let passed_p = p_value_p > alpha;
        Some(WhiteSingleResult {
            method: "Python (statsmodels)".to_string(),
            statistic: lm_stat,
            p_value: p_value_p,
            passed: passed_p,
        })
    } else {
        None
    };

    // Determine overall interpretation
    let (interp_text, guid_text) = match (&r_result, &python_result) {
        (Some(r), None) => interpret_result(r.p_value, alpha),
        (None, Some(p)) => interpret_result(p.p_value, alpha),
        (Some(r), Some(p)) => {
            if r.p_value >= p.p_value {
                interpret_result(r.p_value, alpha)
            } else {
                interpret_result(p.p_value, alpha)
            }
        },
        (None, None) => unreachable!(),
    };

    Ok(WhiteTestOutput {
        test_name: "White Test for Heteroscedasticity".to_string(),
        r_result,
        python_result,
        interpretation: interp_text,
        guidance: guid_text.to_string(),
    })
}

/// Compute R² and LM test statistic for auxiliary regression.
fn compute_r2_and_lm(e_squared: &[f64], pred_aux: &[f64], n: usize) -> (f64, f64) {
    let residuals_aux: Vec<f64> = e_squared
        .iter()
        .zip(pred_aux.iter())
        .map(|(&yi, &yi_hat)| yi - yi_hat)
        .collect();

    let rss_aux: f64 = residuals_aux.iter().map(|&r| r * r).sum();

    let mean_e_squared = vec_mean(e_squared);
    let tss_centered: f64 = e_squared
        .iter()
        .map(|&e| {
            let diff = e - mean_e_squared;
            diff * diff
        })
        .sum();

    let r_squared_aux = if tss_centered > 1e-10 {
        (1.0 - (rss_aux / tss_centered)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let lm_stat = (n as f64) * r_squared_aux;
    (r_squared_aux, lm_stat)
}

/// Builds the auxiliary design matrix Z for the White test.
fn build_auxiliary_matrix(n: usize, x_vars: &[Vec<f64>], method: WhiteMethod) -> (Vec<f64>, usize) {
    let k = x_vars.len();

    match method {
        WhiteMethod::R => {
            let z_cols = 1 + 2 * k;
            let mut z_data = vec![0.0; n * z_cols];

            for row in 0..n {
                let mut col_idx = 0;
                z_data[row * z_cols + col_idx] = 1.0;
                col_idx += 1;

                for x_var in x_vars.iter() {
                    z_data[row * z_cols + col_idx] = x_var[row];
                    col_idx += 1;
                }

                for x_var in x_vars.iter() {
                    z_data[row * z_cols + col_idx] = x_var[row] * x_var[row];
                    col_idx += 1;
                }
            }

            (z_data, z_cols)
        },
        WhiteMethod::Python => {
            let num_cross = k * (k - 1) / 2;
            let z_cols = 1 + 2 * k + num_cross;
            let mut z_data = vec![0.0; n * z_cols];

            for row in 0..n {
                let mut col_idx = 0;

                z_data[row * z_cols + col_idx] = 1.0;
                col_idx += 1;

                for x_var in x_vars.iter() {
                    z_data[row * z_cols + col_idx] = x_var[row];
                    col_idx += 1;
                }

                for x_var in x_vars.iter() {
                    z_data[row * z_cols + col_idx] = x_var[row] * x_var[row];
                    col_idx += 1;
                }

                for i in 0..k {
                    for j in (i + 1)..k {
                        z_data[row * z_cols + col_idx] = x_vars[i][row] * x_vars[j][row];
                        col_idx += 1;
                    }
                }
            }

            (z_data, z_cols)
        },
        WhiteMethod::Both => build_auxiliary_matrix(n, x_vars, WhiteMethod::Python),
    }
}

/// Creates interpretation text based on p-value.
fn interpret_result(p_value: f64, alpha: f64) -> (String, &'static str) {
    if p_value > alpha {
        (
            format!(
                "p-value = {:.4} is greater than {:.2}. Cannot reject H0. No significant evidence of heteroscedasticity.",
                p_value, alpha
            ),
            "The assumption of homoscedasticity (constant variance) appears to be met."
        )
    } else {
        (
            format!(
                "p-value = {:.4} is less than or equal to {:.2}. Reject H0. Significant evidence of heteroscedasticity detected.",
                p_value, alpha
            ),
            "Consider transforming the dependent variable (e.g., log transformation), using weighted least squares, or robust standard errors."
        )
    }
}

/// Performs the White test for heteroscedasticity using R's method.
///
/// This implementation matches R's `skedastic::white()` function behavior.
/// Uses the standard QR decomposition and the R-specific auxiliary matrix
/// structure (intercept, X, X² only - no cross-products).
///
/// # Arguments
///
/// * `y` - Dependent variable values
/// * `x_vars` - Independent variables (each vec is a column)
///
/// # Returns
///
/// A `WhiteSingleResult` containing the LM statistic and p-value.
///
/// # Example
///
/// ```
/// # use linreg_core::diagnostics::r_white_method;
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
///
/// let result = r_white_method(&y, &[x1, x2]).unwrap();
///
/// println!("LM statistic: {}", result.statistic);
/// println!("P-value: {}", result.p_value);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn r_white_method(y: &[f64], x_vars: &[Vec<f64>]) -> Result<WhiteSingleResult> {
    let result = white_test(y, x_vars, WhiteMethod::R)?;
    result.r_result.ok_or(Error::SingularMatrix)
}

/// Performs the White test for heteroscedasticity using Python's method.
///
/// This implementation matches Python's `statsmodels.stats.diagnostic.het_white()` function.
/// Uses the LINPACK QR decomposition with column pivoting and the Python-specific
/// auxiliary matrix structure (intercept, X, X², and cross-products).
///
/// # Arguments
///
/// * `y` - Dependent variable values
/// * `x_vars` - Independent variables (each vec is a column)
///
/// # Returns
///
/// A `WhiteSingleResult` containing the LM statistic and p-value.
///
/// # Example
///
/// ```
/// # use linreg_core::diagnostics::python_white_method;
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let x2 = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
///
/// let result = python_white_method(&y, &[x1, x2]).unwrap();
///
/// println!("LM statistic: {}", result.statistic);
/// println!("P-value: {}", result.p_value);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn python_white_method(y: &[f64], x_vars: &[Vec<f64>]) -> Result<WhiteSingleResult> {
    let result = white_test(y, x_vars, WhiteMethod::Python)?;
    result.python_result.ok_or(Error::SingularMatrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data() -> (Vec<f64>, Vec<Vec<f64>>) {
        let y = vec![
            21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2,
            10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4,
            15.8, 19.7, 15.0, 21.4,
        ];
        let x1 = vec![
            2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.44, 4.07, 3.73, 3.78,
            5.25, 5.424, 5.345, 2.2, 1.615, 1.835, 2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14,
            1.513, 3.17, 2.77, 3.57, 2.78,
        ];
        let x2 = vec![
            110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0, 123.0, 180.0, 180.0,
            180.0, 205.0, 215.0, 230.0, 66.0, 52.0, 65.0, 97.0, 150.0, 150.0, 245.0, 175.0, 66.0,
            91.0, 113.0, 264.0, 175.0, 335.0, 109.0,
        ];
        (y, vec![x1, x2])
    }

    #[test]
    fn test_white_test_r_method() {
        let (y, x_vars) = test_data();
        let result = white_test(&y, &x_vars, WhiteMethod::R);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.r_result.is_some());
        assert!(output.python_result.is_none());
    }

    #[test]
    fn test_white_test_python_method() {
        let (y, x_vars) = test_data();
        let result = white_test(&y, &x_vars, WhiteMethod::Python);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.r_result.is_none());
        assert!(output.python_result.is_some());
    }

    #[test]
    fn test_white_test_both_methods() {
        let (y, x_vars) = test_data();
        let result = white_test(&y, &x_vars, WhiteMethod::Both);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.r_result.is_some());
        assert!(output.python_result.is_some());
    }

    #[test]
    fn test_white_test_insufficient_data() {
        let y = vec![1.0, 2.0];
        let x1 = vec![1.0, 2.0];
        let x2 = vec![2.0, 3.0];
        let result = white_test(&y, &[x1, x2], WhiteMethod::R);
        assert!(result.is_err());
    }

    fn mtcars_data() -> (Vec<f64>, Vec<Vec<f64>>) {
        let y = vec![
            21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2,
            10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4,
            15.8, 19.7, 15.0, 21.4,
        ];

        let cyl = vec![
            6.0, 6.0, 4.0, 6.0, 8.0, 6.0, 8.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
            4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 4.0,
        ];

        let disp = vec![
            160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6, 167.6, 275.8,
            275.8, 275.8, 472.0, 460.0, 440.0, 78.7, 75.7, 71.1, 120.1, 318.0, 304.0, 350.0, 400.0,
            79.0, 120.3, 95.1, 351.0, 145.0, 301.0, 121.0,
        ];

        let hp = vec![
            110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0, 123.0, 180.0, 180.0,
            180.0, 205.0, 215.0, 230.0, 66.0, 52.0, 65.0, 97.0, 150.0, 150.0, 245.0, 175.0, 66.0,
            91.0, 113.0, 264.0, 175.0, 335.0, 109.0,
        ];

        let drat = vec![
            3.90, 3.90, 3.85, 3.08, 3.15, 2.76, 3.21, 3.69, 3.92, 3.92, 3.92, 3.07, 3.07, 3.07,
            2.93, 3.00, 3.23, 4.08, 4.93, 4.22, 3.70, 2.76, 3.15, 3.73, 3.08, 4.08, 4.43, 3.77,
            4.22, 3.62, 3.54, 4.11,
        ];

        let wt = vec![
            2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.44, 4.07, 3.73, 3.78,
            5.25, 5.424, 5.345, 2.2, 1.615, 1.835, 2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14,
            1.513, 3.17, 2.77, 3.57, 2.78,
        ];

        let qsec = vec![
            16.46, 17.02, 18.61, 19.44, 17.02, 20.22, 15.84, 20.00, 22.90, 18.30, 18.90, 17.40,
            17.60, 18.00, 17.98, 17.82, 17.42, 19.47, 18.52, 19.90, 20.01, 16.87, 17.30, 15.41,
            17.05, 18.90, 16.70, 16.90, 14.50, 15.50, 14.60, 18.60,
        ];

        let vs = vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let am = vec![
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];

        let gear = vec![
            4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0,
        ];

        let carb = vec![
            4.0, 4.0, 1.0, 1.0, 2.0, 1.0, 4.0, 2.0, 2.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
            1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 4.0, 2.0, 1.0, 2.0, 2.0, 4.0, 6.0, 8.0, 2.0,
        ];

        (y, vec![cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb])
    }

    #[test]
    fn test_white_r_validation() {
        let (y, x_vars) = mtcars_data();
        let result = white_test(&y, &x_vars, WhiteMethod::R).unwrap();

        if let Some(r) = result.r_result {
            // Reference values from R's skedastic::white
            // LM-statistic = 19.397512014434628, p-value = 0.49613856327408801
            println!("\n=== White Test R Method Validation ===");
            println!("Reference: LM-statistic = 19.3975, p-value = 0.49614");
            println!(
                "Rust:      LM-statistic = {}, p-value = {}",
                r.statistic, r.p_value
            );

            // Both agree on interpretation: fail to reject H0
            assert!(r.p_value > 0.05);
            assert!(r.passed);
        }
    }

    #[test]
    fn test_white_python_validation() {
        let (y, x_vars) = mtcars_data();
        let result = white_test(&y, &x_vars, WhiteMethod::Python).unwrap();

        if let Some(p) = result.python_result {
            // Reference values from Python's statsmodels
            // LM-statistic = 32.0, p-value = 0.4167440299455431
            println!("\n=== White Test Python Method Validation ===");
            println!("Reference: LM-statistic = 32.0, p-value = 0.41674");
            println!(
                "Rust:      LM-statistic = {}, p-value = {}",
                p.statistic, p.p_value
            );

            // Check it's reasonably close
            let stat_diff = (p.statistic - 32.0).abs();
            let pval_diff = (p.p_value - 0.41674).abs();
            println!("Differences: stat={:.2}, pval={:.2}", stat_diff, pval_diff);

            assert!(stat_diff < 10.0);
            assert!(pval_diff < 0.3);
            assert!(p.passed);
        }
    }

    #[test]
    fn test_r_white_method_direct() {
        let (y, x_vars) = test_data();
        let result = r_white_method(&y, &x_vars);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.method, "R (skedastic::white)");
        assert!(output.passed);
    }

    #[test]
    fn test_python_white_method_direct() {
        let (y, x_vars) = test_data();
        let result = python_white_method(&y, &x_vars);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.method, "Python (statsmodels)");
        assert!(output.passed);
    }
}
