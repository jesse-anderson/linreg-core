// ============================================================================
// Breusch-Pagan Test for Heteroscedasticity
// ============================================================================
//
// H0: Homoscedasticity (constant variance of residuals)
// H1: Heteroscedasticity (non-constant variance of residuals)
//
// Implementation: Studentized (Koenker) variant
// Reference: lmtest::bptest(model, studentize=TRUE) in R
//            statsmodels.stats.diagnostic.het_breuschpagan in Python
//
// Algorithm:
// 1. Fit OLS model and compute residuals e_i
// 2. Compute sigma² = RSS / n, where RSS = sum of squared residuals
// 3. Compute normalized squared residuals: u_i = e_i² / sigma²
// 4. Compute mean of u: ū = sum(u_i) / n
// 5. Auxiliary regression: (u_i / ū) on original predictors X
// 6. Test statistic: LM = n * R²_auxiliary
// 7. Under H0, LM follows chi-squared distribution with df = #predictors

use crate::error::{Error, Result};
use crate::linalg::{Matrix, vec_mean};
use super::types::DiagnosticTestResult;
use super::helpers::{fit_ols, chi_squared_p_value};

/// Performs the Breusch-Pagan test for heteroscedasticity.
///
/// This test checks whether the variance of the residuals is constant across
/// all levels of the independent variables (homoscedasticity). The studentized
/// (Koenker) variant is more robust than the original Breusch-Pagan formulation.
///
/// # Arguments
///
/// * `y` - Dependent variable values
/// * `x_vars` - Independent variables (each vec is a column)
///
/// # Returns
///
/// A [`DiagnosticTestResult`] containing the test statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if n ≤ k + 2.
///
/// # Reference
///
/// - Breusch, T. S., & Pagan, A. R. (1979). "A Simple Test for Heteroscedasticity
///   and Random Coefficient Variation". Econometrica, 47(5), 1287-1294.
/// - Koenker, R. (1981). "A Note on Studentizing a Test for Heteroscedasticity".
///   Journal of Econometrics, 17(1), 107-112.
pub fn breusch_pagan_test(
    y: &[f64],
    x_vars: &[Vec<f64>],
) -> Result<DiagnosticTestResult> {
    let n = y.len();
    let k = x_vars.len();  // number of non-intercept predictors
    let p = k + 1;         // total parameters including intercept

    // Validate inputs - need at least p + 1 observations
    if n <= p {
        return Err(Error::InsufficientData { required: p + 1, available: n });
    }

    // Validate dimensions and finite values using shared helper
    super::helpers::validate_regression_data(y, x_vars)?;

    // Create design matrix with intercept
    let mut x_data = vec![1.0; n * p];
    for row in 0..n {
        x_data[row * p] = 1.0;  // intercept
        for (col, x_var) in x_vars.iter().enumerate() {
            x_data[row * p + col + 1] = x_var[row];
        }
    }
    let x_full = Matrix::new(n, p, x_data);

    // Fit OLS on full data
    let beta = fit_ols(y, &x_full)?;

    // Compute residuals and RSS
    let predictions = x_full.mul_vec(&beta);
    let residuals: Vec<f64> = y.iter().zip(predictions.iter())
        .map(|(&yi, &yi_hat)| yi - yi_hat)
        .collect();

    // RSS = sum of squared residuals
    let rss: f64 = residuals.iter().map(|&r| r * r).sum();

    // sigma² = RSS / n (note: using n, not n-p, for BP test)
    let sigma2 = rss / (n as f64);

    // Compute normalized squared residuals: u_i = e_i² / sigma²
    let u: Vec<f64> = residuals.iter().map(|&e| (e * e) / sigma2).collect();

    // Compute mean of u
    let u_mean = vec_mean(&u);

    if u_mean <= 0.0 || !u_mean.is_finite() {
        return Err(Error::InvalidInput("Invalid mean of normalized squared residuals".to_string()));
    }

    // Auxiliary regression: (u_i / u_mean) on X
    // Dependent variable for auxiliary regression
    let y_aux: Vec<f64> = u.iter().map(|&ui| ui / u_mean).collect();

    // Fit auxiliary OLS
    let beta_aux = fit_ols(&y_aux, &x_full)?;

    // Compute predictions and R² for auxiliary regression
    let pred_aux = x_full.mul_vec(&beta_aux);
    let mean_y_aux = vec_mean(&y_aux);

    // Total sum of squares (TSS) and regression sum of squares (RegSS)
    let mut tss_aux = 0.0;
    let mut regss_aux = 0.0;
    for i in 0..n {
        let diff = y_aux[i] - mean_y_aux;
        tss_aux += diff * diff;
        let pred_diff = pred_aux[i] - mean_y_aux;
        regss_aux += pred_diff * pred_diff;
    }

    // R² = RegSS / TSS
    let r_squared_aux = if tss_aux > 1e-10 {
        regss_aux / tss_aux
    } else {
        0.0
    };

    // Clamp R² to [0, 1]
    let r_squared_aux = r_squared_aux.clamp(0.0, 1.0);

    // LM test statistic: n * R²_auxiliary
    let lm_stat = (n as f64) * r_squared_aux;

    // Degrees of freedom: number of predictors (excluding intercept)
    let df = k as f64;

    // P-value from chi-squared distribution (upper tail)
    let p_value = chi_squared_p_value(lm_stat, df);

    let alpha = 0.05;
    let passed = p_value > alpha;

    let (interpretation, guidance) = if passed {
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
    };

    Ok(DiagnosticTestResult {
        test_name: "Breusch-Pagan Test for Heteroscedasticity".to_string(),
        statistic: lm_stat,
        p_value,
        passed,
        interpretation,
        guidance: guidance.to_string(),
    })
}
