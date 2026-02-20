//! Weighted Least Squares (WLS) regression example.
//!
//! Run with:
//!   cargo run --example weighted_regression --no-default-features
//!
//! Demonstrates:
//! - Why OLS fails on heteroscedastic data (variance grows with the predictor)
//! - How WLS corrects for this by down-weighting high-variance observations
//! - Using precision weights (w = 1/variance) as the theoretically correct choice
//! - Comparing OLS vs WLS coefficients and standard errors side-by-side

use linreg_core::core::ols_regression;
use linreg_core::weighted_regression::wls_regression;

fn main() {
    // ── Dataset: income ($k) vs. spending ($k) ─────────────────────────────────
    // Variance in spending grows with income — classic heteroscedasticity.
    // Low-income households spend predictably; high-income households vary widely.
    let income  = vec![ 20.0,  25.0,  30.0,  35.0,  40.0,  50.0,  60.0,  75.0,  90.0, 110.0];
    let spending = vec![ 18.0,  21.0,  25.0,  27.0,  31.0,  38.0,  50.0,  55.0,  72.0,  95.0];

    // Approximate standard deviations: higher income -> higher variance.
    // weights = 1 / variance = 1 / sd²  (precision weighting)
    let std_devs = vec![1.0, 1.2, 1.5, 2.0, 2.5, 3.5, 5.0, 7.0, 9.0, 12.0];
    let weights: Vec<f64> = std_devs.iter().map(|s| 1.0 / (s * s)).collect();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║            WEIGHTED LEAST SQUARES (WLS) REGRESSION                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Dataset: income vs. spending (10 households)");
    println!("Heteroscedasticity: spending variance grows with income.");
    println!();

    // ── 1. Show the data and weights ───────────────────────────────────────────
    println!("━━━ 1. Data and Precision Weights ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {:>8}  {:>10}  {:>8}  {:>8}", "Income", "Spending", "Std Dev", "Weight");
    println!("  {}", "─".repeat(42));
    for i in 0..10 {
        println!(
            "  {:>8.1}  {:>10.1}  {:>8.1}  {:>8.4}",
            income[i], spending[i], std_devs[i], weights[i]
        );
    }
    println!();
    println!("  Weights = 1/variance. Low-income obs get high weight (more reliable).");
    println!("  High-income obs get low weight (noisy, less influential).");
    println!();

    // ── 2. OLS fit (ignores heteroscedasticity) ────────────────────────────────
    println!("━━━ 2. OLS Regression (ignores heteroscedasticity) ━━━━━━━━━━━━━━━━━━");
    let names = vec!["Intercept".to_string(), "Income".to_string()];
    let ols = ols_regression(&spending, &[income.clone()], &names)
        .expect("OLS failed");

    println!("  Intercept:  {:.4}  (SE: {:.4})", ols.coefficients[0], ols.std_errors[0]);
    println!("  Income:     {:.4}  (SE: {:.4})", ols.coefficients[1], ols.std_errors[1]);
    println!("  R²:         {:.4}", ols.r_squared);
    println!("  F-stat:     {:.4}  (p = {:.6})", ols.f_statistic, ols.f_p_value);
    println!("  MSE:        {:.4}", ols.mse);
    println!();
    println!("  Problem: OLS gives equal weight to all observations. The noisy");
    println!("  high-income points pull the line and inflate standard errors.");
    println!();

    // ── 3. WLS fit (precision-weighted) ───────────────────────────────────────
    println!("━━━ 3. WLS Regression (precision-weighted, 1/variance) ━━━━━━━━━━━━━━");
    let wls = wls_regression(&spending, &[income.clone()], &weights)
        .expect("WLS failed");

    println!("  Intercept:  {:.4}  (SE: {:.4})", wls.coefficients[0], wls.standard_errors[0]);
    println!("  Income:     {:.4}  (SE: {:.4})", wls.coefficients[1], wls.standard_errors[1]);
    println!("  R²:         {:.4}", wls.r_squared);
    println!("  F-stat:     {:.4}  (p = {:.6})", wls.f_statistic, wls.f_p_value);
    println!("  Residual SE:{:.4}", wls.residual_std_error);
    println!();
    println!("  Fix: WLS down-weights noisy high-income observations. The fit is");
    println!("  driven by the reliable low-income points, tightening standard errors.");
    println!();

    // ── 4. Side-by-side comparison ────────────────────────────────────────────
    println!("━━━ 4. OLS vs WLS Comparison ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {:20}  {:>12}  {:>12}", "", "OLS", "WLS");
    println!("  {}", "─".repeat(48));
    println!("  {:20}  {:>12.4}  {:>12.4}", "Intercept",      ols.coefficients[0], wls.coefficients[0]);
    println!("  {:20}  {:>12.4}  {:>12.4}", "Intercept SE",   ols.std_errors[0],   wls.standard_errors[0]);
    println!("  {:20}  {:>12.4}  {:>12.4}", "Income slope",   ols.coefficients[1], wls.coefficients[1]);
    println!("  {:20}  {:>12.4}  {:>12.4}", "Income slope SE",ols.std_errors[1],   wls.standard_errors[1]);
    println!("  {:20}  {:>12.4}  {:>12.4}", "R²",             ols.r_squared,       wls.r_squared);
    println!("  {:20}  {:>12.4}  {:>12.4}", "MSE",            ols.mse,             wls.residual_std_error.powi(2));
    println!();

    // ── 5. Fitted values comparison ───────────────────────────────────────────
    println!("━━━ 5. Fitted Values vs Actual ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
             "Income", "Actual", "OLS fit", "WLS fit", "Weight");
    println!("  {}", "─".repeat(54));
    for i in 0..10 {
        let ols_fit = ols.coefficients[0] + ols.coefficients[1] * income[i];
        println!(
            "  {:>8.1}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.4}",
            income[i], spending[i], ols_fit, wls.fitted_values[i], weights[i]
        );
    }
    println!();

    // ── 6. Equal weights = OLS (sanity check) ────────────────────────────────
    println!("━━━ 6. Sanity Check: Equal Weights Reproduces OLS ━━━━━━━━━━━━━━━━━━━");
    let equal_weights = vec![1.0; 10];
    let wls_equal = wls_regression(&spending, &[income.clone()], &equal_weights)
        .expect("WLS equal weights failed");
    println!("  OLS intercept:        {:.6}", ols.coefficients[0]);
    println!("  WLS (equal) intercept:{:.6}", wls_equal.coefficients[0]);
    println!("  OLS slope:            {:.6}", ols.coefficients[1]);
    println!("  WLS (equal) slope:    {:.6}", wls_equal.coefficients[1]);
    let max_diff = (ols.coefficients[0] - wls_equal.coefficients[0]).abs()
        .max((ols.coefficients[1] - wls_equal.coefficients[1]).abs());
    println!("  Max coefficient difference: {:.2e}", max_diff);
    println!("  WLS with equal weights matches OLS: {}", max_diff < 1e-8);
}
