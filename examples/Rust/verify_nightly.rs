// Verifies every golden value asserted in .github/workflows/nightly.yml.
// Run with: cargo run --example verify_nightly --no-default-features

use linreg_core::core::ols_regression;
use linreg_core::distributions::{chi_squared_survival, normal_cdf, student_t_cdf};
use linreg_core::linalg::Matrix;
use linreg_core::regularized::{
    elastic_net_fit, lasso_fit, ridge_fit, ElasticNetOptions, LassoFitOptions, RidgeFitOptions,
};

fn check(label: &str, actual: f64, expected: f64, tol: f64) {
    let diff = (actual - expected).abs();
    if diff <= tol {
        println!("  PASS  {:<40}  actual={:.16}  expected={:.16}  diff={:.2e}", label, actual, expected, diff);
    } else {
        println!("  FAIL  {:<40}  actual={:.16}  expected={:.16}  diff={:.2e}  tol={:.2e}", label, actual, expected, diff, tol);
        std::process::exit(1);
    }
}

fn check_gt(label: &str, actual: f64, min: f64) {
    if actual > min {
        println!("  PASS  {:<40}  actual={:.16}  > {}", label, actual, min);
    } else {
        println!("  FAIL  {:<40}  actual={:.16}  not > {}", label, actual, min);
        std::process::exit(1);
    }
}

fn main() {
    println!("\n=== Verifying nightly.yml golden values ===\n");

    // ── OLS simple linear ──────────────────────────────────────────────────────
    println!("--- OLS (simple linear) ---");
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let names = vec!["Intercept".to_string(), "X1".to_string()];
    let r = ols_regression(&y, &x, &names).expect("OLS failed");
    println!("  raw intercept  = {:.16}", r.coefficients[0]);
    println!("  raw slope      = {:.16}", r.coefficients[1]);
    println!("  raw r_squared  = {:.16}", r.r_squared);
    check("intercept",  r.coefficients[0], 1.66,                1e-10);
    check("slope",      r.coefficients[1], 0.9,                 1e-10);
    check("R-squared",  r.r_squared,       0.9839650145772594,  1e-8);
    println!();

    // ── OLS housing dataset ────────────────────────────────────────────────────
    println!("--- OLS (housing dataset) ---");
    let y2 = vec![245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1];
    let x2 = vec![
        vec![1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0],
        vec![3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0],
    ];
    let names2 = vec!["Intercept".into(), "SqFt".into(), "Bed".into()];
    let r2 = ols_regression(&y2, &x2, &names2).expect("OLS housing failed");
    println!("  raw intercept  = {:.16}", r2.coefficients[0]);
    println!("  raw SqFt       = {:.16}", r2.coefficients[1]);
    println!("  raw Bed        = {:.16}", r2.coefficients[2]);
    println!("  raw r_squared  = {:.16}", r2.r_squared);
    check("housing intercept", r2.coefficients[0], 15.6480854, 1e-4);
    check("housing SqFt",      r2.coefficients[1],  0.1638012, 1e-4);
    check("housing Bed",       r2.coefficients[2],  4.8496809, 1e-4);
    check_gt("housing R²",     r2.r_squared, 0.95);
    println!();

    // ── Ridge ──────────────────────────────────────────────────────────────────
    println!("--- Ridge Regression ---");
    let x_mat = Matrix::new(10, 3, {
        let mut data = Vec::new();
        for i in 0..10 {
            data.push(1.0);
            data.push(x2[0][i]);
            data.push(x2[1][i]);
        }
        data
    });
    let ridge = ridge_fit(&x_mat, &y2, &RidgeFitOptions {
        lambda: 1.0,
        standardize: true,
        intercept: true,
        ..Default::default()
    }).expect("Ridge failed");
    println!("  raw r_squared  = {:.16}", ridge.r_squared);
    check_gt("Ridge R²", ridge.r_squared, 0.90);
    println!();

    // ── Lasso ──────────────────────────────────────────────────────────────────
    println!("--- Lasso Regression ---");
    let lasso = lasso_fit(&x_mat, &y2, &LassoFitOptions {
        lambda: 0.1,
        standardize: true,
        intercept: true,
        ..Default::default()
    }).expect("Lasso failed");
    println!("  converged      = {}", lasso.converged);
    println!("  raw r_squared  = {:.16}", lasso.r_squared);
    if !lasso.converged { println!("  FAIL  Lasso did not converge"); std::process::exit(1); }
    check_gt("Lasso R²", lasso.r_squared, 0.85);
    println!();

    // ── Elastic Net ────────────────────────────────────────────────────────────
    println!("--- Elastic Net ---");
    let enet = elastic_net_fit(&x_mat, &y2, &ElasticNetOptions {
        lambda: 0.1,
        alpha: 0.5,
        standardize: true,
        intercept: true,
        ..Default::default()
    }).expect("Elastic Net failed");
    println!("  converged      = {}", enet.converged);
    println!("  raw r_squared  = {:.16}", enet.r_squared);
    if !enet.converged { println!("  FAIL  Elastic Net did not converge"); std::process::exit(1); }
    check_gt("Elastic Net R²", enet.r_squared, 0.85);
    println!();

    // ── Statistical Distributions ──────────────────────────────────────────────
    println!("--- Statistical Distributions ---");
    let chi2_cdf = 1.0 - chi_squared_survival(5.991, 2.0);
    println!("  raw t_cdf(1.96, 20)     = {:.16}", student_t_cdf(1.96, 20.0));
    println!("  raw normal_cdf(1.96)    = {:.16}", normal_cdf(1.96));
    println!("  raw normal_cdf(0.0)     = {:.16}", normal_cdf(0.0));
    println!("  raw chi2_cdf(5.991, 2)  = {:.16}", chi2_cdf);
    check("t CDF(1.96, df=20)",    student_t_cdf(1.96, 20.0), 0.9681,  1e-3);
    check("Normal CDF(1.96)",      normal_cdf(1.96),           0.97500, 1e-4);
    check("Normal CDF(0.0)",       normal_cdf(0.0),            0.5,     1e-8);
    check("Chi² CDF(5.991, df=2)", chi2_cdf,                   0.9500,  1e-3);
    println!();

    println!("=== All nightly golden values verified ===\n");
}
