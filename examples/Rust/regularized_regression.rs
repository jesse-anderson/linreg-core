//! Regularized regression example: Ridge, Lasso, and Elastic Net.
//!
//! Run with:
//!   cargo run --example regularized_regression --no-default-features
//!
//! Demonstrates:
//! - Ridge regression (L2 penalty) for handling multicollinearity
//! - Lasso regression (L1 penalty) for automatic variable selection
//! - Elastic Net (L1 + L2) as a blend of both
//! - Lambda path generation for exploring regularization strength
//! - Comparing all three methods on the same dataset

use linreg_core::linalg::Matrix;
use linreg_core::regularized::{
    elastic_net_fit, lasso_fit, ridge_fit, ElasticNetOptions, LassoFitOptions, RidgeFitOptions,
};
use linreg_core::regularized::path::{make_lambda_path, LambdaPathOptions};

fn main() {
    // ── Dataset: Boston-style housing (10 observations, 3 predictors) ──────────
    // y  = house price ($k)
    // x1 = square footage (hundreds)
    // x2 = number of bedrooms
    // x3 = age of house (years)
    let y = vec![245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1];
    let x1 = vec![12.0, 18.0, 9.5, 24.0, 14.5, 20.0, 11.0, 28.0, 13.5, 16.5];
    let x2 = vec![3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0];
    let x3 = vec![15.0, 8.0, 30.0, 5.0, 12.0, 3.0, 25.0, 2.0, 18.0, 10.0];

    // Build design matrix (n=10, p=3 cols: intercept + x1 + x2 + x3)
    let mut data = Vec::with_capacity(10 * 4);
    for i in 0..10 {
        data.push(1.0);    // intercept column
        data.push(x1[i]);
        data.push(x2[i]);
        data.push(x3[i]);
    }
    let x_mat = Matrix::new(10, 4, data);

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              REGULARIZED REGRESSION COMPARISON                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Dataset: 10 houses, predictors: SqFt (hundreds), Bedrooms, Age");
    println!();

    // ── 1. Ridge Regression ────────────────────────────────────────────────────
    println!("━━━ 1. Ridge Regression (L2 penalty — shrinks all coefficients) ━━━━━━");
    let ridge = ridge_fit(&x_mat, &y, &RidgeFitOptions {
        lambda: 1.0,
        standardize: true,
        intercept: true,
        ..Default::default()
    }).expect("Ridge failed");

    println!("  Lambda:        1.0");
    println!("  Intercept:     {:.4}", ridge.intercept);
    println!("  SqFt coef:     {:.4}", ridge.coefficients[0]);
    println!("  Bedrooms coef: {:.4}", ridge.coefficients[1]);
    println!("  Age coef:      {:.4}", ridge.coefficients[2]);
    println!("  R²:            {:.4}", ridge.r_squared);
    println!("  MSE:           {:.4}", ridge.mse);
    println!("  AIC:           {:.4}", ridge.aic);
    println!("  Note: All coefficients shrunk toward zero but none zeroed out.");
    println!();

    // ── 2. Lasso Regression ────────────────────────────────────────────────────
    println!("━━━ 2. Lasso Regression (L1 penalty — zeros out weak predictors) ━━━━━");
    let lasso = lasso_fit(&x_mat, &y, &LassoFitOptions {
        lambda: 0.5,
        standardize: true,
        intercept: true,
        ..Default::default()
    }).expect("Lasso failed");

    println!("  Lambda:        0.5");
    println!("  Intercept:     {:.4}", lasso.intercept);
    println!("  SqFt coef:     {:.4}", lasso.coefficients[0]);
    println!("  Bedrooms coef: {:.4}", lasso.coefficients[1]);
    println!("  Age coef:      {:.4}", lasso.coefficients[2]);
    println!("  R²:            {:.4}", lasso.r_squared);
    println!("  MSE:           {:.4}", lasso.mse);
    println!("  AIC:           {:.4}", lasso.aic);
    println!("  Non-zero coefs: {}/{}", lasso.n_nonzero, lasso.coefficients.len());
    println!("  Converged:     {}", lasso.converged);
    println!("  Note: Coefficients exactly zero = variable excluded from model.");
    println!();

    // ── 3. Elastic Net ─────────────────────────────────────────────────────────
    println!("━━━ 3. Elastic Net (alpha=0.5, equal L1+L2 mix) ━━━━━━━━━━━━━━━━━━━━━");
    let enet = elastic_net_fit(&x_mat, &y, &ElasticNetOptions {
        lambda: 0.5,
        alpha: 0.5,
        standardize: true,
        intercept: true,
        ..Default::default()
    }).expect("Elastic Net failed");

    println!("  Lambda:        0.5  Alpha: 0.5");
    println!("  Intercept:     {:.4}", enet.intercept);
    println!("  SqFt coef:     {:.4}", enet.coefficients[0]);
    println!("  Bedrooms coef: {:.4}", enet.coefficients[1]);
    println!("  Age coef:      {:.4}", enet.coefficients[2]);
    println!("  R²:            {:.4}", enet.r_squared);
    println!("  MSE:           {:.4}", enet.mse);
    println!("  AIC:           {:.4}", enet.aic);
    println!("  Non-zero coefs: {}/{}", enet.n_nonzero, enet.coefficients.len());
    println!("  Converged:     {}", enet.converged);
    println!("  Note: Blends Ridge's grouping effect with Lasso's sparsity.");
    println!();

    // ── 4. Lambda Path — how coefficients evolve with regularization ───────────
    // Use make_lambda_path to find lambda_max, then build a readable descending
    // sequence from just below lambda_max down to a small value.
    println!("━━━ 4. Lambda Path (Lasso — how coefficients shrink) ━━━━━━━━━━━━━━━━━");
    let path_opts = LambdaPathOptions {
        nlambda: 400,
        lambda_min_ratio: Some(0.0001),
        alpha: 1.0,
        eps_for_ridge: 0.0001,
    };
    let full_path = make_lambda_path(&x_mat, &y, &path_opts, None, Some(0));
    // Skip the leading inf. Sample evenly across the full finite range so the
    // table spans from near-zero coefficients (high lambda) to near-OLS (low lambda).
    let finite: Vec<f64> = full_path.into_iter().filter(|v| v.is_finite()).collect();
    let n_display = 20;
    let step = (finite.len() - 1) / (n_display - 1);
    // Reverse so the table reads high-lambda -> low-lambda (shrinkage direction)
    let display: Vec<f64> = (0..n_display)
        .map(|i| finite[i * step])
        .collect();

    println!("  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}", "Lambda", "SqFt", "Bedrooms", "Age", "R²");
    println!("  {}", "─".repeat(56));
    for lambda in &display {
        if let Ok(f) = lasso_fit(&x_mat, &y, &LassoFitOptions {
            lambda: *lambda,
            standardize: true,
            intercept: true,
            ..Default::default()
        }) {
            println!(
                "  {:>10.2}  {:>10.4}  {:>10.4}  {:>10.4}  {:>8.4}",
                lambda, f.coefficients[0], f.coefficients[1], f.coefficients[2], f.r_squared
            );
        }
    }
    println!();
    println!("  Note: As lambda decreases, coefficients grow from zero.");
    println!("        Variables that appear last are the weakest predictors.");
    println!();

    // ── 5. Comparing AIC across methods ───────────────────────────────────────
    println!("━━━ 5. Model Comparison (same lambda=0.5) ━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {:12}  {:>8}  {:>8}  {:>8}", "Method", "R²", "MSE", "AIC");
    println!("  {}", "─".repeat(44));
    println!("  {:12}  {:>8.4}  {:>8.4}  {:>8.2}", "Ridge",       ridge.r_squared, ridge.mse, ridge.aic);
    println!("  {:12}  {:>8.4}  {:>8.4}  {:>8.2}", "Lasso",       lasso.r_squared, lasso.mse, lasso.aic);
    println!("  {:12}  {:>8.4}  {:>8.4}  {:>8.2}", "Elastic Net", enet.r_squared,  enet.mse,  enet.aic);
    println!();

}
