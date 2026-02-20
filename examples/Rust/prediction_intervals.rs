//! Prediction intervals example.
//!
//! Run with:
//!   cargo run --example prediction_intervals --no-default-features
//!
//! Demonstrates:
//! - OLS prediction intervals (exact, using leverage)
//! - Ridge / Lasso / Elastic Net prediction intervals (conservative approximation)
//! - Difference between prediction intervals (new obs) vs confidence intervals (mean)
//! - How intervals widen for out-of-sample (extrapolation) points

use linreg_core::linalg::Matrix;
use linreg_core::prediction_intervals::{
    elastic_net_prediction_intervals, lasso_prediction_intervals, prediction_intervals,
    ridge_prediction_intervals,
};
use linreg_core::regularized::{
    elastic_net_fit, lasso_fit, ridge_fit, ElasticNetOptions, LassoFitOptions, RidgeFitOptions,
};

fn main() {
    // ── Training data: house size (sqft/100) vs price ($k) ─────────────────────
    let sqft  = vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0];
    let price = vec![150.0, 175.0, 210.0, 240.0, 265.0, 295.0, 330.0, 360.0, 400.0, 430.0];

    // New points to predict: in-sample range and one extrapolation
    let new_sqft_vals = vec![11.0, 15.0, 19.0, 25.0, 35.0]; // 35 is extrapolation
    let new_sqft: Vec<&[f64]> = vec![&new_sqft_vals];

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                    PREDICTION INTERVALS                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Training: 10 houses, sqft (hundreds) -> price ($k)");
    println!("Predicting at: {:?}", new_sqft_vals);
    println!("Note: sqft=35 is extrapolation (training range: 10-28)");
    println!();

    // ── 1. OLS Prediction Intervals ────────────────────────────────────────────
    println!("━━━ 1. OLS Prediction Intervals (95%, exact) ━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Formula: PI = ŷ ± t(α/2, df) × √(MSE × (1 + leverage))");
    println!();

    let ols_pi = prediction_intervals(&price, &[sqft.clone()], &new_sqft, 0.05)
        .expect("OLS prediction intervals failed");

    println!("  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
             "SqFt", "Predicted", "Lower 95%", "Upper 95%", "Width", "Leverage");
    println!("  {}", "─".repeat(64));
    for i in 0..new_sqft_vals.len() {
        let width = ols_pi.upper_bound[i] - ols_pi.lower_bound[i];
        let extrap = if new_sqft_vals[i] > 28.0 { " <-- extrapolation" } else { "" };
        println!(
            "  {:>8.1}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}  {:>8.4}{}",
            new_sqft_vals[i],
            ols_pi.predicted[i],
            ols_pi.lower_bound[i],
            ols_pi.upper_bound[i],
            width,
            ols_pi.leverage[i],
            extrap
        );
    }
    println!();
    println!("  Note: Interval width grows at the extrapolation point (sqft=35)");
    println!("        because leverage is high far from the training data center.");
    println!("  df_residuals: {}", ols_pi.df_residuals);
    println!();

    // ── 2. Effect of alpha (confidence level) ─────────────────────────────────
    println!("━━━ 2. Interval Width vs Confidence Level (at sqft=19) ━━━━━━━━━━━━━━");
    println!("  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}",
             "Confidence", "Lower", "Predicted", "Upper", "Width");
    println!("  {}", "─".repeat(56));
    for &conf in &[0.50, 0.80, 0.90, 0.95, 0.99] {
        let alpha = 1.0 - conf;
        let pi = prediction_intervals(&price, &[sqft.clone()], &[&[19.0_f64][..]], alpha)
            .expect("PI failed");
        let width = pi.upper_bound[0] - pi.lower_bound[0];
        println!(
            "  {:>11.0}%  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}",
            conf * 100.0, pi.lower_bound[0], pi.predicted[0], pi.upper_bound[0], width
        );
    }
    println!();

    // ── 3. Regularized model prediction intervals ──────────────────────────────
    println!("━━━ 3. Regularized Model Prediction Intervals (at sqft=19) ━━━━━━━━━━");
    println!("  (Conservative approximation using unpenalized leverage + fit MSE)");
    println!();

    let mut x_data = Vec::with_capacity(10 * 2);
    for i in 0..10 {
        x_data.push(1.0);
        x_data.push(sqft[i]);
    }
    let x_mat = Matrix::new(10, 2, x_data);
    let x_vars = vec![sqft.clone()];
    let new_x_single: Vec<&[f64]> = vec![&new_sqft_vals[2..3]]; // sqft=19

    // Ridge
    let ridge = ridge_fit(&x_mat, &price, &RidgeFitOptions {
        lambda: 1.0, standardize: true, intercept: true, ..Default::default()
    }).expect("Ridge failed");
    let ridge_pi = ridge_prediction_intervals(&ridge, &x_vars, &new_x_single, 0.05)
        .expect("Ridge PI failed");

    // Lasso
    let lasso = lasso_fit(&x_mat, &price, &LassoFitOptions {
        lambda: 0.5, standardize: true, intercept: true, ..Default::default()
    }).expect("Lasso failed");
    let lasso_pi = lasso_prediction_intervals(&lasso, &x_vars, &new_x_single, 0.05)
        .expect("Lasso PI failed");

    // Elastic Net
    let enet = elastic_net_fit(&x_mat, &price, &ElasticNetOptions {
        lambda: 0.5, alpha: 0.5, standardize: true, intercept: true, ..Default::default()
    }).expect("Elastic Net failed");
    let enet_pi = elastic_net_prediction_intervals(&enet, &x_vars, &new_x_single, 0.05)
        .expect("Elastic Net PI failed");

    println!("  {:14}  {:>10}  {:>10}  {:>10}  {:>10}",
             "Method", "Predicted", "Lower 95%", "Upper 95%", "Width");
    println!("  {}", "─".repeat(58));

    let ols_at_19 = &ols_pi; // already computed at index 2 (sqft=19)
    let ols_width = ols_at_19.upper_bound[2] - ols_at_19.lower_bound[2];
    println!("  {:14}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}",
             "OLS (exact)",
             ols_at_19.predicted[2], ols_at_19.lower_bound[2],
             ols_at_19.upper_bound[2], ols_width);

    let r_width = ridge_pi.upper_bound[0] - ridge_pi.lower_bound[0];
    println!("  {:14}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}",
             "Ridge",
             ridge_pi.predicted[0], ridge_pi.lower_bound[0],
             ridge_pi.upper_bound[0], r_width);

    let l_width = lasso_pi.upper_bound[0] - lasso_pi.lower_bound[0];
    println!("  {:14}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}",
             "Lasso",
             lasso_pi.predicted[0], lasso_pi.lower_bound[0],
             lasso_pi.upper_bound[0], l_width);

    let e_width = enet_pi.upper_bound[0] - enet_pi.lower_bound[0];
    println!("  {:14}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}",
             "Elastic Net",
             enet_pi.predicted[0], enet_pi.lower_bound[0],
             enet_pi.upper_bound[0], e_width);

    println!();
    println!("  Note: Regularized intervals are conservative (wider) because they");
    println!("        use unpenalized leverage with the penalized model's MSE.");
}
