//! Polynomial regression example.
//!
//! Demonstrates polynomial regression with OLS and regularized variants.
//! Shows how centering reduces multicollinearity in higher-degree models.

use linreg_core::polynomial::{
    predict, polynomial_elastic_net, polynomial_lasso, polynomial_ridge,
    polynomial_regression, PolynomialOptions,
};

fn main() {
    println!("=== Polynomial Regression ===");
    println!();

    // Sample data: enzyme reaction rate vs temperature (quadratic relationship)
    // Real data follows a curve that increases then decreases at high temperatures
    let temperature = vec![
        10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0,
    ];
    let rate = vec![
        12.5, 19.8, 28.3, 37.1, 45.2, 51.8, 56.4, 58.9, 59.2, 57.1, 52.3, 44.8, 34.7,
    ];

    // 1. Quadratic Polynomial (degree 2)
    println!("1. Quadratic Polynomial Regression (degree = 2)");
    println!();

    let options_quad = PolynomialOptions {
        degree: 2,
        center: true, // Centering reduces correlation between x and x²
        ..Default::default()
    };

    match polynomial_regression(&rate, &temperature, &options_quad) {
        Ok(fit) => {
            println!("  Model: rate = b0 + b1*temp + b2*temp^2");
            println!();
            println!("  Coefficients:");
            for (i, name) in fit.feature_names.iter().enumerate() {
                let coef = fit.ols_output.coefficients[i];
                let se = fit.ols_output.std_errors[i];
                let t = fit.ols_output.t_stats[i];
                let p = fit.ols_output.p_values[i];
                let sig = if p < 0.001 {
                    "***"
                } else if p < 0.01 {
                    "**"
                } else if p < 0.05 {
                    "*"
                } else {
                    ""
                };
                println!(
                    "    {:<12} {:>8.4}  (SE: {:>6.4}, t: {:>6.3}, p: {:>.4e}) {}",
                    name, coef, se, t, p, sig
                );
            }
            println!();
            println!("  Model Fit:");
            println!("    R²:           {:.4}", fit.ols_output.r_squared);
            println!(
                "    Adjusted R²:  {:.4}",
                fit.ols_output.adj_r_squared
            );
            println!(
                "    F-statistic:  {:.2} (p = {:.4e})",
                fit.ols_output.f_statistic, fit.ols_output.f_p_value
            );
            println!();

            // Predictions at new temperatures
            let new_temps = vec![32.0, 48.0, 62.0];
            match predict(&fit, &new_temps) {
                Ok(preds) => {
                    println!("  Predictions:");
                    for (temp, pred) in new_temps.iter().zip(preds.iter()) {
                        println!("    At {} C: rate = {:.2}", temp, pred);
                    }
                }
                Err(e) => eprintln!("    Prediction error: {}", e),
            }
        }
        Err(e) => eprintln!("  Error: {}", e),
    }

    println!();
    println!("  {}", "-".repeat(60));

    // 2. Cubic Polynomial (degree 3) - shows centering benefit
    println!();
    println!("2. Cubic Polynomial (degree = 3)");
    println!();

    // Without centering: high multicollinearity
    let options_cubic_no_center = PolynomialOptions {
        degree: 3,
        center: false,
        ..Default::default()
    };

    match polynomial_regression(&rate, &temperature, &options_cubic_no_center) {
        Ok(fit) => {
            println!("  WITHOUT centering:");
            println!(
                "    R²: {:.4},  Max VIF: {:.2}",
                fit.ols_output.r_squared,
                fit.ols_output.vif.iter().map(|v| v.vif).fold(0.0f64, f64::max)
            );
        }
        Err(e) => eprintln!("    Error: {}", e),
    }

    // With centering: reduced multicollinearity
    let options_cubic_centered = PolynomialOptions {
        degree: 3,
        center: true,
        ..Default::default()
    };

    match polynomial_regression(&rate, &temperature, &options_cubic_centered) {
        Ok(fit) => {
            println!();
            println!("  WITH centering (recommended for degree >= 3):");
            println!(
                "    R²: {:.4},  Max VIF: {:.2}",
                fit.ols_output.r_squared,
                fit.ols_output.vif.iter().map(|v| v.vif).fold(0.0f64, f64::max)
            );
            println!();
            println!("  Coefficients:");
            for (i, name) in fit.feature_names.iter().enumerate() {
                println!(
                    "    {:<12} {:>8.4}",
                    name, fit.ols_output.coefficients[i]
                );
            }
        }
        Err(e) => eprintln!("    Error: {}", e),
    }

    println!();
    println!("  {}", "-".repeat(60));

    // 3. Regularized Polynomial (Ridge)
    println!();
    println!("3. Regularized Polynomial: Ridge (degree = 4)");
    println!();

    // Degree 4 with Ridge to handle extreme multicollinearity
    match polynomial_ridge(&rate, &temperature, 4, 0.1, true, true) {
        Ok(ridge) => {
            println!("  Ridge handles high multicollinearity in degree 4+");
            println!();
            println!("  Model: rate = b0 + b1*temp + b2*temp^2 + b3*temp^3 + b4*temp^4");
            println!("  Lambda (L2 penalty): 0.1");
            println!();
            println!("  Coefficients:");
            for (i, coef) in ridge.coefficients.iter().enumerate() {
                let name = if i == 0 {
                    "Intercept"
                } else if i == 1 {
                    "temp (centered)"
                } else {
                    "temp^x"
                };
                println!("    {:<16} {:>8.4}", name, coef);
            }
            println!();
            println!("  Fit:");
            println!("    R²:           {:.4}", ridge.r_squared);
            println!(
                "    Adjusted R²:  {:.4}",
                ridge.adj_r_squared
            );
            println!(
                "    Effective df: {:.2}",
                ridge.df
            );
        }
        Err(e) => eprintln!("  Error: {}", e),
    }

    println!();
    println!("  {}", "-".repeat(60));

    // 4. Regularized Polynomial: Lasso (variable selection)
    println!();
    println!("4. Regularized Polynomial: Lasso (degree = 5)");
    println!();

    // Higher degree with Lasso to automatically select relevant terms
    match polynomial_lasso(&rate, &temperature, 5, 0.5, true, true) {
        Ok(lasso) => {
            println!("  Lasso can eliminate unnecessary higher-order terms");
            println!();
            println!("  Model: degree 5 polynomial");
            println!("  Lambda (L1 penalty): 0.5");
            println!();
            println!("  Coefficients (zero = term eliminated):");
            for (i, coef) in lasso.coefficients.iter().enumerate() {
                let name = if i == 0 {
                    "Intercept".to_string()
                } else {
                    format!("temp^{}", i)
                };
                let status = if coef.abs() < 1e-10 { " (zero)" } else { "" };
                println!("    {:<16} {:>8.4}{}", name, coef, status);
            }
            println!();
            println!("  Fit:");
            println!("    R²:            {:.4}", lasso.r_squared);
            println!(
                "    Non-zero terms: {} / {}",
                lasso.n_nonzero,
                lasso.coefficients.len()
            );
            println!("    Converged:     {}", lasso.converged);
        }
        Err(e) => eprintln!("  Error: {}", e),
    }

    println!();
    println!("  {}", "-".repeat(60));

    // 5. Elastic Net (balanced L1 + L2)
    println!();
    println!("5. Regularized Polynomial: Elastic Net (degree = 4)");
    println!();

    match polynomial_elastic_net(&rate, &temperature, 4, 0.1, 0.5, true, true) {
        Ok(enet) => {
            println!("  Elastic Net combines Ridge (L2) and Lasso (L1) penalties");
            println!();
            println!("  Alpha (L1/L2 mix): 0.5 (equal blend)");
            println!("  Lambda (strength):  0.1");
            println!();
            println!("  Coefficients:");
            for (i, coef) in enet.coefficients.iter().enumerate() {
                let name = if i == 0 {
                    "Intercept".to_string()
                } else {
                    format!("temp^{}", i)
                };
                println!("    {:<16} {:>8.4}", name, coef);
            }
            println!();
            println!("  Fit:");
            println!("    R²:            {:.4}", enet.r_squared);
            println!(
                "    Non-zero terms: {} / {}",
                enet.n_nonzero,
                enet.coefficients.len()
            );
            println!("    Converged:     {}", enet.converged);
        }
        Err(e) => eprintln!("  Error: {}", e),
    }

    println!();
    println!("=== Key Takeaways ===");
    println!("  - Use degree 2-3 for most real-world curved relationships");
    println!("  - Centering (center: true) reduces multicollinearity for degree >= 3");
    println!("  - Ridge/Lasso/Elastic Net help with high-degree models");
}
