//! Feature importance example.
//!
//! Demonstrates various methods for assessing feature importance in regression models:
//! - Standardized coefficients (scale-invariant importance)
//! - SHAP values (local and global explanations)
//! - Permutation importance (model-agnostic importance)
//! - VIF ranking (multicollinearity assessment)

use linreg_core::core::ols_regression;
use linreg_core::{
    permutation_importance_ols, shap_values_linear_named, standardized_coefficients, vif_ranking,
    PermutationImportanceOptions,
};

fn main() {
    println!("=== Feature Importance ===");
    println!();

    // Housing price data: predicting price from multiple features
    // Features are on different scales, making this ideal for demonstrating
    // the difference between raw and standardized coefficients
    let price = vec![
        245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1, 387.2, 421.5,
        267.3, 334.8, 298.5,
    ];
    let sqft = vec![
        1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0,
        2100.0, 2500.0, 1500.0, 1900.0, 1700.0,
    ];
    let bedrooms = vec![
        3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0, 4.0, 4.0, 3.0, 4.0, 3.0,
    ];
    let age = vec![
        15.0, 5.0, 25.0, 2.0, 18.0, 10.0, 30.0, 1.0, 20.0, 12.0, 8.0, 3.0, 22.0, 7.0, 14.0,
    ];

    let x_vars = vec![sqft.clone(), bedrooms.clone(), age.clone()];
    let names = vec![
        "Intercept".to_string(),
        "SqFt".to_string(),
        "Bedrooms".to_string(),
        "Age".to_string(),
    ];

    // Fit the model
    let fit = match ols_regression(&price, &x_vars, &names) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error fitting model: {}", e);
            return;
        }
    };

    println!("Model: Price ~ SqFt + Bedrooms + Age");
    println!("  Observations: {}", fit.n);
    println!("  R²: {:.4}", fit.r_squared);
    println!("  Adjusted R²: {:.4}", fit.adj_r_squared);
    println!();

    // 1. Raw Coefficients (NOT comparable across features)
    println!("1. Raw Coefficients");
    println!();
    println!("  NOTE: Raw coefficients are NOT directly comparable because predictors");
    println!("        have different units and scales.");
    println!();

    println!("  {:<12} {:>12}  {:>12}", "Variable", "Coefficient", "Scale");
    println!("  {}", "-".repeat(40));

    let sqft_mean: f64 = sqft.iter().sum::<f64>() / sqft.len() as f64;
    let bedrooms_mean: f64 = bedrooms.iter().sum::<f64>() / bedrooms.len() as f64;
    let age_mean: f64 = age.iter().sum::<f64>() / age.len() as f64;

    for (i, name) in names.iter().enumerate() {
        let scale = if i == 0 {
            "N/A (intercept)".to_string()
        } else if i == 1 {
            format!("~{:.0}", sqft_mean)
        } else if i == 2 {
            format!("~{:.1}", bedrooms_mean)
        } else {
            format!("~{:.1}", age_mean)
        };
        println!(
            "  {:<12} {:>12.4}  {:>12}",
            name, fit.coefficients[i], scale
        );
    }
    println!();
    println!("  Problem: SqFt coefficient looks small (0.16) but it's in $/sqft units.");
    println!("           Age coefficient looks larger (-2.15) but it's in $/year units.");
    println!("           We cannot compare them directly!");

    println!();
    println!("  {}", "-".repeat(60));

    // 2. Standardized Coefficients (COMPARABLE)
    println!();
    println!("2. Standardized Coefficients");
    println!();
    println!("  Standardized coefficients (beta*) represent the change in Y (in SDs)");
    println!("  for a 1 SD change in X. NOW we can compare importance.");
    println!();

    match standardized_coefficients(&fit.coefficients, &x_vars) {
        Ok(std_coefs) => {
            println!("  {:<12} {:>12}  {:>10}", "Variable", "Beta*", "Rank");
            println!("  {}", "─".repeat(38));

            let ranking = std_coefs.ranking();
            for (rank, (name, _abs_val)) in ranking.iter().enumerate() {
                let idx = std_coefs
                    .variable_names
                    .iter()
                    .position(|n| n == name)
                    .unwrap();
                let beta = std_coefs.standardized_coefficients[idx];
                println!(
                    "  {:<12} {:>12.4}  #{:2}",
                    name,
                    beta,
                    rank + 1
                );
            }

            println!();
            println!("  Interpretation:");
            println!("    • SqFt has the LARGEST effect (|beta*| = {:.2})", {
                let abs_val: f64 = std_coefs.standardized_coefficients.iter().map(|&v| v.abs()).fold(
                    0.0/0.0,
                    |m, v| v.max(m)
                );
                abs_val
            });
            println!("    • Age has a moderate negative effect");
            println!("    • Bedrooms has relatively small effect");
        }
        Err(e) => eprintln!("  Error: {}", e),
    }

    println!();
    println!("  {}", "-".repeat(60));

    // 3. SHAP Values (Local Explanations)
    println!();
    println!("3. SHAP Values (Local + Global Importance)");
    println!();
    println!("  SHAP values decompose each prediction into feature contributions.");
    println!("  Sum of SHAP + base_value = predicted price.");
    println!();

    match shap_values_linear_named(&x_vars, &fit.coefficients, &names[1..]) {
        Ok(shap) => {
            // Show global importance (mean |SHAP|)
            println!("  Global Importance (mean |SHAP|):");
            println!("  {}", "─".repeat(40));

            let ranking = shap.ranking();
            let total_shap: f64 = shap.mean_abs_shap.iter().map(|&v| v.abs()).sum();
            for (rank, (name, mean_abs)) in ranking.iter().enumerate() {
                let pct = if total_shap > 0.0 {
                    mean_abs / total_shap * 100.0
                } else {
                    0.0
                };
                println!("    #{:1} {:<12}  mean|SHAP| = {:.4}  (≈{:>5.1}%)", rank + 1, name, mean_abs, pct);
            }

            println!();
            println!("  Local Explanation (first 3 houses):");
            println!("  {}", "─".repeat(70));

            for i in 0..3_usize.min(price.len()) {
                println!();
                println!("    House #{} (Actual: ${:.1}k)", i + 1, price[i]);

                let mut pred = shap.base_value;
                print!("      Base:    ${:.2}k", shap.base_value);

                for (j, name) in shap.variable_names.iter().enumerate() {
                    let contribution = shap.shap_values[i][j];
                    pred += contribution;
                    let sign = if contribution >= 0.0 { "+" } else { "" };
                    println!();
                    print!(
                        "      {:<8}:  ${:.2}k {}",
                        name, contribution, sign
                    );
                }
                println!();
                println!("      ──────────────────────────────");
                println!("      Pred:    ${:.2}k", pred);
                println!("      Resid:   ${:.2}k", price[i] - pred);
            }
        }
        Err(e) => eprintln!("  Error: {}", e),
    }

    println!();
    println!("  {}", "-".repeat(60));

    // 4. Permutation Importance (Model-Agnostic)
    println!();
    println!("4. Permutation Importance");
    println!();
    println!("  Measures: How much R² drops when each feature is shuffled.");
    println!("  Higher drop = more important feature.");
    println!();

    let perm_options = PermutationImportanceOptions {
        n_permutations: 25,
        seed: Some(42),
        compute_intervals: false,
        ..Default::default()
    };

    match permutation_importance_ols(&price, &x_vars, &fit, &perm_options) {
        Ok(perm) => {
            println!("  Baseline R² = {:.4}", perm.baseline_score);
            println!();
            println!("  {:<12} {:>12}  {:>10}", "Variable", "Importance", "Rank");
            println!("  {}", "─".repeat(38));

            let ranking = perm.ranking();
            for (rank, (name, importance)) in ranking.iter().enumerate() {
                let drop_pct = importance * 100.0;
                println!(
                    "  {:<12} {:>12.4}  #{:2}  (R² ↓ {:.1}%)",
                    name, importance, rank + 1, drop_pct
                );
            }

            println!();
            println!("  Interpretation:");
            println!("    • Shuffling '{}' causes {:.1}% R² drop → MOST important",
                ranking[0].0, ranking[0].1 * 100.0
            );
            println!("    • Shuffling '{}' causes only {:.1}% R² drop → LEAST important",
                ranking.last().unwrap().0, ranking.last().unwrap().1 * 100.0
            );
        }
        Err(e) => eprintln!("  Error: {}", e),
    }

    println!();
    println!("  {}", "-".repeat(60));

    // 5. VIF Ranking (Multicollinearity Check)
    println!();
    println!("5. VIF Ranking (Multicollinearity)");
    println!();
    println!("  VIF measures how much the variance of a coefficient is inflated due to");
    println!("  correlation with other predictors. LOWER VIF is better.");
    println!();
    println!("  Guidelines:");
    println!("    • VIF < 5:  Low multicollinearity (good)");
    println!("    • VIF 5-10: Moderate (review)");
    println!("    • VIF > 10: High multicollinearity (problematic)");
    println!();

    let vif_rank = vif_ranking(&fit.vif);
    println!("  {:<12} {:>12}  {:>20}", "Variable", "VIF", "Status");
    println!("  {}", "─".repeat(48));

    for (name, vif_val) in vif_rank.variable_names.iter().zip(vif_rank.vif_values.iter()) {
        let status = if *vif_val < 5.0 {
            "[OK]"
        } else if *vif_val < 10.0 {
            "[MODERATE]"
        } else {
            "[HIGH]"
        };
        println!("  {:<12} {:>12.4}  {:>20}", name, vif_val, status);
    }

    println!();
    println!("=== Summary: Which metric to use? ===");
    println!("  - Standardized Coefs: Quick, scale-invariant comparison");
    println!("  - SHAP: Local explanations + global importance (recommended)");
    println!("  - Permutation: Model-agnostic, works for any model type");
    println!("  - VIF: Detect multicollinearity (not importance per se)");
}
