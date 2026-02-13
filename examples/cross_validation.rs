//! K-Fold Cross Validation example.
//!
//! This example demonstrates how to use K-Fold Cross Validation for model evaluation
//! and hyperparameter tuning with OLS, Ridge, Lasso, and Elastic Net regression.

use linreg_core::cross_validation::{
    kfold_cv_elastic_net, kfold_cv_lasso, kfold_cv_ols, kfold_cv_ridge, KFoldOptions,
};

fn main() {
    // Housing price dataset (square feet, bedrooms, age -> price)
    // This is a realistic dataset with mild multicollinearity
    let y = vec![
        245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1, 445.8, 167.9,
        367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9, 223.4, 312.5, 156.8, 423.7,
        267.9,
    ];

    let square_feet = vec![
        1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0, 2200.0,
        900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0, 1250.0, 1700.0,
        850.0, 2350.0, 1400.0,
    ];

    let bedrooms = vec![
        3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0, 4.0, 2.0, 4.0, 3.0, 2.0, 5.0, 3.0,
        4.0, 2.0, 4.0, 3.0, 3.0, 2.0, 4.0, 3.0,
    ];

    let age = vec![
        15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0, 3.0, 30.0, 6.0, 14.0, 22.0,
        1.0, 16.0, 9.0, 28.0, 4.0, 19.0, 11.0, 35.0, 3.0, 13.0,
    ];

    let names = vec![
        "Intercept".to_string(),
        "SqFt".to_string(),
        "Bedrooms".to_string(),
        "Age".to_string(),
    ];

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              K-FOLD CROSS VALIDATION EXAMPLE                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // ==========================================================================
    // 1. OLS Cross Validation
    // ==========================================================================
    println!("1. OLS CROSS VALIDATION");
    println!("{}\n", "─".repeat(70));

    let options = KFoldOptions::new(5).with_shuffle(true).with_seed(42);

    match kfold_cv_ols(&y, &[square_feet.clone(), bedrooms.clone(), age.clone()], &names, &options) {
        Ok(result) => {
            print_cv_summary(&result);
        }
        Err(e) => {
            eprintln!("Error in OLS CV: {}", e);
        }
    }

    // ==========================================================================
    // 2. Ridge Regression Cross Validation (Lambda Selection)
    // ==========================================================================
    println!("\n2. RIDGE REGRESSION - LAMBDA SELECTION");
    println!("{}\n", "─".repeat(70));

    let lambdas = [0.01, 0.1, 1.0, 10.0, 100.0];
    let mut best_lambda = 0.0;
    let mut best_rmse = f64::INFINITY;

    println!("Testing lambda values for Ridge regression:");
    println!("{:<12} {:>12} {:>12} {:>12}", "Lambda", "Mean RMSE", "Mean R²", "Std RMSE");
    println!("{}", "─".repeat(50));

    for &lambda in &lambdas {
        match kfold_cv_ridge(
            &[square_feet.clone(), bedrooms.clone(), age.clone()],
            &y,
            lambda,
            true,
            &options,
        ) {
            Ok(result) => {
                println!(
                    "{:<12.2} {:>12.4} {:>12.4} {:>12.4}",
                    lambda, result.mean_rmse, result.mean_r_squared, result.std_rmse
                );
                if result.mean_rmse < best_rmse {
                    best_rmse = result.mean_rmse;
                    best_lambda = lambda;
                }
            }
            Err(e) => {
                eprintln!("Error with lambda={}: {}", lambda, e);
            }
        }
    }

    println!("\n  Best lambda: {:.2} (RMSE: {:.4})", best_lambda, best_rmse);

    // ==========================================================================
    // 3. Lasso Regression Cross Validation (Lambda Selection)
    // ==========================================================================
    println!("\n3. LASSO REGRESSION - LAMBDA SELECTION");
    println!("{}\n", "─".repeat(70));

    let lasso_lambdas = [0.01, 0.1, 0.5, 1.0, 2.0];
    let mut best_lasso_lambda = 0.0;
    let mut best_lasso_rmse = f64::INFINITY;

    println!("Testing lambda values for Lasso regression:");
    println!("{:<12} {:>12} {:>12} {:>12}", "Lambda", "Mean RMSE", "Mean R²", "Std RMSE");
    println!("{}", "─".repeat(50));

    for &lambda in &lasso_lambdas {
        match kfold_cv_lasso(
            &[square_feet.clone(), bedrooms.clone(), age.clone()],
            &y,
            lambda,
            true,
            &options,
        ) {
            Ok(result) => {
                println!(
                    "{:<12.2} {:>12.4} {:>12.4} {:>12.4}",
                    lambda, result.mean_rmse, result.mean_r_squared, result.std_rmse
                );
                if result.mean_rmse < best_lasso_rmse {
                    best_lasso_rmse = result.mean_rmse;
                    best_lasso_lambda = lambda;
                }
            }
            Err(e) => {
                eprintln!("Error with lambda={}: {}", lambda, e);
            }
        }
    }

    println!(
        "\n  Best lambda: {:.2} (RMSE: {:.4})",
        best_lasso_lambda, best_lasso_rmse
    );

    // ==========================================================================
    // 4. Elastic Net Cross Validation (Alpha and Lambda Selection)
    // ==========================================================================
    println!("\n4. ELASTIC NET - ALPHA AND LAMBDA SELECTION");
    println!("{}\n", "─".repeat(70));

    let alphas = [0.0, 0.25, 0.5, 0.75, 1.0]; // 0=Ridge, 1=Lasso
    let en_lambda = 0.1;

    println!("Testing alpha values (lambda = {:.2}):", en_lambda);
    println!("Alpha: 0 = Ridge, 1 = Lasso");
    println!("{:<12} {:>12} {:>12} {:>12}", "Alpha", "Mean RMSE", "Mean R²", "Std RMSE");
    println!("{}", "─".repeat(50));

    let mut best_alpha = 0.0;
    let mut best_en_rmse = f64::INFINITY;

    for &alpha in &alphas {
        match kfold_cv_elastic_net(
            &[square_feet.clone(), bedrooms.clone(), age.clone()],
            &y,
            en_lambda,
            alpha,
            true,
            &options,
        ) {
            Ok(result) => {
                println!(
                    "{:<12.2} {:>12.4} {:>12.4} {:>12.4}",
                    alpha, result.mean_rmse, result.mean_r_squared, result.std_rmse
                );
                if result.mean_rmse < best_en_rmse {
                    best_en_rmse = result.mean_rmse;
                    best_alpha = alpha;
                }
            }
            Err(e) => {
                eprintln!("Error with alpha={}: {}", alpha, e);
            }
        }
    }

    println!(
        "\n  Best alpha: {:.2} (RMSE: {:.4})",
        best_alpha, best_en_rmse
    );

    // ==========================================================================
    // 5. Coefficient Stability Analysis
    // ==========================================================================
    println!("\n5. COEFFICIENT STABILITY ANALYSIS");
    println!("{}\n", "─".repeat(70));

    match kfold_cv_ols(
        &y,
        &[square_feet.clone(), bedrooms.clone(), age.clone()],
        &names,
        &options,
    ) {
        Ok(result) => {
            println!("Coefficient variability across {} folds:", result.n_folds);
            println!();
            println!(
                "{:<12} {:>12} {:>12} {:>12} {:>12}",
                "Variable", "Mean", "Std", "Min", "Max"
            );
            println!("{}", "─".repeat(60));

            for (i, name) in names.iter().enumerate() {
                let coeffs: Vec<f64> = result.fold_coefficients.iter().map(|c| c[i]).collect();
                let mean = coeffs.iter().sum::<f64>() / coeffs.len() as f64;
                let variance =
                    coeffs.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / coeffs.len() as f64;
                let std = variance.sqrt();
                let min = coeffs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = coeffs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                println!(
                    "{:<12} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                    name, mean, std, min, max
                );
            }

            // Calculate coefficient of variation for stability assessment
            println!("\nCoefficient Stability (CV = Std/Mean):");
            for (i, name) in names.iter().enumerate() {
                if i == 0 {
                    continue; // Skip intercept
                }
                let coeffs: Vec<f64> = result.fold_coefficients.iter().map(|c| c[i]).collect();
                let mean = coeffs.iter().sum::<f64>() / coeffs.len() as f64;
                let variance =
                    coeffs.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / coeffs.len() as f64;
                let std = variance.sqrt();
                let cv = if mean.abs() > 1e-10 {
                    std / mean.abs()
                } else {
                    f64::INFINITY
                };

                let status = if cv < 0.1 {
                    "Very Stable"
                } else if cv < 0.2 {
                    "Stable"
                } else {
                    "Variable"
                };

                println!("  {:<12}: CV = {:.3} ({})", name, cv, status);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    // ==========================================================================
    // 6. Reproducibility Demonstration
    // ==========================================================================
    println!("\n6. REPRODUCIBILITY WITH SEED");
    println!("{}\n", "─".repeat(70));

    let options1 = KFoldOptions::new(4).with_shuffle(true).with_seed(12345);
    let options2 = KFoldOptions::new(4).with_shuffle(true).with_seed(12345);

    let result1 =
        kfold_cv_ols(&y, &[square_feet.clone(), bedrooms.clone(), age.clone()], &names, &options1)
            .unwrap();
    let result2 = kfold_cv_ols(&y, &[square_feet, bedrooms, age], &names, &options2).unwrap();

    println!("Two runs with same seed (12345):");
    println!("  Run 1 - RMSE: {:.6}, R²: {:.6}", result1.mean_rmse, result1.mean_r_squared);
    println!("  Run 2 - RMSE: {:.6}, R²: {:.6}", result2.mean_rmse, result2.mean_r_squared);
    println!();
    println!(
        "  Difference - RMSE: {:.2e}, R²: {:.2e}",
        (result1.mean_rmse - result2.mean_rmse).abs(),
        (result1.mean_r_squared - result2.mean_r_squared).abs()
    );
    println!();
    println!("   Identical results demonstrate reproducibility");
}

// ============================================================================
// Helper Functions
// ============================================================================

fn print_cv_summary(result: &linreg_core::cross_validation::CVResult) {
    println!("  Cross-Validation Summary ({:} folds)", result.n_folds);
    println!("  ──────────────────────────────────────────");
    println!("  Samples:            {}", result.n_samples);
    println!();
    println!(
        "  Mean RMSE:          {:.4} (±{:.4})",
        result.mean_rmse, result.std_rmse
    );
    println!(
        "  Mean MAE:           {:.4} (±{:.4})",
        result.mean_mae, result.std_mae
    );
    println!(
        "  Mean R²:            {:.4} (±{:.4})",
        result.mean_r_squared, result.std_r_squared
    );
    println!(
        "  Mean Train R²:      {:.4}",
        result.mean_train_r_squared
    );
    println!();

    // Check for overfitting
    let overfitting_gap = result.mean_train_r_squared - result.mean_r_squared;
    if overfitting_gap > 0.1 {
        println!(
            "  Warning: Train R² is significantly higher than Test R²"
        );
        println!(
            "    This suggests potential overfitting (gap: {:.3})",
            overfitting_gap
        );
    } else {
        println!(
            "   Good generalization (train-test R² gap: {:.3})",
            overfitting_gap
        );
    }
    println!();

    // Per-fold results
    println!("  Fold Results:");
    println!("  ──────────────────────────────────────────");
    println!(
        "  {:<6} {:>8} {:>8} {:>10} {:>10}",
        "Fold", "Train", "Test", "RMSE", "R²"
    );
    println!("{}", "─".repeat(50));

    for fold in &result.fold_results {
        println!(
            "  {:<6} {:>8} {:>8} {:>10.4} {:>10.4}",
            fold.fold_index, fold.train_size, fold.test_size, fold.rmse, fold.r_squared
        );
    }
}
