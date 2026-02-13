// ============================================================================
// Model Serialization Example
// ============================================================================
//
// This example demonstrates how to save and load trained regression models
// using the ModelSave and ModelLoad traits.
//
// Run with:
//     cargo run --example serialization_example

use linreg_core::core::ols_regression;
use linreg_core::loess::{loess_fit, LoessOptions};
use linreg_core::regularized::{elastic_net_fit, lasso_fit, ridge_fit, ElasticNetOptions, LassoFitOptions, RidgeFitOptions};
use linreg_core::serialization::{ModelLoad, ModelSave};
use linreg_core::weighted_regression::wls_regression;
use linreg_core::linalg::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Model Serialization Example ===\n");

    // Sample data
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3, 7.0, 8.1, 9.2];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x2 = vec![1.5, 2.1, 3.2, 3.9, 5.1, 6.2, 7.0, 8.1];

    // ========================================================================
    // OLS Regression Example
    // ========================================================================

    println!("--- OLS Regression ---");
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];
    let ols_result = ols_regression(&y, &[x1.clone(), x2.clone()], &names)?;
    println!("R²: {:.4}", ols_result.r_squared);
    println!("Coefficients: {:?}", ols_result.coefficients);

    // Save with custom name
    let ols_path = "ols_model.json";
    ols_result.save_with_name(ols_path, Some("My OLS Housing Model".to_string()))?;
    println!("Saved to: {}", ols_path);

    // Load back
    let ols_loaded = linreg_core::core::RegressionOutput::load(ols_path)?;
    println!("Loaded model R²: {:.4}", ols_loaded.r_squared);
    println!();

    // ========================================================================
    // Ridge Regression Example
    // ========================================================================

    println!("--- Ridge Regression ---");
    let x_matrix = create_matrix(&y.len(), &[&x1, &x2]);
    let ridge_options = RidgeFitOptions {
        lambda: 1.0,
        standardize: true,
        intercept: true,
        ..Default::default()
    };
    let ridge_result = ridge_fit(&x_matrix, &y, &ridge_options)?;
    println!("Lambda: {}", ridge_result.lambda);
    println!("Intercept: {:.4}", ridge_result.intercept);
    println!("Coefficients: {:?}", ridge_result.coefficients);

    let ridge_path = "ridge_model.json";
    ridge_result.save(ridge_path)?;
    println!("Saved to: {}", ridge_path);

    let ridge_loaded = linreg_core::regularized::ridge::RidgeFit::load(ridge_path)?;
    println!("Loaded model Lambda: {}", ridge_loaded.lambda);
    println!();

    // ========================================================================
    // Lasso Regression Example
    // ========================================================================

    println!("--- Lasso Regression ---");
    let lasso_options = LassoFitOptions {
        lambda: 0.1,
        standardize: true,
        intercept: true,
        ..Default::default()
    };
    let lasso_result = lasso_fit(&x_matrix, &y, &lasso_options)?;
    println!("Lambda: {}", lasso_result.lambda);
    println!("Non-zero coefficients: {}", lasso_result.n_nonzero);
    println!("Coefficients: {:?}", lasso_result.coefficients);

    let lasso_path = "lasso_model.json";
    lasso_result.save(lasso_path)?;
    println!("Saved to: {}", lasso_path);
    println!();

    // ========================================================================
    // Elastic Net Regression Example
    // ========================================================================

    println!("--- Elastic Net Regression ---");
    let enet_options = ElasticNetOptions {
        lambda: 0.1,
        alpha: 0.5,
        standardize: true,
        intercept: true,
        ..Default::default()
    };
    let enet_result = elastic_net_fit(&x_matrix, &y, &enet_options)?;
    println!("Lambda: {}, Alpha: {}", enet_result.lambda, enet_result.alpha);
    println!("Non-zero coefficients: {}", enet_result.n_nonzero);
    println!("Coefficients: {:?}", enet_result.coefficients);

    let enet_path = "elastic_net_model.json";
    enet_result.save(enet_path)?;
    println!("Saved to: {}", enet_path);
    println!();

    // ========================================================================
    // WLS (Weighted Least Squares) Example
    // ========================================================================

    println!("--- Weighted Least Squares (WLS) ---");
    let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let wls_result = wls_regression(&y, &[x1.clone(), x2.clone()], &weights)?;
    println!("R²: {:.4}", wls_result.r_squared);
    println!("Coefficients: {:?}", wls_result.coefficients);

    let wls_path = "wls_model.json";
    wls_result.save(wls_path)?;
    println!("Saved to: {}", wls_path);
    println!();

    // ========================================================================
    // LOESS Regression Example
    // ========================================================================

    println!("--- LOESS Regression ---");
    let loess_options = LoessOptions {
        span: 0.75,
        degree: 2,
        robust_iterations: 2,
        n_predictors: 1,
        surface: linreg_core::loess::LoessSurface::Direct,
    };
    let loess_result = loess_fit(&y, &[x1.clone()], &loess_options)?;
    println!("Span: {}, Degree: {}", loess_result.span, loess_result.degree);
    println!("First 3 fitted values: {:?}", &loess_result.fitted[..3]);

    let loess_path = "loess_model.json";
    loess_result.save(loess_path)?;
    println!("Saved to: {}", loess_path);
    println!();

    // ========================================================================
    // Error Handling Examples
    // ========================================================================

    println!("--- Error Handling ---");

    // Try to load with wrong type
    match linreg_core::regularized::ridge::RidgeFit::load(ols_path) {
        Ok(_) => println!("Unexpectedly loaded OLS as Ridge"),
        Err(e) => println!("Expected error loading OLS as Ridge: {}", e),
    }

    // Try to load non-existent file
    match linreg_core::core::RegressionOutput::load("nonexistent.json") {
        Ok(_) => println!("Unexpectedly loaded nonexistent file"),
        Err(e) => println!("Expected error for nonexistent file: {}", e),
    }

    println!();

    // ========================================================================
    // Metadata Inspection
    // ========================================================================

    println!("--- Metadata Inspection ---");
    // The saved files contain metadata that can be inspected
    let content = std::fs::read_to_string(ols_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&content)?;
    if let Some(metadata) = parsed.get("metadata") {
        println!("Model metadata:");
        println!("  Type: {}", metadata.get("model_type").unwrap());
        println!("  Format version: {}", metadata.get("format_version").unwrap());
        println!("  Library version: {}", metadata.get("library_version").unwrap());
        println!("  Created at: {}", metadata.get("created_at").unwrap());
        println!("  Name: {:?}", metadata.get("name").unwrap());
    }

    println!();

    // ========================================================================
    // Cleanup
    // ========================================================================

    println!("--- Cleanup ---");
    for path in &[ols_path, ridge_path, lasso_path, enet_path, wls_path, loess_path] {
        let _ = std::fs::remove_file(path);
        println!("Removed: {}", path);
    }

    println!("\n=== Example Complete ===");

    Ok(())
}

/// Helper function to create a design matrix from predictor columns
fn create_matrix(n: &usize, x_cols: &[&Vec<f64>]) -> Matrix {
    let p = x_cols.len();
    let mut data = vec![1.0; n * (p + 1)]; // First column is intercept

    for (col_idx, col) in x_cols.iter().enumerate() {
        for (row_idx, &val) in col.iter().enumerate() {
            data[row_idx * (p + 1) + col_idx + 1] = val;
        }
    }

    Matrix::new(*n, p + 1, data)
}
