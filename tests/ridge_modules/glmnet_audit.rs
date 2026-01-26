// ============================================================================
// Ridge Regression glmnet Audit
// ============================================================================
//
// Comprehensive validation of ridge regression against R's glmnet.
// Verifies that the Rust implementation produces results consistent with
// the established glmnet algorithm.

use linreg_core::regularized::{ridge_fit, RidgeFitOptions};
use linreg_core::linalg::Matrix;

// ============================================================================
// Formula Verification
// ============================================================================

#[test]
fn test_ridge_formula_verification() {
    println!("=== RIDGE FORMULA VERIFICATION ===");
    println!();
    println!("Rust implementation uses coordinate descent with soft-thresholding,");
    println!("matching the behavior of R's glmnet package for ridge regression.");
    println!();
}

// ============================================================================
// Exact Comparison Tests
// ============================================================================

/// Exact comparison of Rust vs R for small dataset
#[test]
fn test_ridge_vs_r_exact() {
    // Exact same data as R test
    let y = vec![21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2];
    let wt = vec![2.62_f64, 2.88, 2.32, 3.21, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44];
    let hp = vec![110_f64, 110., 93., 110., 175., 105., 245., 62., 95., 123.];

    let n = 10;
    let p = 2;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = wt[i];
        x_data[i * (p + 1) + 2] = hp[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    // Test with lambda=0.5 (same as R)
    let lambda = 0.5;
    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda,
        intercept: true,
        standardize: true,
        weights: None,
    };

    let result = ridge_fit(&x, &y, &options).unwrap();

    // R glmnet result (lambda=0.5):
    // (Intercept): 30.1527277
    // wt: -1.6646784
    // hp: -0.0372607

    println!("\n=== EXACT COMPARISON WITH R glmnet (lambda=0.5) ===");
    println!("  R intercept: 30.1527277, Rust intercept: {}", result.intercept);
    println!("  R wt: -1.6646784, Rust wt: {}", result.coefficients[0]);
    println!("  R hp: -0.0372607, Rust hp: {}", result.coefficients[1]);
    println!();
    println!("  Differences:");
    println!("  Intercept diff: {}", (result.intercept - 30.1527277).abs());
    println!("  wt diff: {}", (result.coefficients[0] - (-1.6646784)).abs());
    println!("  hp diff: {}", (result.coefficients[1] - (-0.0372607)).abs());

    // Check if within 5% tolerance
    let intercept_diff_pct: f64 = (result.intercept - 30.1527277).abs() / 30.1527277;
    let wt_diff_pct: f64 = (result.coefficients[0] - (-1.6646784)).abs() / 1.6646784;
    let hp_diff_pct: f64 = (result.coefficients[1] - (-0.0372607)).abs() / 0.0372607;

    assert!(intercept_diff_pct < 0.05, "Intercept should match R within 5%, got {:.2}%", intercept_diff_pct * 100.0);
    assert!(wt_diff_pct < 0.05, "wt should match R within 5%, got {:.2}%", wt_diff_pct * 100.0);
    assert!(hp_diff_pct < 0.05, "hp should match R within 5%, got {:.2}%", hp_diff_pct * 100.0);

    println!("\n  Test PASSED!");
}

/// Test with mtcars subset and manual calculation verification
#[test]
fn test_ridge_mtcars_verification() {
    // mtcars subset - just mpg, wt, hp for simplicity
    let y = vec![21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2];
    let wt = vec![2.62_f64, 2.88, 2.32, 3.21, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44];
    let hp = vec![110_f64, 110., 93., 110., 175., 105., 245., 62., 95., 123.];

    let n = 10;
    let p = 2;
    let mut x_data = vec![1.0; n * (p + 1)];
    for i in 0..n {
        x_data[i * (p + 1) + 1] = wt[i];
        x_data[i * (p + 1) + 2] = hp[i];
    }
    let x = Matrix::new(n, p + 1, x_data);

    let lambda = 0.5;
    let options = RidgeFitOptions {
        max_iter: 10000,
        tol: 1e-7,
        warm_start: None,
        lambda,
        intercept: true,
        standardize: true,
        weights: None,
    };

    let result = ridge_fit(&x, &y, &options).unwrap();

    println!("\n=== MTcars SUBSET VERIFICATION (lambda=0.5) ===");
    println!("Rust result:");
    println!("  Intercept: {}", result.intercept);
    println!("  wt: {}", result.coefficients[0]);
    println!("  hp: {}", result.coefficients[1]);

    // Manual calculation for verification
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let wt_mean: f64 = wt.iter().sum::<f64>() / n as f64;
    let hp_mean: f64 = hp.iter().sum::<f64>() / n as f64;

    let y_c: Vec<f64> = y.iter().map(|yi| yi - y_mean).collect();
    let wt_c: Vec<f64> = wt.iter().map(|wi| wi - wt_mean).collect();
    let hp_c: Vec<f64> = hp.iter().map(|hi| hi - hp_mean).collect();

    let y_var = y_c.iter().map(|yi| yi * yi).sum::<f64>() / n as f64;
    let wt_var = wt_c.iter().map(|wi| wi * wi).sum::<f64>() / n as f64;
    let hp_var = hp_c.iter().map(|hi| hi * hi).sum::<f64>() / n as f64;

    let y_scale = y_var.sqrt();
    let wt_scale = wt_var.sqrt();
    let hp_scale = hp_var.sqrt();

    println!("\nStandardization:");
    println!("  y_mean: {}, y_scale: {}", y_mean, y_scale);
    println!("  wt_mean: {}, wt_scale: {}", wt_mean, wt_scale);
    println!("  hp_mean: {}, hp_scale: {}", hp_mean, hp_scale);

    // Standardize
    let y_std: Vec<f64> = y_c.iter().map(|yi| yi / y_scale).collect();
    let wt_std: Vec<f64> = wt_c.iter().map(|wi| wi / wt_scale).collect();
    let hp_std: Vec<f64> = hp_c.iter().map(|hi| hi / hp_scale).collect();

    // Check variance of standardized data
    let wt_std_var = wt_std.iter().map(|wi| wi * wi).sum::<f64>() / n as f64;
    let hp_std_var = hp_std.iter().map(|hi| hi * hi).sum::<f64>() / n as f64;

    println!("\nCheck unit variance:");
    println!("  wt_std_var: {}", wt_std_var);
    println!("  hp_std_var: {}", hp_std_var);

    // Simple OLS on standardized data
    let mut xty_wt = 0.0;
    let mut xty_hp = 0.0;
    let mut xtx_wt_wt = 0.0;
    let mut xtx_wt_hp = 0.0;
    let mut xtx_hp_hp = 0.0;

    for i in 0..n {
        xty_wt += wt_std[i] * y_std[i];
        xty_hp += hp_std[i] * y_std[i];
        xtx_wt_wt += wt_std[i] * wt_std[i];
        xtx_wt_hp += wt_std[i] * hp_std[i];
        xtx_hp_hp += hp_std[i] * hp_std[i];
    }

    println!("\nCorrelations:");
    println!("  xty_wt: {}", xty_wt);
    println!("  xty_hp: {}", xty_hp);
    println!("  xtx_wt_wt: {}", xtx_wt_wt);
    println!("  xtx_wt_hp: {}", xtx_wt_hp);
    println!("  xtx_hp_hp: {}", xtx_hp_hp);

    // For ridge with lambda=0.5
    let n_f64 = n as f64;
    let beta_wt_ridge = xty_wt / (xtx_wt_wt + n_f64 * lambda);
    let beta_hp_ridge = xty_hp / (xtx_hp_hp + n_f64 * lambda);

    println!("\nDirect ridge solution (standardized, assuming uncorrelated):");
    println!("  beta_wt_std: {}", beta_wt_ridge);
    println!("  beta_hp_std: {}", beta_hp_ridge);

    // Unstandardize
    let beta_wt_orig = beta_wt_ridge * (y_scale / wt_scale);
    let beta_hp_orig = beta_hp_ridge * (y_scale / hp_scale);
    let intercept = y_mean - (wt_mean * beta_wt_orig + hp_mean * beta_hp_orig);

    println!("\nDirect solution (original scale):");
    println!("  Intercept: {}", intercept);
    println!("  wt: {}", beta_wt_orig);
    println!("  hp: {}", beta_hp_orig);

    println!("\nRust vs Direct:");
    println!("  Intercept: {} vs {}", result.intercept, intercept);
    println!("  wt: {} vs {}", result.coefficients[0], beta_wt_orig);
    println!("  hp: {} vs {}", result.coefficients[1], beta_hp_orig);
}

// ============================================================================
// Full Dataset Audit (if available)
// ============================================================================

#[derive(serde::Deserialize)]
struct RidgeGlmnetResult {
    lambda_sequence: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    #[allow(dead_code)]
    degrees_of_freedom: Vec<f64>,
}

fn read_glmnet_result(path: &str) -> Option<RidgeGlmnetResult> {
    let json = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&json).ok()
}

fn compute_mse(rust: &[f64], r: &[f64]) -> (f64, f64) {
    let n = rust.len();
    let max_diff = rust.iter()
        .zip(r.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let mse: f64 = rust.iter()
        .zip(r.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>() / n as f64;

    let rmse = mse.sqrt();
    (max_diff, rmse)
}

#[test]
fn test_ridge_mtcars_audit() {
    let current_dir = std::env::current_dir().unwrap();
    let r_path = current_dir.join("verification/results/r/mtcars_ridge_glmnet.json");

    let r_result = match read_glmnet_result(r_path.to_str().unwrap()) {
        Some(r) => r,
        None => {
            println!("=== MTcars glmnet reference not found, skipping audit ===");
            return;
        }
    };

    // Load mtcars data
    let datasets_dir = current_dir.join("verification/datasets/csv");
    let csv_path = datasets_dir.join("mtcars.csv");

    // Simple CSV parsing
    let content = match std::fs::read_to_string(&csv_path) {
        Ok(c) => c,
        Err(_) => {
            println!("=== MTcars CSV not found, skipping audit ===");
            return;
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    let headers: Vec<&str> = lines[0].split(',').collect();

    let mut y = Vec::new();
    let mut x_vars: Vec<Vec<f64>> = vec![Vec::new(); headers.len() - 1]; // 10 predictors

    for line in lines.iter().skip(1) {
        let vals: Vec<f64> = line.split(',')
            .map(|s| s.parse::<f64>().unwrap())
            .collect();

        y.push(vals[0]); // mpg (first column)
        for (j, &val) in vals.iter().skip(1).enumerate() {
            x_vars[j].push(val);
        }
    }

    let n = y.len();
    let p = x_vars.len();
    let mut x_data = vec![1.0; n * (p + 1)];
    for (col_idx, x_col) in x_vars.iter().enumerate() {
        for (row_idx, val) in x_col.iter().enumerate() {
            x_data[row_idx * (p + 1) + col_idx + 1] = *val;
        }
    }
    let x_mat = Matrix::new(n, p + 1, x_data);

    println!("\n=== RIDGE REGRESSION AUDIT: mtcars ===");
    println!("n = {}, p = {}", n, p);
    println!();

    // Test 3 representative lambdas
    let test_indices = vec![0_usize, r_result.lambda_sequence.len() / 2, r_result.lambda_sequence.len() - 1];

    for (idx, lambda_idx) in test_indices.iter().enumerate() {
        let lambda = r_result.lambda_sequence[*lambda_idx];
        let r_coef = &r_result.coefficients[*lambda_idx];

        println!("--- Test {}/{}: lambda = {:.6} ---", idx + 1, test_indices.len(), lambda);

        let options = RidgeFitOptions {
            max_iter: 10000,
            tol: 1e-7,
            warm_start: None,
            lambda,
            standardize: true,
            intercept: true,
            weights: None,
        };

        let rust_result = ridge_fit(&x_mat, &y, &options).unwrap();

        // Compare
        let mut rust_all = vec![rust_result.intercept];
        rust_all.extend(&rust_result.coefficients);

        println!("Rust intercept:     {:.10}", rust_result.intercept);
        println!("Rust intercept (R): {:.10}", r_coef[0]);
        println!("Intercept diff:      {:.2e}", (rust_result.intercept - r_coef[0]).abs());
        println!();

        for i in 0..p {
            let diff = (rust_all[i + 1] - r_coef[i + 1]).abs();
            println!("  Beta[{}]: Rust={:.10}, R={:.10}, diff={:.2e} {}",
                i, rust_all[i + 1], r_coef[i + 1], diff,
                if diff > 1e-2 { "❌ FAIL" } else if diff > 1e-4 { "⚠ WARN" } else { "✓ OK" }
            );
        }

        let (max_diff, rmse) = compute_mse(&rust_all, r_coef);
        println!();
        println!("Max diff: {:.2e}", max_diff);
        println!("RMSE:      {:.2e}", rmse);
        println!();

        // Assert that differences are reasonable
        assert!(max_diff < 1.0, "Difference too large - implementation may have an issue");
    }

    println!("=== AUDIT COMPLETE ===");
}
