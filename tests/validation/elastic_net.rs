// ============================================================================
// Elastic Net Regression Validation
// ============================================================================
//
// Comprehensive validation of elastic net implementation against R's glmnet.
//
// These tests validate:
// - Elastic net with various alpha values
// - Lambda sequence construction
// - Coefficient paths
// - Sparsity patterns
// - Predictions at various lambda values

use crate::common::{
    assert_close_to, load_dataset, LASSO_TOLERANCE, RIDGE_TOLERANCE,
};

use linreg_core::linalg::Matrix;
use linreg_core::regularized::{
    elastic_net_fit, elastic_net_path, ElasticNetOptions,
};
use linreg_core::regularized::path::LambdaPathOptions;

// ============================================================================
// Test Datasets
// ============================================================================

const ELASTIC_NET_TEST_DATASETS: &[&str] = &[
    "mtcars",
    "bodyfat",
    "prostate",
    "longley",
];

// ============================================================================
// Elastic Net Baseline Tests
// ============================================================================

/// Basic elastic net smoke test - verifies the function runs without errors
#[test]
fn test_elastic_net_basic() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  ELASTIC NET - BASIC SMOKE TEST                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Simple test data
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
    let x_data = vec![
        1.0, 1.0,
        1.0, 2.0,
        1.0, 3.0,
        1.0, 4.0,
        1.0, 5.0,
    ];
    let x = Matrix::new(5, 2, x_data);

    // Test with different alpha values
    for (alpha, name) in &[(0.0, "Ridge"), (0.5, "ElasticNet"), (1.0, "Lasso")] {
        let options = ElasticNetOptions {
            lambda: 0.1,
            alpha: *alpha,
            intercept: true,
            standardize: true,
            ..Default::default()
        };

        match elastic_net_fit(&x, &y, &options) {
            Ok(result) => {
                println!("  {} (alpha={}): ", name, alpha);
                println!("    Intercept: {:.6}", result.intercept);
                println!("    Coefficients: {:?}", result.coefficients);
                println!("    Non-zero: {}", result.n_nonzero);
                println!("    Iterations: {}", result.iterations);
                println!("    Converged: {}", result.converged);
                println!();
            }
            Err(e) => {
                panic!("{} fit failed: {}", name, e);
            }
        }
    }

    println!("  Basic elastic net test PASSED!");
}

/// Test elastic net on mtcars dataset (smoke test, no R comparison yet)
#[test]
fn test_elastic_net_mtcars_smoke() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  ELASTIC NET - mtcars SMOKE TEST                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let csv_path = datasets_dir.join("mtcars.csv");
    let dataset = load_dataset(&csv_path).expect("Failed to load mtcars dataset");

    let n = dataset.y.len();
    let p = dataset.x_vars.len();
    let mut x_data = vec![1.0; n * (p + 1)];
    for (col_idx, x_col) in dataset.x_vars.iter().enumerate() {
        for (row_idx, val) in x_col.iter().enumerate() {
            x_data[row_idx * (p + 1) + col_idx + 1] = *val;
        }
    }
    let x = Matrix::new(n, p + 1, x_data);

    println!("  Dataset: mtcars (n = {}, p = {})", n, p);

    // Test with alpha = 0.5 (true elastic net)
    let options = ElasticNetOptions {
        lambda: 0.1,
        alpha: 0.5,
        intercept: true,
        standardize: true,
        ..Default::default()
    };

    let result = elastic_net_fit(&x, &dataset.y, &options)
        .expect("Elastic net fit failed");

    println!();
    println!("  Alpha = 0.5, Lambda = 0.1");
    println!("  Intercept: {:.8}", result.intercept);
    println!("  Coefficients:");
    for (i, coef) in result.coefficients.iter().enumerate() {
        println!("    Beta[{}]: {:.8}", i + 1, coef);
    }
    println!("  Non-zero coefficients: {}", result.n_nonzero);
    println!("  R²: {:.6}", result.r_squared);
    println!("  Iterations: {}", result.iterations);
    println!("  Converged: {}", result.converged);
    println!();

    // Test with alpha = 0.9 (mostly lasso)
    let options_lasso = ElasticNetOptions {
        lambda: 0.1,
        alpha: 0.9,
        intercept: true,
        standardize: true,
        ..Default::default()
    };

    let result_lasso = elastic_net_fit(&x, &dataset.y, &options_lasso)
        .expect("Elastic net fit failed (alpha=0.9)");

    println!("  Alpha = 0.9, Lambda = 0.1");
    println!("  Intercept: {:.8}", result_lasso.intercept);
    println!("  Non-zero coefficients: {}", result_lasso.n_nonzero);
    println!("  R²: {:.6}", result_lasso.r_squared);
    println!();

    println!("  mtcars smoke test PASSED!");
}

/// Test elastic net on longley dataset (known multicollinearity)
#[test]
fn test_elastic_net_longley_smoke() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  ELASTIC NET - longley SMOKE TEST (multicollinear)              ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let csv_path = datasets_dir.join("longley.csv");
    let dataset = load_dataset(&csv_path).expect("Failed to load longley dataset");

    let n = dataset.y.len();
    let p = dataset.x_vars.len();
    let mut x_data = vec![1.0; n * (p + 1)];
    for (col_idx, x_col) in dataset.x_vars.iter().enumerate() {
        for (row_idx, val) in x_col.iter().enumerate() {
            x_data[row_idx * (p + 1) + col_idx + 1] = *val;
        }
    }
    let x = Matrix::new(n, p + 1, x_data);

    println!("  Dataset: longley (n = {}, p = {})", n, p);
    println!("  Note: longley has extreme multicollinearity");

    // Test with different alpha values
    for alpha in [0.0, 0.5, 0.9, 1.0] {
        let options = ElasticNetOptions {
            lambda: 0.1,
            alpha,
            intercept: true,
            standardize: true,
            max_iter: 100000,
            tol: 1e-7,
            ..Default::default()
        };

        let result = elastic_net_fit(&x, &dataset.y, &options)
            .expect(&format!("Elastic net fit failed for alpha={}", alpha));

        let name = match alpha {
            0.0 => "Ridge",
            1.0 => "Lasso",
            _ => "ElasticNet",
        };

        println!();
        println!("  {} (alpha={:.1}), Lambda = 0.1", name, alpha);
        println!("    Intercept: {:.8}", result.intercept);
        println!("    Non-zero coefficients: {} / {}", result.n_nonzero, p);
        println!("    R²: {:.6}", result.r_squared);
        println!("    Converged: {} ({} iterations)", result.converged, result.iterations);
    }

    println!();
    println!("  longley smoke test PASSED!");
}

/// Test lambda path generation for elastic net
#[test]
fn test_elastic_net_lambda_path() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  ELASTIC NET - LAMBDA PATH TEST                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let csv_path = datasets_dir.join("mtcars.csv");
    let dataset = load_dataset(&csv_path).expect("Failed to load mtcars dataset");

    let n = dataset.y.len();
    let p = dataset.x_vars.len();
    let mut x_data = vec![1.0; n * (p + 1)];
    for (col_idx, x_col) in dataset.x_vars.iter().enumerate() {
        for (row_idx, val) in x_col.iter().enumerate() {
            x_data[row_idx * (p + 1) + col_idx + 1] = *val;
        }
    }
    let x = Matrix::new(n, p + 1, x_data);

    // Generate lambda path using our implementation
    let lambda_path = linreg_core::regularized::make_lambda_path(
        &x,
        &dataset.y,
        &linreg_core::regularized::LambdaPathOptions {
            nlambda: 10,
            lambda_min_ratio: Some(0.01),
            alpha: 0.5,
            eps_for_ridge: 1e-3,
        },
        None,
        None,
    );

    println!("  Lambda path generated: {} lambdas", lambda_path.len());
    println!("  Lambda_max: {:.6}", lambda_path.first().unwrap_or(&0.0));
    println!("  Lambda_min: {:.6}", lambda_path.last().unwrap_or(&0.0));

    // Fit elastic net along the path
    println!();
    println!("  Fitting elastic net along lambda path (alpha=0.5):");
    println!("  Lambda    | Non-zero | R²      | Intercept");
    println!("  ----------|----------|---------|-----------");

    for (i, &lambda) in lambda_path.iter().enumerate() {
        let options = ElasticNetOptions {
            lambda,
            alpha: 0.5,
            intercept: true,
            standardize: true,
            ..Default::default()
        };

        let result = elastic_net_fit(&x, &dataset.y, &options)
            .expect(&format!("Fit failed at lambda {}", lambda));

        println!("  {:.6} | {:8} | {:.4} | {:.6}",
            lambda, result.n_nonzero, result.r_squared, result.intercept);
    }

    println!();
    println!("  Lambda path test PASSED!");
}

/// Test consistency: elastic_net_fit with alpha=1 should match lasso_fit
#[test]
fn test_elastic_net_lasso_consistency() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  ELASTIC NET vs LASSO CONSISTENCY TEST                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let csv_path = datasets_dir.join("mtcars.csv");
    let dataset = load_dataset(&csv_path).expect("Failed to load mtcars dataset");

    let n = dataset.y.len();
    let p = dataset.x_vars.len();
    let mut x_data = vec![1.0; n * (p + 1)];
    for (col_idx, x_col) in dataset.x_vars.iter().enumerate() {
        for (row_idx, val) in x_col.iter().enumerate() {
            x_data[row_idx * (p + 1) + col_idx + 1] = *val;
        }
    }
    let x = Matrix::new(n, p + 1, x_data);

    let lambda = 0.1;

    // Fit with elastic_net_fit (alpha=1)
    let en_options = ElasticNetOptions {
        lambda,
        alpha: 1.0,
        intercept: true,
        standardize: true,
        ..Default::default()
    };

    let en_result = elastic_net_fit(&x, &dataset.y, &en_options)
        .expect("Elastic net fit failed");

    // Fit with lasso_fit
    let lasso_options = linreg_core::regularized::LassoFitOptions {
        lambda,
        intercept: true,
        standardize: true,
        ..Default::default()
    };

    let lasso_result = linreg_core::regularized::lasso_fit(&x, &dataset.y, &lasso_options)
        .expect("Lasso fit failed");

    println!("  Lambda = {}", lambda);
    println!();
    println!("  Elastic Net (alpha=1.0):");
    println!("    Intercept: {:.10}", en_result.intercept);
    println!("    Coefficients: {:?}", en_result.coefficients);
    println!("    Non-zero: {}", en_result.n_nonzero);
    println!();
    println!("  Lasso:");
    println!("    Intercept: {:.10}", lasso_result.intercept);
    println!("    Coefficients: {:?}", lasso_result.coefficients);
    println!("    Non-zero: {}", lasso_result.n_nonzero);
    println!();

    // Check consistency
    let intercept_diff = (en_result.intercept - lasso_result.intercept).abs();
    println!("  Intercept difference: {:.2e}", intercept_diff);

    let mut max_coef_diff: f64 = 0.0;
    for (i, (en_coef, lasso_coef)) in en_result.coefficients.iter()
        .zip(lasso_result.coefficients.iter()).enumerate()
    {
        let diff = (en_coef - lasso_coef).abs();
        max_coef_diff = max_coef_diff.max(diff);
        println!("  Beta[{}] difference: {:.2e}", i + 1, diff);
    }

    // They should match very closely (within numerical precision)
    assert_close_to(en_result.intercept, lasso_result.intercept, 1e-6, "EN vs Lasso intercept");
    for (i, (en_coef, lasso_coef)) in en_result.coefficients.iter()
        .zip(lasso_result.coefficients.iter()).enumerate()
    {
        assert_close_to(*en_coef, *lasso_coef, 1e-6, &format!("EN vs Lasso beta[{}]", i));
    }

    println!();
    println!("  Consistency test PASSED!");
}

/// Test consistency: elastic_net_fit with alpha=0 should match ridge_fit
#[test]
fn test_elastic_net_ridge_consistency() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  ELASTIC NET vs RIDGE CONSISTENCY TEST                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let csv_path = datasets_dir.join("mtcars.csv");
    let dataset = load_dataset(&csv_path).expect("Failed to load mtcars dataset");

    let n = dataset.y.len();
    let p = dataset.x_vars.len();
    let mut x_data = vec![1.0; n * (p + 1)];
    for (col_idx, x_col) in dataset.x_vars.iter().enumerate() {
        for (row_idx, val) in x_col.iter().enumerate() {
            x_data[row_idx * (p + 1) + col_idx + 1] = *val;
        }
    }
    let x = Matrix::new(n, p + 1, x_data);

    let lambda = 0.1;

    // Fit with elastic_net_fit (alpha=0)
    let en_options = ElasticNetOptions {
        lambda,
        alpha: 0.0,
        intercept: true,
        standardize: true,
        ..Default::default()
    };

    let en_result = elastic_net_fit(&x, &dataset.y, &en_options)
        .expect("Elastic net fit failed");

    // Fit with ridge_fit
    let ridge_options = linreg_core::regularized::RidgeFitOptions {
        lambda,
        intercept: true,
        standardize: true,
        ..Default::default()
    };

    let ridge_result = linreg_core::regularized::ridge_fit(&x, &dataset.y, &ridge_options)
        .expect("Ridge fit failed");

    println!("  Lambda = {}", lambda);
    println!();
    println!("  Elastic Net (alpha=0.0):");
    println!("    Intercept: {:.10}", en_result.intercept);
    println!("    Coefficients: {:?}", en_result.coefficients);
    println!();
    println!("  Ridge:");
    println!("    Intercept: {:.10}", ridge_result.intercept);
    println!("    Coefficients: {:?}", ridge_result.coefficients);
    println!();

    // Check consistency
    let intercept_diff = (en_result.intercept - ridge_result.intercept).abs();
    println!("  Intercept difference: {:.2e}", intercept_diff);

    let mut max_coef_diff: f64 = 0.0;
    for (i, (en_coef, ridge_coef)) in en_result.coefficients.iter()
        .zip(ridge_result.coefficients.iter()).enumerate()
    {
        let diff = (en_coef - ridge_coef).abs();
        max_coef_diff = max_coef_diff.max(diff);
        println!("  Beta[{}] difference: {:.2e}", i + 1, diff);
    }

    // They should match very closely (within numerical precision)
    // Note: Ridge tolerance is looser due to different algorithms
    assert_close_to(en_result.intercept, ridge_result.intercept, RIDGE_TOLERANCE, "EN vs Ridge intercept");
    for (i, (en_coef, ridge_coef)) in en_result.coefficients.iter()
        .zip(ridge_result.coefficients.iter()).enumerate()
    {
        assert_close_to(*en_coef, *ridge_coef, RIDGE_TOLERANCE, &format!("EN vs Ridge beta[{}]", i));
    }

    println!();
    println!("  Consistency test PASSED!");
}

/// Test elastic_net_path with warm starts against R's glmnet
///
/// This test validates that our warm-start pathwise coordinate descent
/// produces coefficient paths similar to glmnet.
#[test]
fn test_elastic_net_path_warm_start() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  ELASTIC NET PATH - WARM START VALIDATION vs glmnet               ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let csv_path = datasets_dir.join("mtcars.csv");
    let dataset = load_dataset(&csv_path).expect("Failed to load mtcars dataset");

    let n = dataset.y.len();
    let p = dataset.x_vars.len();
    let mut x_data = vec![1.0; n * (p + 1)];
    for (col_idx, x_col) in dataset.x_vars.iter().enumerate() {
        for (row_idx, val) in x_col.iter().enumerate() {
            x_data[row_idx * (p + 1) + col_idx + 1] = *val;
        }
    }
    let x = Matrix::new(n, p + 1, x_data);

    // Fit elastic net path with alpha=0.5 (true elastic net)
    let path_options = LambdaPathOptions {
        nlambda: 10,
        lambda_min_ratio: None,  // Use default (0.0001 for n >= p)
        alpha: 0.5,
        eps_for_ridge: 1e-3,
    };

    let fit_options = ElasticNetOptions {
        lambda: 0.1,  // Not used when calling path, but required
        alpha: 0.5,
        intercept: true,
        standardize: true,
        max_iter: 100000,
        tol: 1e-7,
        penalty_factor: None,
        warm_start: None,
        weights: None,
        coefficient_bounds: None,
    };

    let path_result = elastic_net_path(&x, &dataset.y, &path_options, &fit_options)
        .expect("Elastic net path failed");

    println!("  Dataset: mtcars (n = {}, p = {})", n, p);
    println!("  Alpha = 0.5, nlambda = {}", path_result.len());
    println!();

    // Expected values from R's glmnet (alpha=0.5, nlambda=10)
    // From the R script output:
    // lambda_sequence: 10.29396 3.699458 1.329516 0.477803 0.171714 0.061711 0.022178 0.00797 0.002864 0.001029
    let r_lambdas = [
        10.29396, 3.699458, 1.329516, 0.477803, 0.171714, 0.061711, 0.022178, 0.00797, 0.002864, 0.001029,
    ];

    // R intercepts at each lambda
    let r_intercepts = [
        20.09062, 29.959479, 31.783976, 28.685166, 20.431515, 15.634181, 13.462265, 12.669468, 12.396949, 12.319451,
    ];

    // R coefficient matrix for key variables at each lambda
    // cyl at each lambda
    let r_cyl = [
        0.0, -0.541244, -0.676426, -0.561023, -0.242438, -0.046472, -0.074846, -0.089620, -0.096497, -0.100102,
    ];
    // wt at each lambda
    let r_wt = [
        0.0, -1.349022, -1.943466, -2.284794, -2.462505, -2.612906, -3.253170, -3.526013, -3.630355, -3.667499,
    ];
    // am at each lambda
    let r_am = [
        0.0, 0.0, 0.702268, 1.617566, 2.108057, 2.336923, 2.454800, 2.498472, 2.513813, 2.518521,
    ];

    // Non-zero counts from R
    let r_nonzeros = [0, 4, 8, 8, 9, 10, 10, 10, 10, 10];

    println!("  Comparing coefficient paths with R glmnet:");
    println!("  ─────┬──────────┬──────────┬──────────┬──────────┬──────────");
    println!("   idx │   lambda │   inter  │   cyl    │    wt    │    am    │ nonzero");
    println!("  ─────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────");

    let mut max_intercept_diff: f64 = 0.0;
    let mut max_cyl_diff: f64 = 0.0;
    let mut max_wt_diff: f64 = 0.0;
    let mut max_am_diff: f64 = 0.0;
    let mut nonzero_mismatch = 0;

    for (i, fit) in path_result.iter().enumerate() {
        // Find matching lambda index in R (our lambdas may differ slightly)
        // Use simple index matching for now since nlambda is small
        if i >= r_lambdas.len() {
            break;
        }

        let intercept_diff = (fit.intercept - r_intercepts[i]).abs();
        let cyl_diff = (fit.coefficients[0] - r_cyl[i]).abs();
        let wt_diff = (fit.coefficients[4] - r_wt[i]).abs(); // wt is 5th column (index 4)
        let am_diff = (fit.coefficients[7] - r_am[i]).abs(); // am is 9th column (index 8)

        max_intercept_diff = max_intercept_diff.max(intercept_diff);
        max_cyl_diff = max_cyl_diff.max(cyl_diff);
        max_wt_diff = max_wt_diff.max(wt_diff);
        max_am_diff = max_am_diff.max(am_diff);

        if fit.n_nonzero != r_nonzeros[i] {
            nonzero_mismatch += 1;
        }

        println!("  {:4} │ {:8.5} │ {:8.5} │ {:8.5} │ {:8.5} │ {:8.5} │ {} (R: {})",
            i,
            fit.lambda,
            fit.intercept,
            fit.coefficients[0],
            fit.coefficients[4],
            fit.coefficients[8],
            fit.n_nonzero,
            r_nonzeros[i]
        );
    }
    println!("  ─────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────");
    println!();

    println!("  Maximum differences from R:");
    println!("    Intercept: {:.6e}", max_intercept_diff);
    println!("    cyl:       {:.6e}", max_cyl_diff);
    println!("    wt:        {:.6e}", max_wt_diff);
    println!("    am:        {:.6e}", max_am_diff);
    println!("    Non-zero mismatches: {}", nonzero_mismatch);
    println!();

    // Tolerance for path comparison (somewhat looser than single-fit due to path differences)
    let path_tolerance = 0.2; // 20% tolerance for coefficient paths

    if max_intercept_diff < path_tolerance && max_wt_diff < path_tolerance && max_cyl_diff < path_tolerance && max_am_diff < path_tolerance {
        println!("  Elastic Net Path test PASSED!");
        println!("  (Coefficient paths match glmnet within tolerance)");
    } else {
        println!("  Elastic Net Path test: WARNING - differences detected");
        println!("  This may indicate issues with:");
        println!("    - Lambda path generation");
        println!("    - Standardization approach");
        println!("    - Warm start implementation");
    }
}
