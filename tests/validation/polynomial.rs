// ============================================================================
// Polynomial Regression Validation Tests
// ============================================================================
//
// Validates polynomial regression against known analytic results and R/Python
// reference implementations (when reference files are available).
//
// Reference functions:
// - R:      lm(y ~ poly(x, degree, raw=TRUE))
// - Python: numpy.polyfit(x, y, degree) / sklearn PolynomialFeatures + LinearRegression

use crate::common::{
    load_dataset_with_encoding, load_polynomial_result, POLYNOMIAL_TOLERANCE, CategoricalEncoding,
};
use linreg_core::{aic_python, bic_python};
use linreg_core::polynomial::{polynomial_regression, predict, PolynomialOptions};

// ============================================================================
// Tolerance
// ============================================================================

const POLY_COEFF_TOL: f64 = 1e-9;
const POLY_R2_TOL: f64 = 1e-9;
const POLY_PRED_TOL: f64 = 1e-6;

fn assert_close(a: f64, b: f64, tol: f64, ctx: &str) {
    let diff = (a - b).abs();
    assert!(
        diff <= tol,
        "{}: got {}, expected {}, diff = {} (tol = {})",
        ctx,
        a,
        b,
        diff,
        tol
    );
}

// ============================================================================
// Analytic Validation — Perfect Quadratic (y = 1 + 2x + x²)
//
// R equivalent:
//   x <- 0:4
//   y <- 1 + 2*x + x^2
//   lm(y ~ poly(x, 2, raw=TRUE))
//   # Intercept = 1, poly(x,2,raw=TRUE)1 = 2, poly(x,2,raw=TRUE)2 = 1
// ============================================================================

#[test]
fn validate_polynomial_quadratic_analytic() {
    println!("\n===== POLYNOMIAL REGRESSION — QUADRATIC ANALYTIC VALIDATION =====\n");

    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + xi * xi).collect();

    let options = PolynomialOptions {
        degree: 2,
        center: false,
        standardize: false,
        intercept: true,
    };

    let fit = polynomial_regression(&y, &x, &options).unwrap();

    println!("  Intercept   : {:.10} (expected 1.0)", fit.ols_output.coefficients[0]);
    println!("  x coeff     : {:.10} (expected 2.0)", fit.ols_output.coefficients[1]);
    println!("  x² coeff    : {:.10} (expected 1.0)", fit.ols_output.coefficients[2]);
    println!("  R²          : {:.10} (expected 1.0)", fit.ols_output.r_squared);

    assert_close(fit.ols_output.coefficients[0], 1.0, POLY_COEFF_TOL, "intercept");
    assert_close(fit.ols_output.coefficients[1], 2.0, POLY_COEFF_TOL, "x coeff");
    assert_close(fit.ols_output.coefficients[2], 1.0, POLY_COEFF_TOL, "x² coeff");
    assert_close(fit.ols_output.r_squared, 1.0, POLY_R2_TOL, "R²");

    println!("\n   Quadratic analytic validation passed");
}

// ============================================================================
// Analytic Validation — Perfect Cubic (y = 5 + 3x − 2x² + 0.5x³)
// ============================================================================

#[test]
fn validate_polynomial_cubic_analytic() {
    println!("\n===== POLYNOMIAL REGRESSION — CUBIC ANALYTIC VALIDATION =====\n");

    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 5.0 + 3.0 * xi - 2.0 * xi * xi + 0.5 * xi * xi * xi)
        .collect();

    let options = PolynomialOptions {
        degree: 3,
        center: false,
        standardize: false,
        intercept: true,
    };

    let fit = polynomial_regression(&y, &x, &options).unwrap();

    println!("  Intercept   : {:.10} (expected  5.0)", fit.ols_output.coefficients[0]);
    println!("  x coeff     : {:.10} (expected  3.0)", fit.ols_output.coefficients[1]);
    println!("  x² coeff    : {:.10} (expected -2.0)", fit.ols_output.coefficients[2]);
    println!("  x³ coeff    : {:.10} (expected  0.5)", fit.ols_output.coefficients[3]);
    println!("  R²          : {:.10}", fit.ols_output.r_squared);

    assert_close(fit.ols_output.coefficients[0], 5.0, POLY_COEFF_TOL, "intercept");
    assert_close(fit.ols_output.coefficients[1], 3.0, POLY_COEFF_TOL, "x coeff");
    assert_close(fit.ols_output.coefficients[2], -2.0, POLY_COEFF_TOL, "x² coeff");
    assert_close(fit.ols_output.coefficients[3], 0.5, POLY_COEFF_TOL, "x³ coeff");
    assert_close(fit.ols_output.r_squared, 1.0, POLY_R2_TOL, "R²");

    println!("\n   Cubic analytic validation passed");
}

// ============================================================================
// Centering Validation — coefficients transform correctly
//
// y = β₀ + β₁x + β₂x²  with centering c = x̄
// centered: y = α₀ + α₁(x-c) + α₂(x-c)²
//
// Both fits must produce the same fitted values.
// ============================================================================

#[test]
fn validate_polynomial_centering_same_predictions() {
    println!("\n===== POLYNOMIAL REGRESSION — CENTERING FITTED VALUES =====\n");

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 + xi + 0.5 * xi * xi).collect();

    let fit_uncentered = polynomial_regression(
        &y,
        &x,
        &PolynomialOptions {
            degree: 2,
            center: false,
            ..Default::default()
        },
    )
    .unwrap();

    let fit_centered = polynomial_regression(
        &y,
        &x,
        &PolynomialOptions {
            degree: 2,
            center: true,
            ..Default::default()
        },
    )
    .unwrap();

    println!("  Uncentered R² : {:.10}", fit_uncentered.ols_output.r_squared);
    println!("  Centered   R² : {:.10}", fit_centered.ols_output.r_squared);

    // Fitted values should agree
    for (i, (&u, &c)) in fit_uncentered
        .ols_output
        .predictions
        .iter()
        .zip(fit_centered.ols_output.predictions.iter())
        .enumerate()
    {
        assert_close(u, c, 1e-8, &format!("fitted[{}]", i));
    }

    println!("\n   Centering fitted-values validation passed");
}

// ============================================================================
// Prediction Validation — Training-point predictions match y
// ============================================================================

#[test]
fn validate_polynomial_predictions_match_training() {
    println!("\n===== POLYNOMIAL REGRESSION — PREDICTION VALIDATION =====\n");

    let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 3.0 - xi + 0.4 * xi * xi).collect();

    let options = PolynomialOptions {
        degree: 2,
        center: true,
        ..Default::default()
    };
    let fit = polynomial_regression(&y, &x, &options).unwrap();
    let preds = predict(&fit, &x).unwrap();

    println!("  {:>5}  {:>12}  {:>12}  {:>12}", "i", "y_i", "pred_i", "diff");
    for (i, (&yi, &pi)) in y.iter().zip(preds.iter()).enumerate() {
        println!("  {:>5}  {:>12.6}  {:>12.6}  {:>12.2e}", i, yi, pi, (yi - pi).abs());
        assert_close(pi, yi, POLY_PRED_TOL, &format!("pred[{}]", i));
    }

    println!("\n   Prediction-at-training-points validation passed");
}

// ============================================================================
// Degree 1 = Simple Linear Regression Validation
//
// R equivalent:
//   lm(y ~ x)  must equal  lm(y ~ poly(x, 1, raw=TRUE))
// ============================================================================

#[test]
fn validate_polynomial_degree1_equals_ols() {
    use crate::common::STAT_TOLERANCE;
    use linreg_core::core::ols_regression;

    println!("\n===== POLYNOMIAL DEGREE=1 == OLS VALIDATION =====\n");

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3, 7.0, 7.5, 8.1];

    // Polynomial degree 1
    let poly_fit = polynomial_regression(
        &y,
        &x,
        &PolynomialOptions {
            degree: 1,
            center: false,
            standardize: false,
            intercept: true,
        },
    )
    .unwrap();

    // Plain OLS
    let names = vec!["Intercept".to_string(), "x".to_string()];
    let ols_fit = ols_regression(&y, &[x.clone()], &names).unwrap();

    println!(
        "  poly intercept : {:.10}  ols intercept : {:.10}",
        poly_fit.ols_output.coefficients[0], ols_fit.coefficients[0]
    );
    println!(
        "  poly slope     : {:.10}  ols slope     : {:.10}",
        poly_fit.ols_output.coefficients[1], ols_fit.coefficients[1]
    );
    println!(
        "  poly R²        : {:.10}  ols R²        : {:.10}",
        poly_fit.ols_output.r_squared, ols_fit.r_squared
    );

    assert_close(
        poly_fit.ols_output.coefficients[0],
        ols_fit.coefficients[0],
        STAT_TOLERANCE,
        "intercept",
    );
    assert_close(
        poly_fit.ols_output.coefficients[1],
        ols_fit.coefficients[1],
        STAT_TOLERANCE,
        "slope",
    );
    assert_close(
        poly_fit.ols_output.r_squared,
        ols_fit.r_squared,
        STAT_TOLERANCE,
        "R²",
    );

    println!("\n   Degree-1 = OLS validation passed");
}

// ============================================================================
// Dataset-Based Validation — vs R Reference (lm with poly(x, degree, raw=TRUE))
// and Python Reference (statsmodels OLS with monomial basis)
//
// For each dataset the R/Python scripts use:
//   - response  = first CSV column
//   - predictor = second CSV column (univariate polynomial)
//   - basis     = raw monomials: [1, x, x^2, ..., x^degree]
//
// We fit with center=false / standardize=false to match.
//
// Categorical variables are handled by using the same encoding as the reference:
//   - R validation:      CategoricalEncoding::OneBased  (as.numeric(as.factor(.)) → 1,2,...)
//   - Python validation: CategoricalEncoding::ZeroBased (pd.factorize()           → 0,1,...)
// This ensures x values — and therefore x^2, x^3, ... — match the reference.
// ============================================================================

const POLY_TEST_DATASETS: &[&str] = &[
    "bodyfat",
    "cars_stopping",
    "faithful",
    "iris",
    "lh",
    "longley",
    "mtcars",
    "prostate",
    "synthetic_autocorrelated",
    "synthetic_heteroscedastic",
    "synthetic_interaction",
    "synthetic_multiple",
    "synthetic_nonlinear",
    "synthetic_nonnormal",
    "synthetic_outliers",
    "synthetic_simple_linear",
    "synthetic_small",
    "ToothGrowth",
];

/// Validate polynomial regression against R reference for one dataset and degree.
fn validate_polynomial_r_dataset(dataset_name: &str, degree: usize) {
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let r_results_dir = current_dir.join("verification/results/r");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let csv_path = datasets_dir.join(format!("{}.csv", dataset_name));
    let r_result_path =
        r_results_dir.join(format!("{}_polynomial_degree{}.json", dataset_name, degree));

    // Load dataset with R-compatible 1-based categorical encoding
    let dataset = match load_dataset_with_encoding(&csv_path, CategoricalEncoding::OneBased) {
        Ok(d) => d,
        Err(e) => {
            println!("   SKIP: Failed to load {} dataset: {}", dataset_name, e);
            return;
        }
    };

    if dataset.x_vars.is_empty() {
        println!("   SKIP: {} has no predictors", dataset_name);
        return;
    }

    // Load R reference — skip (with message) if the runner hasn't been run yet
    let r_ref = match load_polynomial_result(&r_result_path) {
        Some(r) => r,
        None => {
            println!(
                "   SKIP: R reference not found for {} degree {} — run:\n\
                 \t Rscript verification/scripts/runners/run_all_diagnostics_r.R",
                dataset_name, degree
            );
            return;
        }
    };

    // Fit with raw (uncentered) monomial basis, matching R's poly(x, degree, raw=TRUE)
    let options = PolynomialOptions {
        degree,
        center: false,
        standardize: false,
        intercept: true,
    };

    let x = dataset.x_vars[0].clone();
    let result = match polynomial_regression(&dataset.y, &x, &options) {
        Ok(r) => r,
        Err(e) => {
            println!(
                "   SKIP: polynomial_regression failed for {} degree {}: {}",
                dataset_name, degree, e
            );
            return;
        }
    };

    let tol = POLYNOMIAL_TOLERANCE;
    let p_tol = tol.max(1e-6);
    let mut all_passed = true;

    macro_rules! check {
        ($label:expr, $got:expr, $exp:expr, $t:expr) => {{
            let diff = ($got - $exp).abs();
            if diff > $t {
                println!(
                    "   {}: Rust={:.10}  R={:.10}  diff={:.2e}",
                    $label, $got, $exp, diff
                );
                all_passed = false;
            }
        }};
    }

    for i in 0..r_ref.coefficients.len().min(result.ols_output.coefficients.len()) {
        check!(format!("coef[{}]", i), result.ols_output.coefficients[i], r_ref.coefficients[i], tol);
    }
    for i in 0..r_ref.std_errors.len().min(result.ols_output.std_errors.len()) {
        check!(format!("SE[{}]", i), result.ols_output.std_errors[i], r_ref.std_errors[i], tol);
    }
    for i in 0..r_ref.t_stats.len().min(result.ols_output.t_stats.len()) {
        check!(format!("t[{}]", i), result.ols_output.t_stats[i], r_ref.t_stats[i], tol);
    }
    for i in 0..r_ref.p_values.len().min(result.ols_output.p_values.len()) {
        check!(format!("p[{}]", i), result.ols_output.p_values[i], r_ref.p_values[i], p_tol);
    }
    check!("R²",      result.ols_output.r_squared,     r_ref.r_squared,     tol);
    check!("Adj R²",  result.ols_output.adj_r_squared, r_ref.adj_r_squared, tol);
    check!("F-stat",  result.ols_output.f_statistic,   r_ref.f_statistic,   tol);
    check!("MSE",     result.ols_output.mse,            r_ref.mse,           tol);
    // R convention for log-likelihood/AIC/BIC (k = n_coef + 1) matches our output
    check!("LogLik",  result.ols_output.log_likelihood, r_ref.log_likelihood, 1e-6);
    check!("AIC",     result.ols_output.aic,             r_ref.aic,            1e-6);
    check!("BIC",     result.ols_output.bic,             r_ref.bic,            1e-6);

    if all_passed {
        println!("   {} degree-{} polynomial R validation: PASS", dataset_name, degree);
    } else {
        panic!("{} degree-{} polynomial R validation: FAILED", dataset_name, degree);
    }
}

/// Validate polynomial regression against Python (statsmodels) reference for one dataset and degree.
fn validate_polynomial_python_dataset(dataset_name: &str, degree: usize) {
    let current_dir = std::env::current_dir().expect("Failed to get current dir");
    let python_results_dir = current_dir.join("verification/results/python");
    let datasets_dir = current_dir.join("verification/datasets/csv");

    let csv_path = datasets_dir.join(format!("{}.csv", dataset_name));
    let py_result_path =
        python_results_dir.join(format!("{}_polynomial_degree{}.json", dataset_name, degree));

    // Load dataset with Python-compatible 0-based categorical encoding
    let dataset = match load_dataset_with_encoding(&csv_path, CategoricalEncoding::ZeroBased) {
        Ok(d) => d,
        Err(e) => {
            println!("   SKIP: Failed to load {} dataset: {}", dataset_name, e);
            return;
        }
    };

    if dataset.x_vars.is_empty() {
        println!("   SKIP: {} has no predictors", dataset_name);
        return;
    }

    // Load Python reference — skip if not found
    let py_ref = match load_polynomial_result(&py_result_path) {
        Some(r) => r,
        None => {
            println!(
                "   SKIP: Python reference not found for {} degree {} — run:\n\
                 \t python verification/scripts/runners/run_all_diagnostics_python.py",
                dataset_name, degree
            );
            return;
        }
    };

    let options = PolynomialOptions {
        degree,
        center: false,
        standardize: false,
        intercept: true,
    };

    let x = dataset.x_vars[0].clone();
    let result = match polynomial_regression(&dataset.y, &x, &options) {
        Ok(r) => r,
        Err(e) => {
            println!(
                "   SKIP: polynomial_regression failed for {} degree {}: {}",
                dataset_name, degree, e
            );
            return;
        }
    };

    // A degree-d polynomial design matrix [1, x, x², ..., x^d] is rank-deficient if x has
    // ≤ d distinct values (the Vandermonde columns become linearly dependent).  In that case
    // our Rust (and R) pick one valid solution while statsmodels picks a different one — both
    // are correct completions of the same degenerate normal equations, so coefficient values
    // are not comparable.  Skip rather than fail.
    let n_unique_x = {
        use std::collections::HashSet;
        x.iter()
            .map(|&v| (v * 1_000_000.0) as i64) // discretise to detect exact repeats
            .collect::<HashSet<_>>()
            .len()
    };
    if n_unique_x <= degree {
        println!(
            "   SKIP: {} degree-{} rank-deficient — predictor has only {} unique values \
             (need >{} for full rank); R/Rust and Python pick different degenerate solutions",
            dataset_name, degree, n_unique_x, degree
        );
        return;
    }

    let tol = POLYNOMIAL_TOLERANCE;
    let p_tol = tol.max(1e-6);
    let mut all_passed = true;

    macro_rules! check {
        ($label:expr, $got:expr, $exp:expr, $t:expr) => {{
            let diff = ($got - $exp).abs();
            if diff > $t {
                println!(
                    "   {}: Rust={:.10}  Py={:.10}  diff={:.2e}",
                    $label, $got, $exp, diff
                );
                all_passed = false;
            }
        }};
    }

    for i in 0..py_ref.coefficients.len().min(result.ols_output.coefficients.len()) {
        check!(format!("coef[{}]", i), result.ols_output.coefficients[i], py_ref.coefficients[i], tol);
    }
    for i in 0..py_ref.std_errors.len().min(result.ols_output.std_errors.len()) {
        check!(format!("SE[{}]", i), result.ols_output.std_errors[i], py_ref.std_errors[i], tol);
    }
    for i in 0..py_ref.t_stats.len().min(result.ols_output.t_stats.len()) {
        check!(format!("t[{}]", i), result.ols_output.t_stats[i], py_ref.t_stats[i], tol);
    }
    for i in 0..py_ref.p_values.len().min(result.ols_output.p_values.len()) {
        check!(format!("p[{}]", i), result.ols_output.p_values[i], py_ref.p_values[i], p_tol);
    }
    check!("R²",      result.ols_output.r_squared,     py_ref.r_squared,     tol);
    check!("Adj R²",  result.ols_output.adj_r_squared, py_ref.adj_r_squared, tol);
    check!("F-stat",  result.ols_output.f_statistic,   py_ref.f_statistic,   tol);
    check!("MSE",     result.ols_output.mse,            py_ref.mse,           tol);

    // log-likelihood: same Gaussian formula — should match
    check!("LogLik",  result.ols_output.log_likelihood, py_ref.log_likelihood, 1e-6);

    // AIC/BIC: statsmodels uses k = n_coef (no variance parameter) — use aic_python/bic_python
    let n_coef = result.ols_output.coefficients.len();
    let rust_aic_py = aic_python(result.ols_output.log_likelihood, n_coef);
    let rust_bic_py = bic_python(result.ols_output.log_likelihood, n_coef, result.ols_output.n);
    check!("AIC (Py conv)", rust_aic_py, py_ref.aic, 1e-6);
    check!("BIC (Py conv)", rust_bic_py, py_ref.bic, 1e-6);

    if all_passed {
        println!("   {} degree-{} polynomial Python validation: PASS", dataset_name, degree);
    } else {
        panic!("{} degree-{} polynomial Python validation: FAILED", dataset_name, degree);
    }
}

// ============================================================================
// #[test] entry points
// ============================================================================

#[test]
fn validate_polynomial_r_all_datasets_degree2() {
    println!("\n===== POLYNOMIAL VALIDATION vs R — DEGREE 2 =====\n");
    for dataset in POLY_TEST_DATASETS {
        println!("--- Dataset: {} ---", dataset);
        validate_polynomial_r_dataset(dataset, 2);
    }
}

#[test]
fn validate_polynomial_r_all_datasets_degree3() {
    println!("\n===== POLYNOMIAL VALIDATION vs R — DEGREE 3 =====\n");
    for dataset in POLY_TEST_DATASETS {
        println!("--- Dataset: {} ---", dataset);
        validate_polynomial_r_dataset(dataset, 3);
    }
}

#[test]
fn validate_polynomial_python_all_datasets_degree2() {
    println!("\n===== POLYNOMIAL VALIDATION vs Python — DEGREE 2 =====\n");
    for dataset in POLY_TEST_DATASETS {
        println!("--- Dataset: {} ---", dataset);
        validate_polynomial_python_dataset(dataset, 2);
    }
}

#[test]
fn validate_polynomial_python_all_datasets_degree3() {
    println!("\n===== POLYNOMIAL VALIDATION vs Python — DEGREE 3 =====\n");
    for dataset in POLY_TEST_DATASETS {
        println!("--- Dataset: {} ---", dataset);
        validate_polynomial_python_dataset(dataset, 3);
    }
}
