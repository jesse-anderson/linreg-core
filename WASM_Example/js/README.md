# linreg-core

[![CI](https://github.com/jesse-anderson/linreg-core/actions/workflows/ci.yml/badge.svg)](https://github.com/jesse-anderson/linreg-core/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/jesse-anderson/linreg-core/main/.github/coverage-badge.json)](https://github.com/jesse-anderson/linreg-core/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)
[![Crates.io](https://img.shields.io/crates/v/linreg-core?color=orange)](https://crates.io/crates/linreg-core)
[![npm](https://img.shields.io/npm/v/linreg-core?color=red)](https://www.npmjs.com/package/linreg-core)
[![PyPI](https://img.shields.io/pypi/v/linreg-core)](https://pypi.org/project/linreg-core/)
[![docs.rs](https://img.shields.io/badge/docs.rs-linreg__core-green)](https://docs.rs/linreg-core)
[![Live Demo](https://img.shields.io/badge/demo-online-brightgreen)](https://jesse-anderson.net/linreg-core/)


A lightweight, self-contained linear regression library written in Rust. Compiles to WebAssembly for browser use, Python bindings via PyO3, a native Windows DLL for Excel VBA, or runs as a native Rust crate.

**Key design principle:** All linear algebra and statistical distribution functions are implemented from scratch — no external math libraries required. This keeps binary sizes small and makes the crate highly portable.

**[Live Demo Link](https://jesse-anderson.net/linreg-core/)**
---

## Table of Contents

| Section | Description |
|---------|-------------|
| [Features](#features) | Regression methods, model statistics, diagnostic tests |
| [Rust Usage](#rust-usage) | Native Rust crate usage |
| [WebAssembly Usage](#webassembly-usage) | Browser/JavaScript usage |
| [Python Usage](#python-usage) | Python bindings via PyO3 |
| [VBA / Excel Usage](#vba--excel-usage) | Excel VBA via native Windows DLL |
| [Feature Flags](#feature-flags) | Build configuration options |
| [Validation](#validation) | Testing and verification |
| [Implementation Notes](#implementation-notes) | Technical details |

---

## Features

### Regression Methods
- **OLS Regression:** Coefficients, standard errors, t-statistics, p-values, confidence intervals, model selection criteria (AIC, BIC, log-likelihood)
- **Ridge Regression:** L2-regularized regression with optional standardization, effective degrees of freedom, model selection criteria
- **Lasso Regression:** L1-regularized regression via coordinate descent with automatic variable selection, convergence tracking, model selection criteria
- **Elastic Net:** Combined L1 + L2 regularization for variable selection with multicollinearity handling, active set convergence, model selection criteria
- **LOESS:** Locally estimated scatterplot smoothing for non-parametric curve fitting with configurable span, polynomial degree, and robust fitting
- **WLS (Weighted Least Squares):** Regression with observation weights for heteroscedastic data, includes confidence intervals
- **K-Fold Cross Validation:** Model evaluation and hyperparameter tuning for all regression types (OLS, Ridge, Lasso, Elastic Net) with customizable folds, shuffling, and seeding
- **Lambda Path Generation:** Create regularization paths for cross-validation

### Model Statistics
- **Fit Metrics:** R-squared, Adjusted R-squared, F-statistic, F-test p-value
- **Error Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)
- **Model Selection:** Log-likelihood, AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion)
- **Residuals:** Raw residuals, standardized residuals, fitted values, leverage (hat matrix diagonal)
- **Multicollinearity:** Variance Inflation Factor (VIF) for each predictor

### Diagnostic Tests
| Category | Tests |
|----------|-------|
| **Linearity** | Rainbow Test, Harvey-Collier Test, RESET Test |
| **Heteroscedasticity** | Breusch-Pagan (Koenker variant), White Test (R & Python methods) |
| **Normality** | Jarque-Bera, Shapiro-Wilk (n ≤ 5000), Anderson-Darling |
| **Autocorrelation** | Durbin-Watson, Breusch-Godfrey (higher-order) |
| **Multicollinearity** | Variance Inflation Factor (VIF) |
| **Influence** | Cook's Distance, DFBETAS, DFFITS |

---

## Rust Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
linreg-core = { version = "0.6", default-features = false }
```

### OLS Regression (Rust)

```rust
use linreg_core::core::ols_regression;

fn main() -> Result<(), linreg_core::Error> {
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
    let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let names = vec!["Intercept".to_string(), "X1".to_string()];

    let result = ols_regression(&y, &x, &names)?;

    println!("Coefficients: {:?}", result.coefficients);
    println!("R-squared: {:.4}", result.r_squared);
    println!("F-statistic: {:.4}", result.f_statistic);
    println!("Log-likelihood: {:.4}", result.log_likelihood);
    println!("AIC: {:.4}", result.aic);
    println!("BIC: {:.4}", result.bic);

    Ok(())
}
```

### Ridge Regression (Rust)

```rust,no_run
use linreg_core::regularized::{ridge_fit, RidgeFitOptions};
use linreg_core::linalg::Matrix;

fn main() -> Result<(), linreg_core::Error> {
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
    let x = Matrix::new(5, 2, vec![
        1.0, 1.0,  // row 0: intercept, x1
        1.0, 2.0,  // row 1
        1.0, 3.0,  // row 2
        1.0, 4.0,  // row 3
        1.0, 5.0,  // row 4
    ]);

    let options = RidgeFitOptions {
        lambda: 1.0,
        standardize: true,
        intercept: true,
    };

    let result = ridge_fit(&x, &y, &options)?;
    println!("Intercept: {}", result.intercept);
    println!("Coefficients: {:?}", result.coefficients);
    println!("R-squared: {:.4}", result.r_squared);
    println!("Effective degrees of freedom: {:.2}", result.effective_df);
    println!("AIC: {:.4}", result.aic);
    println!("BIC: {:.4}", result.bic);

    Ok(())
}
```

### Lasso Regression (Rust)

```rust,no_run
use linreg_core::regularized::{lasso_fit, LassoFitOptions};
use linreg_core::linalg::Matrix;

fn main() -> Result<(), linreg_core::Error> {
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
    let x = Matrix::new(5, 3, vec![
        1.0, 1.0, 0.5,
        1.0, 2.0, 1.0,
        1.0, 3.0, 1.5,
        1.0, 4.0, 2.0,
        1.0, 5.0, 2.5,
    ]);

    let options = LassoFitOptions {
        lambda: 0.1,
        standardize: true,
        intercept: true,
        ..Default::default()
    };

    let result = lasso_fit(&x, &y, &options)?;
    println!("Intercept: {}", result.intercept);
    println!("Coefficients: {:?}", result.coefficients);
    println!("Non-zero coefficients: {}", result.n_nonzero);
    println!("AIC: {:.4}", result.aic);
    println!("BIC: {:.4}", result.bic);

    Ok(())
}
```

### Elastic Net Regression (Rust)

```rust,no_run
use linreg_core::regularized::{elastic_net_fit, ElasticNetOptions};
use linreg_core::linalg::Matrix;

fn main() -> Result<(), linreg_core::Error> {
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
    let x = Matrix::new(5, 3, vec![
        1.0, 1.0, 0.5,
        1.0, 2.0, 1.0,
        1.0, 3.0, 1.5,
        1.0, 4.0, 2.0,
        1.0, 5.0, 2.5,
    ]);

    let options = ElasticNetOptions {
        lambda: 0.1,
        alpha: 0.5,   // 0 = Ridge, 1 = Lasso, 0.5 = balanced
        standardize: true,
        intercept: true,
        ..Default::default()
    };

    let result = elastic_net_fit(&x, &y, &options)?;
    println!("Intercept: {}", result.intercept);
    println!("Coefficients: {:?}", result.coefficients);
    println!("Non-zero coefficients: {}", result.n_nonzero);
    println!("AIC: {:.4}", result.aic);
    println!("BIC: {:.4}", result.bic);

    Ok(())
}
```

### Diagnostic Tests (Rust)

```rust
use linreg_core::diagnostics::{
    breusch_pagan_test, durbin_watson_test, jarque_bera_test,
    shapiro_wilk_test, rainbow_test, harvey_collier_test,
    white_test, anderson_darling_test, breusch_godfrey_test,
    cooks_distance_test, dfbetas_test, dffits_test, vif_test,
    reset_test, BGTestType, RainbowMethod, ResetType, WhiteMethod
};

fn main() -> Result<(), linreg_core::Error> {
    let y = vec![/* your data */];
    let x = vec![vec![/* predictor 1 */], vec![/* predictor 2 */]];

    // Heteroscedasticity tests
    let bp = breusch_pagan_test(&y, &x)?;
    println!("Breusch-Pagan: LM={:.4}, p={:.4}", bp.statistic, bp.p_value);

    let white = white_test(&y, &x, WhiteMethod::R)?;
    println!("White: statistic={:.4}, p={:.4}", white.statistic, white.p_value);

    // Autocorrelation tests
    let dw = durbin_watson_test(&y, &x)?;
    println!("Durbin-Watson: {:.4}", dw.statistic);

    let bg = breusch_godfrey_test(&y, &x, 2, BGTestType::Chisq)?;
    println!("Breusch-Godfrey (order 2): statistic={:.4}, p={:.4}", bg.statistic, bg.p_value);

    // Normality tests
    let jb = jarque_bera_test(&y, &x)?;
    println!("Jarque-Bera: JB={:.4}, p={:.4}", jb.statistic, jb.p_value);

    let sw = shapiro_wilk_test(&y, &x)?;
    println!("Shapiro-Wilk: W={:.4}, p={:.4}", sw.statistic, sw.p_value);

    let ad = anderson_darling_test(&y, &x)?;
    println!("Anderson-Darling: A={:.4}, p={:.4}", ad.statistic, ad.p_value);

    // Linearity tests
    let rainbow = rainbow_test(&y, &x, 0.5, RainbowMethod::R)?;
    println!("Rainbow: F={:.4}, p={:.4}",
        rainbow.r_result.as_ref().unwrap().statistic,
        rainbow.r_result.as_ref().unwrap().p_value);

    let hc = harvey_collier_test(&y, &x)?;
    println!("Harvey-Collier: t={:.4}, p={:.4}", hc.statistic, hc.p_value);

    let reset = reset_test(&y, &x, &[2, 3], ResetType::Fitted)?;
    println!("RESET: F={:.4}, p={:.4}", reset.f_statistic, reset.p_value);

    // Influence diagnostics
    let cd = cooks_distance_test(&y, &x)?;
    println!("Cook's Distance: {} influential points", cd.influential_4_over_n.len());

    let dfbetas = dfbetas_test(&y, &x)?;
    println!("DFBETAS: {} influential observations", dfbetas.influential_observations.len());

    let dffits = dffits_test(&y, &x)?;
    println!("DFFITS: {} influential observations", dffits.influential_observations.len());

    // Multicollinearity
    let vif = vif_test(&y, &x)?;
    println!("VIF: {:?}", vif.vif_values);

    Ok(())
}
```

### WLS Regression (Rust)

```rust,no_run
use linreg_core::weighted_regression::wls_regression;

fn main() -> Result<(), linreg_core::Error> {
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Equal weights = OLS
    let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    let fit = wls_regression(&y, &[x1], &weights)?;

    println!("Intercept: {} (SE: {}, t: {}, p: {})",
        fit.coefficients[0],
        fit.standard_errors[0],
        fit.t_statistics[0],
        fit.p_values[0]
    );
    println!("F-statistic: {} (p: {})", fit.f_statistic, fit.f_p_value);
    println!("R-squared: {:.4}", fit.r_squared);

    // Access confidence intervals
    for (i, (&coef, &lower, &upper)) in fit.coefficients.iter()
        .zip(fit.conf_int_lower.iter())
        .zip(fit.conf_int_upper.iter())
        .enumerate()
    {
        println!("Coefficient {}: [{}, {}]", i, lower, upper);
    }

    Ok(())
}
```

### LOESS Regression (Rust)

```rust,no_run
use linreg_core::loess::{loess_fit, LoessOptions};

fn main() -> Result<(), linreg_core::Error> {
    // Single predictor only
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = vec![1.0, 3.5, 4.8, 6.2, 8.5, 11.0, 13.2, 14.8, 17.5, 19.0, 22.0];

    // Default options: span=0.75, degree=2, robust iterations=0
    let options = LoessOptions::default();

    let result = loess_fit(&y, &[x], &options)?;

    println!("Fitted values: {:?}", result.fitted_values);
    println!("Residuals: {:?}", result.residuals);

    Ok(())
}
```

**Custom LOESS options:**

```rust,no_run
use linreg_core::loess::{loess_fit, LoessOptions, LoessSurface};

let options = LoessOptions {
    span: 0.5,              // Smoothing parameter (0-1, smaller = less smooth)
    degree: 1,              // Polynomial degree (0=constant, 1=linear, 2=quadratic)
    surface: LoessSurface::Direct,  // Note: only "direct" is currently supported; "interpolate" is planned
    robust_iterations: 3,   // Number of robust fitting iterations (0 = disabled)
};

let result = loess_fit(&y, &[x], &options)?;
```

### K-Fold Cross Validation (Rust)

Cross-validation is used for model evaluation and hyperparameter tuning. The library supports K-Fold CV for all regression types:

```rust,no_run
use linreg_core::cross_validation::{kfold_cv_ols, kfold_cv_ridge, kfold_cv_lasso, kfold_cv_elastic_net, KFoldOptions};

fn main() -> Result<(), linreg_core::Error> {
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3, 7.0, 7.5, 8.1];
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0, 4.5, 5.5, 6.0];
    let names = vec!["Intercept".to_string(), "X1".to_string(), "X2".to_string()];

    // Configure CV options
    let options = KFoldOptions {
        n_folds: 5,
        shuffle: true,
        seed: Some(42),  // For reproducibility
    };

    // OLS cross-validation
    let ols_cv = kfold_cv_ols(&y, &[x1.clone(), x2.clone()], &names, &options)?;
    println!("OLS CV RMSE: {:.4} (±{:.4})", ols_cv.mean_rmse, ols_cv.std_rmse);
    println!("OLS CV R²: {:.4} (±{:.4})", ols_cv.mean_r_squared, ols_cv.std_r_squared);

    // Ridge cross-validation (for lambda selection)
    let lambda = 1.0;
    let ridge_cv = kfold_cv_ridge(&[x1.clone(), x2.clone()], &y, lambda, true, &options)?;
    println!("Ridge CV RMSE: {:.4}", ridge_cv.mean_rmse);

    // Lasso cross-validation
    let lasso_cv = kfold_cv_lasso(&[x1.clone(), x2.clone()], &y, 0.1, true, &options)?;
    println!("Lasso CV RMSE: {:.4}", lasso_cv.mean_rmse);

    // Elastic Net cross-validation
    let enet_cv = kfold_cv_elastic_net(&[x1, x2], &y, 0.1, 0.5, true, &options)?;
    println!("Elastic Net CV RMSE: {:.4}", enet_cv.mean_rmse);

    // Access per-fold results
    for fold in &ols_cv.fold_results {
        println!("Fold {}: train={}, test={}, R²={:.4}",
            fold.fold_index, fold.train_size, fold.test_size, fold.r_squared);
    }

    Ok(())
}
```

**CV Result fields:**
- `mean_rmse`, `std_rmse` - Mean and std of RMSE across folds
- `mean_mae`, `std_mae` - Mean and std of MAE across folds
- `mean_r_squared`, `std_r_squared` - Mean and std of R² across folds
- `mean_train_r_squared` - Mean training R² (for overfitting detection)
- `fold_results` - Per-fold metrics (train/test sizes, MSE, RMSE, MAE, R²)
- `fold_coefficients` - Coefficients from each fold (for stability analysis)

### Lambda Path Generation (Rust)

```rust,no_run
use linreg_core::regularized::{make_lambda_path, LambdaPathOptions};
use linreg_core::linalg::Matrix;

let x = Matrix::new(100, 5, vec![0.0; 500]);
let y = vec![0.0; 100];

let options = LambdaPathOptions {
    nlambda: 100,
    lambda_min_ratio: Some(0.01),
    alpha: 1.0,  // Lasso
    ..Default::default()
};

let lambdas = make_lambda_path(&x, &y, &options, None, Some(0));

for &lambda in lambdas.iter() {
    // Fit model with this lambda
}
```

### Model Save/Load (Rust)

All trained models can be saved to disk and loaded back later:

```rust,no_run
use linreg_core::{ModelSave, ModelLoad};

// Train a model
let result = ols_regression(&y, &[x1], &names)?;

// Save to file
result.save("my_model.json")?;

// Or with a custom name
result.save_with_name("my_model.json", Some("My Housing Model".to_string()))?;

// Load back
let loaded = linreg_core::core::RegressionOutput::load("my_model.json")?;
```

The same `save()` and `load()` methods work for all model types: `RegressionOutput`, `RidgeFit`, `LassoFit`, `ElasticNetFit`, `WlsFit`, and `LoessFit`.

---

## WebAssembly Usage

**[Live Demo →](https://jesse-anderson.net/linreg-core/)**

Build with wasm-pack:

```bash
wasm-pack build --release --target web
```

### OLS Regression (WASM)

```javascript
import init, { ols_regression } from './pkg/linreg_core.js';

async function run() {
    await init();

    const y = [1, 2, 3, 4, 5];
    const x = [[1, 2, 3, 4, 5]];
    const names = ["Intercept", "X1"];

    const resultJson = ols_regression(
        JSON.stringify(y),
        JSON.stringify(x),
        JSON.stringify(names)
    );

    const result = JSON.parse(resultJson);
    console.log("Coefficients:", result.coefficients);
    console.log("R-squared:", result.r_squared);
    console.log("Log-likelihood:", result.log_likelihood);
    console.log("AIC:", result.aic);
    console.log("BIC:", result.bic);
}

run();
```

### Ridge Regression (WASM)

```javascript
const result = JSON.parse(ridge_regression(
    JSON.stringify(y),
    JSON.stringify(x),
    JSON.stringify(["Intercept", "X1", "X2"]),
    1.0,      // lambda
    true      // standardize
));

console.log("Coefficients:", result.coefficients);
console.log("R-squared:", result.r_squared);
console.log("Effective degrees of freedom:", result.effective_df);
console.log("AIC:", result.aic);
console.log("BIC:", result.bic);
```

### Lasso Regression (WASM)

```javascript
const result = JSON.parse(lasso_regression(
    JSON.stringify(y),
    JSON.stringify(x),
    JSON.stringify(["Intercept", "X1", "X2"]),
    0.1,      // lambda
    true,     // standardize
    100000,   // max_iter
    1e-7      // tol
));

console.log("Coefficients:", result.coefficients);
console.log("Non-zero coefficients:", result.n_nonzero);
console.log("AIC:", result.aic);
console.log("BIC:", result.bic);
```

### Elastic Net Regression (WASM)

```javascript
const result = JSON.parse(elastic_net_regression(
    JSON.stringify(y),
    JSON.stringify(x),
    JSON.stringify(["Intercept", "X1", "X2"]),
    0.1,      // lambda
    0.5,      // alpha (0 = Ridge, 1 = Lasso, 0.5 = balanced)
    true,     // standardize
    100000,   // max_iter
    1e-7      // tol
));

console.log("Coefficients:", result.coefficients);
console.log("Non-zero coefficients:", result.n_nonzero);
console.log("AIC:", result.aic);
console.log("BIC:", result.bic);
```

### Lambda Path Generation (WASM)

```javascript
const path = JSON.parse(make_lambda_path(
    JSON.stringify(y),
    JSON.stringify(x),
    100,              // n_lambda
    0.01              // lambda_min_ratio (as fraction of lambda_max)
));

console.log("Lambda sequence:", path.lambda_path);
console.log("Lambda max:", path.lambda_max);
```

### WLS Regression (WASM)

```javascript
const result = JSON.parse(wls_regression(
    JSON.stringify([2, 4, 6, 8, 10]),
    JSON.stringify([[1, 2, 3, 4, 5]]),
    JSON.stringify([1, 1, 1, 1, 1])  // weights (equal weights = OLS)
));

console.log("Coefficients:", result.coefficients);
console.log("Standard errors:", result.standard_errors);
console.log("P-values:", result.p_values);
console.log("R-squared:", result.r_squared);
console.log("F-statistic:", result.f_statistic);
console.log("Confidence intervals (lower):", result.conf_int_lower);
console.log("Confidence intervals (upper):", result.conf_int_upper);
```

### LOESS Regression (WASM)

```javascript
const result = JSON.parse(loess_fit(
    JSON.stringify(y),
    JSON.stringify(x[0]),    // Single predictor only (flattened array)
    0.5,      // span (smoothing parameter: 0-1)
    1,        // degree (0=constant, 1=linear, 2=quadratic)
    "direct", // surface method ("direct" only; "interpolate" is planned)
    0         // robust iterations (0=disabled, >0=number of iterations)
));

console.log("Fitted values:", result.fitted_values);
console.log("Residuals:", result.residuals);
```

### K-Fold Cross Validation (WASM)

```javascript
// OLS cross-validation
const ols_cv = JSON.parse(kfold_cv_ols(
    JSON.stringify(y),
    JSON.stringify(x),
    JSON.stringify(["Intercept", "X1", "X2"]),
    5,         // n_folds
    "true",    // shuffle (JSON boolean)
    "42"       // seed (JSON string number, or "null" for no seed)
));

console.log("OLS CV RMSE:", ols_cv.mean_rmse, "±", ols_cv.std_rmse);
console.log("OLS CV R²:", ols_cv.mean_r_squared, "±", ols_cv.std_r_squared);

// Ridge cross-validation
const ridge_cv = JSON.parse(kfold_cv_ridge(
    JSON.stringify(y),
    JSON.stringify(x),
    1.0,       // lambda
    true,      // standardize
    5,         // n_folds
    "true",    // shuffle
    "42"       // seed
));

// Lasso cross-validation
const lasso_cv = JSON.parse(kfold_cv_lasso(
    JSON.stringify(y),
    JSON.stringify(x),
    0.1,       // lambda
    true,      // standardize
    5,         // n_folds
    "true",    // shuffle
    "42"       // seed
));

// Elastic Net cross-validation
const enet_cv = JSON.parse(kfold_cv_elastic_net(
    JSON.stringify(y),
    JSON.stringify(x),
    0.1,       // lambda
    0.5,       // alpha (0 = Ridge, 1 = Lasso)
    true,      // standardize
    5,         // n_folds
    "true",    // shuffle
    "42"       // seed
));

// Access per-fold results
ols_cv.fold_results.forEach(fold => {
    console.log(`Fold ${fold.fold_index}: R²=${fold.r_squared.toFixed(4)}`);
});
```

**Note:** In WASM, boolean and seed parameters are passed as JSON strings. Use `"true"`/`"false"` for shuffle and `"42"` or `"null"` for seed.

### Diagnostic Tests (WASM)

```javascript
// Rainbow test
const rainbow = JSON.parse(rainbow_test(
    JSON.stringify(y),
    JSON.stringify(x),
    0.5,      // fraction
    "r"       // method: "r", "python", or "both"
));

// Harvey-Collier test
const hc = JSON.parse(harvey_collier_test(
    JSON.stringify(y),
    JSON.stringify(x)
));

// Breusch-Pagan test
const bp = JSON.parse(breusch_pagan_test(
    JSON.stringify(y),
    JSON.stringify(x)
));

// White test (method selection: "r", "python", or "both")
const white = JSON.parse(white_test(
    JSON.stringify(y),
    JSON.stringify(x),
    "r"
));

// White test - R-specific method
const whiteR = JSON.parse(r_white_test(
    JSON.stringify(y),
    JSON.stringify(x)
));

// White test - Python-specific method
const whitePy = JSON.parse(python_white_test(
    JSON.stringify(y),
    JSON.stringify(x)
));

// Jarque-Bera test
const jb = JSON.parse(jarque_bera_test(
    JSON.stringify(y),
    JSON.stringify(x)
));

// Durbin-Watson test
const dw = JSON.parse(durbin_watson_test(
    JSON.stringify(y),
    JSON.stringify(x)
));

// Shapiro-Wilk test
const sw = JSON.parse(shapiro_wilk_test(
    JSON.stringify(y),
    JSON.stringify(x)
));

// Anderson-Darling test
const ad = JSON.parse(anderson_darling_test(
    JSON.stringify(y),
    JSON.stringify(x)
));

// Cook's Distance
const cd = JSON.parse(cooks_distance_test(
    JSON.stringify(y),
    JSON.stringify(x)
));

// DFBETAS (influence on coefficients)
const dfbetas = JSON.parse(dfbetas_test(
    JSON.stringify(y),
    JSON.stringify(x)
));

// DFFITS (influence on fitted values)
const dffits = JSON.parse(dffits_test(
    JSON.stringify(y),
    JSON.stringify(x)
));

// VIF test (multicollinearity)
const vif = JSON.parse(vif_test(
    JSON.stringify(y),
    JSON.stringify(x)
));
console.log("VIF values:", vif.vif_values);

// RESET test (functional form)
const reset = JSON.parse(reset_test(
    JSON.stringify(y),
    JSON.stringify(x),
    JSON.stringify([2, 3]),  // powers
    "fitted"                  // type: "fitted", "regressor", or "princomp"
));

// Breusch-Godfrey test (higher-order autocorrelation)
const bg = JSON.parse(breusch_godfrey_test(
    JSON.stringify(y),
    JSON.stringify(x),
    1,        // order
    "chisq"   // test_type: "chisq" or "f"
));
```

### Statistical Utilities (WASM)

```javascript
// Student's t CDF: P(T <= t)
const tCDF = get_t_cdf(1.96, 20);

// Critical t-value for two-tailed test
const tCrit = get_t_critical(0.05, 20);

// Normal inverse CDF (probit)
const zScore = get_normal_inverse(0.975);

// Descriptive statistics (all return JSON strings)
const mean = JSON.parse(stats_mean(JSON.stringify([1, 2, 3, 4, 5])));
const variance = JSON.parse(stats_variance(JSON.stringify([1, 2, 3, 4, 5])));
const stddev = JSON.parse(stats_stddev(JSON.stringify([1, 2, 3, 4, 5])));
const median = JSON.parse(stats_median(JSON.stringify([1, 2, 3, 4, 5])));
const quantile = JSON.parse(stats_quantile(JSON.stringify([1, 2, 3, 4, 5]), 0.5));
const correlation = JSON.parse(stats_correlation(
    JSON.stringify([1, 2, 3, 4, 5]),
    JSON.stringify([2, 4, 6, 8, 10])
));
```

### CSV Parsing (WASM)

```javascript
const csv = parse_csv(csvContent);
const parsed = JSON.parse(csv);
console.log("Headers:", parsed.headers);
console.log("Numeric columns:", parsed.numeric_columns);
```

### Helper Functions (WASM)

```javascript
const version = get_version();  // e.g., "0.5.0"
const msg = test();             // "Rust WASM is working!"
```

### Model Serialization (WASM)

```javascript
// Train a model
const resultJson = ols_regression(
    JSON.stringify(y),
    JSON.stringify(x),
    JSON.stringify(names)
);
const result = JSON.parse(resultJson);

// Serialize with metadata
const serialized = serialize_model(
    resultJson,        // model JSON
    "OLS",             // model type: "OLS", "Ridge", "Lasso", "ElasticNet", "WLS", "LOESS"
    "My Model"         // optional name (null to omit)
);

// Get metadata without loading full model
const metadataJson = get_model_metadata(serialized);
const metadata = JSON.parse(metadataJson);
console.log("Model type:", metadata.model_type);
console.log("Created:", metadata.created_at);

// Deserialize to get model data back
const modelJson = deserialize_model(serialized);
const model = JSON.parse(modelJson);

// Download in browser
const blob = new Blob([serialized], { type: 'application/json' });
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'model.json';
a.click();
```

### Domain Security (WASM)

Optional domain restriction via build-time environment variable:

```bash
LINREG_DOMAIN_RESTRICT=example.com,mysite.com wasm-pack build --release --target web
```

When NOT set (default), all domains are allowed.

---

## Python Usage

Install from PyPI:

```bash
pip install linreg-core
```

### Quick Start (Python)

The recommended way to use `linreg-core` in Python is with native types (lists or numpy arrays):

```python
import linreg_core

# Works with Python lists
y = [1, 2, 3, 4, 5]
x = [[1, 2, 3, 4, 5]]
names = ["Intercept", "X1"]

result = linreg_core.ols_regression(y, x, names)

# Access attributes directly
print(f"R²: {result.r_squared}")
print(f"Coefficients: {result.coefficients}")
print(f"F-statistic: {result.f_statistic}")

# Get a formatted summary
print(result.summary())
```

**With NumPy arrays:**

```python
import numpy as np
import linreg_core

y = np.array([1, 2, 3, 4, 5])
x = np.array([[1, 2, 3, 4, 5]])

result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
print(result.summary())
```

**Result objects** provide:
- Direct attribute access (`result.r_squared`, `result.coefficients`, `result.aic`, `result.bic`, `result.log_likelihood`)
- `summary()` method for formatted output
- `to_dict()` method for JSON serialization

### OLS Regression (Python)

```python
import linreg_core

y = [1, 2, 3, 4, 5]
x = [[1, 2, 3, 4, 5]]
names = ["Intercept", "X1"]

result = linreg_core.ols_regression(y, x, names)
print(f"Coefficients: {result.coefficients}")
print(f"R-squared: {result.r_squared}")
print(f"F-statistic: {result.f_statistic}")
print(f"Log-likelihood: {result.log_likelihood}")
print(f"AIC: {result.aic}")
print(f"BIC: {result.bic}")
```

### Ridge Regression (Python)

```python
result = linreg_core.ridge_regression(
    y, x, ["Intercept", "X1"],
    1.0,      # lambda
    True      # standardize
)
print(f"Intercept: {result.intercept}")
print(f"Coefficients: {result.coefficients}")
print(f"Effective degrees of freedom: {result.effective_df:.2f}")
print(f"AIC: {result.aic}")
print(f"BIC: {result.bic}")
```

### Lasso Regression (Python)

```python
result = linreg_core.lasso_regression(
    y, x, ["Intercept", "X1"],
    0.1,      # lambda
    True,     # standardize
    100000,   # max_iter
    1e-7      # tol
)
print(f"Intercept: {result.intercept}")
print(f"Coefficients: {result.coefficients}")
print(f"Non-zero: {result.n_nonzero}")
print(f"Converged: {result.converged}")
print(f"AIC: {result.aic}")
print(f"BIC: {result.bic}")
```

### Elastic Net Regression (Python)

```python
result = linreg_core.elastic_net_regression(
    y, x, ["Intercept", "X1"],
    0.1,      # lambda
    0.5,      # alpha (0 = Ridge, 1 = Lasso, 0.5 = balanced)
    True,     # standardize
    100000,   # max_iter
    1e-7      # tol
)
print(f"Intercept: {result.intercept}")
print(f"Coefficients: {result.coefficients}")
print(f"Non-zero: {result.n_nonzero}")
print(f"AIC: {result.aic}")
print(f"BIC: {result.bic}")
```

### LOESS Regression (Python)

```python
result = linreg_core.loess_fit(
    y,           # Single predictor only
    [0.5],       # span (smoothing parameter: 0-1)
    2,           # degree (0=constant, 1=linear, 2=quadratic)
    "direct",    # surface ("direct" only; "interpolate" is planned)
    0            # robust iterations (0=disabled, >0=number of iterations)
)
print(f"Fitted values: {result.fitted_values}")
print(f"Residuals: {result.residuals}")
```

### Lambda Path Generation (Python)

```python
path = linreg_core.make_lambda_path(
    y, x,
    100,              # n_lambda
    0.01              # lambda_min_ratio
)
print(f"Lambda max: {path.lambda_max}")
print(f"Lambda min: {path.lambda_min}")
print(f"Number: {path.n_lambda}")
```

### Diagnostic Tests (Python)

```python
# Breusch-Pagan test (heteroscedasticity)
bp = linreg_core.breusch_pagan_test(y, x)
print(f"Statistic: {bp.statistic}, p-value: {bp.p_value}")

# Harvey-Collier test (linearity)
hc = linreg_core.harvey_collier_test(y, x)

# Rainbow test (linearity) - supports "r", "python", or "both" methods
rainbow = linreg_core.rainbow_test(y, x, 0.5, "r")

# White test - choose method: "r", "python", or "both"
white = linreg_core.white_test(y, x, "r")
# Or use specific method functions
white_r = linreg_core.r_white_test(y, x)
white_py = linreg_core.python_white_test(y, x)

# Jarque-Bera test (normality)
jb = linreg_core.jarque_bera_test(y, x)

# Durbin-Watson test (autocorrelation)
dw = linreg_core.durbin_watson_test(y, x)
print(f"DW statistic: {dw.statistic}")

# Shapiro-Wilk test (normality)
sw = linreg_core.shapiro_wilk_test(y, x)

# Anderson-Darling test (normality)
ad = linreg_core.anderson_darling_test(y, x)

# Cook's Distance (influential observations)
cd = linreg_core.cooks_distance_test(y, x)
print(f"Influential points: {cd.influential_4_over_n}")

# DFBETAS (influence on each coefficient)
dfbetas = linreg_core.dfbetas_test(y, x)
print(f"Threshold: {dfbetas.threshold}")
print(f"Influential obs: {dfbetas.influential_observations}")

# DFFITS (influence on fitted values)
dffits = linreg_core.dffits_test(y, x)
print(f"Threshold: {dffits.threshold}")
print(f"Influential obs: {dffits.influential_observations}")

# RESET test (model specification)
reset = linreg_core.reset_test(y, x, [2, 3], "fitted")

# Breusch-Godfrey test (higher-order autocorrelation)
bg = linreg_core.breusch_godfrey_test(y, x, 1, "chisq")
```

### Statistical Utilities (Python)

```python
# Student's t CDF
t_cdf = linreg_core.get_t_cdf(1.96, 20)

# Critical t-value (two-tailed)
t_crit = linreg_core.get_t_critical(0.05, 20)

# Normal inverse CDF (probit)
z_score = linreg_core.get_normal_inverse(0.975)

# Library version
version = linreg_core.get_version()
```

### Descriptive Statistics (Python)

```python
import numpy as np

# All return float directly (no parsing needed)
mean = linreg_core.stats_mean([1, 2, 3, 4, 5])
variance = linreg_core.stats_variance([1, 2, 3, 4, 5])
stddev = linreg_core.stats_stddev([1, 2, 3, 4, 5])
median = linreg_core.stats_median([1, 2, 3, 4, 5])
quantile = linreg_core.stats_quantile([1, 2, 3, 4, 5], 0.5)
correlation = linreg_core.stats_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])

# Works with numpy arrays too
mean = linreg_core.stats_mean(np.array([1, 2, 3, 4, 5]))
```

### CSV Parsing (Python)

```python
csv_content = '''name,value,category
Alice,42.5,A
Bob,17.3,B
Charlie,99.9,A'''

result = linreg_core.parse_csv(csv_content)
print(f"Headers: {result.headers}")
print(f"Numeric columns: {result.numeric_columns}")
print(f"Data rows: {result.n_rows}")
```

### Model Save/Load (Python)

```python
# Train a model
result = linreg_core.ols_regression(y, x, names)

# Save to file
linreg_core.save_model(result, "my_model.json", name="My Housing Model")

# Load back
loaded = linreg_core.load_model("my_model.json")
print(f"R²: {loaded.r_squared}")
print(f"Coefficients: {loaded.coefficients}")
```

The `save_model()` and `load_model()` functions work with all result types: `OLSResult`, `RidgeResult`, `LassoResult`, `ElasticNetResult`, `LoessResult`, and `WlsResult`.

---

## VBA / Excel Usage

The library ships as a native Windows DLL, letting you call it directly from Excel VBA via `Declare` statements. Prebuilt binaries are included in the `VBA_Example/` directory:

| File | Architecture |
|------|-------------|
| `linreg_core_x64.dll` | 64-bit Excel (Office 2010+) |
| `linreg_core_x86.dll` | 32-bit Excel (legacy) |

### Installation

1. Copy `linreg_core_x64.dll` (and/or `linreg_core_x86.dll`) to the same folder as your `.xlsm` workbook.
2. Import `LinregCore.bas` into your VBA project (ALT+F11 → File → Import File).
3. Optionally import `ExampleMacros.bas` for ready-to-run demo macros. Once both files are imported, run `SetupWorkbook()` from the Immediate Window or a button to automatically create example sheets and load sample data.

### Building from Source

```bash
# 64-bit (modern Excel)
cargo build --release --target x86_64-pc-windows-msvc --features ffi

# 32-bit (legacy Excel)
cargo build --release --target i686-pc-windows-msvc --features ffi
```

The 32-bit build automatically uses `linreg_core.def` to strip stdcall decoration, so VBA `Declare` statements work without modification.

### High-Level Wrappers

`LinregCore.bas` exposes friendly wrapper functions that return 2D Excel arrays you can drop straight into cells with `Application.Transpose`:

```vba
' OLS regression - returns (k+6)×5 summary array
Dim result As Variant
result = LinReg_OLS(y, X)

' Regularized regression
result = LinReg_Ridge(y, X, lambda:=1.0, standardize:=True)
result = LinReg_Lasso(y, X, lambda:=0.1)
result = LinReg_ElasticNet(y, X, lambda:=0.1, alpha:=0.5)

' Weighted OLS
result = LinReg_WLS(y, X, weights)

' Prediction intervals (n_new × 4: predicted, lower, upper, SE)
result = LinReg_PredictionIntervals(y, X, newX, alpha:=0.05)

' Diagnostic tests - each returns 1×3: {statistic, p-value, df}
result = LinReg_BreuschPagan(y, X)
result = LinReg_White(y, X)
result = LinReg_JarqueBera(y, X)
result = LinReg_ShapiroWilk(y, X)
result = LinReg_AndersonDarling(y, X)
result = LinReg_HarveyCollier(y, X)
result = LinReg_Rainbow(y, X, fraction:=0.5)
result = LinReg_Reset(y, X)
result = LinReg_DurbinWatson(y, X)   ' {DW statistic, ρ, ""}
result = LinReg_BreuschGodfrey(y, X, lagOrder:=1)

' Influence diagnostics
result = LinReg_VIF(y, X)            ' p×1
result = LinReg_CooksDistance(y, X)  ' n×1
result = LinReg_DFFITS(y, X)         ' n×1
result = LinReg_DFBETAS(y, X)        ' (n+1)×(p+1) with header row/col

' Regularization path and cross-validation
result = LinReg_LambdaPath(y, X, nLambda:=100, lmr:=0.01, alpha:=1.0)
result = LinReg_KFoldOLS(y, X, nFolds:=5)        ' 1×6 CV metrics
result = LinReg_KFoldRidge(y, X, lambda:=1.0)
result = LinReg_KFoldLasso(y, X, lambda:=0.1)
result = LinReg_KFoldElasticNet(y, X, lambda:=0.1, alpha:=0.5)
```

All wrappers return a 1-element array containing an error string on failure:

```vba
If IsArray(result) And UBound(result, 1) = 0 Then
    MsgBox "Error: " & result(0)
    Exit Sub
End If
```

### Low-Level Handle API

The DLL uses an opaque handle pattern. All `LR_*` functions return a `usize` handle (0 = error); call `LR_Free` when done:

```vba
' --- declarations already in LinregCore.bas ---
' Private Declare PtrSafe Function LR_OLS Lib "linreg_core_x64.dll" ...
' Private Declare PtrSafe Sub LR_Free Lib "linreg_core_x64.dll" ...

Sub LowLevelExample()
    Dim n As Long, p As Long
    n = 5 : p = 1

    Dim y(4) As Double
    y(0) = 2.5 : y(1) = 3.7 : y(2) = 4.2 : y(3) = 5.1 : y(4) = 6.3

    ' X is row-major, no intercept column (added automatically)
    Dim X(4) As Double
    X(0) = 1 : X(1) = 2 : X(2) = 3 : X(3) = 4 : X(4) = 5

    Dim h As LongPtr
    h = LR_OLS(VarPtr(y(0)), n, VarPtr(X(0)), n, p)

    If h = 0 Then
        MsgBox "Regression failed"
        Exit Sub
    End If

    Dim r2 As Double, mse As Double
    r2  = LR_GetRSquared(h)
    mse = LR_GetMSE(h)

    ' Retrieve coefficient vector (intercept + slopes = p+1 values)
    Dim coefs(1) As Double
    LR_GetCoefficients h, VarPtr(coefs(0)), p + 1

    Debug.Print "R²=" & r2 & "  MSE=" & mse
    Debug.Print "Intercept=" & coefs(0) & "  Slope=" & coefs(1)

    LR_Free h
End Sub
```

### Key FFI Functions

| Category | Functions |
|----------|-----------|
| **Lifecycle** | `LR_Init`, `LR_Free`, `LR_GetLastError`, `LR_Version` |
| **Regression** | `LR_OLS`, `LR_Ridge`, `LR_Lasso`, `LR_ElasticNet`, `LR_WLS` |
| **Predictions** | `LR_PredictionIntervals` |
| **Diagnostics** | `LR_BreuschPagan`, `LR_White`, `LR_JarqueBera`, `LR_ShapiroWilk`, `LR_AndersonDarling`, `LR_HarveyCollier`, `LR_Rainbow`, `LR_Reset`, `LR_DurbinWatson`, `LR_BreuschGodfrey` |
| **Influence** | `LR_CooksDistance`, `LR_DFFITS`, `LR_DFBETAS`, `LR_VIF` |
| **Path / CV** | `LR_LambdaPath`, `LR_KFoldOLS`, `LR_KFoldRidge`, `LR_KFoldLasso`, `LR_KFoldElasticNet` |
| **Scalar getters** | `LR_GetRSquared`, `LR_GetAdjRSquared`, `LR_GetFStatistic`, `LR_GetFPValue`, `LR_GetMSE`, `LR_GetIntercept`, `LR_GetDF`, `LR_GetNNonzero`, `LR_GetStatistic`, `LR_GetPValue`, `LR_GetTestDF`, `LR_GetAutocorrelation` |
| **Vector getters** | `LR_GetCoefficients`, `LR_GetStdErrors`, `LR_GetTStats`, `LR_GetPValues`, `LR_GetResiduals`, `LR_GetFittedValues`, `LR_GetVector`, `LR_GetMatrix`, `LR_GetPredicted`, `LR_GetLowerBound`, `LR_GetUpperBound`, `LR_GetSEPred` |

### Running FFI Tests

```bash
cargo test --features ffi --test ffi_tests
cargo test --features ffi --test ffi_vba_tests
```

---

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `wasm` | Yes | Enables WASM bindings and browser support |
| `python` | No | Enables Python bindings via PyO3 |
| `ffi` | No | Enables Windows DLL bindings for VBA/Excel use |
| `validation` | No | Includes test data for validation tests |

For native Rust without WASM overhead:

```toml
linreg-core = { version = "0.6", default-features = false }
```

For Python bindings (built with maturin):

```bash
pip install linreg-core
```

---

## Validation

Results are validated against R (`lmtest`, `car`, `skedastic`, `nortest`, `glmnet`) and Python (`statsmodels`, `scipy`, `sklearn`). See the `verification/` directory for test scripts and reference outputs.

### Running Tests

```bash
# Unit tests
cargo test

# WASM tests
wasm-pack test --node

# All tests including doctests
cargo test --all-features
```

---

## Implementation Notes

### Regularization

The Ridge and Lasso implementations follow the glmnet formulation:

```
minimize (1/(2n)) * Σ(yᵢ - β₀ - xᵢᵀβ)² + λ * [(1 - α) * ||β||₂² / 2 + α * ||β||₁]
```

- **Ridge** (α = 0): Closed-form solution with (X'X + λI)⁻¹X'y
- **Lasso** (α = 1): Coordinate descent algorithm

### Numerical Precision

- QR decomposition used throughout for numerical stability
- Anderson-Darling uses Abramowitz & Stegun 7.1.26 for normal CDF (differs from R's Cephes by ~1e-6)
- Shapiro-Wilk implements Royston's 1995 algorithm matching R's implementation

### Known Limitations

- Harvey-Collier test may fail on high-VIF datasets (VIF > 5) due to numerical instability in recursive residuals
- Shapiro-Wilk limited to n <= 5000 (matching R's limitation)
- White test may differ from R on collinear datasets due to numerical precision in near-singular matrices

---

## Disclaimer

This library is under active development and has not reached 1.0 stability. While outputs are validated against R and Python implementations, **do not use this library for critical applications** (medical, financial, safety-critical systems) without independent verification. See the [LICENSE](LICENSE-MIT) for full terms. The software is provided "as is" without warranty of any kind.

---

## Benchmarks

<details>
<summary><strong>Click to expand v0.6.0 benchmark results</strong></summary>

Benchmark results run on Windows with `cargo bench --no-default-features`. Times are median values.

### Core Regression Benchmarks

| Benchmark | Size (n × p) | Time | Throughput |
|-----------|--------------|------|------------|
| OLS Regression | 10 × 2 | 12.46 µs | 802.71 Kelem/s |
| OLS Regression | 50 × 3 | 53.72 µs | 930.69 Kelem/s |
| OLS Regression | 100 × 5 | 211.09 µs | 473.73 Kelem/s |
| OLS Regression | 500 × 10 | 7.46 ms | 67.04 Kelem/s |
| OLS Regression | 1000 × 20 | 47.81 ms | 20.91 Kelem/s |
| OLS Regression | 5000 × 50 | 2.86 s | 1.75 Kelem/s |
| Ridge Regression | 50 × 3 | 9.61 µs | 5.20 Melem/s |
| Ridge Regression | 100 × 5 | 70.41 µs | 1.42 Melem/s |
| Ridge Regression | 500 × 10 | 842.37 µs | 593.56 Kelem/s |
| Ridge Regression | 1000 × 20 | 1.38 ms | 724.71 Kelem/s |
| Ridge Regression | 5000 × 50 | 10.25 ms | 487.78 Kelem/s |
| Lasso Regression | 50 × 3 | 258.82 µs | 193.18 Kelem/s |
| Lasso Regression | 100 × 5 | 247.89 µs | 403.41 Kelem/s |
| Lasso Regression | 500 × 10 | 3.58 ms | 139.86 Kelem/s |
| Lasso Regression | 1000 × 20 | 1.54 ms | 651.28 Kelem/s |
| Lasso Regression | 5000 × 50 | 12.52 ms | 399.50 Kelem/s |
| Elastic Net Regression | 50 × 3 | 46.15 µs | 1.08 Melem/s |
| Elastic Net Regression | 100 × 5 | 358.07 µs | 279.27 Kelem/s |
| Elastic Net Regression | 500 × 10 | 1.61 ms | 310.18 Kelem/s |
| Elastic Net Regression | 1000 × 20 | 1.60 ms | 623.66 Kelem/s |
| Elastic Net Regression | 5000 × 50 | 12.57 ms | 397.77 Kelem/s |
| WLS Regression | 50 × 3 | 32.92 µs | 1.52 Melem/s |
| WLS Regression | 100 × 5 | 155.30 µs | 643.93 Kelem/s |
| WLS Regression | 500 × 10 | 6.63 ms | 75.37 Kelem/s |
| WLS Regression | 1000 × 20 | 42.68 ms | 23.43 Kelem/s |
| WLS Regression | 5000 × 50 | 2.64 s | 1.89 Kelem/s |
| LOESS Fit | 50 × 1 | 132.83 µs | 376.42 Kelem/s |
| LOESS Fit | 100 × 1 | 1.16 ms | 86.00 Kelem/s |
| LOESS Fit | 500 × 1 | 28.42 ms | 17.59 Kelem/s |
| LOESS Fit | 1000 × 1 | 113.00 ms | 8.85 Kelem/s |
| LOESS Fit | 100 × 2 | 7.10 ms | 14.09 Kelem/s |
| LOESS Fit | 500 × 2 | 1.05 s | 476.19 elem/s |

### Lambda Path & Elastic Net Path Benchmarks

| Benchmark | Size (n × p) | Time | Throughput |
|-----------|--------------|------|------------|
| Elastic Net Path | 100 × 5 | 198.60 ms | 503.52 elem/s |
| Elastic Net Path | 500 × 10 | 69.46 ms | 7.20 Kelem/s |
| Elastic Net Path | 1000 × 20 | 39.08 ms | 25.59 Kelem/s |
| Make Lambda Path | 100 × 5 | 1.09 µs | 91.58 Melem/s |
| Make Lambda Path | 500 × 10 | 8.10 µs | 61.70 Melem/s |
| Make Lambda Path | 1000 × 20 | 29.96 µs | 33.37 Melem/s |
| Make Lambda Path | 5000 × 50 | 424.18 µs | 11.79 Melem/s |

### Diagnostic Test Benchmarks

| Benchmark | Size (n × p) | Time |
|-----------|--------------|------|
| Rainbow Test | 50 × 3 | 40.34 µs |
| Rainbow Test | 100 × 5 | 187.94 µs |
| Rainbow Test | 500 × 10 | 8.63 ms |
| Rainbow Test | 1000 × 20 | 60.09 ms |
| Rainbow Test | 5000 × 50 | 3.45 s |
| Harvey-Collier Test | 50 × 1 | 15.26 µs |
| Harvey-Collier Test | 100 × 1 | 30.32 µs |
| Harvey-Collier Test | 500 × 1 | 138.44 µs |
| Harvey-Collier Test | 1000 × 1 | 298.33 µs |
| Breusch-Pagan Test | 50 × 3 | 58.07 µs |
| Breusch-Pagan Test | 100 × 5 | 296.74 µs |
| Breusch-Pagan Test | 500 × 10 | 13.79 ms |
| Breusch-Pagan Test | 1000 × 20 | 96.49 ms |
| Breusch-Pagan Test | 5000 × 50 | 5.56 s |
| White Test | 50 × 3 | 14.31 µs |
| White Test | 100 × 5 | 44.25 µs |
| White Test | 500 × 10 | 669.40 µs |
| White Test | 1000 × 20 | 4.89 ms |
| Jarque-Bera Test | 50 × 3 | 30.13 µs |
| Jarque-Bera Test | 100 × 5 | 149.29 µs |
| Jarque-Bera Test | 500 × 10 | 6.64 ms |
| Jarque-Bera Test | 1000 × 20 | 47.89 ms |
| Jarque-Bera Test | 5000 × 50 | 2.75 s |
| Durbin-Watson Test | 50 × 3 | 31.80 µs |
| Durbin-Watson Test | 100 × 5 | 152.56 µs |
| Durbin-Watson Test | 500 × 10 | 6.87 ms |
| Durbin-Watson Test | 1000 × 20 | 48.65 ms |
| Durbin-Watson Test | 5000 × 50 | 2.76 s |
| Breusch-Godfrey Test | 50 × 3 | 71.73 µs |
| Breusch-Godfrey Test | 100 × 5 | 348.94 µs |
| Breusch-Godfrey Test | 500 × 10 | 14.77 ms |
| Breusch-Godfrey Test | 1000 × 20 | 100.08 ms |
| Breusch-Godfrey Test | 5000 × 50 | 5.64 s |
| Shapiro-Wilk Test | 10 × 2 | 2.04 µs |
| Shapiro-Wilk Test | 50 × 3 | 4.87 µs |
| Shapiro-Wilk Test | 100 × 5 | 10.67 µs |
| Shapiro-Wilk Test | 500 × 10 | 110.02 µs |
| Shapiro-Wilk Test | 1000 × 20 | 635.13 µs |
| Shapiro-Wilk Test | 5000 × 50 | 17.53 ms |
| Anderson-Darling Test | 50 × 3 | 34.02 µs |
| Anderson-Darling Test | 100 × 5 | 162.28 µs |
| Anderson-Darling Test | 500 × 10 | 6.95 ms |
| Anderson-Darling Test | 1000 × 20 | 48.15 ms |
| Anderson-Darling Test | 5000 × 50 | 2.78 s |
| Cook's Distance Test | 50 × 3 | 64.52 µs |
| Cook's Distance Test | 100 × 5 | 297.69 µs |
| Cook's Distance Test | 500 × 10 | 12.73 ms |
| Cook's Distance Test | 1000 × 20 | 94.02 ms |
| Cook's Distance Test | 5000 × 50 | 5.31 s |
| DFBETAS Test | 50 × 3 | 46.34 µs |
| DFBETAS Test | 100 × 5 | 185.52 µs |
| DFBETAS Test | 500 × 10 | 7.04 ms |
| DFBETAS Test | 1000 × 20 | 49.68 ms |
| DFFITS Test | 50 × 3 | 33.56 µs |
| DFFITS Test | 100 × 5 | 157.62 µs |
| DFFITS Test | 500 × 10 | 6.82 ms |
| DFFITS Test | 1000 × 20 | 48.35 ms |
| VIF Test | 50 × 3 | 5.36 µs |
| VIF Test | 100 × 5 | 12.68 µs |
| VIF Test | 500 × 10 | 128.04 µs |
| VIF Test | 1000 × 20 | 807.30 µs |
| VIF Test | 5000 × 50 | 26.33 ms |
| RESET Test | 50 × 3 | 77.85 µs |
| RESET Test | 100 × 5 | 359.12 µs |
| RESET Test | 500 × 10 | 14.40 ms |
| RESET Test | 1000 × 20 | 100.52 ms |
| RESET Test | 5000 × 50 | 5.67 s |
| Full Diagnostics | 100 × 5 | 2.75 ms |
| Full Diagnostics | 500 × 10 | 104.01 ms |
| Full Diagnostics | 1000 × 20 | 740.52 ms |

### Linear Algebra Benchmarks

| Benchmark | Size | Time |
|-----------|------|------|
| Matrix Transpose | 10 × 10 | 209.50 ns |
| Matrix Transpose | 50 × 50 | 3.67 µs |
| Matrix Transpose | 100 × 100 | 14.92 µs |
| Matrix Transpose | 500 × 500 | 924.23 µs |
| Matrix Transpose | 1000 × 1000 | 5.56 ms |
| Matrix Multiply (matmul) | 10 × 10 × 10 | 1.54 µs |
| Matrix Multiply (matmul) | 50 × 50 × 50 | 144.15 µs |
| Matrix Multiply (matmul) | 100 × 100 × 100 | 1.39 ms |
| Matrix Multiply (matmul) | 200 × 200 × 200 | 11.90 ms |
| Matrix Multiply (matmul) | 1000 × 100 × 100 | 13.94 ms |
| QR Decomposition | 10 × 5 | 1.41 µs |
| QR Decomposition | 50 × 10 | 14.81 µs |
| QR Decomposition | 100 × 20 | 57.61 µs |
| QR Decomposition | 500 × 50 | 2.19 ms |
| QR Decomposition | 1000 × 100 | 19.20 ms |
| QR Decomposition | 5000 × 100 | 1.48 s |
| QR Decomposition | 10000 × 100 | 8.09 s |
| QR Decomposition | 1000 × 500 | 84.48 ms |
| SVD | 10 × 5 | 150.36 µs |
| SVD | 50 × 10 | 505.41 µs |
| SVD | 100 × 20 | 2.80 ms |
| SVD | 500 × 50 | 60.00 ms |
| SVD | 1000 × 100 | 513.35 ms |
| Matrix Invert | 5 × 5 | 877.32 ns |
| Matrix Invert | 10 × 10 | 2.48 µs |
| Matrix Invert | 20 × 20 | 5.46 µs |
| Matrix Invert | 50 × 50 | 31.94 µs |
| Matrix Invert | 100 × 100 | 141.38 µs |
| Matrix Invert | 200 × 200 | 647.03 µs |

### Pressure Benchmarks (Large Datasets)

| Benchmark | Size (n) | Time |
|-----------|----------|------|
| Pressure (OLS + all diagnostics) | 10000 | 11.28 s |

</details>

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
