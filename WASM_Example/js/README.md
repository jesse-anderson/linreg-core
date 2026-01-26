# linreg-core

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)
[![Crates.io](https://img.shields.io/crates/v/linreg-core?color=orange)](https://crates.io/crates/linreg-core)
[![docs.rs](https://img.shields.io/badge/docs.rs-linreg__core-green)](https://docs.rs/linreg-core)

A lightweight, self-contained linear regression library written in Rust. Compiles to WebAssembly for browser use or runs as a native Rust crate.

**Key design principle:** All linear algebra and statistical distribution functions are implemented from scratch — no external math libraries required. This keeps binary sizes small and makes the crate highly portable.

## Features

### Regression Methods
- **OLS Regression:** Coefficients, standard errors, t-statistics, p-values, confidence intervals
- **Ridge Regression:** L2-regularized regression with optional standardization
- **Lasso Regression:** L1-regularized regression via coordinate descent
- **Lambda Path Generation:** Create regularization paths for cross-validation

### Model Statistics
- R-squared, Adjusted R-squared, F-statistic, F-test p-value
- Residuals, fitted values, leverage (hat matrix diagonal)
- Mean Squared Error (MSE)
- Variance Inflation Factor (VIF) for multicollinearity detection

### Diagnostic Tests
| Category | Tests |
|----------|-------|
| **Linearity** | Rainbow Test, Harvey-Collier Test |
| **Heteroscedasticity** | Breusch-Pagan (Koenker variant), White Test (R & Python methods) |
| **Normality** | Jarque-Bera, Shapiro-Wilk (n ≤ 5000), Anderson-Darling |
| **Autocorrelation** | Durbin-Watson |
| **Influence** | Cook's Distance |

### Dual Target
- Browser (WASM) and server (native Rust)
- Optional domain restriction for WASM builds

## Quick Start

### Native Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
linreg-core = { version = "0.3", default-features = false }
```

#### OLS Regression

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

    Ok(())
}
```

#### Ridge Regression

```rust,no_run
use linreg_core::regularized::{ridge_fit, RidgeFitOptions};
use linreg_core::linalg::Matrix;

fn main() -> Result<(), linreg_core::Error> {
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
    // Matrix: 5 rows × 2 cols (intercept + 1 predictor), row-major order
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

    Ok(())
}
```

#### Lasso Regression

```rust,no_run
use linreg_core::regularized::{lasso_fit, LassoFitOptions};
use linreg_core::linalg::Matrix;

fn main() -> Result<(), linreg_core::Error> {
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3];
    // Matrix: 5 rows × 3 cols (intercept + 2 predictors), row-major order
    let x = Matrix::new(5, 3, vec![
        1.0, 1.0, 0.5,  // row 0: intercept, x1, x2
        1.0, 2.0, 1.0,  // row 1
        1.0, 3.0, 1.5,  // row 2
        1.0, 4.0, 2.0,  // row 3
        1.0, 5.0, 2.5,  // row 4
    ]);

    let options = LassoFitOptions {
        lambda: 0.1,
        standardize: true,
        intercept: true,
        ..Default::default()  // uses default max_iter=1000, tol=1e-7
    };

    let result = lasso_fit(&x, &y, &options)?;

    println!("Intercept: {}", result.intercept);
    println!("Coefficients: {:?}", result.coefficients);
    println!("Non-zero coefficients: {}", result.n_nonzero);

    Ok(())
}
```

### WebAssembly (Browser)

Build with wasm-pack:

```bash
wasm-pack build --release --target web
```

#### OLS in JavaScript

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
}

run();
```

#### Ridge Regression in JavaScript

```javascript
const result = JSON.parse(ridge_regression(
    JSON.stringify(y),
    JSON.stringify(x),
    JSON.stringify(["Intercept", "X1", "X2"]),
    1.0,      // lambda
    true      // standardize
));

console.log("Coefficients:", result.coefficients);
```

#### Lasso Regression in JavaScript

```javascript
const result = JSON.parse(lasso_regression(
    JSON.stringify(y),
    JSON.stringify(x),
    JSON.stringify(["Intercept", "X1", "X2"]),
    0.1,      // lambda
    true      // standardize
));

console.log("Coefficients:", result.coefficients);
console.log("Non-zero coefficients:", result.n_nonzero_coeffs);
```

#### Lambda Path Generation

```javascript
const path = JSON.parse(make_lambda_path(
    JSON.stringify(y),
    JSON.stringify(x),
    100,              // n_lambda
    0.01              // lambda_min_ratio (as fraction of lambda_max)
));

console.log("Lambda sequence:", path.lambdas);
console.log("Lambda max:", path.lambda_max);
```

## Diagnostic Tests

### Native Rust

```rust
use linreg_core::diagnostics::{
    breusch_pagan_test, durbin_watson_test, jarque_bera_test,
    shapiro_wilk_test, RainbowMethod, rainbow_test
};

fn main() -> Result<(), linreg_core::Error> {
    let y = vec![/* your data */];
    let x = vec![vec![/* predictor 1 */], vec![/* predictor 2 */]];

    // Heteroscedasticity
    let bp = breusch_pagan_test(&y, &x)?;
    println!("Breusch-Pagan: LM={:.4}, p={:.4}", bp.statistic, bp.p_value);

    // Autocorrelation
    let dw = durbin_watson_test(&y, &x)?;
    println!("Durbin-Watson: {:.4}", dw.statistic);

    // Normality
    let jb = jarque_bera_test(&y, &x)?;
    println!("Jarque-Bera: JB={:.4}, p={:.4}", jb.statistic, jb.p_value);

    // Linearity
    let rainbow = rainbow_test(&y, &x, 0.5, RainbowMethod::R)?;
    println!("Rainbow: F={:.4}, p={:.4}",
        rainbow.r_result.as_ref().unwrap().statistic,
        rainbow.r_result.as_ref().unwrap().p_value);

    Ok(())
}
```

### WebAssembly

All diagnostic tests are available in WASM:

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

// White test (method selection)
const white = JSON.parse(white_test(
    JSON.stringify(y),
    JSON.stringify(x),
    "r"       // "r", "python", or "both"
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
```

## Statistical Utilities (WASM)

```javascript
// Student's t CDF: P(T <= t)
const tCDF = get_t_cdf(1.96, 20);

// Critical t-value for two-tailed test
const tCrit = get_t_critical(0.05, 20);

// Normal inverse CDF (probit)
const zScore = get_normal_inverse(0.975);
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `wasm` | Yes | Enables WASM bindings and browser support |
| `validation` | No | Includes test data for validation tests |

For native Rust without WASM overhead:

```toml
linreg-core = { version = "0.3", default-features = false }
```

## Regularization Path

Generate a sequence of lambda values for regularization path analysis:

```rust,no_run
use linreg_core::regularized::{make_lambda_path, LambdaPathOptions};
use linreg_core::linalg::Matrix;

// Assume x is your standardized design matrix and y is centered
let x = Matrix::new(100, 5, vec![0.0; 500]);
let y = vec![0.0; 100];

let options = LambdaPathOptions {
    nlambda: 100,
    lambda_min_ratio: Some(0.01),
    alpha: 1.0,  // Lasso
    ..Default::default()
};

let lambdas = make_lambda_path(&x, &y, &options, None, Some(0));

// Use each lambda for cross-validation or plotting regularization paths
for &lambda in lambdas.iter() {
    // Fit model with this lambda
    // ...
}
```

## Domain Security (WASM)

Optional domain restriction via build-time environment variable:

```bash
LINREG_DOMAIN_RESTRICT=example.com,mysite.com wasm-pack build --release --target web
```

When NOT set (default), all domains are allowed. When set, only the specified domains can use the WASM module.

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

## Disclaimer

This library is under active development and has not reached 1.0 stability. While outputs are validated against R and Python implementations, **do not use this library for critical applications** (medical, financial, safety-critical systems) without independent verification. See the [LICENSE](LICENSE-MIT) for full terms. The software is provided "as is" without warranty of any kind.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
