# linreg-core

[![CI](https://github.com/jesse-anderson/linreg-core/actions/workflows/ci.yml/badge.svg)](https://github.com/jesse-anderson/linreg-core/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/jesse-anderson/linreg-core/main/.github/coverage-badge.json)](https://github.com/jesse-anderson/linreg-core/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)
[![Crates.io](https://img.shields.io/crates/v/linreg-core?color=orange)](https://crates.io/crates/linreg-core)
[![npm](https://img.shields.io/npm/v/linreg-core?color=red)](https://www.npmjs.com/package/linreg-core)
[![docs.rs](https://img.shields.io/badge/docs.rs-linreg__core-green)](https://docs.rs/linreg-core)
[![PyPI](https://img.shields.io/badge/pypi-0.4.0-blue)](https://pypi.org/project/linreg-core/)

A lightweight, self-contained linear regression library written in Rust. Compiles to WebAssembly for browser use, Python bindings via PyO3, or runs as a native Rust crate.

**Key design principle:** All linear algebra and statistical distribution functions are implemented from scratch — no external math libraries required. This keeps binary sizes small and makes the crate highly portable.

---

## Table of Contents

| Section | Description |
|---------|-------------|
| [Features](#features) | Regression methods, model statistics, diagnostic tests |
| [Rust Usage](#rust-usage) | Native Rust crate usage |
| [WebAssembly Usage](#webassembly-usage) | Browser/JavaScript usage |
| [Python Usage](#python-usage) | Python bindings via PyO3 |
| [Feature Flags](#feature-flags) | Build configuration options |
| [Validation](#validation) | Testing and verification |
| [Implementation Notes](#implementation-notes) | Technical details |

---

## Features

### Regression Methods
- **OLS Regression:** Coefficients, standard errors, t-statistics, p-values, confidence intervals
- **Ridge Regression:** L2-regularized regression with optional standardization
- **Lasso Regression:** L1-regularized regression via coordinate descent
- **Elastic Net:** Combined L1 + L2 regularization for variable selection with multicollinearity handling
- **Lambda Path Generation:** Create regularization paths for cross-validation

### Model Statistics
- R-squared, Adjusted R-squared, F-statistic, F-test p-value
- Residuals, fitted values, leverage (hat matrix diagonal)
- Mean Squared Error (MSE)
- Variance Inflation Factor (VIF) for multicollinearity detection

### Diagnostic Tests
| Category | Tests |
|----------|-------|
| **Linearity** | Rainbow Test, Harvey-Collier Test, RESET Test |
| **Heteroscedasticity** | Breusch-Pagan (Koenker variant), White Test (R & Python methods) |
| **Normality** | Jarque-Bera, Shapiro-Wilk (n ≤ 5000), Anderson-Darling |
| **Autocorrelation** | Durbin-Watson, Breusch-Godfrey (higher-order) |
| **Influence** | Cook's Distance |

---

## Rust Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
linreg-core = { version = "0.4", default-features = false }
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

    Ok(())
}
```

### Diagnostic Tests (Rust)

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

---

## WebAssembly Usage

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
const version = get_version();  // e.g., "0.4.0"
const msg = test();             // "Rust WASM is working!"
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
- Direct attribute access (`result.r_squared`, `result.coefficients`)
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

---

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `wasm` | Yes | Enables WASM bindings and browser support |
| `python` | No | Enables Python bindings via PyO3 |
| `validation` | No | Includes test data for validation tests |

For native Rust without WASM overhead:

```toml
linreg-core = { version = "0.4", default-features = false }
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

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
