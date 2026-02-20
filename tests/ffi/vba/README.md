# FFI Tests for VBA/C#/C++ Consumers

This directory contains Rust-native tests for the FFI layer that validate compatibility with external consumers like VBA, C#, and C++.

## Why This Approach?

Instead of using C++ testing frameworks like GoogleTest (which would require separate build systems, C++ wrappers, and cross-platform complexity), we test the FFI layer directly from Rust. This approach:

- **Uses existing Cargo infrastructure** - No new build systems required
- **Runs with `cargo test`** - Integrates seamlessly with CI/CD
- **Tests the actual FFI boundary** - Validates raw pointer handling, calling conventions
- **Cross-platform** - Works on Windows, Linux, macOS
- **Fast** - No C++ compilation step

## Running the Tests

```bash
# Run all FFI tests
cargo test --features ffi --test ffi_tests

# Run only OLS FFI tests
cargo test --features ffi --test ffi_tests ols

# Run with output
cargo test --features ffi --test ffi_tests -- --nocapture

# Run a specific test
cargo test --features ffi --test ffi_tests test_ols_simple_linear_regression
```

## Test Structure

```
tests/ffi/vba/
├── mod.rs              # Module exports
├── common.rs           # Shared utilities and FFI declarations
├── ols.rs              # OLS regression tests
├── regularized.rs      # Ridge/Lasso/ElasticNet tests
├── diagnostics.rs      # Diagnostic test validations
├── prediction_intervals.rs  # Prediction interval tests
├── cross_validation.rs # Cross-validation tests
├── utilities.rs        # Version, init, error handling tests
└── lib.rs              # Integration entry point
```

## What Gets Tested

### OLS Regression (`ols.rs`)
- LR_OLS function with various input sizes
- All scalar getters (R², MSE, F-statistic, etc.)
- Vector getters (coefficients, std errors, t-stats, p-values, residuals, fitted values)
- Error handling (null pointers, invalid dimensions)
- Buffer size handling
- Handle management (creation, freeing, uniqueness)

### Regularized Regression (`regularized.rs`)
- LR_Ridge with lambda and standardization options
- LR_Lasso with convergence tracking and sparsity
- LR_ElasticNet with alpha parameter continuum
- Special getters (intercept, effective df, non-zero count, converged)
- Lambda effect on coefficient magnitudes
- Standardized vs unstandardized behavior

### Diagnostic Tests (`diagnostics.rs`)
- All diagnostic functions (Breusch-Pagan, Jarque-Bera, etc.)
- Special case handling (Durbin-Watson autocorrelation, RESET powers)
- Influence diagnostics (Cook's Distance, DFFITS, VIF, DFBETAS)
- Vector and matrix result handling
- Error handling for invalid inputs

### Prediction Intervals (`prediction_intervals.rs`)
- LR_PredictionIntervals with alpha levels
- Multiple predictors
- Consistency checks (lower ≤ predicted ≤ upper)
- Buffer size handling
- Extrapolation behavior

### Cross-Validation (`cross_validation.rs`)
- K-fold CV for OLS, Ridge, Lasso, ElasticNet
- CV statistics (mean/std MSE, RMSE, R²)
- Lambda and alpha effects
- Edge cases (small datasets, k > n)

### Utilities (`utilities.rs`)
- LR_Version string retrieval
- LR_Init behavior
- LR_GetLastError message handling
- Thread safety basics
- NaN/Infinity handling

## Test Data Fixtures

Common test data is defined in `common.rs`:

- `simple_linear_data()` - Perfect linear relationship (y = 2 + 3x)
- `mtcars_subset()` - Real regression data (20 observations, 4 predictors)
- `collinear_data()` - For testing multicollinearity handling
- `heteroscedastic_data()` - For heteroscedasticity tests

## Helper Functions

- `assert_valid_handle()` - Asserts non-zero handle with error message
- `columns_to_row_major()` - Converts column vectors to FFI's row-major format
- `read_vector_from_ffi()` - Generic vector reader from FFI
- `read_vector_result()` - Reads Vector results via LR_GetVector
- `read_matrix_result()` - Reads Matrix results via LR_GetMatrix
- `read_prediction_intervals()` - Reads all 4 prediction interval arrays
- `HandleGuard` - RAII wrapper for automatic handle cleanup

## Adding New Tests

When adding new FFI functions:

1. **Add declaration in `common.rs`**:
   ```rust
   extern "system" {
       pub fn LR_NewFunction(...) -> usize;
   }
   ```

2. **Add tests in the appropriate module**:
   ```rust
   #[test]
   fn test_new_function_basic() {
       // Test basic functionality
       let handle = unsafe { LR_NewFunction(...) };
       let _guard = HandleGuard::new(handle);
       // Validate results
   }
   ```

3. **Add error handling tests**:
   ```rust
   #[test]
   fn test_new_function_null_pointer() {
       let handle = unsafe { LR_NewFunction(std::ptr::null(), ...) };
       assert_eq!(handle, 0);
   }
   ```

## Future Consumer Directories

The `vba/` directory name reflects the primary consumer (VBA), but the same FFI functions can be used by:

- **C#** via P/Invoke (`[DllImport("linreg_core_x64.dll")]`)
- **C++** via `__declspec(dllimport)` or `LoadLibrary`
- **Delphi** via `external` declarations
- **Any** language that supports calling C-style DLL functions

If testing for specific consumers is needed, create parallel directories:
- `tests/ffi/csharp/` - C#-specific integration tests
- `tests/ffi/cpp/` - C++-specific integration tests
- `tests/ffi/delphi/` - Delphi-specific integration tests

But for most purposes, the `vba/` tests validate the FFI contract adequately for all consumers.
