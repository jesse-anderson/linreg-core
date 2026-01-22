# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - 2025-01-22

### Added
- WASM: `max_iter` and `tol` parameters to `lasso_regression` for fine-grained convergence control
- WASM: `get_version()` function to query library version at runtime
- WASM Demo: Individual diagnostic test category buttons (Linearity, Heteroscedasticity, Normality, Autocorrelation, Influence)
- WASM Demo: Lasso-specific parameter controls (max iterations slider, tolerance selector)
- Comprehensive unit tests: New `regularized_tests.rs` module with 50+ tests for Ridge/Lasso (1500+ lines)
- `rustfmt.toml`: Rust formatter configuration for consistent code style

### Fixed
- **Regularized regression**: Fixed standardization logic to always center predictors when `intercept=true` (required for correct coordinate descent behavior)
- **Regularized regression**: Fixed `unstandardize_coefficients` to correctly return only slope coefficients (previously included intercept column coefficient)
- **Regularized regression**: Fixed Ridge/Lasso OLS fallback to avoid incorrect double-unstandardization
- **Anderson-Darling test**: Fixed `-inf` return from `log_normal_cdf` for extreme z-scores by clamping to MIN_LOG (-745)
- **Cook's Distance**: Fixed numerical instability for perfect-fit models (MSE ≈ 0) that caused 0/0 situations
- **Validation tests**: Fixed R JSON deserialization - Cook's distances are flat arrays, not nested
- **Validation tests**: Added proper tracking of missing reference files in failure reporting

### Changed
- WASM Demo: Diagnostic tests now run on-demand via category buttons (no longer auto-run after regression)
- WASM Demo: Added version display on WASM initialization
- WASM Demo: Improved console logging with structured messages
- Validation tests: Raised pass rate threshold from 80% to 90%
- Python White test script: Use `pd.factorize()` for categorical encoding (was `pd.to_numeric` with error coercion)

## [0.2.0] - 2025-01-21

### Added
- Ridge regression
- Lasso regression
- Improved documentation with 40+ runnable code examples

### Fixed
- Enhanced numerical stability in diagonal tolerance calculations
- Various documentation improvements

### Changed
- Bumped minimum documentation coverage significantly
