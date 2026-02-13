# Changelog

All notable changes to this project will be documented in this file.

## [0.6.1] - 2026-02-12

### Added
- **K-Fold Cross Validation** - K-fold CV for OLS, Ridge, Lasso, and Elastic Net with configurable folds, shuffling, and seeding
- **Model Serialization** - Save/load trained models to/from JSON files for all model types (OLS, Ridge, Lasso, ElasticNet, WLS, LOESS) with version compatibility and metadata support
- **Code refactoring** - Reduced ~260 lines through helper functions and `impl_serialization!` macro
- **Test coverage** - Increased to 94.13%

## [0.6.0] - 2026-02-10

### Added
- **Weighted Least Squares (WLS)** - Full WLS implementation with confidence intervals, Python/WASM bindings
- **LOESS point fitting** - `fit_at_point()` for single-point LOESS prediction
- **Statistics module** - Public `stats` module with mean, variance, stddev, median, quantile, correlation (Rust/Python/WASM)
- **WASM module restructure** - Split monolithic `wasm.rs` into modular `src/wasm/` directory (csv, diagnostics, loess, regression, regularized, stats, tests)
- **Enhanced benchmarks** - New `benches/linalg.rs` with SVD, matrix operations; expanded coverage

### Changed
- README - Added LOESS and WLS Rust examples, expanded diagnostic tests documentation

## [0.5.0] - 2026-01-30

### Added

#### LOESS Regression
- **LOESS (Locally Estimated Scatterplot Smoothing)** — Non-parametric regression method for fitting smooth curves through data without assuming a functional form
  - `loess_fit()` — Main LOESS fitting function with configurable options
  - `LoessOptions` — Configuration including span (smoothing parameter), degree (0=constant, 1=linear, 2=quadratic), robust iterations
  - `LoessFit` — Result struct with fitted values, residuals, diagnostics, and parameters
  - `LoessSurface` enum — Surface method (currently only `Direct` is implemented; `Interpolate` is accepted but behaves as `Direct`)
  - `tricube_weight()` — Tricube kernel weighting function
  - K-nearest neighbors search for local subsets
  - Weighted least squares polynomial fitting
  - Optional robust fitting with biweight function for outlier resistance
  - Multi-predictor support (native Rust; Python/WASM restricted to single predictor)
  - Output validates against R's `stats::loess()` with `surface = "direct"`

#### New Diagnostic Tests
- **DFBETAS** — Measures influence of each observation on each regression coefficient
  - `dfbetas_test()` — Compute DFBETAS matrix (n × p)
  - `DfbetasResult` — Includes threshold (2/√n), influential observations by coefficient, interpretation guidance
  - Efficient closed-form computation using leave-one-out formula (avoids refitting n times)
  - Output validates against R's `stats::dfbetas()`

- **DFFITS** — Measures influence of each observation on its own fitted value
  - `dffits_test()` — Compute DFFITS vector (one per observation)
  - `DffitsResult` — Includes threshold (2*√(p/n)), list of influential observations, interpretation guidance
  - Efficient computation using studentized residuals and leverage
  - Output validates against R's `stats::dffits()`

- **VIF Diagnostic Test** — Standalone VIF analysis for multicollinearity detection
  - `vif_test()` — Compute Variance Inflation Factor for all predictors
  - `VifDiagnosticResult` — Detailed VIF results with interpretation by threshold (acceptable <5, concerning 5-10, severe >10)
  - Output validates against R's `car::vif()`

#### WASM/JavaScript Bindings
- **LOESS regression** — `loess_fit()` function with span, degree, surface parameter (stored but only Direct method implemented), and robust fitting parameters
- **LOESS-specific UI** — Parameter controls (span slider, degree selector, surface method selector, robust checkbox)
- **Model comparison panel** — Save multiple models and compare side-by-side with sortable results table
- **Enhanced residuals table** — Sortable columns, observation detail panel, leverage highlighting, quick filter buttons (influential observations, outliers, reset)
- **Residuals vs Leverage chart** — New visualization for detecting high-leverage/influential points
- **Export augmented data** — Export residuals with all diagnostic metrics (Cook's D, DFFITS, DFBETAS, leverage)
- **Quick action buttons** — Filter by influential observations, outliers; save/compare models

#### Python Bindings (PyO3)
- **LOESS support** — `loess_fit(y, x, span, degree, surface, robust)` and `loess_predict()` functions (single predictor only; surface parameter accepted but only Direct method implemented)
- **New diagnostic tests** — `dfbetas()`, `dffits()`, `vif()` functions with NumPy array support
- **Updated result types** — `PyDfbetasResult`, `PyDffitsResult`, `PyVifTestResult`, `PyVifDetail`

#### Testing & Validation
- **LOESS integration tests** — `tests/loess_integration_test.rs` with comprehensive tests
- **WASM diagnostic tests** — Added tests for LOESS, DFBETAS, DFFITS, VIF in `tests/wasm/diagnostic_tests.rs`
- **Python tests** — Added LOESS and new diagnostic tests in `tests/python/test_diagnostics.py`
- **Validation runners** — R and Python scripts for running all diagnostics across 19 datasets
  - `verification/scripts/runners/run_all_diagnostics_r.R` — Runs all R reference tests
  - `verification/scripts/runners/run_all_diagnostics_python.py` — Runs all Python reference tests
- **Reference result scripts** — Individual R and Python scripts for LOESS (6 variants per dataset), DFBETAS, DFFITS, VIF

#### Documentation
- **npm and PyPI badges** — Added to README for better package visibility
- **README updates** — Added LOESS to regression methods; updated diagnostic tests table

### Changed
- **Performance test thresholds** — Relaxed from 5s to 10s for large dataset tests to accommodate CI environment variability
- **Python type documentation** — Added proper backtick formatting to type references in `src/python/types.rs`

### Fixed
- **CI configuration** — Fixed Rust test command from `--all-features` to `--features wasm` to avoid Python feature conflicts
- **CI Python wheel installation** — Added explicit wheel installation step before running Python tests

## [0.4.0] - 2026-01-27

### Added
- **Python bindings via PyO3** — Full Python API matching WASM and Rust functionality
- **NumPy array support** — Efficient data passing between Python and Rust
- **Type stubs** — `linreg_core.pyi` for IDE autocomplete and type checking
- **Descriptive statistics** — `stats_mean`, `stats_variance`, `stats_stddev`, `stats_median`, `stats_quantile`, `stats_correlation`
- **CSV parsing utility** — `parse_csv()` for WASM and Python
- **Model selection criteria** — `log_likelihood`, `aic`, `bic` for OLS, Ridge, Lasso, and Elastic Net (R convention)

### Changed
- Streamlined and reorganized README with clearer structure
- Improved diagnostic tests documentation

### Fixed
- **Harvey-Collier test** — Fixed numerical stability issues on high-VIF datasets

### Priority Items
- **Performance** - We really need to improve native rust/python performance. We will get ~10x with FAER in theory and same with SIMD FAER for WASM, but for now this feels like running the slow DBSCAN MATLAB implementation.

## [0.3.0] - 2026-01-26

### Added

#### Elastic Net Regression
- **Elastic Net regression** - Combined L1 (Lasso) and L2 (Ridge) penalties via cyclical coordinate descent
  - `elastic_net_fit()` - Fit elastic net for single (lambda, alpha) pair
  - `elastic_net_path()` - Fit entire regularization path efficiently
  - `ElasticNetOptions` - Configuration with lambda, alpha, standardize, intercept, max_iter, tol
  - `ElasticNetFit` - Result struct with coefficients, convergence info, nonzero count, r_squared
  - `soft_threshold()` - Soft-thresholding operator (public API)
  - Supports penalty factors, observation weights, warm starts, and coefficient bounds
  - Follows glmnet conventions for objective function and standardization

#### Lambda Path Generation
- **Lambda path utilities** for regularization path analysis
  - `make_lambda_path()` - Generate logarithmically-spaced lambda sequences
  - `compute_lambda_max()` - Find maximum lambda (where all penalized coefficients are zero)
  - `LambdaPathOptions` - Configuration (nlambda, lambda_min_ratio, alpha)
  - Automatic lambda_min_ratio selection based on sample size

#### Diagnostic Tests
- **Breusch-Godfrey test** - LM test for higher-order serial correlation
  - `breusch_godfrey_test()` - Test autocorrelation at any lag order
  - `BGTestType` enum - Chi-squared (asymptotic) or F (finite sample) test variants
  - `BreuschGodfreyResult` - Complete result with statistic, p-value, degrees of freedom
  - Validates against R's `lmtest::bgtest()`
  - Supports any lag order (not just first-order like Durbin-Watson)

- **RESET test** - Ramsey's Regression Specification Error Test
  - `reset_test()` - Test for omitted variables or incorrect functional form
  - `ResetType` enum - Three variants: Fitted, Regressor, PrincipalComponent
  - Tests if powers of fitted values/regressors/PC significantly improve model fit
  - Validates against R's `lmtest::resettest()`

#### WASM Bindings
- `elastic_net_regression()` - Elastic net regression in browser
- `make_lambda_path()` - Lambda sequence generation in browser
- `reset_test()` - RESET test with power/type configuration
- `breusch_godfrey_test()` - Higher-order autocorrelation test with order/test type
- All new WASM functions include domain checking and JSON error handling

#### Validation Tests
- **glmnet algorithm tests** - Verify Elastic Net matches R's glmnet package
  - `tests/unit/glmnet_algorithm_tests.rs` - 20+ tests for coordinate descent behavior
  - Tests soft-thresholding, coordinate descent updates, active set convergence
  - Verifies penalty factors, observation weights, warm starts, coefficient bounds

- **Elastic Net validation** - `tests/validation/elastic_net.rs`
  - Validates against R's `glmnet()` with alpha=0.5
  - 19+ datasets tested (standard and synthetic)

- **Breusch-Godfrey validation** - `tests/validation/breusch_godfrey.rs`
  - Validates against R's `lmtest::bgtest()`
  - Both Chi-squared and F test variants

- **RESET validation** - `tests/validation/reset.rs`
  - Validates against R's `lmtest::resettest()`
  - All three test types (fitted, regressor, princomp)

#### WASM-Specific Tests
- **WASM test suite** - `tests/wasm/` module with comprehensive browser testing
  - `mod.rs` - WASM test module organization
  - `fixtures.rs` - Shared test fixtures and utilities
  - `diagnostic_tests.rs` - Diagnostic tests in WASM environment (17.6 KB)
  - `integration_tests.rs` - End-to-end integration tests
  - `ols_tests.rs` - OLS regression WASM tests
  - `regularized_tests.rs` - Ridge/Lasso/Elastic Net WASM tests (10.9 KB)
  - `utility_tests.rs` - Utility function tests for WASM

#### Ridge-Specific Tests
- **Ridge test suite** - `tests/ridge_modules/` module for Ridge regression validation
  - `mod.rs` - Ridge test module organization
  - `baseline.rs` - Baseline Ridge regression tests (11.9 KB)
  - `glmnet_audit.rs` - Ridge vs glmnet compatibility tests (12.7 KB)
  - `verification.rs` - Reference result verification (8.3 KB)
- `tests/ridge.rs` - Standalone Ridge regression test file

#### Verification Reference Results
- Complete R reference outputs for all new tests across 19+ datasets
- Python reference outputs where applicable
- New datasets added: ToothGrowth, cars_stopping, faithful, lh

### Changed
- **lib.rs**: Added comprehensive diagnostic tests table to module documentation
- **regularized/mod.rs**: Expanded module docs with elastic net objective function
- **diagnostics/mod.rs**: Added Breusch-Godfrey and RESET to available tests documentation
- Improved public API re-exports for easier imports

### Fixed
- Fixed rustdoc warnings in `preprocess.rs` - escaped brackets in documentation comments
- All diagnostic test modules now properly re-exported from `diagnostics` module

## [0.2.3] - 2025-01-22

### Fixed
- Applied `cargo fmt` across the codebase (9000+ lines) for consistent code style
- Fixed clippy warnings: `assign_op_pattern` (use `*=` operator), `manual_memcpy` (use `copy_from_slice`)
- Added function-level `#[allow(clippy::needless_range_loop)]` for numerical code clarity (index-based loops preferred for math algorithms)
- Added `#[allow(clippy::too_many_arguments)]` to `coordinate_descent` (internal algorithm function)

## [0.2.2] - 2025-01-22

### Changed
- Code formatting: Applied `cargo fmt` across the codebase

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
