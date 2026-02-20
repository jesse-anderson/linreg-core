/* tslint:disable */
/* eslint-disable */

/**
 * Performs the Anderson-Darling test for normality via WASM.
 *
 * The Anderson-Darling test checks whether the residuals are normally distributed
 * by comparing the empirical distribution to the expected normal distribution.
 * This test is particularly sensitive to deviations in the tails of the distribution.
 * A significant p-value suggests that the residuals deviate from normality.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing the A² statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function anderson_darling_test(y_json: string, x_vars_json: string): string;

/**
 * Performs the Breusch-Godfrey test for higher-order serial correlation via WASM.
 *
 * Unlike the Durbin-Watson test which only detects first-order autocorrelation,
 * the Breusch-Godfrey test can detect serial correlation at any lag order.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `order` - Maximum order of serial correlation to test (default: 1)
 * * `test_type` - Type of test statistic: "chisq" or "f" (default: "chisq")
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, degrees of freedom, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function breusch_godfrey_test(y_json: string, x_vars_json: string, order: number, test_type: string): string;

/**
 * Performs the Breusch-Pagan test for heteroscedasticity via WASM.
 *
 * The Breusch-Pagan test checks whether the variance of residuals is constant
 * across the range of predicted values (homoscedasticity assumption).
 * A significant p-value suggests heteroscedasticity.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function breusch_pagan_test(y_json: string, x_vars_json: string): string;

/**
 * Computes Cook's distance for identifying influential observations via WASM.
 *
 * Cook's distance measures how much each observation influences the regression
 * model by comparing coefficient estimates with and without that observation.
 * Unlike hypothesis tests, this is an influence measure - not a test with p-values.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing:
 * - Vector of Cook's distances (one per observation)
 * - Thresholds for identifying influential observations
 * - Indices of potentially influential observations
 * - Interpretation and guidance
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function cooks_distance_test(y_json: string, x_vars_json: string): string;

/**
 * Deserialize a serialized model, extracting the inner model data.
 *
 * This function takes a serialized model JSON (as created by serialize_model),
 * validates the format version, and returns the inner model data as JSON.
 *
 * # Arguments
 *
 * * `json_string` - JSON string of the serialized model (with metadata wrapper)
 *
 * # Returns
 *
 * JSON string of the inner model data (coefficients, statistics, etc.),
 * or a JSON error object if the input is invalid, the format version is
 * incompatible, or the domain check fails.
 *
 * # Example
 *
 * ```javascript
 * import { deserialize_model } from './linreg_core.js';
 *
 * // Load from file (browser-side)
 * const response = await fetch('my_model.json');
 * const serialized = await response.text();
 *
 * // Deserialize to get the model data
 * const modelJson = deserialize_model(serialized);
 * const model = JSON.parse(modelJson);
 *
 * console.log(model.coefficients);
 * console.log(model.r_squared);
 * ```
 */
export function deserialize_model(json_string: string): string;

/**
 * Performs DFBETAS analysis via WASM.
 *
 * DFBETAS measures the influence of each observation on each regression coefficient.
 * For each observation and each coefficient, it computes the standardized change
 * in the coefficient when that observation is omitted.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing the DFBETAS matrix, threshold, and influential observations.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function dfbetas_test(y_json: string, x_vars_json: string): string;

/**
 * Performs DFFITS analysis via WASM.
 *
 * DFFITS measures the influence of each observation on its own fitted value.
 * It is the standardized change in the fitted value when that observation
 * is omitted from the model.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing the DFFITS vector, threshold, and influential observations.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function dffits_test(y_json: string, x_vars_json: string): string;

/**
 * Performs the Durbin-Watson test for autocorrelation via WASM.
 *
 * The Durbin-Watson test checks for autocorrelation in the residuals.
 * Values near 2 indicate no autocorrelation, values near 0 suggest positive
 * autocorrelation, and values near 4 suggest negative autocorrelation.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing the DW statistic and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function durbin_watson_test(y_json: string, x_vars_json: string): string;

/**
 * Fits an elastic net regularization path via WASM (Optimized).
 *
 * Computes the coefficient path for a sequence of lambda values.
 * Returns a lightweight summary to avoid excessive JSON serialization overhead.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `n_lambda` - Number of lambda values (default: 100)
 * * `lambda_min_ratio` - Ratio for smallest lambda
 * * `alpha` - Mixing parameter (0 = Ridge, 1 = Lasso)
 * * `standardize` - Whether to standardize predictors
 * * `max_iter` - Maximum iterations per lambda
 * * `tol` - Convergence tolerance
 *
 * # Returns
 *
 * JSON string containing `PathResult` (lambdas, coefficients, stats).
 */
export function elastic_net_path_wasm(y_json: string, x_vars_json: string, n_lambda: number, lambda_min_ratio: number, alpha: number, standardize: boolean, max_iter: number, tol: number): string;

/**
 * Computes approximate Elastic Net regression prediction intervals via WASM.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays (training data)
 * * `new_x_json` - JSON array of predictor arrays (new observations)
 * * `alpha` - Significance level (e.g., 0.05 for 95% PI)
 * * `lambda` - Regularization strength
 * * `enet_alpha` - Elastic net mixing parameter (0 = Ridge, 1 = Lasso)
 * * `standardize` - Whether to standardize predictors
 * * `max_iter` - Maximum coordinate descent iterations
 * * `tol` - Convergence tolerance
 */
export function elastic_net_prediction_intervals(y_json: string, x_vars_json: string, new_x_json: string, alpha: number, lambda: number, enet_alpha: number, standardize: boolean, max_iter: number, tol: number): string;

/**
 * Performs Elastic Net regression via WASM.
 *
 * Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `variable_names` - JSON array of variable names
 * * `lambda` - Regularization strength (>= 0)
 * * `alpha` - Elastic net mixing parameter (0 = Ridge, 1 = Lasso)
 * * `standardize` - Whether to standardize predictors (recommended: true)
 * * `max_iter` - Maximum coordinate descent iterations
 * * `tol` - Convergence tolerance
 *
 * # Returns
 *
 * JSON string containing regression results (same fields as Lasso).
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, parameters are invalid,
 * or domain check fails.
 */
export function elastic_net_regression(y_json: string, x_vars_json: string, _variable_names: string, lambda: number, alpha: number, standardize: boolean, max_iter: number, tol: number): string;

/**
 * Computes complete feature importance analysis for OLS regression.
 *
 * This combines standardized coefficients, SHAP values, VIF ranking,
 * and permutation importance into a single call.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response values
 * * `x_json` - JSON array of predictor arrays (each array is a column)
 * * `variable_names_json` - JSON array of variable names
 * * `y_std` - Standard deviation of response variable
 * * `n_permutations` - Number of permutation iterations
 * * `seed` - Random seed (use 0 for no seed)
 *
 * # Returns
 *
 * JSON string with all feature importance metrics
 *
 * # Example
 *
 * ```javascript
 * const y = [2.5, 3.7, 4.2, 5.1, 6.3];
 * const x = [[1,2,3,4,5], [2,4,5,4,3]];
 * const names = ["Temperature", "Pressure"];
 *
 * const result = JSON.parse(feature_importance_ols(
 *     JSON.stringify(y),
 *     JSON.stringify(x),
 *     JSON.stringify(names),
 *     2.5,   // y_std
 *     50,    // n_permutations
 *     42     // seed
 * ));
 *
 * console.log(result.standardized_coefficients);
 * console.log(result.shap);
 * console.log(result.permutation_importance);
 * console.log(result.vif_ranking);
 * ```
 */
export function feature_importance_ols(y_json: string, x_json: string, variable_names_json: string, y_std: number, n_permutations: number, seed: bigint): string;

/**
 * Extract metadata from a serialized model without deserializing the full model.
 *
 * This function returns only the metadata portion of a serialized model,
 * which includes information like model type, library version, creation time,
 * and optional model name.
 *
 * # Arguments
 *
 * * `json_string` - JSON string of the serialized model
 *
 * # Returns
 *
 * JSON string containing the metadata object with fields:
 * - `format_version` - Format version (e.g., "1.0")
 * - `library_version` - linreg-core version used to create the model
 * - `model_type` - Type of model ("OLS", "Ridge", etc.)
 * - `created_at` - ISO 8601 timestamp of creation
 * - `name` - Optional custom model name
 *
 * Returns a JSON error object if the input is invalid or the domain check fails.
 *
 * # Example
 *
 * ```javascript
 * import { get_model_metadata } from './linreg_core.js';
 *
 * const response = await fetch('my_model.json');
 * const serialized = await response.text();
 *
 * const metadataJson = get_model_metadata(serialized);
 * const metadata = JSON.parse(metadataJson);
 *
 * console.log('Model type:', metadata.model_type);
 * console.log('Created:', metadata.created_at);
 * console.log('Name:', metadata.name || '(unnamed)');
 * ```
 */
export function get_model_metadata(json_string: string): string;

/**
 * Computes the inverse of the standard normal CDF (probit function).
 *
 * Returns the z-score such that P(Z ≤ z) = p for a standard normal distribution.
 *
 * # Arguments
 *
 * * `p` - Probability (0 < p < 1)
 *
 * # Returns
 *
 * The z-score, or `NaN` if domain check fails.
 */
export function get_normal_inverse(p: number): number;

/**
 * Computes the Student's t-distribution cumulative distribution function.
 *
 * Returns P(T ≤ t) for a t-distribution with the given degrees of freedom.
 *
 * # Arguments
 *
 * * `t` - t-statistic value
 * * `df` - Degrees of freedom
 *
 * # Returns
 *
 * The CDF value, or `NaN` if domain check fails.
 */
export function get_t_cdf(t: number, df: number): number;

/**
 * Computes the critical t-value for a given significance level.
 *
 * Returns the t-value such that the area under the t-distribution curve
 * to the right equals alpha/2 (two-tailed test).
 *
 * # Arguments
 *
 * * `alpha` - Significance level (typically 0.05 for 95% confidence)
 * * `df` - Degrees of freedom
 *
 * # Returns
 *
 * The critical t-value, or `NaN` if domain check fails.
 */
export function get_t_critical(alpha: number, df: number): number;

/**
 * Returns the current version of the library.
 *
 * Returns the Cargo package version as a string (e.g., "0.1.0").
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function get_version(): string;

/**
 * Performs the Harvey-Collier test for linearity via WASM.
 *
 * The Harvey-Collier test checks whether the residuals exhibit a linear trend,
 * which would indicate that the model's functional form is misspecified.
 * A significant p-value suggests non-linearity.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function harvey_collier_test(y_json: string, x_vars_json: string): string;

/**
 * Performs the Jarque-Bera test for normality via WASM.
 *
 * The Jarque-Bera test checks whether the residuals are normally distributed
 * by examining skewness and kurtosis. A significant p-value suggests that
 * the residuals deviate from normality.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function jarque_bera_test(y_json: string, x_vars_json: string): string;

/**
 * Performs K-Fold Cross Validation for Elastic Net regression via WASM.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `lambda` - Regularization strength (>= 0)
 * * `alpha` - Mixing parameter (0 = Ridge, 1 = Lasso)
 * * `standardize` - Whether to standardize predictors
 * * `n_folds` - Number of folds (must be >= 2)
 * * `shuffle_json` - JSON boolean: whether to shuffle data before splitting
 * * `seed_json` - JSON string with seed number or "null" for no seed
 *
 * # Returns
 *
 * JSON string containing CV results (same structure as OLS).
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, parameters are invalid,
 * or domain check fails.
 */
export function kfold_cv_elastic_net(y_json: string, x_vars_json: string, lambda: number, alpha: number, standardize: boolean, n_folds: number, shuffle_json: string, seed_json: string): string;

/**
 * Performs K-Fold Cross Validation for Lasso regression via WASM.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `lambda` - Regularization strength (>= 0)
 * * `standardize` - Whether to standardize predictors
 * * `n_folds` - Number of folds (must be >= 2)
 * * `shuffle_json` - JSON boolean: whether to shuffle data before splitting
 * * `seed_json` - JSON string with seed number or "null" for no seed
 *
 * # Returns
 *
 * JSON string containing CV results (same structure as OLS).
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, parameters are invalid,
 * or domain check fails.
 */
export function kfold_cv_lasso(y_json: string, x_vars_json: string, lambda: number, standardize: boolean, n_folds: number, shuffle_json: string, seed_json: string): string;

/**
 * Performs K-Fold Cross Validation for OLS regression via WASM.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `variable_names_json` - JSON array of variable names
 * * `n_folds` - Number of folds (must be >= 2)
 * * `shuffle_json` - JSON boolean: whether to shuffle data before splitting
 * * `seed_json` - JSON string with seed number or "null" for no seed
 *
 * # Returns
 *
 * JSON string containing CV results:
 * - `n_folds` - Number of folds used
 * - `n_samples` - Total number of observations
 * - `mean_mse`, `std_mse` - Mean and std of MSE across folds
 * - `mean_rmse`, `std_rmse` - Mean and std of RMSE across folds
 * - `mean_mae`, `std_mae` - Mean and std of MAE across folds
 * - `mean_r_squared`, `std_r_squared` - Mean and std of R² across folds
 * - `fold_results` - Array of individual fold results
 * - `fold_coefficients` - Array of coefficient arrays from each fold
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, parameters are invalid,
 * or domain check fails.
 */
export function kfold_cv_ols(y_json: string, x_vars_json: string, variable_names_json: string, n_folds: number, shuffle_json: string, seed_json: string): string;

/**
 * Performs K-Fold Cross Validation for Ridge regression via WASM.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `lambda` - Regularization strength (>= 0)
 * * `standardize` - Whether to standardize predictors
 * * `n_folds` - Number of folds (must be >= 2)
 * * `shuffle_json` - JSON boolean: whether to shuffle data before splitting
 * * `seed_json` - JSON string with seed number or "null" for no seed
 *
 * # Returns
 *
 * JSON string containing CV results (same structure as OLS).
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, parameters are invalid,
 * or domain check fails.
 */
export function kfold_cv_ridge(y_json: string, x_vars_json: string, lambda: number, standardize: boolean, n_folds: number, shuffle_json: string, seed_json: string): string;

/**
 * Computes approximate Lasso regression prediction intervals via WASM.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays (training data)
 * * `new_x_json` - JSON array of predictor arrays (new observations)
 * * `alpha` - Significance level (e.g., 0.05 for 95% PI)
 * * `lambda` - Regularization strength
 * * `standardize` - Whether to standardize predictors
 * * `max_iter` - Maximum coordinate descent iterations
 * * `tol` - Convergence tolerance
 */
export function lasso_prediction_intervals(y_json: string, x_vars_json: string, new_x_json: string, alpha: number, lambda: number, standardize: boolean, max_iter: number, tol: number): string;

/**
 * Performs Lasso regression via WASM.
 *
 * Lasso regression adds an L1 penalty to the coefficients, which performs
 * automatic variable selection by shrinking some coefficients to exactly zero.
 * The intercept is never penalized.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `variable_names` - JSON array of variable names
 * * `lambda` - Regularization strength (>= 0, typical range 0.01 to 10)
 * * `standardize` - Whether to standardize predictors (recommended: true)
 * * `max_iter` - Maximum coordinate descent iterations (default: 100000)
 * * `tol` - Convergence tolerance (default: 1e-7)
 *
 * # Returns
 *
 * JSON string containing:
 * - `lambda` - Lambda value used
 * - `intercept` - Intercept coefficient
 * - `coefficients` - Slope coefficients (some may be exactly zero)
 * - `fitted_values` - Predictions on training data
 * - `residuals` - Residuals (y - fitted_values)
 * - `n_nonzero` - Number of non-zero coefficients (excluding intercept)
 * - `iterations` - Number of coordinate descent iterations
 * - `converged` - Whether the algorithm converged
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, lambda is negative,
 * or domain check fails.
 */
export function lasso_regression(y_json: string, x_vars_json: string, _variable_names: string, lambda: number, standardize: boolean, max_iter: number, tol: number): string;

/**
 * Performs LOESS regression via WASM.
 *
 * LOESS (Locally Estimated Scatterplot Smoothing) is a non-parametric
 * regression method that fits multiple regressions in local subsets
 * of data to create a smooth curve through the data points.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `span` - Fraction of data used in each local fit (0.0 to 1.0)
 * * `degree` - Degree of local polynomial: 0 (constant), 1 (linear), or 2 (quadratic)
 * * `robust_iterations` - Number of robustness iterations (0 for non-robust fit)
 * * `surface` - Surface computation method: "direct" or "interpolate"
 *
 * # Returns
 *
 * JSON string containing:
 * - `fitted` - Fitted values at each observation point
 * - `span` - Span parameter used
 * - `degree` - Degree of polynomial used
 * - `robust_iterations` - Number of robustness iterations performed
 * - `surface` - Surface computation method used
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function loess_fit(y_json: string, x_vars_json: string, span: number, degree: number, robust_iterations: number, surface: string): string;

/**
 * Performs LOESS prediction at new query points via WASM.
 *
 * Predicts LOESS fitted values at arbitrary new points by redoing the
 * local fitting at each query point using the original training data.
 *
 * # Arguments
 *
 * * `new_x_json` - JSON array of new predictor values (p vectors, each of length m)
 * * `original_x_json` - JSON array of original training predictors
 * * `original_y_json` - JSON array of original training response values
 * * `span` - Span parameter (must match the original fit)
 * * `degree` - Degree of polynomial (must match the original fit)
 * * `robust_iterations` - Robustness iterations (must match the original fit)
 * * `surface` - Surface computation method: "direct" or "interpolate"
 *
 * # Returns
 *
 * JSON string containing:
 * - `predictions` - Vector of predicted values at query points
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, parameters don't match
 * the original fit, or domain check fails.
 */
export function loess_predict(new_x_json: string, original_x_json: string, original_y_json: string, span: number, degree: number, robust_iterations: number, surface: string): string;

/**
 * Generates a lambda path for regularized regression via WASM.
 *
 * Creates a logarithmically-spaced sequence of lambda values from lambda_max
 * (where all penalized coefficients are zero) down to lambda_min. This is
 * useful for fitting regularization paths and selecting optimal lambda via
 * cross-validation.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `n_lambda` - Number of lambda values to generate (default: 100)
 * * `lambda_min_ratio` - Ratio for smallest lambda (default: 0.0001 if n >= p, else 0.01)
 *
 * # Returns
 *
 * JSON string containing:
 * - `lambda_path` - Array of lambda values in decreasing order
 * - `lambda_max` - Maximum lambda value
 * - `lambda_min` - Minimum lambda value
 * - `n_lambda` - Number of lambda values
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function make_lambda_path(y_json: string, x_vars_json: string, n_lambda: number, lambda_min_ratio: number): string;

/**
 * Computes OLS prediction intervals via WASM.
 *
 * Fits an OLS model to the training data and computes prediction intervals
 * for the new observations.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays (training data)
 * * `new_x_json` - JSON array of predictor arrays (new observations)
 * * `alpha` - Significance level (e.g., 0.05 for 95% PI)
 *
 * # Returns
 *
 * JSON string containing predicted values, lower/upper bounds, SE, leverage.
 */
export function ols_prediction_intervals(y_json: string, x_vars_json: string, new_x_json: string, alpha: number): string;

/**
 * Performs OLS regression via WASM.
 *
 * All parameters and return values are JSON-encoded strings for JavaScript
 * interoperability. Returns regression output including coefficients,
 * standard errors, diagnostic statistics, and VIF analysis.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values: `[1.0, 2.0, 3.0]`
 * * `x_vars_json` - JSON array of predictor arrays: `[[1.0, 2.0], [0.5, 1.0]]`
 * * `variable_names` - JSON array of variable names: `["Intercept", "X1", "X2"]`
 *
 * # Returns
 *
 * JSON string containing the complete regression output with coefficients,
 * standard errors, t-statistics, p-values, R², F-statistic, residuals, leverage, VIF, etc.
 *
 * # Errors
 *
 * Returns a JSON error object if:
 * - JSON parsing fails
 * - Insufficient data (n ≤ k + 1)
 * - Matrix is singular
 * - Domain check fails
 */
export function ols_regression(y_json: string, x_vars_json: string, variable_names: string): string;

/**
 * Parses CSV data and returns it as a JSON string.
 *
 * Parses the CSV content and identifies numeric columns. Returns a JSON object
 * with headers, data rows, and a list of numeric column names.
 *
 * # Arguments
 *
 * * `content` - CSV content as a string
 *
 * # Returns
 *
 * JSON string with structure:
 * ```json
 * {
 *   "headers": ["col1", "col2", ...],
 *   "data": [{"col1": 1.0, "col2": "text"}, ...],
 *   "numeric_columns": ["col1", ...]
 * }
 * ```
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function parse_csv(content: string): string;

/**
 * Computes permutation importance for OLS regression.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response values
 * * `x_json` - JSON array of predictor arrays (each array is a column)
 * * `fit_json` - JSON string of OLS fit result
 * * `n_permutations` - Number of permutation iterations
 * * `seed` - Random seed (use 0 for no seed)
 *
 * # Returns
 *
 * JSON string of [`PermutationImportanceOutput`]
 *
 * # Example
 *
 * ```javascript
 * const y = [2.5, 3.7, 4.2, 5.1, 6.3];
 * const x = [[1,2,3,4,5], [2,4,5,4,3]];
 * const fit = JSON.parse(ols_regression(...)); // from regression module
 *
 * const result = JSON.parse(permutation_importance_ols(
 *     JSON.stringify(y),
 *     JSON.stringify(x),
 *     JSON.stringify(fit),
 *     50,  // n_permutations
 *     42   // seed
 * ));
 * console.log(result.importance);
 * ```
 */
export function permutation_importance_ols(y_json: string, x_json: string, fit_json: string, n_permutations: number, seed: bigint): string;

/**
 * Fit polynomial Elastic Net regression via WASM.
 *
 * Elastic Net combines L1 and L2 penalties, balancing variable selection
 * with multicollinearity handling.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response values, e.g. `[1.0, 4.0, 9.0]`
 * * `x_json` - JSON array of predictor values, e.g. `[1.0, 2.0, 3.0]`
 * * `degree` - Polynomial degree (≥ 1)
 * * `lambda` - Regularization strength (≥ 0)
 * * `alpha` - Mixing parameter: 0 = Ridge, 1 = Lasso
 * * `center` - Whether to center x before expansion (reduces multicollinearity)
 * * `standardize` - Whether to standardize features (recommended)
 *
 * # Returns
 *
 * JSON string of the [`ElasticNetFit`] result, which includes:
 * - `intercept`, `coefficients`
 * - `fitted_values`, `residuals`
 * - `r_squared`, `adj_r_squared`, `mse`, `rmse`, `mae`
 * - `n_nonzero`, `converged`, `n_iterations`
 * - `log_likelihood`, `aic`, `bic`
 */
export function polynomial_elastic_net_wasm(y_json: string, x_json: string, degree: number, lambda: number, alpha: number, center: boolean, standardize: boolean): string;

/**
 * Fit polynomial Lasso regression via WASM.
 *
 * Lasso can perform variable selection among polynomial terms,
 * potentially eliminating higher-order terms.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response values, e.g. `[1.0, 4.0, 9.0]`
 * * `x_json` - JSON array of predictor values, e.g. `[1.0, 2.0, 3.0]`
 * * `degree` - Polynomial degree (≥ 1)
 * * `lambda` - Regularization strength (≥ 0)
 * * `center` - Whether to center x before expansion (reduces multicollinearity)
 * * `standardize` - Whether to standardize features (recommended)
 *
 * # Returns
 *
 * JSON string of the [`LassoFit`] result, which includes:
 * - `intercept`, `coefficients`
 * - `fitted_values`, `residuals`
 * - `r_squared`, `adj_r_squared`, `mse`, `rmse`, `mae`
 * - `n_nonzero`, `converged`, `n_iterations`
 * - `log_likelihood`, `aic`, `bic`
 */
export function polynomial_lasso_wasm(y_json: string, x_json: string, degree: number, lambda: number, center: boolean, standardize: boolean): string;

/**
 * Predict using a fitted polynomial model via WASM.
 *
 * # Arguments
 *
 * * `fit_json` - JSON string of the `PolynomialFit` returned by [`polynomial_regression_wasm`]
 * * `x_new_json` - JSON array of new predictor values, e.g. `[6.0, 7.0]`
 *
 * # Returns
 *
 * JSON array of predicted values, or a JSON error object on failure.
 */
export function polynomial_predict_wasm(fit_json: string, x_new_json: string): string;

/**
 * Fit polynomial regression via WASM.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response values, e.g. `[1.0, 4.0, 9.0]`
 * * `x_json` - JSON array of predictor values, e.g. `[1.0, 2.0, 3.0]`
 * * `degree` - Polynomial degree (≥ 1)
 * * `center` - Whether to center x before expanding (reduces multicollinearity)
 * * `standardize` - Whether to standardize polynomial features
 *
 * # Returns
 *
 * JSON string of the complete [`PolynomialFit`], which includes:
 * - `ols_output` — full OLS regression output (coefficients, R², F-stat, etc.)
 * - `degree`, `centered`, `x_mean`, `x_std`, `standardized`
 * - `feature_names`, `feature_means`, `feature_stds`
 *
 * The returned JSON can be passed directly to [`polynomial_predict_wasm`].
 *
 * # Errors
 *
 * Returns a JSON error object `{"error": "…"}` if:
 * - JSON parsing fails
 * - `degree` is 0
 * - `y` and `x` have different lengths
 * - Insufficient data
 * - Domain check fails
 */
export function polynomial_regression_wasm(y_json: string, x_json: string, degree: number, center: boolean, standardize: boolean): string;

/**
 * Fit polynomial Ridge regression via WASM.
 *
 * Ridge regularization helps with multicollinearity in polynomial features.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response values, e.g. `[1.0, 4.0, 9.0]`
 * * `x_json` - JSON array of predictor values, e.g. `[1.0, 2.0, 3.0]`
 * * `degree` - Polynomial degree (≥ 1)
 * * `lambda` - Regularization strength (≥ 0)
 * * `center` - Whether to center x before expansion (reduces multicollinearity)
 * * `standardize` - Whether to standardize features (recommended)
 *
 * # Returns
 *
 * JSON string of the [`RidgeFit`] result, which includes:
 * - `intercept`, `coefficients`
 * - `fitted_values`, `residuals`
 * - `r_squared`, `adj_r_squared`, `mse`, `rmse`, `mae`
 * - `effective_df`, `log_likelihood`, `aic`, `bic`
 */
export function polynomial_ridge_wasm(y_json: string, x_json: string, degree: number, lambda: number, center: boolean, standardize: boolean): string;

/**
 * Performs the Python method White test for heteroscedasticity via WASM.
 *
 * This implementation matches Python's `statsmodels.stats.diagnostic.het_white()` function.
 * Uses the LINPACK QR decomposition with column pivoting and the Python-specific
 * auxiliary matrix structure (intercept, X, X², and cross-products).
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays (each array is a column)
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function python_white_test(y_json: string, x_vars_json: string): string;

/**
 * Performs the R method White test for heteroscedasticity via WASM.
 *
 * This implementation matches R's `skedastic::white()` function behavior.
 * Uses the standard QR decomposition and the R-specific auxiliary matrix
 * structure (intercept, X, X² only - no cross-products).
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays (each array is a column)
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function r_white_test(y_json: string, x_vars_json: string): string;

/**
 * Performs the Rainbow test for linearity via WASM.
 *
 * The Rainbow test checks whether the relationship between predictors and response
 * is linear. A significant p-value suggests non-linearity.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `fraction` - Fraction of data to use in the central subset (0.0 to 1.0, typically 0.5)
 * * `method` - Method to use: "r", "python", or "both" (case-insensitive, defaults to "r")
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function rainbow_test(y_json: string, x_vars_json: string, fraction: number, method: string): string;

/**
 * Performs the RESET test for model specification error via WASM.
 *
 * The RESET (Regression Specification Error Test) test checks whether the model
 * is correctly specified by testing if additional terms (powers of fitted values,
 * regressors, or first principal component) significantly improve the model fit.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `powers_json` - JSON array of powers to use (e.g., [2, 3] for ŷ², ŷ³)
 * * `type_` - Type of terms to add: "fitted", "regressor", or "princomp"
 *
 * # Returns
 *
 * JSON string containing the F-statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function reset_test(y_json: string, x_vars_json: string, powers_json: string, type_: string): string;

/**
 * Computes approximate Ridge regression prediction intervals via WASM.
 *
 * Fits a Ridge model and computes conservative prediction intervals using
 * leverage from unpenalized X'X and MSE from the ridge fit.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays (training data)
 * * `new_x_json` - JSON array of predictor arrays (new observations)
 * * `alpha` - Significance level (e.g., 0.05 for 95% PI)
 * * `lambda` - Regularization strength
 * * `standardize` - Whether to standardize predictors
 */
export function ridge_prediction_intervals(y_json: string, x_vars_json: string, new_x_json: string, alpha: number, lambda: number, standardize: boolean): string;

/**
 * Performs Ridge regression via WASM.
 *
 * Ridge regression adds an L2 penalty to the coefficients, which helps with
 * multicollinearity and overfitting. The intercept is never penalized.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `variable_names` - JSON array of variable names
 * * `lambda` - Regularization strength (>= 0, typical range 0.01 to 100)
 * * `standardize` - Whether to standardize predictors (recommended: true)
 *
 * # Returns
 *
 * JSON string containing:
 * - `lambda` - Lambda value used
 * - `intercept` - Intercept coefficient
 * - `coefficients` - Slope coefficients
 * - `fitted_values` - Predictions on training data
 * - `residuals` - Residuals (y - fitted_values)
 * - `df` - Effective degrees of freedom
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, lambda is negative,
 * or domain check fails.
 */
export function ridge_regression(y_json: string, x_vars_json: string, _variable_names: string, lambda: number, standardize: boolean): string;

/**
 * Serialize a model by wrapping its JSON data with metadata.
 *
 * This function takes a model's JSON representation (as returned by regression
 * functions), wraps it with version and type metadata, and returns a serialized
 * JSON string suitable for storage or download.
 *
 * # Arguments
 *
 * * `model_json` - JSON string of the model result (e.g., from ols_regression)
 * * `model_type` - Type of model: "OLS", "Ridge", "Lasso", "ElasticNet", "WLS", or "LOESS"
 * * `name` - Optional custom name for the model
 *
 * # Returns
 *
 * JSON string containing the serialized model with metadata, or a JSON error object
 * if the input is invalid or the domain check fails.
 *
 * # Example
 *
 * ```javascript
 * import { serialize_model, ols_regression } from './linreg_core.js';
 *
 * // Train a model
 * const resultJson = ols_regression(yJson, xJson, namesJson);
 *
 * // Serialize it
 * const serialized = serialize_model(resultJson, "OLS", "My Housing Model");
 *
 * // Download (browser-side)
 * const blob = new Blob([serialized], { type: 'application/json' });
 * const url = URL.createObjectURL(blob);
 * const a = document.createElement('a');
 * a.href = url;
 * a.download = 'my_model.json';
 * a.click();
 * ```
 */
export function serialize_model(model_json: string, model_type: string, name?: string | null): string;

/**
 * Computes SHAP (SHapley Additive exPlanations) values for linear models.
 *
 * # Arguments
 *
 * * `x_json` - JSON array of predictor arrays (each array is a column)
 * * `coefficients_json` - JSON array of coefficients including intercept
 * * `variable_names_json` - JSON array of variable names
 *
 * # Returns
 *
 * JSON string of [`ShapOutput`]
 *
 * # Example
 *
 * ```javascript
 * const result = JSON.parse(shap_values_linear(
 *     JSON.stringify([[1,2,3], [2,4,6]]),
 *     JSON.stringify([5, 2, 3]),
 *     JSON.stringify(["X1", "X2"])
 * ));
 * console.log(result.mean_abs_shap); // Global importance
 * console.log(result.shap_values); // Local contributions
 * ```
 */
export function shap_values_linear(x_json: string, coefficients_json: string, variable_names_json: string): string;

/**
 * Performs the Shapiro-Wilk test for normality via WASM.
 *
 * The Shapiro-Wilk test is a powerful test for normality,
 * especially for small to moderate sample sizes (3 ≤ n ≤ 5000). It tests
 * the null hypothesis that the residuals are normally distributed.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing the W statistic (ranges from 0 to 1), p-value,
 * and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function shapiro_wilk_test(y_json: string, x_vars_json: string): string;

/**
 * Computes standardized coefficients for feature importance.
 *
 * # Arguments
 *
 * * `x_json` - JSON array of predictor arrays (each array is a column)
 * * `coefficients_json` - JSON array of coefficients including intercept
 * * `variable_names_json` - JSON array of variable names
 * * `y_std` - Standard deviation of response variable
 *
 * # Returns
 *
 * JSON string of [`StandardizedCoefficientsOutput`]
 *
 * # Example
 *
 * ```javascript
 * const result = JSON.parse(standardized_coefficients(
 *     JSON.stringify([[1,2,3,4,5], [10,20,30,40,50]]),
 *     JSON.stringify([1, 0.5, -0.3]),
 *     JSON.stringify(["Temperature", "Pressure"]),
 *     2.5
 * ));
 * console.log(result.standardized_coefficients);
 * ```
 */
export function standardized_coefficients(x_json: string, coefficients_json: string, variable_names_json: string, y_std: number): string;

/**
 * Computes the correlation coefficient between two JSON arrays of f64 values.
 *
 * # Arguments
 *
 * * `x_json` - JSON string representing the first array of f64 values
 * * `y_json` - JSON string representing the second array of f64 values
 *
 * # Returns
 *
 * JSON string with the correlation coefficient, or "null" if input is invalid
 */
export function stats_correlation(x_json: string, y_json: string): string;

/**
 * Computes the arithmetic mean of a JSON array of f64 values.
 *
 * # Arguments
 *
 * * `data_json` - JSON string representing an array of f64 values
 *
 * # Returns
 *
 * JSON string with the mean, or "null" if input is invalid/empty
 */
export function stats_mean(data_json: string): string;

/**
 * Computes the median of a JSON array of f64 values.
 *
 * # Arguments
 *
 * * `data_json` - JSON string representing an array of f64 values
 *
 * # Returns
 *
 * JSON string with the median, or "null" if input is invalid/empty
 */
export function stats_median(data_json: string): string;

/**
 * Computes a quantile of a JSON array of f64 values.
 *
 * # Arguments
 *
 * * `data_json` - JSON string representing an array of f64 values
 * * `q` - Quantile to calculate (0.0 to 1.0)
 *
 * # Returns
 *
 * JSON string with the quantile value, or "null" if input is invalid
 */
export function stats_quantile(data_json: string, q: number): string;

/**
 * Computes the sample standard deviation of a JSON array of f64 values.
 *
 * Uses the (n-1) denominator for unbiased estimation.
 *
 * # Arguments
 *
 * * `data_json` - JSON string representing an array of f64 values
 *
 * # Returns
 *
 * JSON string with the standard deviation, or "null" if input is invalid
 */
export function stats_stddev(data_json: string): string;

/**
 * Computes the sample variance of a JSON array of f64 values.
 *
 * Uses the (n-1) denominator for unbiased estimation.
 *
 * # Arguments
 *
 * * `data_json` - JSON string representing an array of f64 values
 *
 * # Returns
 *
 * JSON string with the variance, or "null" if input is invalid
 */
export function stats_variance(data_json: string): string;

/**
 * Simple test function to verify WASM is working.
 *
 * Returns a success message confirming the WASM module loaded correctly.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function test(): string;

/**
 * Test function for confidence interval computation.
 *
 * Returns JSON with the computed confidence interval for a coefficient.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function test_ci(coef: number, se: number, df: number, alpha: number): string;

/**
 * Test function for regression validation against R reference values.
 *
 * Runs a regression on a housing dataset and compares results against R's lm() output.
 * Returns JSON with status "PASS" or "FAIL" with details.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function test_housing_regression(): string;

/**
 * Test function for R accuracy validation.
 *
 * Returns JSON comparing our statistical functions against R reference values.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function test_r_accuracy(): string;

/**
 * Test function for t-critical value computation.
 *
 * Returns JSON with the computed t-critical value for the given parameters.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function test_t_critical(df: number, alpha: number): string;

/**
 * Computes VIF (Variance Inflation Factor) ranking.
 *
 * # Arguments
 *
 * * `vif_json` - JSON array of VIF results from OLS output
 *
 * # Returns
 *
 * JSON string of [`VifRankingOutput`]
 *
 * # Example
 *
 * ```javascript
 * const fit = JSON.parse(ols_regression(...));
 * const result = JSON.parse(vif_ranking(JSON.stringify(fit.vif)));
 * console.log(result.ranking); // Sorted by VIF (lowest first)
 * ```
 */
export function vif_ranking(vif_json: string): string;

/**
 * Performs Variance Inflation Factor (VIF) analysis via WASM.
 *
 * VIF measures how much the variance of regression coefficients is inflated
 * due to multicollinearity among predictor variables. High VIF values indicate
 * that a predictor is highly correlated with other predictors.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing the maximum VIF, detailed VIF results for each predictor,
 * interpretation, and guidance.
 *
 * # Interpretation
 *
 * - VIF = 1: No correlation with other predictors
 * - VIF > 5: Moderate multicollinearity (concerning)
 * - VIF > 10: High multicollinearity (severe)
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function vif_test(y_json: string, x_vars_json: string): string;

/**
 * Performs the White test for heteroscedasticity via WASM.
 *
 * The White test is a more general test for heteroscedasticity that does not
 * assume a specific form of heteroscedasticity. A significant p-value suggests
 * that the error variance is not constant.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `method` - Method to use: "r", "python", or "both" (case-insensitive, defaults to "r")
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 */
export function white_test(y_json: string, x_vars_json: string, method: string): string;

/**
 * Performs Weighted Least Squares (WLS) regression via WASM.
 *
 * WLS regression allows each observation to have a different weight, which is
 * useful for handling heteroscedasticity or when observations have different
 * precision/variances.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `weights_json` - JSON array of observation weights (must be non-negative)
 *
 * # Returns
 *
 * JSON string containing:
 * - `coefficients` - Coefficient values (including intercept as first element)
 * - `standard_errors` - Standard errors of the coefficients
 * - `t_statistics` - t-statistics for coefficient significance tests
 * - `p_values` - Two-tailed p-values for coefficients
 * - `r_squared` - R-squared (coefficient of determination)
 * - `adj_r_squared` - Adjusted R-squared
 * - `f_statistic` - F-statistic for overall model significance
 * - `f_p_value` - p-value for F-statistic
 * - `residual_std_error` - Residual standard error (sigma-hat estimate)
 * - `df_residuals` - Degrees of freedom for residuals
 * - `df_model` - Degrees of freedom for the model
 * - `fitted_values` - Fitted values (predicted values)
 * - `residuals` - Residuals (y - ŷ)
 * - `mse` - Mean squared error
 * - `rmse` - Root mean squared error
 * - `mae` - Mean absolute error
 * - `n` - Number of observations
 * - `k` - Number of predictors (excluding intercept)
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, parameters are invalid,
 * or domain check fails.
 */
export function wls_regression(y_json: string, x_vars_json: string, weights_json: string): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly anderson_darling_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly breusch_godfrey_test: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly breusch_pagan_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly cooks_distance_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly deserialize_model: (a: number, b: number) => [number, number];
    readonly dfbetas_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly dffits_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly durbin_watson_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly elastic_net_path_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
    readonly elastic_net_prediction_intervals: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => [number, number];
    readonly elastic_net_regression: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly feature_importance_ols: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: bigint) => [number, number];
    readonly get_model_metadata: (a: number, b: number) => [number, number];
    readonly get_version: () => [number, number];
    readonly harvey_collier_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly jarque_bera_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly kfold_cv_elastic_net: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => [number, number];
    readonly kfold_cv_lasso: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly kfold_cv_ols: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly kfold_cv_ridge: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly lasso_prediction_intervals: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly lasso_regression: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
    readonly loess_fit: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly loess_predict: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly make_lambda_path: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly ols_prediction_intervals: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly ols_regression: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly parse_csv: (a: number, b: number) => [number, number];
    readonly permutation_importance_ols: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: bigint) => [number, number];
    readonly polynomial_elastic_net_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly polynomial_lasso_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly polynomial_predict_wasm: (a: number, b: number, c: number, d: number) => [number, number];
    readonly polynomial_regression_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly polynomial_ridge_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly python_white_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly r_white_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly rainbow_test: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly reset_test: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly ridge_prediction_intervals: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly ridge_regression: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly serialize_model: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly shap_values_linear: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly shapiro_wilk_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly standardized_coefficients: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly stats_correlation: (a: number, b: number, c: number, d: number) => [number, number];
    readonly stats_mean: (a: number, b: number) => [number, number];
    readonly stats_median: (a: number, b: number) => [number, number];
    readonly stats_quantile: (a: number, b: number, c: number) => [number, number];
    readonly stats_stddev: (a: number, b: number) => [number, number];
    readonly stats_variance: (a: number, b: number) => [number, number];
    readonly test: () => [number, number];
    readonly test_ci: (a: number, b: number, c: number, d: number) => [number, number];
    readonly test_housing_regression: () => [number, number];
    readonly test_r_accuracy: () => [number, number];
    readonly test_t_critical: (a: number, b: number) => [number, number];
    readonly vif_ranking: (a: number, b: number) => [number, number];
    readonly vif_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly white_test: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly wls_regression: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly get_t_cdf: (a: number, b: number) => number;
    readonly get_t_critical: (a: number, b: number) => number;
    readonly get_normal_inverse: (a: number) => number;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
